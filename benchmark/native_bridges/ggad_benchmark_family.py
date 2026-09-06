"""Canonical bridges for the GGAD benchmark AEGIS and GAAN implementations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .common import (
    git_head,
    load_label_free_bundle,
    load_score_nodes,
    seed_everything,
    sha256,
    write_evidence,
    write_scores,
)


UPSTREAM_COMMIT = "358fb4d4b4ee5b8b195445791a9e2b6d52487f2b"
EPOCHS = {
    "AEGIS": {"Amazon": 800, "T-Finance": 1500},
    "GAAN": {"Amazon": 800, "T-Finance": 1500, "Elliptic": 600},
}


def _git(source: Path, *args: str) -> subprocess.CompletedProcess[str]:
    executable = "/data1/anaconda3/bin/git" if Path("/data1/anaconda3/bin/git").is_file() else "git"
    return subprocess.run([executable, "-C", str(source), *args], check=False, capture_output=True, text=True)


def _import_upstream(source: Path, method: str):
    if _git(source, "merge-base", "--is-ancestor", UPSTREAM_COMMIT, "HEAD").returncode:
        raise RuntimeError("GGAD benchmark checkout does not contain the frozen source commit")
    sys.path.insert(0, str(source))
    try:
        if method == "AEGIS":
            from model_AEGIS import Model  # type: ignore

            module = sys.modules["model_AEGIS"]
        else:
            from model_gaan import Model  # type: ignore

            module = sys.modules["model_gaan"]
        from utils import normalize_adj  # type: ignore
    finally:
        sys.path.pop(0)
    return Model, module, normalize_adj


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device.index or 0)
    return device


def _adjacency(edge_index: torch.Tensor, num_nodes: int, normalize_adj) -> torch.Tensor:
    rows, columns = edge_index.numpy()
    adjacency = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, columns)), shape=(num_nodes, num_nodes))
    adjacency.data[:] = 1.0
    normalized = normalize_adj(adjacency) + sp.eye(num_nodes, dtype=np.float32, format="coo")
    return torch.from_numpy(normalized.toarray().astype(np.float32, copy=False)).unsqueeze(0)


def _vectorized_edge_list(adjacency: torch.Tensor, nodes: list[int]) -> np.ndarray:
    node_tensor = torch.as_tensor(nodes, dtype=torch.long, device=adjacency.device)
    positions = torch.nonzero(adjacency[node_tensor] > 0, as_tuple=False)
    rows = node_tensor[positions[:, 0]]
    return torch.stack([rows, positions[:, 1]], dim=1).cpu().numpy()


def preflight(bundle_path: str | Path, method: str) -> dict[str, object]:
    bundle = load_label_free_bundle(bundle_path)
    n = int(bundle["x"].shape[0])
    dense_bytes = n * n * 4
    return {
        "method": method,
        "num_nodes": n,
        "num_edges": int(bundle["edge_index"].shape[1]),
        "dense_n_by_n_path": True,
        "pairwise_discriminator_or_reconstruction": True,
        "minimum_named_dense_bytes": 4 * dense_bytes,
        "preflight_status": "NOT_RUN_RESOURCE_RISK" if 4 * dense_bytes >= 14 * 1024**3 else "PREFLIGHT_WITHIN_SINGLE_GPU_CAPACITY",
        "edge_count_convention": "directed_edge_index_entries",
    }


def _train_gaan(model, features, adjacency, nodes: list[int], score_nodes: list[int], epochs: int, lr: float) -> tuple[list[float], torch.Tensor]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr)
    losses: list[float] = []
    scores: torch.Tensor | None = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        generator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss, generator_loss, score = model(features, adjacency, nodes, score_nodes)
        discriminator_loss.backward()
        generator_loss.backward()
        optimizer.step()
        generator_optimizer.step()
        total = discriminator_loss.detach() + generator_loss.detach()
        if not torch.isfinite(total):
            raise FloatingPointError("GAAN produced a non-finite native loss")
        losses.append(float(total.cpu()))
        scores = score.detach()
    if scores is None:
        raise RuntimeError("GAAN did not execute its optimizer loop")
    model.eval()
    with torch.no_grad():
        _, _, scores = model(features, adjacency, nodes, score_nodes)
    return losses, scores.reshape(-1)


def _train_aegis(model, features, adjacency, nodes: list[int], score_nodes: list[int], epochs: int, lr: float, reconstruction_epochs: int) -> tuple[list[float], list[float], torch.Tensor]:
    reconstruction_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pretrain_losses: list[float] = []
    for _ in range(reconstruction_epochs):
        _, _, reconstruction_loss, _, _ = model(features, adjacency, nodes, score_nodes)
        if not torch.isfinite(reconstruction_loss):
            raise FloatingPointError("AEGIS pretraining produced a non-finite loss")
        reconstruction_loss.backward()
        reconstruction_optimizer.step()
        pretrain_losses.append(float(reconstruction_loss.detach().cpu()))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr)
    losses: list[float] = []
    scores: torch.Tensor | None = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        generator_optimizer.zero_grad(set_to_none=True)
        benchmark_loss, generator_loss, _, score, _ = model(features, adjacency, nodes, score_nodes)
        generator_loss.backward(retain_graph=True)
        benchmark_loss.backward(retain_graph=True)
        optimizer.step()
        generator_optimizer.step()
        total = benchmark_loss.detach() + generator_loss.detach()
        if not torch.isfinite(total):
            raise FloatingPointError("AEGIS produced a non-finite native loss")
        losses.append(float(total.cpu()))
        scores = score.detach()
    if scores is None:
        raise RuntimeError("AEGIS did not execute its optimizer loop")
    model.eval()
    with torch.no_grad():
        _, _, _, scores, _ = model(features, adjacency, nodes, score_nodes)
    return pretrain_losses, losses, scores.reshape(-1)


def run(args: argparse.Namespace) -> int:
    started = time.time()
    seed_everything(args.seed)
    source = Path(args.source).resolve()
    Model, module, normalize_adj = _import_upstream(source, args.method)
    bundle = load_label_free_bundle(args.bundle)
    dataset = str(bundle["provenance"].get("dataset"))
    epochs = args.epochs or EPOCHS[args.method].get(dataset, 500)
    lr = args.lr if args.lr > 0 else ({"Amazon": 1e-3, "T-Finance": 5e-4, "Elliptic": 5e-3}.get(dataset, 1e-3))
    score_nodes_array = load_score_nodes(args.score_nodes, num_nodes=bundle["x"].shape[0])
    device = _device(args.device)
    previous_default = torch.get_default_device()
    torch.set_default_device(device)
    try:
        features = bundle["x"].unsqueeze(0).to(device)
        adjacency = _adjacency(bundle["edge_index"], features.shape[1], normalize_adj).to(device)
        all_nodes = list(range(features.shape[1]))
        score_nodes = score_nodes_array.tolist()
        model = Model(features.shape[-1], args.embedding_dim, "prelu", 1, "avg").to(device)
        if args.method == "GAAN":
            module.neighList_to_edgeList_train = _vectorized_edge_list
            pretrain_losses: list[float] = []
            losses, score = _train_gaan(model, features, adjacency, all_nodes, score_nodes, epochs, lr)
        else:
            pretrain_losses, losses, score = _train_aegis(
                model, features, adjacency, all_nodes, score_nodes, epochs, lr, args.reconstruction_epochs
            )
        scores = score.cpu().numpy()
    finally:
        torch.set_default_device(previous_default)
    score_path = write_scores(args.scores, score_nodes_array, scores)
    evidence = {
        "method": args.method,
        "source_kind": "OFFICIAL_BENCHMARK_REIMPLEMENTATION",
        "implementation_source": "mala-lab/GGAD",
        "source_url": "https://github.com/mala-lab/GGAD.git",
        "source_commit": UPSTREAM_COMMIT,
        "checkout_head": git_head(source),
        "native_model_import": str(source / ("model_AEGIS.py" if args.method == "AEGIS" else "model_gaan.py")) + ":Model",
        "native_optimizer_and_objective_preserved": True,
        "mechanical_compatibility": "PyTorch default tensor device and vectorized row-major edge enumeration; no equation or loss change",
        "bundle": str(bundle["bundle_path"]),
        "bundle_sha256": sha256(bundle["bundle_path"]),
        "bundle_keys": ["edge_index", "features", "node_id", "provenance"],
        "ground_truth_anomaly_label_present_or_accessed": False,
        "normal_support_used": False,
        "test_metrics_or_labels_called_during_training": False,
        "early_stopping_or_checkpoint_selection": False,
        "coReGAD_cross_fitting_injected": False,
        "dataset": dataset,
        "seed": args.seed,
        "epochs": epochs,
        "reconstruction_pretrain_epochs": len(pretrain_losses),
        "schedule_source": "upstream dataset default" if dataset in EPOCHS[args.method] else "fixed upstream Reddit default",
        "hyperparameters": {"embedding_dim": args.embedding_dim, "lr": lr},
        "pretrain_loss_trace": pretrain_losses,
        "loss_trace": losses,
        "device": str(device),
        "score_file": str(score_path.resolve()),
        "score_count": int(score_nodes_array.size),
        "score_direction": "higher_is_more_anomalous",
        "scores_finite": bool(np.isfinite(scores).all()),
        "elapsed_seconds": time.time() - started,
        "preflight": preflight(args.bundle, args.method),
    }
    write_evidence(args.evidence, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="GGAD benchmark-family canonical bridge")
    result.add_argument("--method", choices=("AEGIS", "GAAN"), required=True)
    result.add_argument("--bundle", required=True)
    result.add_argument("--score-nodes", required=True)
    result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--epochs", type=int, default=0)
    result.add_argument("--reconstruction-epochs", type=int, default=10)
    result.add_argument("--embedding-dim", type=int, default=300)
    result.add_argument("--lr", type=float, default=0.0)
    result.add_argument("--preflight", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.bundle, args.method), indent=2, sort_keys=True))
        return 0
    if args.epochs < 0 or args.reconstruction_epochs < 0 or args.lr < 0:
        raise SystemExit("invalid schedule override")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
