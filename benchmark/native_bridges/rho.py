"""Canonical normal-support bridge for the author-official RHO model."""

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
    load_normal_support,
    load_score_nodes,
    seed_everything,
    sha256,
    write_evidence,
    write_scores,
)


UPSTREAM_COMMIT = "a394d5575dea6745215b15e0453e1f925ffcc1f2"
OFFICIAL_EPOCHS = {"Amazon": 500, "Tolokers": 500}
OFFICIAL_FULL_NCE = {"Amazon", "Tolokers"}


def _git(source: Path, *args: str) -> subprocess.CompletedProcess[str]:
    executable = "/data1/anaconda3/bin/git" if Path("/data1/anaconda3/bin/git").is_file() else "git"
    return subprocess.run([executable, "-C", str(source), *args], check=False, capture_output=True, text=True)


def _import_upstream(source: Path):
    if _git(source, "merge-base", "--is-ancestor", UPSTREAM_COMMIT, "HEAD").returncode:
        raise RuntimeError("RHO checkout does not contain the frozen author commit")
    sys.path.insert(0, str(source))
    try:
        from model import RHO  # type: ignore
        from utils import init_params  # type: ignore
    finally:
        sys.path.pop(0)
    return RHO, init_params


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device.index or 0)
    return device


def _laplacian(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    rows = edge_index[0].numpy()
    columns = edge_index[1].numpy()
    adjacency = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, columns)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    adjacency.data[:] = 1.0
    transpose = adjacency.T.tocsr()
    adjacency = adjacency + transpose.multiply(transpose > adjacency) - adjacency.multiply(transpose > adjacency)
    adjacency = adjacency + sp.eye(num_nodes, dtype=np.float32, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse_sqrt = np.power(degree, -0.5, where=degree > 0)
    inverse_sqrt[degree <= 0] = 0.0
    scaling = sp.diags(inverse_sqrt)
    laplacian = (sp.eye(num_nodes, dtype=np.float32, format="csr") - scaling @ adjacency @ scaling).tocoo()
    indices = torch.from_numpy(np.vstack((laplacian.row, laplacian.col)).astype(np.int64, copy=False))
    values = torch.from_numpy(laplacian.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()


def _centers(model, laplacian: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        global_output, local_output, _ = model(laplacian, features)
        global_center = global_output.mean(dim=0)
        local_center = local_output.mean(dim=0)
    for center in (global_center, local_center):
        center[(center.abs() < 0.1) & (center < 0)] = -0.1
        center[(center.abs() < 0.1) & (center > 0)] = 0.1
    return local_center, global_center


def preflight(bundle_path: str | Path, batch_size: int = 1024) -> dict[str, object]:
    bundle = load_label_free_bundle(bundle_path)
    n = int(bundle["x"].shape[0])
    selected = n if batch_size == 0 or batch_size > n else batch_size
    pairwise_bytes = 4 * selected * selected * 4
    return {
        "method": "RHO",
        "num_nodes": n,
        "num_edges": int(bundle["edge_index"].shape[1]),
        "sparse_laplacian": True,
        "batched_nce": batch_size > 0 and batch_size < n,
        "nce_batch_size": selected,
        "estimated_pairwise_nce_bytes": pairwise_bytes,
        "preflight_status": "NOT_RUN_RESOURCE_RISK" if pairwise_bytes >= 12 * 1024**3 else "PREFLIGHT_BATCHED_NCE",
        "edge_count_convention": "directed_edge_index_entries",
    }


def run(args: argparse.Namespace) -> int:
    started = time.time()
    seed_everything(args.seed)
    source = Path(args.source).resolve()
    RHO, init_params = _import_upstream(source)
    bundle = load_label_free_bundle(args.bundle)
    dataset = str(bundle["provenance"].get("dataset"))
    epochs = args.epochs or OFFICIAL_EPOCHS.get(dataset, 100)
    batch_size = args.batch_size if args.batch_size >= 0 else (0 if dataset in OFFICIAL_FULL_NCE else 1024)
    lr = args.lr if args.lr > 0 else (0.005 if dataset in {"Amazon", "Tolokers", "T-Finance"} else 0.0005)
    support = load_normal_support(args.support, num_nodes=bundle["x"].shape[0])
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=bundle["x"].shape[0])
    device = _device(args.device)
    features = bundle["x"].to(device)
    laplacian = _laplacian(bundle["edge_index"], features.shape[0]).to(device)
    support_index = torch.as_tensor(support, dtype=torch.long, device=device)
    model = RHO(features.shape[1], args.hidden1, args.hidden2, args.layers, batch_size, args.temperature).to(device)
    model.apply(init_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    losses: list[float] = []
    final_local: torch.Tensor | None = None
    final_global: torch.Tensor | None = None
    for _ in range(epochs):
        local_center, global_center = _centers(model, laplacian, features)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        global_output, local_output, nce_loss = model(laplacian, features)
        global_distance = torch.sum((global_output[support_index] - global_center) ** 2, dim=1)
        local_distance = torch.sum((local_output[support_index] - local_center) ** 2, dim=1)
        loss = (0.5 * global_distance + 0.5 * local_distance).mean() + args.alpha * nce_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("RHO produced a non-finite native loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        final_local, final_global = local_center, global_center
        del global_output, local_output, nce_loss, global_distance, local_distance, loss
    if final_local is None or final_global is None:
        raise RuntimeError("RHO did not execute its training loop")
    model.eval()
    with torch.no_grad():
        global_output, local_output, _ = model(laplacian, features)
        nodes = torch.as_tensor(score_nodes, dtype=torch.long, device=device)
        scores = 0.5 * torch.sum((global_output[nodes] - final_global) ** 2, dim=1)
        scores += 0.5 * torch.sum((local_output[nodes] - final_local) ** 2, dim=1)
        scores_numpy = scores.cpu().numpy()
    score_path = write_scores(args.scores, score_nodes, scores_numpy)
    evidence = {
        "method": "RHO",
        "source_kind": "AUTHOR_OFFICIAL",
        "source_url": "https://github.com/mala-lab/RHO.git",
        "source_commit": UPSTREAM_COMMIT,
        "checkout_head": git_head(source),
        "native_model_import": str(source / "model.py") + ":RHO",
        "native_optimizer": "torch.optim.Adam",
        "native_objective": "mean normal-support local/global hypersphere distance + alpha * batched NCE",
        "sparse_laplacian_preserved": True,
        "batched_nce_preserved": True,
        "bundle": str(bundle["bundle_path"]),
        "bundle_sha256": sha256(bundle["bundle_path"]),
        "bundle_keys": ["edge_index", "features", "node_id", "provenance"],
        "normal_support_manifest": str(Path(args.support).resolve()),
        "normal_support_sha256": sha256(args.support),
        "normal_support_count": int(support.size),
        "ground_truth_anomaly_label_present_or_accessed": False,
        "test_metrics_or_labels_called_during_training": False,
        "early_stopping_or_checkpoint_selection": False,
        "coReGAD_cross_fitting_injected": False,
        "dataset": dataset,
        "seed": args.seed,
        "epochs": epochs,
        "schedule_source": "upstream dataset default" if dataset in OFFICIAL_EPOCHS else "upstream global default",
        "hyperparameters": {"hidden1": args.hidden1, "hidden2": args.hidden2, "layers": args.layers, "batch_size": batch_size, "temperature": args.temperature, "alpha": args.alpha, "lr": lr, "weight_decay": args.weight_decay},
        "loss_trace": losses,
        "device": str(device),
        "score_file": str(score_path.resolve()),
        "score_count": int(score_nodes.size),
        "score_direction": "higher_is_more_anomalous",
        "scores_finite": bool(np.isfinite(scores_numpy).all()),
        "elapsed_seconds": time.time() - started,
        "preflight": preflight(args.bundle, batch_size),
    }
    write_evidence(args.evidence, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="RHO canonical native bridge")
    result.add_argument("--bundle", required=True)
    result.add_argument("--support", required=True)
    result.add_argument("--score-nodes", required=True)
    result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--epochs", type=int, default=0)
    result.add_argument("--hidden1", type=int, default=1024)
    result.add_argument("--hidden2", type=int, default=64)
    result.add_argument("--layers", type=int, default=2)
    result.add_argument("--batch-size", type=int, default=-1, help="-1 selects the frozen dataset/default value")
    result.add_argument("--temperature", type=float, default=0.2)
    result.add_argument("--alpha", type=float, default=1.0)
    result.add_argument("--lr", type=float, default=0.0, help="0 selects the frozen dataset/default value")
    result.add_argument("--weight-decay", type=float, default=5e-5)
    result.add_argument("--preflight", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.bundle), indent=2, sort_keys=True))
        return 0
    if args.epochs < 0 or args.batch_size < -1 or args.lr < 0:
        raise SystemExit("invalid schedule override")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
