"""Label-isolated canonical bridge for the author-official GGAD model.

The imported model, pseudo-anomaly construction, three native losses, and Adam
optimization are preserved.  The adapter replaces only the upstream loader
(which selects normal nodes by opening the anomaly labels) and removes
epoch-time test-label metrics from the training process.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

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


UPSTREAM_COMMIT = "358fb4d4b4ee5b8b195445791a9e2b6d52487f2b"
OFFICIAL_EPOCHS = {"Amazon": 800, "T-Finance": 500, "Elliptic": 150}
LABEL_FREE_FALLBACK_EPOCHS = 300


def _git(source: Path, *args: str) -> subprocess.CompletedProcess[str]:
    executable = "/data1/anaconda3/bin/git" if Path("/data1/anaconda3/bin/git").is_file() else "git"
    return subprocess.run([executable, "-C", str(source), *args], check=False, capture_output=True, text=True)


def _import_upstream(source: Path):
    if _git(source, "merge-base", "--is-ancestor", UPSTREAM_COMMIT, "HEAD").returncode:
        raise RuntimeError("GGAD checkout does not contain the frozen author commit")
    sys.path.insert(0, str(source))
    try:
        from model import Model  # type: ignore
        from utils import normalize_adj  # type: ignore
    finally:
        sys.path.pop(0)
    return Model, normalize_adj


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device.index or 0)
    return device


def _adjacency(edge_index: torch.Tensor, num_nodes: int, normalize_adj) -> tuple[torch.Tensor, torch.Tensor]:
    rows = edge_index[0].numpy()
    columns = edge_index[1].numpy()
    adjacency = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, columns)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    ).tocsr()
    adjacency.data[:] = 1.0
    normalized = normalize_adj(adjacency)
    normalized = (normalized + sp.eye(num_nodes, dtype=np.float32, format="coo")).toarray().astype(np.float32, copy=False)
    raw = (adjacency + sp.eye(num_nodes, dtype=np.float32, format="csr")).toarray().astype(np.float32, copy=False)
    return torch.from_numpy(normalized).unsqueeze(0), torch.from_numpy(raw)


def preflight(bundle_path: str | Path) -> dict[str, object]:
    bundle = load_label_free_bundle(bundle_path)
    n = int(bundle["x"].shape[0])
    dense_bytes = n * n * 4
    return {
        "method": "GGAD",
        "num_nodes": n,
        "num_edges": int(bundle["edge_index"].shape[1]),
        "dense_n_by_n_path": True,
        "named_dense_tensors": ["normalized_adjacency", "raw_adjacency", "embedding_similarity"],
        "minimum_named_dense_bytes": 3 * dense_bytes,
        "minimum_named_dense_gib": 3 * dense_bytes / 1024**3,
        "preflight_status": "NOT_RUN_RESOURCE_RISK" if 3 * dense_bytes >= 12 * 1024**3 else "PREFLIGHT_WITHIN_SINGLE_GPU_CAPACITY",
        "edge_count_convention": "directed_edge_index_entries",
    }


def run(args: argparse.Namespace) -> int:
    started = time.time()
    seed_everything(args.seed)
    source = Path(args.source).resolve()
    Model, normalize_adj = _import_upstream(source)
    bundle = load_label_free_bundle(args.bundle)
    dataset = str(bundle["provenance"].get("dataset"))
    epochs = args.epochs or OFFICIAL_EPOCHS.get(dataset, LABEL_FREE_FALLBACK_EPOCHS)
    x_cpu = bundle["x"].contiguous()
    support = load_normal_support(args.support, num_nodes=x_cpu.shape[0])
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=x_cpu.shape[0])
    normal_nodes = support.tolist()
    random.shuffle(normal_nodes)
    pseudo_count = max(1, int(len(normal_nodes) * (0.05 if dataset == "Amazon" else 0.15)))
    pseudo_nodes = normal_nodes[:pseudo_count]
    device = _device(args.device)
    normalized_adj_cpu, raw_adj_cpu = _adjacency(bundle["edge_index"], x_cpu.shape[0], normalize_adj)
    previous_default = torch.get_default_device()
    torch.set_default_device(device)
    try:
        features = x_cpu.unsqueeze(0).to(device)
        adjacency = normalized_adj_cpu.to(device)
        raw_adjacency = raw_adj_cpu.to(device)
        native_args = SimpleNamespace(var=args.noise_variance, mean=args.noise_mean)
        model = Model(features.shape[-1], args.embedding_dim, "prelu", 1, "avg").to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([1.0], device=device))
        normal_index = torch.as_tensor(normal_nodes, dtype=torch.long, device=device)
        pseudo_index = torch.as_tensor(pseudo_nodes, dtype=torch.long, device=device)
        losses: list[float] = []
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            embedding, _, logits, generated, sampled = model(
                features, adjacency, pseudo_index, normal_index, True, native_args
            )
            target = torch.cat(
                [torch.zeros(normal_index.numel(), device=device), torch.ones(generated.shape[0], device=device)]
            ).view(1, -1, 1)
            loss_bce = bce(logits, target).mean()
            embedding = embedding.squeeze(0)
            embedding_norm = embedding / torch.linalg.vector_norm(embedding, dim=-1, keepdim=True).clamp_min(1e-12)
            similarity = torch.mm(embedding_norm, embedding_norm.T) * raw_adjacency
            affinity = similarity.sum(0) / raw_adjacency.sum(0).clamp_min(1.0)
            normal_affinity = affinity[normal_index].mean()
            pseudo_affinity = affinity[pseudo_index].mean()
            loss_margin = (0.7 - (normal_affinity - pseudo_affinity)).clamp_min(0)
            loss_reconstruction = torch.sqrt(torch.sum((generated - sampled) ** 2, dim=1)).mean()
            loss = loss_margin + loss_bce + loss_reconstruction
            if not torch.isfinite(loss):
                raise FloatingPointError("GGAD produced a non-finite native loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            del embedding, logits, generated, sampled, similarity, affinity, loss
        model.eval()
        with torch.no_grad():
            _, _, logits, _, _ = model(features, adjacency, pseudo_index, normal_index, False, native_args)
            all_scores = logits.squeeze(0).squeeze(-1)
            scores = all_scores[torch.as_tensor(score_nodes, dtype=torch.long, device=device)].detach().cpu().numpy()
    finally:
        torch.set_default_device(previous_default)
    score_path = write_scores(args.scores, score_nodes, scores)
    evidence = {
        "method": "GGAD",
        "source_kind": "AUTHOR_OFFICIAL",
        "source_url": "https://github.com/mala-lab/GGAD.git",
        "source_commit": UPSTREAM_COMMIT,
        "checkout_head": git_head(source),
        "native_model_import": str(source / "model.py") + ":Model",
        "native_loader_replaced": str(source / "utils.py") + ":load_mat opens anomaly labels to select normal support",
        "native_optimizer": "torch.optim.Adam",
        "native_losses": ["BCE pseudo-anomaly discrimination", "local-affinity margin", "generated-outlier reconstruction"],
        "bundle": str(bundle["bundle_path"]),
        "bundle_sha256": sha256(bundle["bundle_path"]),
        "bundle_keys": ["edge_index", "features", "node_id", "provenance"],
        "normal_support_manifest": str(Path(args.support).resolve()),
        "normal_support_sha256": sha256(args.support),
        "normal_support_count": len(normal_nodes),
        "pseudo_anomaly_count": len(pseudo_nodes),
        "pseudo_anomalies_derived_only_from_frozen_normal_support": True,
        "ground_truth_anomaly_label_present_or_accessed": False,
        "test_metrics_or_labels_called_during_training": False,
        "early_stopping_or_checkpoint_selection": False,
        "coReGAD_cross_fitting_injected": False,
        "dataset": dataset,
        "seed": args.seed,
        "epochs": epochs,
        "schedule_source": "upstream dataset default" if dataset in OFFICIAL_EPOCHS else "fixed label-free fallback matching upstream Reddit default",
        "hyperparameters": {"embedding_dim": args.embedding_dim, "lr": args.lr, "weight_decay": args.weight_decay, "noise_mean": args.noise_mean, "noise_variance": args.noise_variance},
        "loss_trace": losses,
        "device": str(device),
        "score_file": str(score_path.resolve()),
        "score_count": int(score_nodes.size),
        "score_direction": "higher_is_more_anomalous",
        "scores_finite": bool(np.isfinite(scores).all()),
        "elapsed_seconds": time.time() - started,
        "preflight": preflight(args.bundle),
    }
    write_evidence(args.evidence, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="GGAD canonical native bridge")
    result.add_argument("--bundle", required=True)
    result.add_argument("--support", required=True)
    result.add_argument("--score-nodes", required=True)
    result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--epochs", type=int, default=0, help="0 selects the frozen dataset/default schedule")
    result.add_argument("--embedding-dim", type=int, default=300)
    result.add_argument("--lr", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--noise-mean", type=float, default=0.0)
    result.add_argument("--noise-variance", type=float, default=0.0)
    result.add_argument("--preflight", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.bundle), indent=2, sort_keys=True))
        return 0
    if args.epochs < 0:
        raise SystemExit("--epochs must be zero (default schedule) or positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
