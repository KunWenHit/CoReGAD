"""Canonical bridge for the author-official DOMINANT PyTorch release.

The bridge imports the upstream model and loss. It preserves the native dense
structure reconstruction; resource preflight reports the resulting N^2 risk
instead of substituting a weaker sparse objective.
"""

from __future__ import annotations

import argparse
import json
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


UPSTREAM_COMMIT = "83fa93931164aee5386fdc74c12a5d5c46a5d62f"


def _device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _import_upstream(source: Path):
    if git_head(source) != UPSTREAM_COMMIT:
        raise RuntimeError("DOMINANT source checkout does not match the frozen author commit")
    sys.path.insert(0, str(source))
    try:
        from model import Dominant  # type: ignore
        from run import loss_func  # type: ignore
        from utils import normalize_adj  # type: ignore
    finally:
        sys.path.pop(0)
    return Dominant, loss_func, normalize_adj


def _adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    normalize_adj,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = edge_index[0].cpu().numpy()
    cols = edge_index[1].cpu().numpy()
    values = np.ones(rows.shape[0], dtype=np.float32)
    raw = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    raw = (raw + sp.eye(num_nodes, dtype=np.float32, format="coo")).tocsr()
    raw.data[:] = 1.0
    normalized = normalize_adj(raw).tocoo()
    indices = torch.from_numpy(np.vstack([normalized.row, normalized.col]).astype(np.int64))
    sparse_adj = torch.sparse_coo_tensor(
        indices,
        torch.from_numpy(normalized.data.astype(np.float32, copy=False)),
        (num_nodes, num_nodes),
    ).coalesce().to(device)
    target = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    target[torch.from_numpy(raw.nonzero()[0]).to(device), torch.from_numpy(raw.nonzero()[1]).to(device)] = 1.0
    return sparse_adj, target


def preflight(bundle_path: str | Path) -> dict[str, object]:
    bundle = load_label_free_bundle(bundle_path)
    n = int(bundle["x"].shape[0])
    dense_bytes = n * n * 4
    minimum_dense_bytes = dense_bytes * 2
    return {
        "method": "DOMINANT",
        "num_nodes": n,
        "num_edges": int(bundle["edge_index"].shape[1]),
        "dense_n_by_n_path": True,
        "dense_tensor_bytes_each": dense_bytes,
        "dense_tensor_gib_each": dense_bytes / 1024**3,
        "minimum_named_dense_tensors": ["adjacency_target", "structure_reconstruction"],
        "minimum_named_dense_bytes": minimum_dense_bytes,
        "preflight_status": "NOT_RUN_RESOURCE_RISK" if minimum_dense_bytes >= 8 * 1024**3 else "PREFLIGHT_WITHIN_SINGLE_GPU_CAPACITY",
        "edge_count_convention": "directed_edge_index_entries",
    }


def run(args: argparse.Namespace) -> int:
    started = time.time()
    seed_everything(args.seed)
    source = Path(args.source).resolve()
    Dominant, loss_func, normalize_adj = _import_upstream(source)
    bundle = load_label_free_bundle(args.bundle)
    x_cpu = bundle["x"]
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=x_cpu.shape[0])
    device = _device(args.device)
    x = x_cpu.to(device)
    adj, target = _adjacency(bundle["edge_index"], x.shape[0], normalize_adj, device)
    model = Dominant(
        feat_size=x.shape[1], hidden_size=args.hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses: list[float] = []
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        structure_hat, x_hat = model(x, adj)
        per_node, _, _ = loss_func(target, structure_hat, x, x_hat, args.alpha)
        loss = per_node.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("DOMINANT produced a non-finite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        del structure_hat, x_hat, per_node, loss
    model.eval()
    with torch.no_grad():
        structure_hat, x_hat = model(x, adj)
        per_node, _, _ = loss_func(target, structure_hat, x, x_hat, args.alpha)
        scores = per_node[torch.as_tensor(score_nodes, device=device)].cpu().numpy()
    score_path = write_scores(args.scores, score_nodes, scores)
    evidence = {
        "method": "DOMINANT",
        "source_kind": "AUTHOR_OFFICIAL",
        "source_url": "https://github.com/kaize0409/GCN_AnomalyDetection_pytorch.git",
        "source_commit": git_head(source),
        "native_model_import": str(source / "model.py"),
        "native_loss_import": str(source / "run.py") + ":loss_func",
        "bundle": str(bundle["bundle_path"]),
        "bundle_sha256": sha256(bundle["bundle_path"]),
        "bundle_keys": sorted({"features", "edge_index", "node_id", "provenance"}),
        "label_tensor_present_or_accessed": False,
        "coReGAD_cross_fitting_injected": False,
        "node_order_preserved": True,
        "score_direction": "higher_is_more_anomalous",
        "score_file": str(score_path.resolve()),
        "score_count": int(score_nodes.size),
        "scores_finite": bool(np.isfinite(scores).all()),
        "seed": args.seed,
        "epochs": args.epochs,
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "alpha": args.alpha,
        },
        "loss_trace": losses,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "preflight": preflight(args.bundle),
    }
    write_evidence(args.evidence, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="DOMINANT canonical native bridge")
    result.add_argument("--bundle", required=True)
    result.add_argument("--score-nodes", required=True)
    result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--hidden-dim", type=int, default=64)
    result.add_argument("--dropout", type=float, default=0.3)
    result.add_argument("--lr", type=float, default=5e-3)
    result.add_argument("--alpha", type=float, default=0.8)
    result.add_argument("--preflight", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.bundle), indent=2, sort_keys=True))
        return 0
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
