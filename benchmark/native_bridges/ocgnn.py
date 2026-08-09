"""Canonical normal-support bridge for the author-official OCGNN release.

The bridge imports the upstream GraphSAGE model initializer and one-class loss.
It replaces only the upstream dataset/label split with the frozen normal-support
IDs; no anomaly label or evaluator is available in this process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


UPSTREAM_COMMIT = "ec7d11c5ae2f91f4165e384131c6a8358836ff58"


def _git(source: Path, *args: str) -> str:
    executable = "/data1/anaconda3/bin/git" if Path("/data1/anaconda3/bin/git").is_file() else "git"
    return subprocess.run(
        [executable, "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _import_upstream(source: Path):
    try:
        _git(source, "merge-base", "--is-ancestor", UPSTREAM_COMMIT, "HEAD")
    except subprocess.CalledProcessError as error:
        raise RuntimeError("OCGNN checkout does not contain the frozen author commit") from error
    sys.path.insert(0, str(source))
    try:
        from networks.init import init_model  # type: ignore
        from optim.loss import anomaly_score, get_radius, init_center, loss_function  # type: ignore
    finally:
        sys.path.pop(0)
    return init_model, anomaly_score, get_radius, init_center, loss_function


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device.index or 0)
    return device


def preflight(bundle_path: str | Path) -> dict[str, object]:
    bundle = load_label_free_bundle(bundle_path)
    n = int(bundle["x"].shape[0])
    e = int(bundle["edge_index"].shape[1])
    f = int(bundle["x"].shape[1])
    return {
        "method": "OCGNN",
        "num_nodes": n,
        "num_edges": e,
        "num_features": f,
        "dense_n_by_n_path": False,
        "full_batch_message_passing": True,
        "estimated_input_bytes": int(e * 2 * 8 + n * f * 4),
        "preflight_status": "PREFLIGHT_SPARSE_FULL_BATCH",
        "edge_count_convention": "directed_edge_index_entries",
    }


def run(args: argparse.Namespace) -> int:
    started = time.time()
    seed_everything(args.seed)
    source = Path(args.source).resolve()
    init_model, anomaly_score, get_radius, init_center, loss_function = _import_upstream(source)
    import dgl

    bundle = load_label_free_bundle(args.bundle)
    features_cpu = bundle["x"]
    support = load_normal_support(args.support, num_nodes=features_cpu.shape[0])
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=features_cpu.shape[0])
    device = _device(args.device)
    edge_index = bundle["edge_index"]
    graph = dgl.graph(
        (edge_index[0], edge_index[1]),
        num_nodes=features_cpu.shape[0],
        idtype=torch.int64,
    )
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph).to(device)
    features = features_cpu.to(device)
    support_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
    support_mask[torch.as_tensor(support, dtype=torch.long, device=device)] = True
    native_args = SimpleNamespace(
        module=args.module,
        gpu=(device.index or 0) if device.type == "cuda" else -1,
        n_hidden=args.hidden_dim,
        n_layers=args.layers,
        dropout=args.dropout,
        nu=args.nu,
    )
    model = init_model(native_args, features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    center = init_center(native_args, graph, features, model)
    radius = torch.tensor(0.0, device=device)
    losses: list[float] = []
    for _ in range(args.epochs):
        model.train()
        outputs = model(graph, features)
        loss, distances, _ = loss_function(args.nu, center, outputs, radius, support_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("OCGNN produced a non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        radius = torch.tensor(float(get_radius(distances, args.nu)), device=device)

    model.eval()
    with torch.no_grad():
        outputs = model(graph, features)
        _, all_scores = anomaly_score(center, outputs, radius, mask=None)
        scores = all_scores[torch.as_tensor(score_nodes, device=device)].cpu().numpy()
    score_path = write_scores(args.scores, score_nodes, scores)
    evidence = {
        "method": "OCGNN",
        "source_kind": "AUTHOR_OFFICIAL",
        "source_url": "https://github.com/WangXuhongCN/OCGNN.git",
        "source_commit": UPSTREAM_COMMIT,
        "checkout_head": git_head(source),
        "native_model_import": str(source / "networks" / "init.py"),
        "native_loss_import": str(source / "optim" / "loss.py"),
        "bundle": str(bundle["bundle_path"]),
        "bundle_sha256": sha256(bundle["bundle_path"]),
        "bundle_keys": sorted({"features", "edge_index", "node_id", "provenance"}),
        "normal_support_manifest": str(Path(args.support).resolve()),
        "normal_support_sha256": sha256(args.support),
        "normal_support_count": int(support.size),
        "label_tensor_present_or_accessed": False,
        "coReGAD_cross_fitting_injected": False,
        "node_order_preserved": True,
        "score_direction": "higher_is_more_anomalous",
        "score_formula": "squared_distance_to_center_minus_radius_squared",
        "score_file": str(score_path.resolve()),
        "score_count": int(score_nodes.size),
        "scores_finite": bool(np.isfinite(scores).all()),
        "seed": args.seed,
        "epochs": args.epochs,
        "hyperparameters": {
            "module": args.module,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "nu": args.nu,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "loss_trace": losses,
        "final_radius": float(radius.detach().cpu()),
        "device": str(device),
        "dgl_version": dgl.__version__,
        "elapsed_seconds": time.time() - started,
        "preflight": preflight(args.bundle),
    }
    write_evidence(args.evidence, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="OCGNN canonical native bridge")
    result.add_argument("--bundle", required=True)
    result.add_argument("--support", required=True)
    result.add_argument("--score-nodes", required=True)
    result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--epochs", type=int, default=5000)
    result.add_argument("--module", default="GraphSAGE", choices=("GraphSAGE", "GCN", "GAT", "GIN"))
    result.add_argument("--hidden-dim", type=int, default=32)
    result.add_argument("--layers", type=int, default=2)
    result.add_argument("--dropout", type=float, default=0.5)
    result.add_argument("--nu", type=float, default=0.2)
    result.add_argument("--lr", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=5e-4)
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
