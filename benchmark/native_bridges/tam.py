"""Label-free canonical bridge for the author-official TAM implementation."""

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


UPSTREAM_COMMIT = "e82ead0584b9b807d4db56a9ffdf89ff94c74dfa"


def _git(source: Path, *args: str) -> subprocess.CompletedProcess[str]:
    executable = "/data1/anaconda3/bin/git" if Path("/data1/anaconda3/bin/git").is_file() else "git"
    return subprocess.run([executable, "-C", str(source), *args], check=False, capture_output=True, text=True)


def _import_upstream(source: Path):
    if _git(source, "merge-base", "--is-ancestor", UPSTREAM_COMMIT, "HEAD").returncode:
        raise RuntimeError("TAM checkout does not contain the frozen author commit")
    sys.path.insert(0, str(source))
    try:
        from model import Model  # type: ignore
        from utils import graph_nsgt  # type: ignore
    finally:
        sys.path.pop(0)
    return Model, graph_nsgt


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device.index or 0)
    return device


def _dense_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    rows, columns = edge_index.numpy()
    adjacency = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, columns)), shape=(num_nodes, num_nodes))
    adjacency.data[:] = 1.0
    adjacency = adjacency + sp.eye(num_nodes, dtype=np.float32, format="csr")
    adjacency.data[:] = 1.0
    return torch.from_numpy(adjacency.toarray().astype(np.float32, copy=False))


def _edge_distance(adjacency: torch.Tensor, features: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """Vectorized equivalent of upstream ``calc_distance`` on nonzero edges."""
    rows, columns = torch.nonzero(adjacency > 0, as_tuple=True)
    distance = torch.zeros_like(adjacency)
    for start in range(0, rows.numel(), chunk):
        stop = min(start + chunk, rows.numel())
        selected_rows = rows[start:stop]
        selected_columns = columns[start:stop]
        values = torch.sqrt(torch.sum((features[selected_rows] - features[selected_columns]) ** 2, dim=1))
        distance[selected_rows, selected_columns] = values
    return distance


def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Algebraically identical to upstream dense diagonal matrix products."""
    degree_inverse_sqrt = adjacency.sum(0).pow(-0.5)
    degree_inverse_sqrt[torch.isinf(degree_inverse_sqrt)] = 0.0
    return (degree_inverse_sqrt[:, None] * adjacency * degree_inverse_sqrt[None, :]).unsqueeze(0)


def _affinity(embedding: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    normalized = embedding / torch.linalg.vector_norm(embedding, dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = torch.mm(normalized, normalized.T) * adjacency
    inverse_degree = adjacency.sum(0).pow(-1)
    inverse_degree[torch.isinf(inverse_degree)] = 0.0
    return similarity.sum(1) * inverse_degree


def _regularization(embedding: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    normalized = embedding / torch.linalg.vector_norm(embedding, dim=-1, keepdim=True).clamp_min(1e-12)
    inverse = 1 - adjacency
    similarity = torch.mm(normalized, normalized.T) * inverse
    inverse_count = inverse.sum(1).pow(-1)
    inverse_count[torch.isinf(inverse_count)] = 0.0
    return torch.sum(similarity.sum(1) * inverse_count)


def _normalize_score(score: np.ndarray) -> np.ndarray:
    span = float(score.max() - score.min())
    return np.zeros_like(score) if span == 0 else (score - score.min()) / span


def preflight(bundle_path: str | Path, trees: int = 3) -> dict[str, object]:
    bundle = load_label_free_bundle(bundle_path)
    n = int(bundle["x"].shape[0])
    dense_bytes = n * n * 4
    return {
        "method": "TAM",
        "num_nodes": n,
        "num_edges": int(bundle["edge_index"].shape[1]),
        "dense_n_by_n_path": True,
        "dense_truncated_adjacency_copies": trees,
        "minimum_named_dense_bytes": (trees + 4) * dense_bytes,
        "preflight_status": "NOT_RUN_RESOURCE_RISK" if (trees + 4) * dense_bytes >= 16 * 1024**3 else "PREFLIGHT_WITHIN_SINGLE_GPU_CAPACITY",
        "edge_count_convention": "directed_edge_index_entries",
    }


def run(args: argparse.Namespace) -> int:
    started = time.time()
    seed_everything(args.seed)
    source = Path(args.source).resolve()
    Model, graph_nsgt = _import_upstream(source)
    bundle = load_label_free_bundle(args.bundle)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=bundle["x"].shape[0])
    device = _device(args.device)
    features_cpu = bundle["x"].contiguous()
    raw_adjacency_cpu = _dense_adjacency(bundle["edge_index"], features_cpu.shape[0])
    distance_cpu = _edge_distance(raw_adjacency_cpu, features_cpu)
    features = features_cpu.unsqueeze(0).to(device)
    raw_adjacency = raw_adjacency_cpu.to(device)
    distance = distance_cpu.to(device)
    models = [Model(features.shape[-1], args.embedding_dim, "prelu", 2, "avg").to(device) for _ in range(args.cutting * args.trees)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) for model in models]
    truncated = torch.stack([raw_adjacency.clone() for _ in range(args.trees)], dim=0)
    loss_trace: list[float] = []
    cut_messages: list[torch.Tensor] = []
    model_index = 0
    for _ in range(args.cutting):
        tree_messages: list[torch.Tensor] = []
        for tree in range(args.trees):
            cut_adjacency = graph_nsgt(distance, truncated[tree]).to(device)
            normalized_adjacency = _normalize_adjacency(cut_adjacency)
            model = models[model_index]
            optimizer = optimizers[model_index]
            optimizer.zero_grad(set_to_none=True)
            final_message: torch.Tensor | None = None
            for _ in range(args.epochs):
                model.train()
                node_embedding, first_projection, _ = model(features, normalized_adjacency)
                message = _affinity(node_embedding.squeeze(0), raw_adjacency)
                loss = -message.sum()
                regularization = _regularization(first_projection.squeeze(0), raw_adjacency)
                loss = loss + args.lambda_regularization * regularization
                if not torch.isfinite(loss):
                    raise FloatingPointError("TAM produced a non-finite native loss")
                loss.backward()
                optimizer.step()
                loss_trace.append(float(loss.detach().cpu()))
                final_message = _affinity(node_embedding.squeeze(0), raw_adjacency).detach()
            if final_message is None:
                raise RuntimeError("TAM did not execute its native optimizer loop")
            tree_messages.append(final_message)
            truncated[tree] = cut_adjacency
            model_index += 1
        cut_messages.append(torch.stack(tree_messages, dim=0).mean(0))
    mean_affinity = torch.stack(cut_messages, dim=0).mean(0).cpu().numpy()
    all_scores = 1.0 - _normalize_score(mean_affinity)
    scores = all_scores[score_nodes]
    score_path = write_scores(args.scores, score_nodes, scores)
    evidence = {
        "method": "TAM",
        "source_kind": "AUTHOR_OFFICIAL",
        "source_url": "https://github.com/mala-lab/TAM-master.git",
        "source_commit": UPSTREAM_COMMIT,
        "checkout_head": git_head(source),
        "native_model_import": str(source / "model.py") + ":Model",
        "native_graph_truncation_import": str(source / "utils.py") + ":graph_nsgt",
        "native_optimizer": "one independent Adam optimizer per cut/tree LAMNet",
        "native_objective": "maximize local node affinity plus lambda * non-edge similarity regularization",
        "truncated_affinity_preserved": True,
        "one_class_homophily_preserved": True,
        "mechanical_compatibility_optimizations": ["vectorized edge-distance calculation", "elementwise symmetric degree normalization instead of multiplying dense diagonal matrices"],
        "numerical_semantics_changed": False,
        "bundle": str(bundle["bundle_path"]),
        "bundle_sha256": sha256(bundle["bundle_path"]),
        "bundle_keys": ["edge_index", "features", "node_id", "provenance"],
        "ground_truth_anomaly_label_present_or_accessed": False,
        "normal_support_used": False,
        "test_metrics_or_labels_called_during_training": False,
        "early_stopping_or_checkpoint_selection": False,
        "coReGAD_cross_fitting_injected": False,
        "dataset": str(bundle["provenance"].get("dataset")),
        "seed": args.seed,
        "epochs_per_lamnet": args.epochs,
        "cutting_rounds": args.cutting,
        "trees": args.trees,
        "optimizer_steps": len(loss_trace),
        "schedule_source": "upstream global defaults",
        "hyperparameters": {"embedding_dim": args.embedding_dim, "lr": args.lr, "weight_decay": args.weight_decay, "lambda_regularization": args.lambda_regularization},
        "loss_trace": loss_trace,
        "device": str(device),
        "score_file": str(score_path.resolve()),
        "score_count": int(score_nodes.size),
        "score_direction": "higher_is_more_anomalous",
        "scores_finite": bool(np.isfinite(scores).all()),
        "elapsed_seconds": time.time() - started,
        "preflight": preflight(args.bundle, args.trees),
    }
    write_evidence(args.evidence, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="TAM canonical native bridge")
    result.add_argument("--bundle", required=True)
    result.add_argument("--score-nodes", required=True)
    result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--epochs", type=int, default=500)
    result.add_argument("--cutting", type=int, default=18)
    result.add_argument("--trees", type=int, default=3)
    result.add_argument("--embedding-dim", type=int, default=128)
    result.add_argument("--lr", type=float, default=1e-5)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--lambda-regularization", type=float, default=0.0)
    result.add_argument("--preflight", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.bundle, args.trees), indent=2, sort_keys=True))
        return 0
    if min(args.epochs, args.cutting, args.trees) <= 0:
        raise SystemExit("epochs, cutting, and trees must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
