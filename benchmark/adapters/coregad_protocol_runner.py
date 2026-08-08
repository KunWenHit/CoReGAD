"""Shared, non-invasive adapter for the CoReGAD five-fold benchmark protocol.

The adapter deliberately separates training bundles from evaluation labels.  An
upstream baseline may consume the generated ``fold_*.npz`` files through a
method-specific loader, but it cannot obtain ``y`` from those bundles.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


OUTER_FOLDS = 5


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset(path: str | Path, *, labels: bool = False) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as stored:
        x = np.asarray(stored["x"], dtype=np.float32)
        edge_index = np.asarray(stored["edge_index"], dtype=np.int64)
        node_id = (
            np.asarray(stored["node_id"], dtype=np.int64)
            if "node_id" in stored
            else np.arange(x.shape[0], dtype=np.int64)
        )
        result = {"x": x, "edge_index": edge_index, "node_id": node_id}
        if labels:
            if "y" not in stored:
                raise ValueError("evaluation requires y in the canonical dataset")
            result["y"] = np.asarray(stored["y"], dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("x must have shape [N, F]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if node_id.shape != (x.shape[0],) or np.unique(node_id).size != node_id.size:
        raise ValueError("node_id must contain one unique id per row")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
        raise ValueError("edge_index contains an out-of-range row index")
    return result


def _split(path: str | Path, n: int) -> dict[str, np.ndarray | int]:
    raw = _read_json(path)
    folds = int(raw.get("folds", OUTER_FOLDS))
    if folds != OUTER_FOLDS:
        raise ValueError("the release protocol requires exactly five outer folds")
    normal = np.asarray(raw["normal_nodes"], dtype=np.int64)
    unlabeled = np.asarray(raw["unlabeled_nodes"], dtype=np.int64)
    ownership = np.asarray(raw["ownership"], dtype=np.int8)
    if ownership.shape != unlabeled.shape:
        raise ValueError("ownership must align with unlabeled_nodes")
    if np.unique(normal).size != normal.size or np.unique(unlabeled).size != unlabeled.size:
        raise ValueError("split node lists must be unique")
    if np.intersect1d(normal, unlabeled).size:
        raise ValueError("labeled-normal and unlabeled nodes must be disjoint")
    joined = np.concatenate([normal, unlabeled])
    if joined.size and (joined.min() < 0 or joined.max() >= n):
        raise ValueError("split contains an out-of-range row index")
    if ownership.size and np.any((ownership < 0) | (ownership >= OUTER_FOLDS)):
        raise ValueError("ownership values must be in [0, 4]")
    return {
        "folds": folds,
        "normal_nodes": normal,
        "unlabeled_nodes": unlabeled,
        "ownership": ownership,
    }


def _support(path: str | Path, split: dict[str, Any]) -> list[np.ndarray]:
    raw = _read_json(path)
    if int(raw.get("folds", OUTER_FOLDS)) != OUTER_FOLDS:
        raise ValueError("support manifest must define five folds")
    rows = raw.get("support_nodes_by_fold")
    if not isinstance(rows, list) or len(rows) != OUTER_FOLDS:
        raise ValueError("support_nodes_by_fold must contain five node lists")
    allowed = np.asarray(split["normal_nodes"], dtype=np.int64)
    budget = int(raw.get("support_budget", len(rows[0])))
    supports: list[np.ndarray] = []
    for row in rows:
        nodes = np.asarray(row, dtype=np.int64)
        if nodes.size != budget or np.unique(nodes).size != nodes.size:
            raise ValueError("every fold must use the same unique support budget")
        if np.setdiff1d(nodes, allowed).size:
            raise ValueError("support nodes must be labeled-normal nodes")
        supports.append(nodes)
    return supports


def validate(dataset: str, split_manifest: str, support_manifest: str) -> dict[str, Any]:
    graph = _dataset(dataset)
    split = _split(split_manifest, graph["x"].shape[0])
    supports = _support(support_manifest, split)
    fold_sizes = [
        int(np.sum(np.asarray(split["ownership"]) == fold))
        for fold in range(OUTER_FOLDS)
    ]
    return {
        "valid": True,
        "num_nodes": int(graph["x"].shape[0]),
        "num_features": int(graph["x"].shape[1]),
        "num_edges": int(graph["edge_index"].shape[1]),
        "normal_nodes": int(np.asarray(split["normal_nodes"]).size),
        "unlabeled_nodes": int(np.asarray(split["unlabeled_nodes"]).size),
        "support_budget": int(supports[0].size),
        "fold_sizes": fold_sizes,
    }


def prepare_fold(
    dataset: str,
    split_manifest: str,
    support_manifest: str,
    fold: int,
    output: str,
) -> Path:
    if fold not in range(OUTER_FOLDS):
        raise ValueError("fold must be in [0, 4]")
    graph = _dataset(dataset, labels=False)
    split = _split(split_manifest, graph["x"].shape[0])
    supports = _support(support_manifest, split)
    normal = np.asarray(split["normal_nodes"], dtype=np.int64)
    unlabeled = np.asarray(split["unlabeled_nodes"], dtype=np.int64)
    ownership = np.asarray(split["ownership"], dtype=np.int8)
    heldout = unlabeled[ownership == fold]
    training_unlabeled = unlabeled[ownership != fold]
    train_visible = np.sort(np.concatenate([normal, training_unlabeled]))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=graph["x"],
        edge_index=graph["edge_index"],
        node_id=graph["node_id"],
        labeled_normal_nodes=normal,
        support_normal_nodes=supports[fold],
        training_unlabeled_nodes=training_unlabeled,
        train_visible_nodes=train_visible,
        score_nodes=heldout,
        owner_fold=np.asarray(fold, dtype=np.int8),
    )
    return output_path


def collect_scores(dataset: str, split_manifest: str, score_dir: str, output: str) -> Path:
    graph = _dataset(dataset)
    split = _split(split_manifest, graph["x"].shape[0])
    unlabeled = np.asarray(split["unlabeled_nodes"], dtype=np.int64)
    ownership = np.asarray(split["ownership"], dtype=np.int8)
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold in range(OUTER_FOLDS):
        path = Path(score_dir) / f"fold_{fold}.npz"
        with np.load(path, allow_pickle=False) as stored:
            nodes = np.asarray(stored["node_id"], dtype=np.int64)
            scores = np.asarray(stored["score"], dtype=np.float64)
        expected = unlabeled[ownership == fold]
        if nodes.shape != scores.shape or not np.array_equal(np.sort(nodes), np.sort(expected)):
            raise ValueError(f"fold {fold} scores do not exactly match its OOF ownership")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"fold {fold} produced non-finite scores")
        rows.append((nodes, scores, np.full(nodes.shape, fold, dtype=np.int8)))
    node_id = np.concatenate([row[0] for row in rows])
    score = np.concatenate([row[1] for row in rows])
    owner = np.concatenate([row[2] for row in rows])
    order = np.argsort(node_id, kind="mergesort")
    if np.unique(node_id).size != unlabeled.size:
        raise ValueError("every unlabeled node must receive exactly one OOF score")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, node_id=node_id[order], score=score[order], owner_fold=owner[order])
    return output_path


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    valid = np.isin(labels, [0, 1]) & np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("evaluation requires both normal and anomaly labels")
    k = max(1, int(labels.sum()))
    top = np.argsort(-scores, kind="mergesort")[:k]
    true_positive = int(labels[top].sum())
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "recall_at_k": float(true_positive / max(int(labels.sum()), 1)),
        "precision_at_k": float(true_positive / k),
        "ndcg_at_k": float(ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1), k=k)),
        "k": float(k),
    }


def evaluate(dataset: str, oof_scores: str, evaluation_manifest: str | None = None) -> dict[str, float]:
    graph = _dataset(dataset, labels=True)
    with np.load(oof_scores, allow_pickle=False) as stored:
        node_id = np.asarray(stored["node_id"], dtype=np.int64)
        score = np.asarray(stored["score"], dtype=np.float64)
        owner = np.asarray(stored["owner_fold"], dtype=np.int8)
    if node_id.shape != score.shape or node_id.shape != owner.shape:
        raise ValueError("OOF node, score, and owner arrays must align")
    if np.unique(node_id).size != node_id.size or np.any((owner < 0) | (owner >= OUTER_FOLDS)):
        raise ValueError("OOF ownership is invalid")
    if evaluation_manifest:
        allowed = np.asarray(_read_json(evaluation_manifest)["evaluation_nodes"], dtype=np.int64)
        keep = np.isin(node_id, allowed)
        node_id, score = node_id[keep], score[keep]
    return _metrics(np.asarray(graph["y"])[node_id], score)


def execute_template(command: str, env: dict[str, str], *, allow_training: bool) -> int:
    rendered = command.format(**env)
    if not allow_training:
        print(rendered)
        return 0
    process_env = os.environ.copy()
    process_env.update({f"COREGAD_{key.upper()}": value for key, value in env.items()})
    return subprocess.run(shlex.split(rendered), check=False, env=process_env).returncode


def build_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"CoReGAD protocol adapter for {method}")
    parser.set_defaults(method=method)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("validate", "prepare"):
        item = sub.add_parser(name)
        item.add_argument("--dataset", required=True)
        item.add_argument("--split-manifest", required=True)
        item.add_argument("--support-manifest", required=True)
        if name == "prepare":
            item.add_argument("--fold", type=int, required=True)
            item.add_argument("--output", required=True)
    item = sub.add_parser("collect")
    item.add_argument("--dataset", required=True)
    item.add_argument("--split-manifest", required=True)
    item.add_argument("--score-dir", required=True)
    item.add_argument("--output", required=True)
    item = sub.add_parser("evaluate")
    item.add_argument("--dataset", required=True)
    item.add_argument("--oof-scores", required=True)
    item.add_argument("--evaluation-manifest")
    item = sub.add_parser("command")
    item.add_argument("--template", required=True)
    item.add_argument("--fold-bundle", required=True)
    item.add_argument("--score-output", required=True)
    item.add_argument("--fold", type=int, required=True)
    item.add_argument("--allow-training", action="store_true")
    return parser


def main(method: str = "baseline") -> int:
    args = build_parser(method).parse_args()
    if args.action == "validate":
        print(json.dumps(validate(args.dataset, args.split_manifest, args.support_manifest), indent=2))
    elif args.action == "prepare":
        print(prepare_fold(args.dataset, args.split_manifest, args.support_manifest, args.fold, args.output))
    elif args.action == "collect":
        print(collect_scores(args.dataset, args.split_manifest, args.score_dir, args.output))
    elif args.action == "evaluate":
        print(json.dumps(evaluate(args.dataset, args.oof_scores, args.evaluation_manifest), indent=2))
    elif args.action == "command":
        return execute_template(
            args.template,
            {
                "method": method,
                "fold": str(args.fold),
                "fold_bundle": str(Path(args.fold_bundle).resolve()),
                "score_output": str(Path(args.score_output).resolve()),
            },
            allow_training=args.allow_training,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
