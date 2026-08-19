"""Label-isolated adapter boundary for STANDARD_TRANSDUCTIVE_NORMAL_ONLY.

Training preparation deliberately has no label argument and never opens ``y``.
The evaluator is a separate call and is the only code path that loads labels.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_training_graph(path: str | Path) -> dict[str, np.ndarray]:
    """Load canonical covariates only, even if the storage file also has y."""
    with np.load(Path(path), allow_pickle=False) as stored:
        x = np.asarray(stored["x"], dtype=np.float32)
        edge_index = np.asarray(stored["edge_index"], dtype=np.int64)
        node_id = (
            np.asarray(stored["node_id"], dtype=np.int64)
            if "node_id" in stored
            else np.arange(x.shape[0], dtype=np.int64)
        )
    if x.ndim != 2:
        raise ValueError("x must have shape [N, F]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if node_id.shape != (x.shape[0],) or np.unique(node_id).size != node_id.size:
        raise ValueError("node_id must contain one unique id per canonical row")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
        raise ValueError("edge_index contains an out-of-range canonical row")
    return {"x": x, "edge_index": edge_index, "node_id": node_id}


def load_normal_support(path: str | Path, *, num_nodes: int) -> np.ndarray:
    raw = _read_json(path)
    values = raw.get("normal_support_ids", raw.get("train_normal_node_ids"))
    if values is None and "support_nodes_by_fold" in raw:
        rows = raw["support_nodes_by_fold"]
        if not rows:
            raise ValueError("support_nodes_by_fold is empty")
        values = rows[0]
    if values is None:
        raise ValueError("support manifest does not contain normal support ids")
    support = np.asarray(values, dtype=np.int64)
    budget = int(raw.get("normal_label_budget", raw.get("support_budget", support.size)))
    if support.shape != (budget,) or np.unique(support).size != support.size:
        raise ValueError("normal support must exactly match its unique frozen budget")
    if support.size and (support.min() < 0 or support.max() >= num_nodes):
        raise ValueError("normal support contains an out-of-range node")
    return support


def load_evaluation_nodes(path: str | Path, *, num_nodes: int) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            nodes = [int(row["node_id"]) for row in reader]
    else:
        raw = _read_json(source)
        nodes = raw.get("evaluation_nodes", raw.get("unlabeled_nodes"))
        if nodes is None:
            raise ValueError("evaluation manifest does not expose node ids")
    result = np.asarray(nodes, dtype=np.int64)
    if np.unique(result).size != result.size:
        raise ValueError("evaluation node ids must be unique")
    if result.size and (result.min() < 0 or result.max() >= num_nodes):
        raise ValueError("evaluation manifest contains an out-of-range node")
    return result


def prepare_transductive_bundle(
    dataset: str | Path,
    support_manifest: str | Path,
    evaluation_manifest: str | Path,
    output: str | Path,
    *,
    uses_normal_support: bool,
) -> Path:
    """Write one full-graph, label-free native-training bundle.

    No outer ownership, fold, held-out covariates, or altered graph is created.
    """
    graph = load_training_graph(dataset)
    support = (
        load_normal_support(support_manifest, num_nodes=graph["x"].shape[0])
        if uses_normal_support
        else np.empty(0, dtype=np.int64)
    )
    score_nodes = load_evaluation_nodes(
        evaluation_manifest, num_nodes=graph["x"].shape[0]
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=graph["x"],
        edge_index=graph["edge_index"],
        node_id=graph["node_id"],
        normal_support_ids=support,
        training_nodes=np.arange(graph["x"].shape[0], dtype=np.int64),
        score_nodes=score_nodes,
        protocol=np.asarray("STANDARD_TRANSDUCTIVE_NORMAL_ONLY"),
    )
    return output_path


def write_scores(
    output: str | Path,
    node_id: Iterable[int] | np.ndarray,
    anomaly_score: Iterable[float] | np.ndarray,
    *,
    expected_nodes: Iterable[int] | np.ndarray | None = None,
) -> Path:
    nodes = np.asarray(node_id, dtype=np.int64)
    scores = np.asarray(anomaly_score, dtype=np.float64)
    if nodes.ndim != 1 or scores.shape != nodes.shape:
        raise ValueError("node_id and anomaly_score must be aligned vectors")
    if np.unique(nodes).size != nodes.size or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be unique by node and finite")
    if expected_nodes is not None:
        expected = np.asarray(expected_nodes, dtype=np.int64)
        if not np.array_equal(np.sort(nodes), np.sort(expected)):
            raise ValueError("scores do not exactly cover the evaluation nodes")
    order = np.argsort(nodes, kind="mergesort")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("node_id", "anomaly_score"))
        writer.writerows(zip(nodes[order].tolist(), scores[order].tolist()))
    return output_path


def read_scores(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["node_id", "anomaly_score"]:
            raise ValueError("score file must contain node_id,anomaly_score")
        rows = list(reader)
    nodes = np.asarray([int(row["node_id"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["anomaly_score"]) for row in rows], dtype=np.float64)
    if np.unique(nodes).size != nodes.size or not np.all(np.isfinite(scores)):
        raise ValueError("score rows must be unique and finite")
    return nodes, scores


def evaluate_scores(
    evaluation_dataset: str | Path,
    score_file: str | Path,
    evaluation_nodes: str | Path | None = None,
) -> dict[str, float]:
    """Independent label-opening evaluator; never import this in native training."""
    with np.load(Path(evaluation_dataset), allow_pickle=False) as stored:
        if "y" not in stored:
            raise ValueError("evaluation dataset must contain y")
        labels = np.asarray(stored["y"], dtype=np.int64)
    nodes, scores = read_scores(score_file)
    if evaluation_nodes is not None:
        expected = load_evaluation_nodes(evaluation_nodes, num_nodes=labels.size)
        if not np.array_equal(np.sort(nodes), np.sort(expected)):
            raise ValueError("score file does not exactly match the evaluation set")
    y = labels[nodes]
    valid = np.isin(y, (0, 1)) & np.isfinite(scores)
    y, scores = y[valid], scores[valid]
    if y.size == 0 or np.unique(y).size != 2:
        raise ValueError("evaluation requires normal and anomaly labels")
    k = max(1, int(y.sum()))
    top = np.argsort(-scores, kind="mergesort")[:k]
    true_positive = int(y[top].sum())
    return {
        "auprc": float(average_precision_score(y, scores)),
        "auroc": float(roc_auc_score(y, scores)),
        "recall_at_k": float(true_positive / int(y.sum())),
        "precision_at_k": float(true_positive / k),
        "ndcg_at_k": float(ndcg_score(y.reshape(1, -1), scores.reshape(1, -1), k=k)),
        "k": float(k),
    }
