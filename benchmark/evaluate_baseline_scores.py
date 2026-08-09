"""Evaluation-only process for unified baseline score artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent


def _manifest(dataset: str) -> dict[str, Any]:
    return json.loads((RELEASE / "datasets" / "manifests" / f"{dataset}.json").read_text(encoding="utf-8"))


def _labels(dataset: str) -> np.ndarray:
    manifest = _manifest(dataset)
    source = Path(manifest["raw_source_path"])
    evaluation_mask: np.ndarray | None = None
    if dataset == "Amazon":
        labels = np.asarray(sio.loadmat(source, variable_names=["label"])["label"]).reshape(-1)
    elif dataset == "Elliptic":
        labels = np.asarray(sio.loadmat(source, variable_names=["Label"])["Label"]).reshape(-1)
    elif dataset == "Tolokers":
        with np.load(source, allow_pickle=False) as stored:
            labels = np.asarray(stored["node_labels"]).reshape(-1)
    else:
        import dgl

        graphs, _ = dgl.load_graphs(str(source))
        graph = graphs[0]
        labels = graph.ndata["label"].detach().cpu().numpy().reshape(-1)
        if dataset == "DGraph-Fin" and "mark" in graph.ndata:
            evaluation_mask = graph.ndata["mark"].detach().cpu().numpy().reshape(-1).astype(bool)
    labels = labels.astype(np.int64, copy=False)
    if labels.shape != (int(manifest["num_nodes"]),):
        raise RuntimeError("evaluation label identity mismatch")
    if evaluation_mask is not None:
        labels = labels.copy()
        labels[~evaluation_mask] = -1
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.scores, allow_pickle=False) as stored:
        node_id = np.asarray(stored["node_id"], dtype=np.int64)
        score = np.asarray(stored["anomaly_score"], dtype=np.float64)
        direction = bool(np.asarray(stored["higher_is_more_anomalous"]).item())
    if not direction:
        raise RuntimeError("score artifact direction is not higher-is-more-anomalous")
    labels = _labels(args.dataset)[node_id]
    valid = np.isin(labels, (0, 1)) & np.isfinite(score)
    labels, score = labels[valid], score[valid]
    if labels.size == 0 or np.unique(labels).size != 2:
        raise RuntimeError("evaluation subset does not contain both classes")
    from coregad.evaluation.metrics import ranking_metrics

    metrics = ranking_metrics(labels, score)
    result = {"resource_status": "PASS", "evaluation_only": True, "evaluated_nodes": int(labels.size), "metrics": metrics}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
