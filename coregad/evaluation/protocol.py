from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import ranking_metrics


def evaluate_oof_file(
    score_path: str | Path,
    labels: np.ndarray,
) -> dict[str, float]:
    with np.load(Path(score_path), allow_pickle=False) as stored:
        node_ids = np.asarray(stored["node_id"], dtype=np.int64)
        scores = np.asarray(stored["score"], dtype=np.float64)
        owners = np.asarray(stored["owner_fold"], dtype=np.int8)
    if node_ids.shape != scores.shape or node_ids.shape != owners.shape:
        raise ValueError("OOF node, score and owner arrays must align")
    if np.unique(node_ids).size != node_ids.size:
        raise ValueError("OOF nodes must be unique")
    if bool(np.any((owners < 0) | (owners >= 5))):
        raise ValueError("OOF owner must be one of five outer folds")
    return ranking_metrics(np.asarray(labels)[node_ids], scores)
