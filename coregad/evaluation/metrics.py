from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isin(labels, [0, 1]) & np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        raise ValueError("evaluation requires both normal and anomaly labels")
    k = max(1, int(labels.sum()))
    order = np.argsort(-scores, kind="mergesort")[:k]
    true_positive = int(labels[order].sum())
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "recall_at_k": float(true_positive / max(int(labels.sum()), 1)),
        "precision_at_k": float(true_positive / max(k, 1)),
        "ndcg_at_k": float(
            ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1), k=k)
        ),
        "k": float(k),
    }
