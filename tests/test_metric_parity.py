import numpy as np

from coregad.evaluation.metrics import ranking_metrics


def test_metric_k_is_the_fixed_anomaly_count() -> None:
    labels = np.asarray([0, 1, 0, 1, 0])
    scores = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3])
    result = ranking_metrics(labels, scores)
    assert result["k"] == 2.0
    assert result["auprc"] == 1.0
    assert result["recall_at_k"] == result["precision_at_k"] == 1.0
