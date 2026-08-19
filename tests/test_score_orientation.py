import numpy as np

from coregad.evaluation.orientation import orient_anomaly_scores


def test_score_orientation_is_registry_driven_not_label_driven() -> None:
    scores = np.asarray([0.2, 0.8])
    assert np.array_equal(orient_anomaly_scores(scores, "higher_is_anomaly"), scores)
    assert np.array_equal(orient_anomaly_scores(scores, "higher_is_normal"), -scores)
