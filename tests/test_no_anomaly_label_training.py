import inspect

from coregad.training.normality import train_cross_fitted_normality_fold
from coregad.training.pipeline import train_fold


def test_no_anomaly_label_training() -> None:
    for function in (train_cross_fitted_normality_fold, train_fold):
        signature = inspect.signature(function)
        assert "labels" not in signature.parameters
