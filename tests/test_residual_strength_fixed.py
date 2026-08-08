import pytest

from coregad.models.structural_residualization import (
    ControlledStructuralResidualizer,
    CrossFittedNuisanceEstimator,
)


def test_residual_strength_fixed() -> None:
    assert ControlledStructuralResidualizer.residual_strength == 0.75
    with pytest.raises(ValueError):
        CrossFittedNuisanceEstimator(rff_dim=128)
