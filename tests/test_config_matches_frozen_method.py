from pathlib import Path

import yaml

from coregad.models.structural_residualization import (
    INNER_FOLDS,
    RFF_DIM,
    RIDGE_LAMBDA,
    STRUCTURAL_RESIDUAL_STRENGTH,
)


def test_config_matches_frozen_method() -> None:
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "coregad.yaml").read_text(encoding="utf-8")
    )
    assert config["method"]["outer_oof_folds"] == 5
    assert config["nuisance_estimator"]["inner_folds"] == INNER_FOLDS
    assert config["nuisance_estimator"]["random_fourier_dimension"] == RFF_DIM
    assert config["nuisance_estimator"]["ridge_lambda"] == RIDGE_LAMBDA
    assert config["controlled_structural_residualization"]["residual_strength"] == STRUCTURAL_RESIDUAL_STRENGTH
    assert config["residual_training"]["epochs"] == 300
