import numpy as np
import torch

from coregad.models.structural_residualization import (
    CrossFittedNuisanceEstimator,
    inner_fold_ownership,
)


def test_nuisance_crossfit_isolation() -> None:
    generator = torch.Generator().manual_seed(11)
    structural = torch.randn(80, 3, generator=generator)
    targets = torch.randn(80, 2, generator=generator)
    node_ids = np.arange(80, dtype=np.int64)
    ownership = inner_fold_ownership(node_ids)
    changed = targets.clone()
    changed[torch.from_numpy(ownership == 2)] += 1.0
    first = CrossFittedNuisanceEstimator().fit(structural, targets, node_ids).predict_oof(structural)
    second = CrossFittedNuisanceEstimator().fit(structural, changed, node_ids).predict_oof(structural)
    held = torch.from_numpy(ownership == 2)
    # The implementation forms global sufficient statistics and subtracts the
    # owned fold, so float32 matrix products can leave only roundoff residue.
    assert torch.allclose(first[held], second[held], atol=2.0e-5, rtol=0.0)
