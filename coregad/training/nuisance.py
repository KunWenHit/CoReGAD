from __future__ import annotations

import numpy as np
import torch

from coregad.models.structural_residualization import ControlledStructuralResidualizer


def fit_controlled_residualizer(
    spectral_discrepancy: torch.Tensor,
    structural_statistics: torch.Tensor,
    node_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[ControlledStructuralResidualizer, dict[str, torch.Tensor]]:
    residualizer = ControlledStructuralResidualizer(seed=seed)
    outputs = residualizer.fit_transform(
        spectral_discrepancy.detach(), structural_statistics.detach(), node_ids
    )
    return residualizer, outputs
