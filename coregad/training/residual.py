from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from coregad.models.coregad import CoReGAD


RESIDUAL_EPOCHS = 300
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
CORRECTION_PENALTY = 1.0e-3


def normal_unlabeled_residual_objective(
    output: dict[str, torch.Tensor],
    roles: torch.Tensor,
) -> torch.Tensor:
    normal = roles == 0
    unlabeled = roles == 1
    objective = output["final_anomaly_logit"].sum() * 0.0
    if bool(normal.any()):
        objective = objective + 0.5 * F.softplus(
            output["final_anomaly_logit"][normal]
        ).mean()
    if bool(unlabeled.any()):
        objective = objective + 0.5 * F.softplus(
            -output["final_anomaly_logit"][unlabeled]
        ).mean()
    return objective + CORRECTION_PENALTY * output["graph_correction"].square().mean()


def train_residual_detector(
    *,
    controlled_spectral_residual: torch.Tensor,
    structural_statistics: torch.Tensor,
    base_anomaly_logit: torch.Tensor,
    roles: torch.Tensor,
    seed: int,
    epochs: int = RESIDUAL_EPOCHS,
) -> CoReGAD:
    if controlled_spectral_residual.requires_grad or structural_statistics.requires_grad:
        raise ValueError("normality and nuisance outputs must be frozen")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = CoReGAD().to(controlled_spectral_residual.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            controlled_spectral_residual,
            structural_statistics,
            base_anomaly_logit,
        )
        loss = normal_unlabeled_residual_objective(output, roles)
        loss.backward()
        optimizer.step()
    model.eval()
    return model
