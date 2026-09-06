from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from coregad.models.coregad import (
    FULL_F2,
    WO_M1_GRAPH_RESIDUAL_EVIDENCE,
    CoReGAD,
    normalize_model_variant,
)
from coregad.models.routing import (
    build_routing_batch,
    fit_normal_channel_percentiles,
)


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
    raw_spectral_evidence: torch.Tensor | None = None,
    model_variant: str = FULL_F2,
) -> CoReGAD:
    if controlled_spectral_residual.requires_grad or structural_statistics.requires_grad:
        raise ValueError("normality and nuisance outputs must be frozen")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    variant = normalize_model_variant(model_variant)
    model = CoReGAD(model_variant=variant).to(controlled_spectral_residual.device)
    if variant == WO_M1_GRAPH_RESIDUAL_EVIDENCE:
        model.eval()
        return model
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    routing_batch = None
    if raw_spectral_evidence is not None:
        normal_mask = roles == 0
        percentiles = fit_normal_channel_percentiles(
            raw_spectral_evidence, normal_mask
        )
        routing_batch = build_routing_batch(
            raw_evidence=raw_spectral_evidence,
            controlled_evidence=controlled_spectral_residual,
            structural_statistics=structural_statistics,
            base_anomaly_logit=base_anomaly_logit,
            normal_mask=normal_mask,
            normal_channel_percentiles=percentiles,
        )
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if routing_batch is None:
            output = model(
                controlled_spectral_residual,
                structural_statistics,
                base_anomaly_logit,
            )
        else:
            output = model.forward_batch(
                routing_batch, calibration_batch=routing_batch
            )
        loss = normal_unlabeled_residual_objective(output, roles)
        loss.backward()
        optimizer.step()
    model.eval()
    return model
