from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from coregad.models.graph_context import (
    GraphContextShaping,
    anchor_preservation_loss,
    context_features_from_index_pack,
    context_rows_for_normal_nodes,
    context_weight,
    drop_context_rows,
    graph_context_loss,
    pack_context_rows,
)
from coregad.models.normality_core import (
    CrossFittedNormalityCore,
    FrozenNormalityCore,
    balanced_normal_unlabeled_loss,
)


BASE_EPOCHS = 200
BASE_LEARNING_RATE = 1.0e-3
CONTEXT_EPOCHS = 200
CONTEXT_LEARNING_RATE = 5.0e-4
WEIGHT_DECAY = 1.0e-4
CONTEXT_WEIGHT = 0.8498013479600357
ANCHOR_WEIGHT = 0.18573695284155523
CONTEXT_WARMUP_RATIO = 0.1
NEIGHBOR_DROPOUT = 0.1


@dataclass
class NormalityFoldResult:
    frozen_core: FrozenNormalityCore
    graph_context_shaping: GraphContextShaping
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor


def fit_feature_scaler(
    features: torch.Tensor,
    normal_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = features[normal_nodes].float()
    mean = selected.mean(dim=0)
    scale = selected.std(dim=0, unbiased=False)
    return mean, torch.where(scale > 1.0e-6, scale, torch.ones_like(scale))


def train_cross_fitted_normality_fold(
    *,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    normal_nodes: torch.Tensor,
    training_unlabeled_nodes: torch.Tensor,
    heldout_nodes: torch.Tensor,
    seed: int,
    device: torch.device | str,
    base_epochs: int = BASE_EPOCHS,
    context_epochs: int = CONTEXT_EPOCHS,
) -> NormalityFoldResult:
    """Train one normal/unlabeled fold without reading anomaly labels."""

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device(device)
    normal_nodes = normal_nodes.long().to(device)
    training_unlabeled_nodes = training_unlabeled_nodes.long().to(device)
    heldout_nodes = heldout_nodes.long().to(device)
    mean, scale = fit_feature_scaler(features, normal_nodes.cpu())
    scaled = ((features.float() - mean) / scale).to(device)
    core = CrossFittedNormalityCore(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        core.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    for _ in range(int(base_epochs)):
        core.train()
        optimizer.zero_grad(set_to_none=True)
        output = core(scaled)
        loss = balanced_normal_unlabeled_loss(
            output["normality_logit"][normal_nodes],
            output["normality_logit"][training_unlabeled_nodes],
        )
        loss.backward()
        optimizer.step()

    teacher = core.freeze()
    with torch.no_grad():
        teacher_output = teacher(scaled)
    student = CrossFittedNormalityCore(features.shape[1]).to(device)
    student.encoder.load_state_dict(copy.deepcopy(teacher.encoder.state_dict()))
    student.normality_head.load_state_dict(
        copy.deepcopy(teacher.normality_head.state_dict())
    )
    student.normality_head.eval()
    for parameter in student.normality_head.parameters():
        parameter.requires_grad_(False)
    shaping = GraphContextShaping(teacher_output["node_embeddings"].shape[1]).to(device)
    context_optimizer = torch.optim.Adam(
        list(student.encoder.parameters()) + list(shaping.parameters()),
        lr=CONTEXT_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    visible_nodes = torch.cat([normal_nodes, training_unlabeled_nodes]).detach().cpu().numpy()
    context_centers, original_rows = context_rows_for_normal_nodes(
        edge_index,
        features.shape[0],
        visible_nodes,
        normal_nodes.detach().cpu().numpy(),
    )
    center_tensor = torch.as_tensor(context_centers, dtype=torch.long, device=device)
    teacher_context_target = teacher_output["node_embeddings"][center_tensor].detach()
    for epoch in range(int(context_epochs)):
        student.train()
        shaping.train()
        context_optimizer.zero_grad(set_to_none=True)
        output = student(scaled)
        normal_unlabeled = balanced_normal_unlabeled_loss(
            output["normality_logit"][normal_nodes],
            output["normality_logit"][training_unlabeled_nodes],
        )
        anchor = anchor_preservation_loss(
            output["normality_logit"], teacher_output["normality_logit"]
        )
        dropped_rows = drop_context_rows(
            original_rows,
            dropout=NEIGHBOR_DROPOUT,
            seed=int(seed) + int(epoch) * 104729,
        )
        context_pack = pack_context_rows(dropped_rows, device)
        context_features = context_features_from_index_pack(
            output["node_embeddings"], context_pack
        )
        context_prediction = shaping(context_features)
        context = graph_context_loss(context_prediction, teacher_context_target)
        weight = context_weight(
            epoch, int(context_epochs), CONTEXT_WEIGHT, CONTEXT_WARMUP_RATIO
        )
        total = normal_unlabeled + ANCHOR_WEIGHT * anchor + weight * context
        total.backward()
        context_optimizer.step()

    frozen = student.freeze()
    shaping.eval()
    for parameter in shaping.parameters():
        parameter.requires_grad_(False)
    return NormalityFoldResult(
        frozen_core=frozen,
        graph_context_shaping=shaping,
        feature_mean=mean.cpu(),
        feature_scale=scale.cpu(),
    )
