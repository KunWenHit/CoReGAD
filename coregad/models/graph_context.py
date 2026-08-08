from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ContextIndexPack:
    flat_indices: torch.Tensor
    segment_ids: torch.Tensor
    counts: torch.Tensor


class GraphContextShaping(nn.Module):
    """Training-only predictor that shapes attribute normality embeddings."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input = nn.Linear(2 * int(hidden_dim), int(hidden_dim))
        self.activation = nn.GELU()
        self.output = nn.Linear(int(hidden_dim), int(hidden_dim))

    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.input(context_features)))


class _ZeroGradientAtZeroSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        result = torch.sqrt(torch.clamp(value, min=0.0))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        (result,) = ctx.saved_tensors
        output = torch.zeros_like(gradient)
        active = result > 0.0
        output[active] = gradient[active] / (2.0 * result[active])
        return (output,)


def context_features_from_index_pack(
    node_embeddings: torch.Tensor,
    context_pack: ContextIndexPack,
) -> torch.Tensor:
    hidden_dim = int(node_embeddings.shape[1])
    context_count = int(context_pack.counts.numel())
    if context_count == 0:
        return node_embeddings.new_zeros((0, 2 * hidden_dim))
    neighbor_embeddings = F.layer_norm(
        node_embeddings[context_pack.flat_indices], (hidden_dim,)
    )
    sums = node_embeddings.new_zeros((context_count, hidden_dim))
    sums.index_add_(0, context_pack.segment_ids, neighbor_embeddings)
    counts = context_pack.counts.to(dtype=node_embeddings.dtype).clamp_min(1.0).unsqueeze(1)
    means = sums / counts
    square_sums = node_embeddings.new_zeros((context_count, hidden_dim))
    square_sums.index_add_(0, context_pack.segment_ids, neighbor_embeddings.square())
    standard_deviations = _ZeroGradientAtZeroSqrt.apply(
        torch.clamp(square_sums / counts - means.square(), min=0.0)
    )
    return torch.cat([means, standard_deviations], dim=1)


def graph_context_loss(
    prediction: torch.Tensor,
    frozen_teacher_target: torch.Tensor,
) -> torch.Tensor:
    if frozen_teacher_target.numel() == 0:
        return frozen_teacher_target.sum() * 0.0
    target = F.layer_norm(
        frozen_teacher_target.detach(), (frozen_teacher_target.shape[1],)
    )
    predicted = F.layer_norm(prediction, (prediction.shape[1],))
    return F.smooth_l1_loss(predicted, target, reduction="none").mean(dim=1).mean()


def anchor_preservation_loss(
    student_logits: torch.Tensor,
    frozen_teacher_logits: torch.Tensor,
) -> torch.Tensor:
    return F.smooth_l1_loss(
        student_logits, frozen_teacher_logits.detach(), reduction="mean"
    )


def context_weight(epoch: int, epochs: int, target: float, warmup_ratio: float) -> float:
    if epochs <= 0 or not 0 <= epoch < epochs:
        raise ValueError("epoch must be inside a positive training schedule")
    warmup_epochs = int(np.ceil(int(epochs) * float(warmup_ratio)))
    if warmup_epochs <= 1 or epoch >= warmup_epochs:
        return float(target)
    return float(target) * float(epoch) / float(warmup_epochs - 1)


def pack_context_rows(rows: list[np.ndarray], device: torch.device | str) -> ContextIndexPack:
    if any(np.asarray(row).size == 0 for row in rows):
        raise ValueError("graph contexts must be nonempty")
    counts = np.asarray([np.asarray(row).size for row in rows], dtype=np.int64)
    flat = np.concatenate([np.asarray(row, dtype=np.int64) for row in rows]) if rows else np.empty(0, dtype=np.int64)
    segments = np.repeat(np.arange(len(rows), dtype=np.int64), counts)
    return ContextIndexPack(
        flat_indices=torch.as_tensor(flat, dtype=torch.long, device=device),
        segment_ids=torch.as_tensor(segments, dtype=torch.long, device=device),
        counts=torch.as_tensor(counts, dtype=torch.float32, device=device),
    )


def training_neighbors(
    edge_index: torch.Tensor,
    num_nodes: int,
    visible_nodes: np.ndarray,
) -> dict[int, np.ndarray]:
    visible = np.zeros(int(num_nodes), dtype=bool)
    visible[np.asarray(visible_nodes, dtype=np.int64)] = True
    result: list[set[int]] = [set() for _ in range(int(num_nodes))]
    for source, target in edge_index.detach().cpu().long().t().tolist():
        source, target = int(source), int(target)
        if source == target or not visible[source] or not visible[target]:
            continue
        result[target].add(source)
    return {
        node: np.asarray(sorted(values), dtype=np.int64)
        for node, values in enumerate(result)
    }


def context_rows_for_normal_nodes(
    edge_index: torch.Tensor,
    num_nodes: int,
    visible_nodes: np.ndarray,
    normal_nodes: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    neighbors = training_neighbors(edge_index, num_nodes, visible_nodes)
    centers: list[int] = []
    rows: list[np.ndarray] = []
    for node in np.asarray(normal_nodes, dtype=np.int64).tolist():
        row = neighbors[int(node)]
        if row.size:
            centers.append(int(node))
            rows.append(row)
    return np.asarray(centers, dtype=np.int64), rows


def drop_context_rows(
    rows: list[np.ndarray],
    *,
    dropout: float,
    seed: int,
) -> list[np.ndarray]:
    if not 0.0 <= float(dropout) <= 1.0:
        raise ValueError("neighbor dropout must be in [0, 1]")
    generator = np.random.default_rng(int(seed))
    result: list[np.ndarray] = []
    for row in rows:
        values = np.asarray(row, dtype=np.int64)
        if values.size <= 1 or dropout == 0.0:
            result.append(values.copy())
            continue
        keep = generator.random(values.size) >= float(dropout)
        if not bool(np.any(keep)):
            keep[int(generator.integers(0, values.size))] = True
        result.append(values[keep].astype(np.int64, copy=True))
    return result
