from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


class AttributeEncoder(nn.Module):
    """Frozen attribute encoder architecture used by the released core."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(in_dim), int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.relu(self.fc1(features)))


class CrossFittedNormalityCore(nn.Module):
    """Attribute-only normality model used independently in every outer fold."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AttributeEncoder(in_dim, hidden_dim, dropout)
        self.normality_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        node_embeddings = self.encoder(features)
        normality_logit = self.normality_head(node_embeddings).reshape(-1)
        base_anomaly_score = 1.0 - torch.sigmoid(normality_logit)
        return {
            "node_embeddings": node_embeddings,
            "normality_logit": normality_logit,
            "base_anomaly_logit": -normality_logit,
            "base_anomaly_score": base_anomaly_score,
        }

    def freeze(self) -> "FrozenNormalityCore":
        return FrozenNormalityCore.from_trained(self)


class FrozenNormalityCore(nn.Module):
    """Read-only normality core used by every graph-residual stage."""

    def __init__(self, encoder: nn.Module, normality_head: nn.Linear) -> None:
        super().__init__()
        self.encoder = encoder
        self.normality_head = normality_head
        self.freeze_parameters()

    @classmethod
    def from_trained(cls, model: CrossFittedNormalityCore) -> "FrozenNormalityCore":
        return cls(copy.deepcopy(model.encoder), copy.deepcopy(model.normality_head))

    def freeze_parameters(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenNormalityCore":
        super().train(False)
        return self

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            node_embeddings = self.encoder(features)
            normality_logit = self.normality_head(node_embeddings).reshape(-1)
            base_anomaly_logit = -normality_logit
            return {
                "node_embeddings": node_embeddings,
                "normality_logit": normality_logit,
                "base_anomaly_logit": base_anomaly_logit,
                "base_anomaly_score": torch.sigmoid(base_anomaly_logit),
            }


def balanced_normal_unlabeled_loss(
    normal_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
) -> torch.Tensor:
    """Balanced normal-positive versus unlabeled-negative objective."""

    normal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        normal_logits, torch.ones_like(normal_logits), reduction="mean"
    )
    unlabeled_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        unlabeled_logits, torch.zeros_like(unlabeled_logits), reduction="mean"
    )
    return 0.5 * normal_loss + 0.5 * unlabeled_loss
