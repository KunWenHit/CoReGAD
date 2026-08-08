from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


GLOBAL_SPECTRAL_LOGITS = (1.0, 0.0, 0.0)
LAYER_NORM_EPS = 1.0e-5


def fold_safe_normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    visible_nodes: torch.Tensor,
    heldout_nodes: torch.Tensor,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the released row-oriented fold-safe normalized graph operator."""

    edge = edge_index.detach().cpu().long()
    visible = torch.zeros(int(num_nodes), dtype=torch.bool)
    heldout = torch.zeros(int(num_nodes), dtype=torch.bool)
    visible[visible_nodes.detach().cpu().long()] = True
    heldout[heldout_nodes.detach().cpu().long()] = True
    source, destination = edge[0], edge[1]
    keep = (
        (source != destination)
        & visible[destination]
        & (visible[source] | heldout[source])
    )
    source, destination = source[keep], destination[keep]
    degree = torch.bincount(source, minlength=int(num_nodes)).to(torch.float32)
    degree_with_self = degree + 1.0
    nodes = torch.arange(int(num_nodes), dtype=torch.long)
    rows = torch.cat([source, nodes])
    columns = torch.cat([destination, nodes])
    values = torch.rsqrt(degree_with_self[rows] * degree_with_self[columns])
    adjacency = torch.sparse_coo_tensor(
        torch.stack([rows, columns]),
        values,
        (int(num_nodes), int(num_nodes)),
        check_invariants=False,
    ).coalesce().to(device)
    original_degree = torch.bincount(
        edge[0][edge[0] != edge[1]], minlength=int(num_nodes)
    ).to(torch.float32)
    support_ratio = degree / original_degree.clamp_min(1.0)
    return adjacency, degree.to(device), support_ratio.clamp(0.0, 1.0).to(device)


class GlobalSpectralReference(nn.Module):
    """Construct the frozen shared low/band/high spectral reference."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "global_spectral_logits",
            torch.tensor(GLOBAL_SPECTRAL_LOGITS, dtype=torch.float32),
        )

    @property
    def global_spectral_weights(self) -> torch.Tensor:
        return torch.softmax(self.global_spectral_logits, dim=0)

    def forward(
        self,
        normalized_adjacency: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        propagate = torch.sparse.mm if normalized_adjacency.is_sparse else torch.matmul
        low = propagate(normalized_adjacency, node_embeddings)
        low_twice = propagate(normalized_adjacency, low)
        band = low - low_twice
        high = node_embeddings - low
        weights = self.global_spectral_weights.to(node_embeddings.device)
        spectral_reference = weights[0] * low + weights[1] * band + weights[2] * high
        return {
            "low_spectral_component": low,
            "band_spectral_component": band,
            "high_spectral_component": high,
            "global_spectral_weights": weights,
            "spectral_reference": spectral_reference,
        }


class SpectralDiscrepancy(nn.Module):
    """Return exactly the two node-varying spectral discrepancy channels."""

    def forward(
        self,
        node_embeddings: torch.Tensor,
        spectral_reference: torch.Tensor,
        normality_head: nn.Linear,
        spectral_components: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_dim = int(node_embeddings.shape[1])
        if spectral_components is None:
            embedding_discrepancy = torch.linalg.vector_norm(
                F.layer_norm(node_embeddings, (hidden_dim,), eps=LAYER_NORM_EPS)
                - F.layer_norm(spectral_reference, (hidden_dim,), eps=LAYER_NORM_EPS),
                dim=1,
            ) / math.sqrt(hidden_dim)
            decision_discrepancy = torch.abs(
                normality_head(node_embeddings).reshape(-1)
                - normality_head(spectral_reference).reshape(-1)
            )
        else:
            basis = torch.stack(
                [
                    spectral_components["low_spectral_component"],
                    spectral_components["band_spectral_component"],
                    spectral_components["high_spectral_component"],
                ],
                dim=1,
            )
            weights = spectral_components["global_spectral_weights"]
            embedding_normalized = F.layer_norm(
                node_embeddings, (hidden_dim,), eps=LAYER_NORM_EPS
            )
            means = basis.mean(dim=2)
            centered = basis - means[:, :, None]
            covariance = torch.einsum(
                "nkd,nld->nkl", centered, centered
            ) / hidden_dim
            dot = torch.einsum(
                "nd,nkd->nk", embedding_normalized, centered
            ) / hidden_dim
            graph_variance = (
                weights[0].square() * covariance[:, 0, 0]
                + 2.0 * weights[0] * weights[1] * covariance[:, 0, 1]
                + 2.0 * weights[0] * weights[2] * covariance[:, 0, 2]
                + weights[1].square() * covariance[:, 1, 1]
                + 2.0 * weights[1] * weights[2] * covariance[:, 1, 2]
                + weights[2].square() * covariance[:, 2, 2]
            ).clamp_min(0.0)
            graph_scale = torch.sqrt(graph_variance + LAYER_NORM_EPS)
            graph_normalized_square = graph_variance / (
                graph_variance + LAYER_NORM_EPS
            )
            weighted_dot = (
                weights[0] * dot[:, 0]
                + weights[1] * dot[:, 1]
                + weights[2] * dot[:, 2]
            )
            embedding_square = embedding_normalized.square().mean(dim=1)
            discrepancy_square = (
                embedding_square
                + graph_normalized_square
                - 2.0 * weighted_dot / graph_scale
            ).clamp_min(0.0)
            embedding_discrepancy = torch.sqrt(
                discrepancy_square.clamp_min(1.0e-12)
            )
            basis_decisions = torch.einsum(
                "nkd,d->nk", basis, normality_head.weight.reshape(-1)
            ) + normality_head.bias.reshape(())
            embedding_decision = torch.einsum(
                "nd,d->n", node_embeddings, normality_head.weight.reshape(-1)
            ) + normality_head.bias.reshape(())
            reference_decision = (
                weights[0] * basis_decisions[:, 0]
                + weights[1] * basis_decisions[:, 1]
                + weights[2] * basis_decisions[:, 2]
            )
            decision_discrepancy = torch.abs(
                embedding_decision - reference_decision
            )
        return {
            "embedding_discrepancy": embedding_discrepancy,
            "decision_discrepancy": decision_discrepancy,
            "spectral_discrepancy": torch.stack(
                [embedding_discrepancy, decision_discrepancy], dim=1
            ),
        }


def structural_statistics(
    normalized_adjacency: torch.Tensor,
    node_embeddings: torch.Tensor,
    visible_degree: torch.Tensor,
    support_ratio: torch.Tensor,
) -> torch.Tensor:
    hidden_dim = int(node_embeddings.shape[1])
    normalized_embeddings = F.layer_norm(
        node_embeddings, (hidden_dim,), eps=LAYER_NORM_EPS
    )
    adjacency = normalized_adjacency.coalesce()
    rows, columns = adjacency.indices()
    nonself = rows != columns
    rows, columns = rows[nonself], columns[nonself]
    local_sum = node_embeddings.new_zeros((node_embeddings.shape[0], hidden_dim))
    local_square_sum = node_embeddings.new_zeros((node_embeddings.shape[0], hidden_dim))
    if rows.numel():
        local_sum.index_add_(0, rows, normalized_embeddings[columns])
        local_square_sum.index_add_(0, rows, normalized_embeddings[columns].square())
    count = visible_degree.to(node_embeddings.device).clamp_min(1.0).unsqueeze(1)
    local_mean = local_sum / count
    local_variance = torch.clamp(local_square_sum / count - local_mean.square(), min=0.0)
    local_embedding_variation = torch.sqrt(local_variance).mean(dim=1)
    local_embedding_variation = torch.where(
        visible_degree.to(node_embeddings.device) > 0,
        local_embedding_variation,
        torch.zeros_like(local_embedding_variation),
    )
    return torch.stack(
        [
            torch.log1p(visible_degree.to(node_embeddings.device)),
            support_ratio.to(node_embeddings.device),
            local_embedding_variation,
        ],
        dim=1,
    )
