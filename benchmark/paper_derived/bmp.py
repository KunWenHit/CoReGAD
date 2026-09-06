"""Paper-derived BMP core from AAAI 2026 equations 3--10.

The paper advertises https://github.com/Thankstaro/BMP, but the repository was
not public/reachable during the 2026-08-09 closure audit. This module therefore
must never be represented as author code. It implements the routing/forest and
loss shape faithfully enough for mathematical and adapter tests; formal use is
blocked until native reproduction sanity is completed.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def sorted_node_permutation(probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if probability.ndim != 1:
        raise ValueError("probability must be a vector")
    order = torch.argsort(probability, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return order, inverse


def _normalized_sparse(indices: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=indices.device),
            torch.empty(0, dtype=dtype, device=indices.device),
            (num_nodes, num_nodes),
        ).coalesce()
    row, col = indices
    degree = torch.bincount(row, minlength=num_nodes).to(dtype=dtype).clamp_min_(1.0)
    values = degree[row].rsqrt() * degree[col].rsqrt()
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()


def build_normalized_routes(
    edge_index: torch.Tensor,
    anomaly_probability: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build normalized upper/lower routes after ascending probability sort."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if anomaly_probability.shape != (num_nodes,):
        raise ValueError("one anomaly probability is required per node")
    order, inverse = sorted_node_permutation(anomaly_probability)
    ranked_edges = inverse[edge_index.long()]
    loops = torch.arange(num_nodes, device=edge_index.device)
    ranked_edges = torch.cat([ranked_edges, torch.stack([loops, loops])], dim=1)
    upper = ranked_edges[:, ranked_edges[0] <= ranked_edges[1]]
    lower = ranked_edges[:, ranked_edges[0] >= ranked_edges[1]]
    return (
        _normalized_sparse(upper, num_nodes, anomaly_probability.dtype),
        _normalized_sparse(lower, num_nodes, anomaly_probability.dtype),
        order,
        inverse,
    )


class BMPGraphLayer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dimension, dimension, bias=False)

    def forward(self, route: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.sparse.mm(route, self.linear(value)))


class BMPTree(nn.Module):
    """T*_(p,q): p upper/right layers followed by q lower/left layers."""

    def __init__(self, dimension: int, p: int, q: int) -> None:
        super().__init__()
        if p < 0 or q < 0:
            raise ValueError("p and q must be non-negative")
        self.right = nn.ModuleList(BMPGraphLayer(dimension) for _ in range(p))
        self.left = nn.ModuleList(BMPGraphLayer(dimension) for _ in range(q))

    def forward(self, value: torch.Tensor, upper: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
        for layer in self.right:
            value = layer(upper, value)
        for layer in self.left:
            value = layer(lower, value)
        return value


class BMPForest(nn.Module):
    """Equation 5: T00, T01, T10 and independently parameterized T(p,1..p)."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, order: int = 3) -> None:
        super().__init__()
        if order < 1:
            raise ValueError("forest order must be positive")
        self.order = order
        self.preprocess = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        specs = [(0, 1), (1, 0)] + [(order, q) for q in range(1, order + 1)]
        self.trees = nn.ModuleList(BMPTree(hidden_dim, p, q) for p, q in specs)
        self.output_dim = hidden_dim * (1 + len(specs))

    def forward(
        self,
        x: torch.Tensor,
        upper: torch.Tensor,
        lower: torch.Tensor,
        order: torch.Tensor,
        inverse: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.preprocess(x)[order]
        pieces = [encoded]
        pieces.extend(tree(encoded, upper, lower) for tree in self.trees)
        return torch.cat(pieces, dim=1)[inverse]


class BMP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, order: int = 3) -> None:
        super().__init__()
        self.forest = BMPForest(input_dim, hidden_dim, order)
        self.predictor = nn.Linear(self.forest.output_dim, 1)
        self.masker = nn.Linear(self.forest.output_dim, self.forest.output_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, route_probability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        upper, lower, order, inverse = build_normalized_routes(
            edge_index, route_probability, x.shape[0]
        )
        embedding = self.forest(x, upper, lower, order, inverse)
        probability = torch.sigmoid(self.predictor(embedding)).squeeze(-1)
        mask = torch.sigmoid(self.masker(embedding))
        masked_probability = torch.sigmoid(self.predictor(mask * embedding)).squeeze(-1)
        return probability, masked_probability, mask


def bmp_loss(
    probability: torch.Tensor,
    masked_probability: torch.Tensor,
    mask: torch.Tensor,
    known_nodes: torch.Tensor,
    known_labels: torch.Tensor,
    *,
    threshold: float = 0.5,
    mask_regularization: float = 1e-4,
) -> torch.Tensor:
    """Equations 7--10 with a training-only fixed/published threshold policy."""
    if probability.ndim != 1 or masked_probability.shape != probability.shape:
        raise ValueError("predictions must be aligned vectors")
    supervised = F.binary_cross_entropy(
        probability[known_nodes], known_labels.to(dtype=probability.dtype)
    )
    pseudo = (probability.detach() >= threshold).to(masked_probability.dtype)
    consistency = F.binary_cross_entropy(masked_probability, pseudo)
    return supervised + consistency + mask_regularization * mask.abs().mean()


PAPER_DEFAULTS = {
    "forest_order": 3,
    "route_update_interval": 50,
    "score_direction": "higher_is_anomaly",
    "source": "AAAI-26 paper/extended-link recovery; no public author checkout",
}
