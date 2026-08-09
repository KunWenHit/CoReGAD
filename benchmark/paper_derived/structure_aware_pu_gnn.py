"""Paper-derived Structure-aware PU-GNN components (CIKM 2023, eqs. 2--3).

No official/author implementation was found in the paper, author pages, GitHub
repository search, or paper indexes during the 2026-08-09 source audit.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


def distance_partition(
    edge_index: torch.Tensor,
    positive_nodes: torch.Tensor,
    num_nodes: int,
    *,
    threshold: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split unlabeled nodes by shortest distance to the nearest positive node."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edge_index.t().detach().cpu().tolist():
        adjacency[source].append(target)
        adjacency[target].append(source)
    positives = set(int(x) for x in positive_nodes.detach().cpu().tolist())
    distance = np.full(num_nodes, np.iinfo(np.int64).max, dtype=np.int64)
    queue: deque[int] = deque()
    for node in positives:
        distance[node] = 0
        queue.append(node)
    while queue:
        node = queue.popleft()
        if distance[node] >= threshold:
            continue
        for neighbor in adjacency[node]:
            if distance[neighbor] > distance[node] + 1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    unlabeled = [node for node in range(num_nodes) if node not in positives]
    near = [node for node in unlabeled if distance[node] <= threshold]
    far = [node for node in unlabeled if distance[node] > threshold]
    device = edge_index.device
    return torch.tensor(near, dtype=torch.long, device=device), torch.tensor(
        far, dtype=torch.long, device=device
    )


def _mean_or_zero(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return values.mean() if values.numel() else reference.new_zeros(())


def distance_aware_pu_loss(
    positive_probability: torch.Tensor,
    labeled_positive: torch.Tensor,
    near_unlabeled: torch.Tensor,
    far_unlabeled: torch.Tensor,
    *,
    near_prior: float = 0.6,
    far_prior: float = 0.3,
) -> torch.Tensor:
    """Equation 2 distribution-alignment loss with paper default priors."""
    if not 0.0 < far_prior < near_prior < 1.0:
        raise ValueError("paper contract requires 0 < far_prior < near_prior < 1")
    positive_term = (positive_probability[labeled_positive].mean() - 1.0).abs()
    near_term = (
        _mean_or_zero(positive_probability[near_unlabeled], positive_probability)
        - near_prior
    ).abs()
    far_term = (
        _mean_or_zero(positive_probability[far_unlabeled], positive_probability)
        - far_prior
    ).abs()
    return 2.0 * (near_prior + far_prior) * positive_term + near_term + far_term


def sample_non_neighbors(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    samples_per_edge: int = 50,
    seed: int = 0,
) -> torch.Tensor:
    """Paper equation 3 negative samples, deterministic for the benchmark seed."""
    if samples_per_edge <= 0:
        raise ValueError("samples_per_edge must be positive")
    neighbors = [set([node]) for node in range(num_nodes)]
    edges = edge_index.t().detach().cpu().tolist()
    for source, target in edges:
        neighbors[source].add(target)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sampled: list[list[int]] = []
    all_nodes = torch.arange(num_nodes)
    for source, _ in edges:
        candidates = all_nodes[[node not in neighbors[source] for node in range(num_nodes)]]
        if candidates.numel() == 0:
            raise ValueError("a complete graph has no structural negative samples")
        index = torch.randint(candidates.numel(), (samples_per_edge,), generator=generator)
        sampled.append(candidates[index].tolist())
    return torch.tensor(sampled, dtype=torch.long, device=edge_index.device)


def structural_regularizer(
    embedding: torch.Tensor,
    edge_index: torch.Tensor,
    negative_nodes: torch.Tensor,
) -> torch.Tensor:
    """Equation 3: neighbor similarity to one, sampled non-neighbor to zero."""
    if negative_nodes.shape[0] != edge_index.shape[1]:
        raise ValueError("negative samples must align with directed edges")
    source, target = edge_index.long()
    positive_similarity = torch.sigmoid((embedding[source] * embedding[target]).sum(dim=1))
    positive_loss = (positive_similarity - 1.0).square().mean()
    source_embedding = embedding[source].unsqueeze(1)
    negative_embedding = embedding[negative_nodes.long()]
    negative_similarity = torch.sigmoid((source_embedding * negative_embedding).sum(dim=2))
    return positive_loss + negative_similarity.square().mean()


def pu_gnn_objective(
    positive_probability: torch.Tensor,
    embedding: torch.Tensor,
    edge_index: torch.Tensor,
    labeled_positive: torch.Tensor,
    near_unlabeled: torch.Tensor,
    far_unlabeled: torch.Tensor,
    negative_nodes: torch.Tensor,
    *,
    alpha: float = 0.01,
) -> torch.Tensor:
    return distance_aware_pu_loss(
        positive_probability, labeled_positive, near_unlabeled, far_unlabeled
    ) + alpha * structural_regularizer(embedding, edge_index, negative_nodes)


PAPER_DEFAULTS = {
    "alpha": 0.01,
    "distance_threshold": 3,
    "negative_samples": 50,
    "near_positive_prior": 0.6,
    "far_positive_prior": 0.3,
    "backbone": "two-layer GCN",
    "hidden_dimension": 16,
    "anomaly_score_conversion": "1 - predicted normal-positive probability",
}
