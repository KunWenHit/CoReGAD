import json
from pathlib import Path

import torch

from benchmark.paper_derived.bmp import BMP, bmp_loss, build_normalized_routes
from benchmark.paper_derived.structure_aware_pu_gnn import (
    distance_aware_pu_loss,
    distance_partition,
    sample_non_neighbors,
    structural_regularizer,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bmp_math_is_archived_provenance_not_active_reproduction():
    registry = json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))
    excluded = json.loads((ROOT / "benchmark" / "excluded_baselines.yaml").read_text(encoding="utf-8"))
    assert "BMP" not in {row["method"] for row in registry["baselines"]}
    bmp = next(row for row in excluded["inactive"] if row["method"] == "BMP")
    assert bmp["status"] == "EXCLUDED_NO_RECOVERABLE_OFFICIAL_SOURCE"
    assert bmp["active_launcher_candidate"] is False


def test_bmp_tree_routes_forest_shape_probability_update_and_loss():
    edge_index = torch.tensor([[0, 1, 2, 3, 1], [1, 2, 3, 4, 4]], dtype=torch.long)
    probability = torch.tensor([0.2, 0.1, 0.8, 0.4, 0.9])
    upper, lower, order, inverse = build_normalized_routes(edge_index, probability, 5)
    assert upper.shape == lower.shape == torch.Size([5, 5])
    assert torch.equal(order[inverse], torch.arange(5))
    model = BMP(input_dim=3, hidden_dim=4, order=2)
    score, masked_score, mask = model(torch.randn(5, 3), edge_index, probability)
    assert score.shape == masked_score.shape == (5,)
    assert mask.shape[0] == 5
    loss = bmp_loss(score, masked_score, mask, torch.tensor([0, 1]), torch.zeros(2))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_structure_aware_pu_equations_and_shapes():
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    positives = torch.tensor([0], dtype=torch.long)
    near, far = distance_partition(edge_index, positives, 6, threshold=2)
    assert near.tolist() == [1, 2]
    assert far.tolist() == [3, 4, 5]
    probability = torch.tensor([0.9, 0.7, 0.6, 0.4, 0.3, 0.2])
    pu_loss = distance_aware_pu_loss(probability, positives, near, far)
    negatives = sample_non_neighbors(edge_index, 6, samples_per_edge=2, seed=0)
    reg = structural_regularizer(torch.randn(6, 4), edge_index, negatives)
    assert pu_loss.ndim == reg.ndim == 0
    assert torch.isfinite(pu_loss) and torch.isfinite(reg)
