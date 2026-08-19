from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
import numpy as np

import coregad.training.normality as normality
from coregad.training.normality import NormalityFoldResult
from coregad.training.pipeline import train_fold


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [0.2, -0.1, 0.4, 0.7],
            [-0.3, 0.8, 0.1, -0.5],
            [0.6, 0.2, -0.4, 0.3],
            [-0.7, 0.5, 0.9, -0.2],
            [0.1, -0.6, 0.2, 0.8],
            [0.9, 0.4, -0.8, 0.1],
            [-0.2, -0.9, 0.5, 0.6],
            [0.7, -0.3, -0.1, -0.4],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [2, 3, 3, 4, 0, 1, 4, 2, 5, 6, 7],
            [0, 0, 1, 1, 2, 3, 2, 4, 0, 1, 4],
        ],
        dtype=torch.long,
    )
    normal_nodes = torch.tensor([0, 1], dtype=torch.long)
    training_unlabeled_nodes = torch.tensor([2, 3, 4], dtype=torch.long)
    heldout_nodes = torch.tensor([5, 6, 7], dtype=torch.long)
    return features, edge_index, normal_nodes, training_unlabeled_nodes, heldout_nodes


def _train(
    features: torch.Tensor,
    *,
    base_epochs: int,
    context_epochs: int,
) -> NormalityFoldResult:
    _, edge_index, normal_nodes, training_unlabeled_nodes, heldout_nodes = _fixture()
    return normality.train_cross_fitted_normality_fold(
        features=features,
        edge_index=edge_index,
        normal_nodes=normal_nodes,
        training_unlabeled_nodes=training_unlabeled_nodes,
        heldout_nodes=heldout_nodes,
        seed=1907,
        device="cpu",
        base_epochs=base_epochs,
        context_epochs=context_epochs,
    )


def _states(result: NormalityFoldResult) -> dict[str, Mapping[str, torch.Tensor]]:
    return {
        "student_encoder": result.frozen_core.encoder.state_dict(),
        "normality_head": result.frozen_core.normality_head.state_dict(),
        "graph_context_shaping": result.graph_context_shaping.state_dict(),
    }


def _assert_states_identical(
    left: NormalityFoldResult,
    right: NormalityFoldResult,
) -> None:
    left_states, right_states = _states(left), _states(right)
    assert left_states.keys() == right_states.keys()
    for group in left_states:
        assert left_states[group].keys() == right_states[group].keys()
        for name in left_states[group]:
            assert torch.equal(left_states[group][name], right_states[group][name]), (
                f"{group}.{name} changed after perturbing heldout attributes"
            )


def _perturbed_heldout_features() -> tuple[torch.Tensor, torch.Tensor]:
    original, _, _, _, heldout_nodes = _fixture()
    perturbed = original.clone()
    perturbed[heldout_nodes] = torch.tensor(
        [
            [1000.0, -2000.0, 3000.0, -4000.0],
            [-5000.0, 6000.0, -7000.0, 8000.0],
            [9000.0, -10000.0, 11000.0, -12000.0],
        ],
        dtype=torch.float32,
    )
    return original, perturbed


def test_anchor_scope_and_outer_fold_ownership_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    features, edge_index, normal_nodes, training_unlabeled_nodes, heldout_nodes = _fixture()
    observed_shapes: list[tuple[int, int]] = []
    original_anchor = normality.anchor_preservation_loss

    def traced_anchor(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        observed_shapes.append((int(student_logits.numel()), int(teacher_logits.numel())))
        return original_anchor(student_logits, teacher_logits)

    monkeypatch.setattr(normality, "anchor_preservation_loss", traced_anchor)
    normality.train_cross_fitted_normality_fold(
        features=features,
        edge_index=edge_index,
        normal_nodes=normal_nodes,
        training_unlabeled_nodes=training_unlabeled_nodes,
        heldout_nodes=heldout_nodes,
        seed=1907,
        device="cpu",
        base_epochs=1,
        context_epochs=2,
    )
    expected_visible = int(normal_nodes.numel() + training_unlabeled_nodes.numel())
    assert expected_visible == 5
    assert observed_shapes == [(expected_visible, expected_visible)] * 2
    assert expected_visible != int(features.shape[0])

    with pytest.raises(ValueError, match="visible_index and heldout_nodes to be disjoint"):
        normality.train_cross_fitted_normality_fold(
            features=features,
            edge_index=edge_index,
            normal_nodes=normal_nodes,
            training_unlabeled_nodes=training_unlabeled_nodes,
            heldout_nodes=torch.tensor([4, 5, 6, 7]),
            seed=1907,
            device="cpu",
            base_epochs=0,
            context_epochs=0,
        )


def test_heldout_attribute_perturbation_does_not_change_trained_state() -> None:
    original, perturbed = _perturbed_heldout_features()
    run_a = _train(original, base_epochs=3, context_epochs=3)
    run_b = _train(perturbed, base_epochs=3, context_epochs=3)
    _assert_states_identical(run_a, run_b)


def test_heldout_attributes_cannot_change_one_context_optimization_step() -> None:
    original, perturbed = _perturbed_heldout_features()
    run_a = _train(original, base_epochs=0, context_epochs=1)
    run_b = _train(perturbed, base_epochs=0, context_epochs=1)
    _assert_states_identical(run_a, run_b)


def test_fold_diagnostics_are_heldout_only_and_never_used_for_training() -> None:
    features, edge_index, normal_nodes, training_unlabeled_nodes, heldout_nodes = _fixture()
    result = train_fold(
        features=features,
        edge_index=edge_index,
        normal_nodes=normal_nodes.numpy(),
        training_unlabeled_nodes=training_unlabeled_nodes.numpy(),
        heldout_nodes=heldout_nodes.numpy(),
        fold=0,
        model_seed=0,
        device="cpu",
        normality_epochs=1,
        context_epochs=1,
        residual_epochs=1,
    )
    diagnostics = result.artifacts["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["used_for_training_or_selection"] is False
    assert np.array_equal(diagnostics["node_id"], heldout_nodes.numpy())
    for name in (
        "pre_context_teacher_base_anomaly_score",
        "post_context_frozen_normality_score",
        "graph_correction",
        "final_anomaly_score",
    ):
        assert diagnostics[name].shape == (heldout_nodes.numel(),)
    for name in ("spectral_discrepancy", "controlled_spectral_residual"):
        assert diagnostics[name].shape == (heldout_nodes.numel(), 2)
    assert np.array_equal(diagnostics["final_anomaly_score"], result.final_anomaly_score)
