from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from coregad.models.coregad import CoReGAD
from coregad.models.spectral_reference import (
    GlobalSpectralReference,
    SpectralDiscrepancy,
    fold_safe_normalized_adjacency,
    structural_statistics,
)
from coregad.training.normality import NormalityFoldResult, train_cross_fitted_normality_fold
from coregad.training.nuisance import fit_controlled_residualizer
from coregad.training.residual import train_residual_detector


@dataclass
class FoldOutput:
    heldout_nodes: np.ndarray
    final_anomaly_score: np.ndarray
    artifacts: dict[str, object]


def train_fold(
    *,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    normal_nodes: np.ndarray,
    training_unlabeled_nodes: np.ndarray,
    heldout_nodes: np.ndarray,
    fold: int,
    model_seed: int,
    device: torch.device | str,
    normality_epochs: int = 200,
    context_epochs: int = 200,
    residual_epochs: int = 300,
) -> FoldOutput:
    device = torch.device(device)
    normal_tensor = torch.as_tensor(normal_nodes, dtype=torch.long)
    training_unlabeled_tensor = torch.as_tensor(
        training_unlabeled_nodes, dtype=torch.long
    )
    heldout_tensor = torch.as_tensor(heldout_nodes, dtype=torch.long)
    normality: NormalityFoldResult = train_cross_fitted_normality_fold(
        features=features,
        edge_index=edge_index,
        normal_nodes=normal_tensor,
        training_unlabeled_nodes=training_unlabeled_tensor,
        heldout_nodes=heldout_tensor,
        seed=int(model_seed) + int(fold),
        device=device,
        base_epochs=normality_epochs,
        context_epochs=context_epochs,
    )
    scaled = (
        (features.float() - normality.feature_mean)
        / normality.feature_scale
    ).to(device)
    with torch.no_grad():
        normality_output = normality.frozen_core(scaled)
    visible_nodes = torch.cat([normal_tensor, training_unlabeled_tensor])
    adjacency, degree, support = fold_safe_normalized_adjacency(
        edge_index,
        features.shape[0],
        visible_nodes,
        heldout_tensor,
        device=device,
    )
    spectral_builder = GlobalSpectralReference().to(device)
    with torch.no_grad():
        reference = spectral_builder(
            adjacency, normality_output["node_embeddings"]
        )
        discrepancy = SpectralDiscrepancy()(
            normality_output["node_embeddings"],
            reference["spectral_reference"],
            normality.frozen_core.normality_head,
            spectral_components=reference,
        )
        statistics = structural_statistics(
            adjacency,
            normality_output["node_embeddings"],
            degree,
            support,
        )
    train_nodes = np.concatenate(
        [
            np.asarray(normal_nodes, dtype=np.int64),
            np.asarray(training_unlabeled_nodes, dtype=np.int64),
        ]
    )
    train_index = torch.as_tensor(train_nodes, dtype=torch.long, device=device)
    residualizer, training_residual = fit_controlled_residualizer(
        discrepancy["spectral_discrepancy"][train_index],
        statistics[train_index],
        train_nodes,
        seed=7319 + int(model_seed) * 100 + int(fold),
    )
    roles = torch.cat(
        [
            torch.zeros(len(normal_nodes), dtype=torch.int8, device=device),
            torch.ones(
                len(training_unlabeled_nodes), dtype=torch.int8, device=device
            ),
        ]
    )
    detector: CoReGAD = train_residual_detector(
        controlled_spectral_residual=training_residual[
            "controlled_spectral_residual"
        ].detach(),
        structural_statistics=training_residual[
            "structural_statistics_normalized"
        ].detach(),
        base_anomaly_logit=normality_output["base_anomaly_logit"][
            train_index
        ].detach(),
        roles=roles,
        seed=2026080700 + int(model_seed) * 100 + int(fold),
        epochs=residual_epochs,
    )
    heldout_index = heldout_tensor.to(device)
    evaluation_residual = residualizer.transform(
        discrepancy["spectral_discrepancy"][heldout_index],
        statistics[heldout_index],
    )
    with torch.no_grad():
        final = detector(
            evaluation_residual["controlled_spectral_residual"],
            evaluation_residual["structural_statistics_normalized"],
            normality_output["base_anomaly_logit"][heldout_index],
        )
    artifacts: dict[str, object] = {
        "normality_core_state_dict": {
            key: value.detach().cpu()
            for key, value in normality.frozen_core.state_dict().items()
        },
        "graph_context_state_dict": {
            key: value.detach().cpu()
            for key, value in normality.graph_context_shaping.state_dict().items()
        },
        "detector_state_dict": {
            key: value.detach().cpu() for key, value in detector.state_dict().items()
        },
        "residualizer": residualizer,
        "feature_mean": normality.feature_mean,
        "feature_scale": normality.feature_scale,
        "fold": int(fold),
        "model_seed": int(model_seed),
        "normal_nodes": np.asarray(normal_nodes, dtype=np.int64),
        "training_unlabeled_nodes": np.asarray(
            training_unlabeled_nodes, dtype=np.int64
        ),
        "heldout_nodes": np.asarray(heldout_nodes, dtype=np.int64),
    }
    return FoldOutput(
        heldout_nodes=np.asarray(heldout_nodes, dtype=np.int64),
        final_anomaly_score=final["final_anomaly_score"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        artifacts=artifacts,
    )
