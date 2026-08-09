from __future__ import annotations

import numpy as np
import torch
from torch import nn

from coregad.models.coregad import CoReGAD
from coregad.models.spectral_reference import (
    GlobalSpectralReference,
    SpectralDiscrepancy,
    fold_safe_normalized_adjacency,
    structural_statistics,
)
from coregad.models.structural_residualization import ControlledStructuralResidualizer
from coregad.scalable import build_sparse_graph_operator, compute_chunked_graph_features


TOLERANCE = 1.0e-6


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.float() - right.float())).item())


def test_standard_scalable_parity(tmp_path) -> None:
    torch.manual_seed(17)
    n, hidden = 23, 8
    embeddings = torch.randn(n, hidden)
    source = torch.arange(n, dtype=torch.long).repeat_interleave(4)
    offsets = torch.tensor([1, 2, 5, 11], dtype=torch.long).repeat(n)
    destination = (source + offsets) % n
    edge_index = torch.stack([source, destination])
    visible = torch.arange(18, dtype=torch.long)
    heldout = torch.arange(18, n, dtype=torch.long)
    normality_head = nn.Linear(hidden, 1)

    adjacency, degree, support = fold_safe_normalized_adjacency(
        edge_index, n, visible, heldout, device="cpu"
    )
    standard_reference = GlobalSpectralReference()(adjacency, embeddings)
    standard_s = SpectralDiscrepancy()(
        embeddings,
        standard_reference["spectral_reference"],
        normality_head,
        spectral_components=standard_reference,
    )["spectral_discrepancy"]
    standard_t = structural_statistics(
        adjacency,
        embeddings,
        degree,
        support,
        standard_reference["low_spectral_component"],
    )

    operator = build_sparse_graph_operator(edge_index, n, visible, heldout)
    scalable = compute_chunked_graph_features(
        operator=operator,
        node_embeddings=embeddings,
        normality_head=normality_head,
        cache_dir=tmp_path,
        row_chunk=7,
        basis_cache_dtype="float32",
    )
    scalable_components = {
        "low_spectral_component": torch.from_numpy(np.array(scalable.low.open(), copy=True)),
        "band_spectral_component": torch.from_numpy(np.array(scalable.band.open(), copy=True)),
        "high_spectral_component": torch.from_numpy(np.array(scalable.high.open(), copy=True)),
        "spectral_reference": torch.from_numpy(
            np.array(scalable.spectral_reference.open(), copy=True)
        ),
    }
    scalable_s = torch.from_numpy(
        np.array(scalable.spectral_discrepancy.open(), copy=True)
    )
    scalable_t = torch.from_numpy(
        np.array(scalable.structural_statistics.open(), copy=True)
    )
    for name in (
        "low_spectral_component",
        "band_spectral_component",
        "high_spectral_component",
        "spectral_reference",
    ):
        assert _max_abs(standard_reference[name], scalable_components[name]) <= TOLERANCE
    assert _max_abs(standard_s[:, 0], scalable_s[:, 0]) <= TOLERANCE
    assert _max_abs(standard_s[:, 1], scalable_s[:, 1]) <= TOLERANCE
    assert _max_abs(standard_t, scalable_t) <= TOLERANCE

    node_ids = np.arange(n, dtype=np.int64)
    standard_residualizer = ControlledStructuralResidualizer(seed=7319)
    scalable_residualizer = ControlledStructuralResidualizer(seed=7319)
    standard_residual = standard_residualizer.fit_transform(
        standard_s, standard_t, node_ids
    )
    scalable_residual = scalable_residualizer.fit_transform(
        scalable_s, scalable_t, node_ids
    )
    assert _max_abs(
        standard_residual["structure_predictable_spectral_component"],
        scalable_residual["structure_predictable_spectral_component"],
    ) <= TOLERANCE
    assert _max_abs(
        standard_residual["controlled_spectral_residual"],
        scalable_residual["controlled_spectral_residual"],
    ) <= TOLERANCE

    torch.manual_seed(29)
    detector = CoReGAD()
    base = torch.linspace(-1.0, 1.0, n)
    standard_final = detector(
        standard_residual["controlled_spectral_residual"],
        standard_residual["structural_statistics_normalized"],
        base,
    )
    scalable_final = detector(
        scalable_residual["controlled_spectral_residual"],
        scalable_residual["structural_statistics_normalized"],
        base,
    )
    for name in (
        "spectral_residual_energy",
        "structural_reliability",
        "graph_correction",
        "final_anomaly_score",
    ):
        assert _max_abs(standard_final[name], scalable_final[name]) <= TOLERANCE


def test_historical_mixed_precision_cache_is_explicit(tmp_path) -> None:
    torch.manual_seed(3)
    embeddings = torch.randn(9, 4)
    nodes = torch.arange(9)
    edge_index = torch.stack([nodes, torch.roll(nodes, -1)])
    operator = build_sparse_graph_operator(edge_index, 9, nodes[:7], nodes[7:])
    result = compute_chunked_graph_features(
        operator=operator,
        node_embeddings=embeddings,
        normality_head=nn.Linear(4, 1),
        cache_dir=tmp_path,
        row_chunk=3,
        basis_cache_dtype="float16",
    )
    assert result.metadata["historical_mixed_precision_cache"] is True
    assert result.metadata["basis_cache_dtype"] == "float16"
    assert result.metadata["dense_nxn_constructed"] is False
