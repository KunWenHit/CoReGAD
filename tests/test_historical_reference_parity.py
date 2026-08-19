from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
from coregad.models.structural_residualization import (
    ControlledStructuralResidualizer,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "frozen_release_reference.npz"
METADATA_PATH = FIXTURE_DIR / "frozen_release_reference.json"


def _tensor(fixture: np.lib.npyio.NpzFile, key: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(fixture[key])).to(dtype=torch.float32)


def _assert_strict(actual: torch.Tensor, fixture: np.lib.npyio.NpzFile, key: str) -> None:
    expected = _tensor(fixture, key)
    assert actual.shape == expected.shape
    actual_cpu = actual.detach().cpu()
    maximum = float(torch.max(torch.abs(actual_cpu - expected)).item())
    assert torch.allclose(actual_cpu, expected, atol=1.0e-6, rtol=0.0), (
        f"{key}: max_abs={maximum:.12g}"
    )


def test_public_pipeline_matches_frozen_historical_release() -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["contains_real_dataset"] is False
    assert metadata["synthetic"] is True
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == metadata["sha256"]

    with np.load(FIXTURE_PATH, allow_pickle=False) as fixture:
        assert str(fixture["fixture_version"]) == "coregad-historical-release-v1"
        assert float(fixture["alpha"]) == 0.75
        embeddings = _tensor(fixture, "node_embeddings")
        edge_index = torch.from_numpy(fixture["edge_index"]).long()
        visible = torch.from_numpy(fixture["visible_node_id"]).long()
        heldout = torch.from_numpy(fixture["heldout_node_id"]).long()
        adjacency, degree, support_ratio = fold_safe_normalized_adjacency(
            edge_index,
            int(embeddings.shape[0]),
            visible,
            heldout,
            device="cpu",
        )

        spectral = GlobalSpectralReference()(adjacency, embeddings)
        _assert_strict(spectral["low_spectral_component"], fixture, "Z_low")
        _assert_strict(spectral["band_spectral_component"], fixture, "Z_band")
        _assert_strict(spectral["high_spectral_component"], fixture, "Z_high")
        _assert_strict(spectral["spectral_reference"], fixture, "spectral_reference")

        normality_head = nn.Linear(int(embeddings.shape[1]), 1)
        with torch.no_grad():
            normality_head.weight.copy_(_tensor(fixture, "normality_weight"))
            normality_head.bias.copy_(_tensor(fixture, "normality_bias"))
        discrepancy = SpectralDiscrepancy()(
            embeddings,
            spectral["spectral_reference"],
            normality_head,
            spectral,
        )["spectral_discrepancy"]
        _assert_strict(discrepancy[:, 0], fixture, "r_emb")
        _assert_strict(discrepancy[:, 1], fixture, "r_dec")

        structural = structural_statistics(
            adjacency,
            embeddings,
            degree,
            support_ratio,
        )
        _assert_strict(structural, fixture, "T")

        # Feed the exact frozen S/T interface into the downstream modules.  The
        # independently checked graph stages above permit 1e-6 sparse-kernel
        # roundoff, while the high-dimensional ridge solve is intentionally
        # verified against identical historical inputs.
        frozen_discrepancy = torch.stack(
            [_tensor(fixture, "r_emb"), _tensor(fixture, "r_dec")], dim=1
        )
        residualized = ControlledStructuralResidualizer(
            seed=int(fixture["nuisance_seed"])
        ).fit_transform(
            frozen_discrepancy,
            _tensor(fixture, "T"),
            np.asarray(fixture["node_id"], dtype=np.int64),
        )
        _assert_strict(
            residualized["structural_statistics_normalized"],
            fixture,
            "T_normalized",
        )
        _assert_strict(
            residualized["structure_predictable_spectral_component"],
            fixture,
            "S_hat",
        )
        _assert_strict(
            residualized["controlled_spectral_residual"], fixture, "S_CR"
        )

        model = CoReGAD().eval()
        state = {
            "spectral_residual_energy.network." + key.removeprefix("state__energy."):
                _tensor(fixture, key)
            for key in fixture.files
            if key.startswith("state__energy.")
        }
        state.update(
            {
                "structural_reliability_gate.network."
                + key.removeprefix("state__gate."): _tensor(fixture, key)
                for key in fixture.files
                if key.startswith("state__gate.")
            }
        )
        state["gamma_raw"] = _tensor(fixture, "state__gamma_raw")
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            output = model(
                residualized["controlled_spectral_residual"],
                residualized["structural_statistics_normalized"],
                _tensor(fixture, "base_logit"),
            )
        _assert_strict(output["spectral_residual_energy"], fixture, "Energy")
        _assert_strict(output["structural_reliability"], fixture, "Reliability")
        _assert_strict(output["graph_correction"], fixture, "correction")
        _assert_strict(output["final_anomaly_score"], fixture, "final_score")
    torch.set_num_threads(previous_threads)
