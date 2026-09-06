from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from coregad.models.coregad import (
    FULL_F2,
    LEGACY_CONTROLLED_TIGHT,
    WO_M1_GRAPH_RESIDUAL_EVIDENCE,
    WO_M2_RELIABILITY,
    WO_M3_FACTORIZED_ROUTING,
    CoReGAD,
)
from coregad.models.routing import (
    F2_ALPHA,
    F2_RHO,
    F2_WIDE_TEMPERATURE,
    NormalCalibratedFactorizedRouter,
    StrictLowerNormalPercentile,
    build_routing_batch,
    correction_endpoints,
    factorized_correction,
    fit_normal_channel_percentiles,
    tail_ramp,
)
from coregad.training.residual import train_residual_detector


ROOT = Path(__file__).parents[1]


def _batch(count: int = 32, *, device: str = "cpu", dtype: torch.dtype = torch.float32):
    generator = torch.Generator().manual_seed(1801)
    raw = torch.randn((count, 2), generator=generator, dtype=dtype).to(device)
    predictable = 0.2 * torch.randn(
        (count, 2), generator=generator, dtype=dtype
    ).to(device)
    controlled = raw - F2_ALPHA * predictable
    structural = torch.randn((count, 3), generator=generator, dtype=dtype).to(device)
    base = torch.randn((count,), generator=generator, dtype=dtype).to(device)
    normal = (torch.arange(count, device=device) < count // 2)
    percentiles = fit_normal_channel_percentiles(raw, normal)
    return build_routing_batch(
        raw_evidence=raw,
        controlled_evidence=controlled,
        structural_statistics=structural,
        base_anomaly_logit=base,
        normal_mask=normal,
        normal_channel_percentiles=percentiles,
    )


def test_strict_lower_ecdf_ties_and_vectorization() -> None:
    reference = torch.tensor([0.0, 0.0, 0.0, 1.0])
    transform = StrictLowerNormalPercentile.fit(
        reference, normal_mask=torch.ones(4, dtype=torch.bool)
    )
    assert transform.transform(torch.tensor([0.0])).item() == 0.0
    assert transform.transform(torch.tensor([1.0])).item() == pytest.approx(0.75)
    observed = transform.transform(torch.tensor([[0.0, 0.5], [1.0, 2.0]]))
    assert torch.equal(
        observed, torch.tensor([[0.0, 0.75], [0.75, 1.0]])
    )


def test_tail_ramp_frozen_values() -> None:
    values = torch.tensor([0.0, 0.8, 0.9, 1.0], dtype=torch.float64)
    assert torch.allclose(
        tail_ramp(values),
        torch.tensor([0.0, 0.0, 0.5, 1.0], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert F2_RHO == 0.8


def test_four_endpoint_formulas() -> None:
    gamma = torch.tensor(0.2, dtype=torch.float64)
    q = torch.tensor([0.5, 0.75], dtype=torch.float64)
    raw = torch.tensor([0.3, 1.2], dtype=torch.float64)
    controlled = torch.tensor([0.2, 0.8], dtype=torch.float64)
    endpoint = correction_endpoints(
        gamma=gamma,
        reliability=q,
        raw_energy=raw,
        controlled_energy=controlled,
    )
    common = gamma * q
    assert torch.allclose(endpoint["controlled_tight"], common * torch.tanh(controlled))
    assert torch.allclose(
        endpoint["controlled_wide"], common * 2.0 * torch.tanh(controlled / 2.0)
    )
    assert torch.allclose(endpoint["raw_tight"], common * torch.tanh(raw))
    assert torch.allclose(
        endpoint["raw_wide"], common * 2.0 * torch.tanh(raw / 2.0)
    )
    assert F2_WIDE_TEMPERATURE == 2.0


@pytest.mark.parametrize(
    ("evidence_route", "bound_route", "expected"),
    [(0.0, 0.0, 1.0), (0.0, 1.0, 2.0), (1.0, 0.0, 3.0), (1.0, 1.0, 4.0)],
)
def test_factorized_mixture_boundaries(
    evidence_route: float, bound_route: float, expected: float
) -> None:
    observed = factorized_correction(
        controlled_tight=torch.tensor([1.0]),
        controlled_wide=torch.tensor([2.0]),
        raw_tight=torch.tensor([3.0]),
        raw_wide=torch.tensor([4.0]),
        evidence_route=torch.tensor([evidence_route]),
        bound_route=torch.tensor([bound_route]),
    )
    assert observed.item() == expected


def test_legacy_reduction_is_controlled_tight() -> None:
    model = CoReGAD(model_variant=LEGACY_CONTROLLED_TIGHT)
    batch = _batch()
    output = model.forward_batch(batch, calibration_batch=batch)
    assert torch.equal(output["evidence_route"], torch.zeros(32))
    assert torch.equal(output["bound_route"], torch.zeros(32))
    assert torch.equal(output["graph_correction"], output["controlled_tight"])


def test_routes_bounded_delta_finite_and_shapes() -> None:
    model = CoReGAD()
    batch = _batch()
    output = model.forward_batch(batch, calibration_batch=batch)
    for name in ("evidence_route", "bound_route"):
        assert output[name].shape == (32,)
        assert bool(torch.all((output[name] >= 0.0) & (output[name] <= 1.0)))
    assert output["graph_correction"].shape == (32,)
    assert bool(torch.isfinite(output["graph_correction"]).all())


def test_raw_and_controlled_share_one_energy_head() -> None:
    model = CoReGAD()
    batch = _batch()
    calls: list[int] = []
    original = model.spectral_residual_energy.forward

    def tracked(values: torch.Tensor) -> torch.Tensor:
        calls.append(id(model.spectral_residual_energy))
        return original(values)

    with patch.object(model.spectral_residual_energy, "forward", side_effect=tracked):
        model.forward_batch(batch, calibration_batch=batch)
    assert len(calls) == 2
    assert len(set(calls)) == 1


def test_m3_has_zero_parameters_and_routes_stop_gradient() -> None:
    router = NormalCalibratedFactorizedRouter()
    assert sum(parameter.numel() for parameter in router.parameters()) == 0
    model = CoReGAD()
    assert model.m3_trainable_parameter_count == 0
    output = model.forward_batch(_batch(), calibration_batch=_batch())
    assert output["evidence_route"].requires_grad is False
    assert output["bound_route"].requires_grad is False


def test_model_has_no_evaluation_label_interface() -> None:
    for callable_ in (
        CoReGAD.forward,
        CoReGAD.forward_batch,
        NormalCalibratedFactorizedRouter.forward,
    ):
        parameters = inspect.signature(callable_).parameters
        assert "labels" not in parameters
        assert "y" not in parameters
    source = (ROOT / "coregad" / "models" / "routing.py").read_text(
        encoding="utf-8"
    )
    assert "ranking_metrics" not in source


@pytest.mark.parametrize("variant", [FULL_F2, WO_M3_FACTORIZED_ROUTING])
def test_cpu_dtype_and_state_roundtrip(variant: str) -> None:
    model = CoReGAD(model_variant=variant).double()
    batch = _batch(dtype=torch.float64)
    output = model.forward_batch(batch, calibration_batch=batch)
    assert output["graph_correction"].dtype == torch.float64
    clone = CoReGAD(model_variant=variant).double()
    clone.load_state_dict(model.state_dict(), strict=True)
    cloned = clone.forward_batch(batch, calibration_batch=batch)
    assert torch.equal(output["final_anomaly_score"], cloned["final_anomaly_score"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_gpu_device_consistency() -> None:
    cpu_model = CoReGAD().eval()
    cpu_batch = _batch()
    cpu_output = cpu_model.forward_batch(cpu_batch, calibration_batch=cpu_batch)
    gpu_model = CoReGAD().eval().cuda()
    gpu_model.load_state_dict(cpu_model.state_dict(), strict=True)
    gpu_batch = _batch(device="cuda")
    gpu_output = gpu_model.forward_batch(gpu_batch, calibration_batch=gpu_batch)
    assert torch.allclose(
        cpu_output["graph_correction"],
        gpu_output["graph_correction"].cpu(),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_ablation_semantics() -> None:
    batch = _batch()
    no_m1 = CoReGAD(model_variant=WO_M1_GRAPH_RESIDUAL_EVIDENCE)
    output_m1 = no_m1.forward_batch(batch, calibration_batch=batch)
    assert torch.count_nonzero(output_m1["graph_correction"]).item() == 0
    assert torch.equal(output_m1["final_anomaly_logit"], batch.base_anomaly_logit)

    no_m2 = CoReGAD(model_variant=WO_M2_RELIABILITY)
    output_m2 = no_m2.forward_batch(batch, calibration_batch=batch)
    assert no_m2.structural_reliability_gate is None
    assert torch.equal(output_m2["structural_reliability"], torch.ones(32))

    no_m3 = CoReGAD(model_variant=WO_M3_FACTORIZED_ROUTING)
    output_m3 = no_m3.forward_batch(batch, calibration_batch=batch)
    assert torch.equal(output_m3["graph_correction"], output_m3["controlled_tight"])


def test_light_deterministic_f2_training_smoke() -> None:
    batch = _batch(12)
    roles = torch.cat(
        [torch.zeros(6, dtype=torch.int8), torch.ones(6, dtype=torch.int8)]
    )
    model = train_residual_detector(
        raw_spectral_evidence=batch.raw_evidence,
        controlled_spectral_residual=batch.controlled_evidence,
        structural_statistics=batch.structural_statistics,
        base_anomaly_logit=batch.base_anomaly_logit,
        roles=roles,
        seed=20260906,
        epochs=1,
        model_variant=FULL_F2,
    )
    output = model.forward_batch(batch, calibration_batch=batch)
    assert bool(torch.isfinite(output["final_anomaly_score"]).all())


def test_frozen_config_roundtrip() -> None:
    path = ROOT / "configs" / "coregad.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    roundtrip = yaml.safe_load(yaml.safe_dump(config, sort_keys=True))
    assert roundtrip == config
    assert config["method"]["model_variant"] == FULL_F2
    assert config["controlled_structural_residualization"]["residual_strength"] == F2_ALPHA
    assert config["factorized_routing"]["normal_tail_rho"] == F2_RHO
    assert config["factorized_routing"]["wide_temperature"] == F2_WIDE_TEMPERATURE
    assert config["factorized_routing"]["ecdf"] == "strict_lower"


def _tensor(value: object) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def test_server_canonical_parity_fixture() -> None:
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "f2_tfinance_seed0_fold0_parity.json").read_text(
            encoding="utf-8"
        )
    )
    assert fixture["canonical_model"] == "F2_TIE_SAFE_FACTORIZED_ROUTER_T2"
    assert fixture["artifact_origin"]["dataset"] == "T-Finance"
    assert fixture["training_executed"] is False

    model = CoReGAD().eval()
    model.load_state_dict(
        {name: _tensor(value) for name, value in fixture["model_state"].items()},
        strict=True,
    )
    calibration = fixture["calibration"]
    normal = torch.tensor(calibration["normal_mask"], dtype=torch.bool)
    raw = _tensor(calibration["raw_evidence"])
    percentiles = fit_normal_channel_percentiles(raw, normal)
    calibration_batch = build_routing_batch(
        raw_evidence=raw,
        controlled_evidence=_tensor(calibration["controlled_evidence"]),
        structural_statistics=_tensor(calibration["structural_statistics"]),
        base_anomaly_logit=_tensor(calibration["base_anomaly_logit"]),
        normal_mask=normal,
        normal_channel_percentiles=percentiles,
    )
    query = fixture["query"]
    query_raw = _tensor(query["raw_evidence"])
    query_batch = build_routing_batch(
        raw_evidence=query_raw,
        controlled_evidence=_tensor(query["controlled_evidence"]),
        structural_statistics=_tensor(query["structural_statistics"]),
        base_anomaly_logit=_tensor(query["base_anomaly_logit"]),
        normal_mask=torch.zeros(query_raw.shape[0], dtype=torch.bool),
        normal_channel_percentiles=percentiles,
    )
    with torch.no_grad():
        observed = model.forward_batch(
            query_batch, calibration_batch=calibration_batch
        )
    for name, expected in fixture["expected"].items():
        difference = (observed[name] - _tensor(expected)).abs()
        assert float(difference.max()) <= 1.0e-6, name
