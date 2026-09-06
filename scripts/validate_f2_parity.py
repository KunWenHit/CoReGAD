from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coregad.models.coregad import CoReGAD
from coregad.models.routing import (
    build_routing_batch,
    fit_normal_channel_percentiles,
)


DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "f2_tfinance_seed0_fold0_parity.json"
)


def _tensor(value: object) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def validate(path: Path, tolerance: float = 1.0e-6) -> dict[str, object]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    model = CoReGAD().eval()
    model.load_state_dict(
        {name: _tensor(value) for name, value in fixture["model_state"].items()},
        strict=True,
    )

    calibration = fixture["calibration"]
    calibration_raw = _tensor(calibration["raw_evidence"])
    normal_mask = torch.tensor(calibration["normal_mask"], dtype=torch.bool)
    percentiles = fit_normal_channel_percentiles(calibration_raw, normal_mask)
    calibration_batch = build_routing_batch(
        raw_evidence=calibration_raw,
        controlled_evidence=_tensor(calibration["controlled_evidence"]),
        structural_statistics=_tensor(calibration["structural_statistics"]),
        base_anomaly_logit=_tensor(calibration["base_anomaly_logit"]),
        normal_mask=normal_mask,
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

    comparisons: dict[str, dict[str, float | bool]] = {}
    all_errors: list[torch.Tensor] = []
    for name, expected_values in fixture["expected"].items():
        expected = _tensor(expected_values)
        difference = (observed[name].detach().cpu() - expected).abs()
        all_errors.append(difference.reshape(-1))
        comparisons[name] = {
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "allclose": bool(
                torch.allclose(
                    observed[name].detach().cpu(),
                    expected,
                    atol=float(tolerance),
                    rtol=0.0,
                )
            ),
        }
    combined = torch.cat(all_errors)
    passed = all(bool(item["allclose"]) for item in comparisons.values())
    return {
        "schema_version": "coregad_f2_production_parity_report_v1",
        "status": "PASS" if passed else "FAIL",
        "model": fixture["canonical_model"],
        "canonical_formal_source_sha256": fixture[
            "canonical_formal_source_sha256"
        ],
        "artifact_origin": fixture["artifact_origin"],
        "training_executed": False,
        "tolerance": float(tolerance),
        "max_abs_error": float(combined.max()),
        "mean_abs_error": float(combined.mean()),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()
    report = validate(args.fixture, args.tolerance)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
