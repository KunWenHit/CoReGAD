import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark.native_bridges.common import load_label_free_bundle, write_scores
from benchmark.native_bridges.dominant import preflight
from benchmark.resource_contract import RESOURCE_STATUSES, matrix_rows, validate_cell


ROOT = Path(__file__).resolve().parents[1]
GATES = {
    "SOURCE_PASS", "ENV_PASS", "NATIVE_SANITY_PASS", "LABEL_AUDIT_PASS",
    "CANONICAL_SMOKE_PASS", "SCORE_CONTRACT_PASS",
}


def _registry():
    return json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))


def test_execute_enabled_implies_six_evidenced_pass_gates_and_native_command():
    execute_enabled = 0
    for row in _registry()["baselines"]:
        if row["protocol_class"] != "PRIMARY_TRANSDUCTIVE_NORMAL_ONLY":
            continue
        gate = json.loads((ROOT / row["gate_file"]).read_text(encoding="utf-8"))
        assert gate["method"] == row["method"]
        assert set(gate["gates"]) == GATES
        if row["execute_enabled"]:
            execute_enabled += 1
            assert row.get("native_command")
            assert all(value["status"] == "PASS" and value["evidence"] for value in gate["gates"].values())
            for value in gate["gates"].values():
                evidence = value["evidence"]
                evidence = evidence if isinstance(evidence, list) else [evidence]
                assert all((ROOT / path).is_file() for path in evidence)
    assert execute_enabled == 2


def test_native_commands_do_not_receive_labels_or_cross_fitting():
    for row in _registry()["baselines"]:
        command = row.get("native_command", [])
        joined = " ".join(command).casefold()
        assert not any(token in joined for token in ("--y ", "--label", "owner_fold", "strict_oof", "cross_fitting"))


def test_label_free_bundle_enforces_keys_node_order_and_score_orientation(tmp_path):
    path = tmp_path / "graph.pt"
    payload = {
        "features": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "node_id": torch.arange(4),
        "provenance": {"contains_y_or_labels": False},
    }
    torch.save(payload, path)
    loaded = load_label_free_bundle(path)
    assert loaded["node_id"].tolist() == [0, 1, 2, 3]
    scores = tmp_path / "scores.csv"
    write_scores(scores, [3, 1], [0.9, 0.1])
    assert scores.read_text(encoding="utf-8").splitlines() == [
        "node_id,anomaly_score", "1,0.1", "3,0.9"
    ]
    payload["y"] = torch.tensor([0, 0, 1, 1])
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="label-free"):
        load_label_free_bundle(tmp_path / "bad.pt")


def test_resource_matrix_is_168_and_failure_contract_requires_empty_metrics_and_evidence():
    rows = matrix_rows(_registry())
    assert len(rows) == 168
    assert {row["resource_status"] for row in rows} <= RESOURCE_STATUSES
    errors = validate_cell({"resource_status": "OOM", "metrics": {"auprc": 0.5}, "resource_evidence": None})
    assert any("empty metrics" in error for error in errors)
    assert any("resource evidence" in error for error in errors)
    errors = validate_cell({"resource_status": "OOT", "metrics": None, "resource_evidence": None})
    assert any("resource evidence" in error for error in errors)


def test_dominant_preflight_records_dense_n_squared_without_claiming_oom(tmp_path):
    bundle = tmp_path / "graph.pt"
    torch.save(
        {
            "features": torch.ones((5, 2)),
            "edge_index": torch.tensor([[0, 1], [1, 2]]),
            "node_id": torch.arange(5),
            "provenance": {"contains_y_or_labels": False},
        },
        bundle,
    )
    result = preflight(bundle)
    assert result["dense_n_by_n_path"] is True
    assert result["dense_tensor_bytes_each"] == 100
    assert result["preflight_status"] != "OOM"


def test_active_launchers_exclude_page_taq_and_bmp():
    launcher = (ROOT / "benchmark" / "run_all_baselines_seed0.sh").read_text(encoding="utf-8")
    for name in ("PAGE", "TAQ", "TAQ-GAD", "BMP"):
        assert f'"{name}"' not in launcher


def test_large_graph_preflight_covers_all_primary_methods_and_never_claims_oom():
    path = ROOT / "benchmark" / "evidence" / "primary_large_graph_preflight_20260809.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    primary = {
        row["method"] for row in _registry()["baselines"]
        if row["protocol_class"] == "PRIMARY_TRANSDUCTIVE_NORMAL_ONLY"
    }
    assert set(audit["methods"]) == primary
    assert set(audit["datasets"]) == {"T-Finance", "T-Social", "DGraph-Fin"}
    for method in audit["methods"].values():
        assert set(method["preflight"]) == {"T-Finance", "T-Social", "DGraph-Fin"}
        assert "OOM" not in method["preflight"].values()
