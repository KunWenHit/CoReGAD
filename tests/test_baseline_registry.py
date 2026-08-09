import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_registry_is_complete_and_portable():
    registry = json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))
    rows = registry["baselines"]
    assert len(rows) == 9
    assert len({row["method"] for row in rows}) == 9
    required = {
        "method",
        "paper_title",
        "protocol_family",
        "official_repo",
        "local_path",
        "upstream_commit",
        "official_repo_verified",
        "repo_reachable",
        "protocol_status",
        "smoke_status",
        "uses_pseudo_anomalies",
        "uses_teacher",
        "supported_datasets",
        "native_score_meaning",
        "notes",
    }
    assert all(required <= row.keys() for row in rows)
    assert all(row["uses_real_anomaly_labels"] is False for row in rows)
    assert {row["method"] for row in rows} == {
        "GGAD", "RHO", "GraphNC", "PAGE", "TAQ-GAD", "TAM", "HUGE", "OCGNN", "GAD-NR"
    }
    text = json.dumps(registry)
    assert "D:\\" not in text
    assert "E:\\" not in text
    assert "/data1/" not in text


def test_protocol_blockers_are_explicit_not_hidden():
    rows = json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))["baselines"]
    blocked = {row["method"]: row["smoke_status"] for row in rows if row["protocol_status"] == "PROTOCOL_INCOMPATIBLE"}
    assert blocked == {
        "PAGE": "BLOCKED_OFFICIAL_TRAINING_CODE_NOT_RELEASED",
        "TAQ-GAD": "BLOCKED_GT_LABEL_EARLY_STOPPING",
    }
