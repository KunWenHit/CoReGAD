import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_registry_is_complete_and_portable():
    registry = json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))
    rows = registry["baselines"]
    assert len(rows) == 24
    assert len({row["method"] for row in rows}) == 24
    required = {
        "method",
        "paper_title",
        "protocol_class",
        "official_repo",
        "local_path",
        "upstream_commit",
        "protocol_branch",
        "adapter_status",
        "requires_anomaly_labels",
        "uses_pseudo_anomalies",
        "uses_external_teacher",
        "datasets_supported",
        "dependencies",
        "notes",
    }
    assert all(required <= row.keys() for row in rows)
    assert all(not (row["fair_ranking"] and row["requires_anomaly_labels"]) for row in rows)
    text = json.dumps(registry)
    assert "D:\\" not in text
    assert "E:\\" not in text
    assert "/data1/" not in text


def test_supervised_references_are_excluded_from_fair_ranking():
    rows = json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))["baselines"]
    external = [row for row in rows if row["protocol_class"] == "EXTERNAL_SUPERVISED_REFERENCE"]
    assert len(external) == 9
    assert all(row["requires_anomaly_labels"] is True for row in external)
    assert all(row["fair_ranking"] is False for row in external)
