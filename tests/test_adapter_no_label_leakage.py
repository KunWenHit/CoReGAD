import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_registry_never_silently_marks_known_leaky_code_ready() -> None:
    rows = json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))["baselines"]
    by_name = {row["method"]: row for row in rows}
    assert by_name["GraphNC"]["protocol_status"] == "ADAPTER_REQUIRED"
    assert by_name["PAGE"]["protocol_status"] == "PROTOCOL_INCOMPATIBLE"
    assert by_name["TAQ-GAD"]["protocol_status"] == "PROTOCOL_INCOMPATIBLE"
    assert all(row["uses_real_anomaly_labels"] is False for row in rows)
