import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_source_provenance_never_overstates_official_identity() -> None:
    rows = json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))["baselines"]
    assert len(rows) == 22
    official = [row for row in rows if row["source_kind"] in {"AUTHOR_OFFICIAL", "OFFICIAL_BENCHMARK_REIMPLEMENTATION"}]
    assert official
    assert all(len(row["source_commit"]) == 40 for row in official)
    assert all(row["source_url"].startswith("https://github.com/") for row in official)
    by_name = {row["method"]: row for row in rows}
    assert by_name["BMP"]["source_kind"] == "PAPER_DERIVED"
    assert "404" in by_name["BMP"]["reproduction_status"]
    assert by_name["Structure-aware PU-GNN"]["source_kind"] == "PAPER_DERIVED"
    assert by_name["HSMAD"]["runnable_status"] == "BLOCKED_SOURCE"
