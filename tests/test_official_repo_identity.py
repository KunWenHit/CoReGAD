import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_active_baselines_have_verified_official_repositories() -> None:
    rows = json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))["baselines"]
    assert len(rows) == 9
    assert all(row["status"] == "ACTIVE" for row in rows)
    assert all(row["official_repo_verified"] and row["repo_reachable"] for row in rows)
    assert all(len(row["upstream_commit"]) == 40 for row in rows)
