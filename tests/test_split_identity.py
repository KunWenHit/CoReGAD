import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_exact_split_identity_is_five_fold_and_not_regenerated() -> None:
    for path in (ROOT / "reproducibility" / "splits").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        assert row["outer_folds"] == 5
        assert row["fold_seed"] == 0
        assert sum(row["fold_counts"].values()) == row["evaluation_node_count"]
        assert "never regenerate" in row["generation_rule"]
