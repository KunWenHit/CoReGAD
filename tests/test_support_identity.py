import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_supports_are_exact_normal_and_disjoint() -> None:
    paths = list((ROOT / "reproducibility" / "supports").glob("*.json"))
    assert len(paths) == 8
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        assert row["normal_label_budget"] > 0
        assert row["support_nodes_are_normal"] is True
        assert row["support_and_evaluation_disjoint"] is True
        assert row["node_ids_public"] is False
