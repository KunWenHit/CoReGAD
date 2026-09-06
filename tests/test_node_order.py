import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_node_order_is_explicit_and_dataset_specific() -> None:
    values = []
    for path in (ROOT / "reproducibility" / "datasets").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        values.append(row["node_order_sha256"])
        assert row["preprocessing"]["node_ids"].startswith("canonical zero-based")
    assert len(values) == len(set(values)) == 8
