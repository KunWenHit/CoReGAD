import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_eight_dataset_hash_manifests_are_complete() -> None:
    index = json.loads((ROOT / "reproducibility" / "index.json").read_text(encoding="utf-8"))
    assert index["dataset_count"] == 8
    for row in index["datasets"]:
        manifest = json.loads((ROOT / "reproducibility" / row["manifest"]).read_text(encoding="utf-8"))
        for key in ("raw_sha256", "node_order_sha256", "edge_order_sha256", "feature_sha256", "label_sha256", "evaluation_mask_sha256", "split_sha256", "support_sha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", manifest[key]), (row["dataset"], key)
        assert manifest["identity_checks"]["exact_match"] is True
