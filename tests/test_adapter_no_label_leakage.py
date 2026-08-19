import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "benchmark" / "adapters" / "transductive_contract.py"


def _module():
    spec = importlib.util.spec_from_file_location("transductive_contract", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normal_only_training_bundle_cannot_expose_y(tmp_path):
    adapter = _module()
    graph = tmp_path / "graph.npz"
    support = tmp_path / "support.json"
    evaluation = tmp_path / "evaluation.json"
    output = tmp_path / "training.npz"
    np.savez_compressed(
        graph,
        x=np.arange(18, dtype=np.float32).reshape(6, 3),
        edge_index=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        node_id=np.arange(6, dtype=np.int64),
        y=np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int64),
    )
    support.write_text(json.dumps({"normal_label_budget": 2, "normal_support_ids": [0, 1]}), encoding="utf-8")
    evaluation.write_text(json.dumps({"evaluation_nodes": [2, 3, 4, 5]}), encoding="utf-8")
    adapter.prepare_transductive_bundle(graph, support, evaluation, output, uses_normal_support=True)
    with np.load(output, allow_pickle=False) as stored:
        assert "y" not in stored.files
        assert "owner_fold" not in stored.files
        assert stored["training_nodes"].tolist() == list(range(6))
        assert stored["normal_support_ids"].tolist() == [0, 1]


def test_all_normal_only_registry_rows_forbid_anomaly_label_training():
    rows = json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))["baselines"]
    normal_only = [row for row in rows if row["protocol_class"] == "PRIMARY_TRANSDUCTIVE_NORMAL_ONLY"]
    assert all(row["ground_truth_anomaly_labels_used_for_training"] is False for row in normal_only)
