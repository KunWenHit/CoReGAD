import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _adapter():
    path = ROOT / "benchmark" / "adapters" / "transductive_contract.py"
    spec = importlib.util.spec_from_file_location("transductive_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_main_protocol_has_no_mandatory_outer_ownership():
    protocol = json.loads((ROOT / "benchmark" / "protocols" / "standard_transductive_normal_only_v1.json").read_text(encoding="utf-8"))
    assert protocol["name"] == "STANDARD_TRANSDUCTIVE_NORMAL_ONLY"
    assert protocol["complete_unlabeled_graph_covariates_visible"] is True
    assert protocol["mandatory_outer_ownership"] is False
    assert protocol["mandatory_baseline_cross_fitting"] is False
    assert protocol["secondary_robustness_protocol"] == "STRICT_OOF"


def test_node_order_support_identity_score_schema_and_orientation(tmp_path):
    adapter = _adapter()
    graph = tmp_path / "graph.npz"
    support = tmp_path / "support.json"
    evaluation = tmp_path / "evaluation.json"
    bundle = tmp_path / "bundle.npz"
    scores = tmp_path / "scores.csv"
    np.savez_compressed(
        graph,
        x=np.arange(20, dtype=np.float32).reshape(5, 4),
        edge_index=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        node_id=np.asarray([10, 11, 12, 13, 14], dtype=np.int64),
        y=np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
    )
    support.write_text(json.dumps({"normal_label_budget": 2, "train_normal_node_ids": [0, 2]}), encoding="utf-8")
    evaluation.write_text(json.dumps({"evaluation_nodes": [0, 1, 2, 3, 4]}), encoding="utf-8")
    adapter.prepare_transductive_bundle(graph, support, evaluation, bundle, uses_normal_support=True)
    with np.load(bundle, allow_pickle=False) as stored:
        assert stored["node_id"].tolist() == [10, 11, 12, 13, 14]
        assert stored["normal_support_ids"].tolist() == [0, 2]
        assert stored["score_nodes"].tolist() == [0, 1, 2, 3, 4]
    adapter.write_scores(scores, [0, 1, 2, 3, 4], [0.0, 0.1, 0.2, 0.9, 0.8], expected_nodes=[0, 1, 2, 3, 4])
    metrics = adapter.evaluate_scores(graph, scores, evaluation)
    assert metrics["auprc"] == 1.0
    assert metrics["auroc"] == 1.0


def test_strict_oof_infrastructure_is_retained():
    assert (ROOT / "benchmark" / "adapters" / "coregad_protocol_runner.py").is_file()
    assert (ROOT / "tests" / "test_strict_oof_anchor.py").is_file()
    assert (ROOT / "tests" / "test_oof_ownership.py").is_file()
