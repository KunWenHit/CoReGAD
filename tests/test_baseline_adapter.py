import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "benchmark" / "adapters" / "coregad_protocol_runner.py"


def _module():
    spec = importlib.util.spec_from_file_location("coregad_baseline_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_baseline_fold_bundle_is_label_free(tmp_path):
    adapter = _module()
    dataset = tmp_path / "graph.npz"
    split_path = tmp_path / "split.json"
    support_path = tmp_path / "support.json"
    np.savez_compressed(
        dataset,
        x=np.arange(24, dtype=np.float32).reshape(8, 3),
        edge_index=np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
        node_id=np.arange(8, dtype=np.int64),
        y=np.asarray([0, 0, 0, 1, 0, 1, 0, 1], dtype=np.int64),
    )
    split_path.write_text(
        json.dumps(
            {
                "folds": 5,
                "normal_nodes": [0, 1, 2],
                "unlabeled_nodes": [3, 4, 5, 6, 7],
                "ownership": [0, 1, 2, 3, 4],
            }
        ),
        encoding="utf-8",
    )
    support_path.write_text(
        json.dumps(
            {
                "folds": 5,
                "support_budget": 2,
                "support_nodes_by_fold": [[0, 1]] * 5,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "fold_0_input.npz"
    adapter.prepare_fold(str(dataset), str(split_path), str(support_path), 0, str(output))
    with np.load(output, allow_pickle=False) as stored:
        assert "y" not in stored.files
        assert stored["score_nodes"].tolist() == [3]
        assert 3 not in stored["train_visible_nodes"].tolist()
