import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from coregad.data.oof import assemble_oof_scores
from coregad.data.splits import build_outer_fold_manifest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "benchmark" / "adapters" / "coregad_protocol_runner.py"


def _module():
    spec = importlib.util.spec_from_file_location("coregad_oof_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_coregad_oof_ownership() -> None:
    manifest = build_outer_fold_manifest(np.arange(5), np.arange(5, 105))
    outputs = []
    observed = []
    for fold in range(5):
        _, training, heldout = manifest.fold(fold)
        assert np.intersect1d(training, heldout).size == 0
        observed.extend(heldout.tolist())
        outputs.append((heldout, heldout.astype(np.float64)))
    assert sorted(observed) == list(range(5, 105))
    scores = assemble_oof_scores(105, outputs)
    assert np.isfinite(scores[5:]).all()


def test_oof_collector_enforces_exact_fold_ownership(tmp_path: Path) -> None:
    adapter = _module()
    dataset = tmp_path / "graph.npz"
    split_path = tmp_path / "split.json"
    score_dir = tmp_path / "scores"
    score_dir.mkdir()

    np.savez_compressed(
        dataset,
        x=np.ones((8, 2), dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        node_id=np.arange(8, dtype=np.int64),
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
    for fold, node_id in enumerate(range(3, 8)):
        np.savez_compressed(
            score_dir / f"fold_{fold}.npz",
            node_id=np.asarray([node_id], dtype=np.int64),
            score=np.asarray([fold / 10], dtype=np.float64),
        )

    output = tmp_path / "oof_scores.npz"
    adapter.collect_scores(str(dataset), str(split_path), str(score_dir), str(output))
    with np.load(output, allow_pickle=False) as stored:
        assert stored["node_id"].tolist() == [3, 4, 5, 6, 7]
        assert stored["owner_fold"].tolist() == [0, 1, 2, 3, 4]

    np.savez_compressed(
        score_dir / "fold_0.npz",
        node_id=np.asarray([4], dtype=np.int64),
        score=np.asarray([0.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="fold 0 scores do not exactly match"):
        adapter.collect_scores(str(dataset), str(split_path), str(score_dir), str(output))
