from inspect import signature

import numpy as np

from coregad.data.datasets import load_graph_npz


def test_training_loader_does_not_open_y(tmp_path) -> None:
    path = tmp_path / "graph.npz"
    np.savez(path, x=np.ones((3, 2), dtype=np.float32), edge_index=np.asarray([[0, 1], [1, 2]]), y=np.asarray([0, 1, 0]))
    graph = load_graph_npz(path, include_labels=False)
    assert graph.labels is None
    assert "labels" not in signature(load_graph_npz).parameters
