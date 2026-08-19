from __future__ import annotations

import numpy as np
import torch

from coregad.scalable import build_sparse_graph_operator


def test_large_graph_path_materializes_only_requested_rows(monkeypatch) -> None:
    n = 20_000
    rows = np.repeat(np.arange(n, dtype=np.int64), 3)
    columns = np.concatenate(
        [
            (np.arange(n, dtype=np.int64) + offset) % n
            for offset in (1, 7, 31)
        ]
    ).reshape(3, n).T.reshape(-1)
    edge_index = np.stack([rows, columns])
    visible = np.arange(n - 1000, dtype=np.int64)
    heldout = np.arange(n - 1000, n, dtype=np.int64)
    operator = build_sparse_graph_operator(edge_index, n, visible, heldout)
    recorded_shapes: list[tuple[int, int]] = []
    original = torch.sparse_coo_tensor

    def recording_sparse_coo_tensor(*args, **kwargs):
        size = kwargs.get("size")
        if size is None and len(args) >= 3:
            size = args[2]
        recorded_shapes.append(tuple(int(value) for value in size))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "sparse_coo_tensor", recording_sparse_coo_tensor)
    features = torch.randn(n, 4)
    output = operator.propagate_rows(features, 123, 1147)
    assert output.shape == (1024, 4)
    assert recorded_shapes == [(1024, n)]
    assert operator.audit()["dense_nxn_constructed"] is False
    assert operator.audit()["full_graph_device_coo_constructed"] is False
    assert operator.row_ptr.shape == (n + 1,)
