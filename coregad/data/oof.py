from __future__ import annotations

import numpy as np


def assemble_oof_scores(
    num_nodes: int,
    fold_outputs: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    scores = np.full(int(num_nodes), np.nan, dtype=np.float64)
    ownership_count = np.zeros(int(num_nodes), dtype=np.int8)
    for nodes, values in fold_outputs:
        nodes = np.asarray(nodes, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if nodes.shape != values.shape:
            raise ValueError("nodes and scores must align")
        scores[nodes] = values
        ownership_count[nodes] += 1
    covered = ownership_count > 0
    if bool(np.any(ownership_count[covered] != 1)):
        raise ValueError("each scored node must have exactly one outer owner")
    return scores
