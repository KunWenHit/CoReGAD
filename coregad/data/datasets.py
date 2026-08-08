from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class GraphDataset:
    features: torch.Tensor
    edge_index: torch.Tensor
    node_ids: np.ndarray
    labels: np.ndarray | None = None


def load_graph_npz(path: str | Path, *, include_labels: bool = False) -> GraphDataset:
    """Load the public canonical format without touching labels during training."""

    with np.load(Path(path), allow_pickle=False) as stored:
        features = torch.from_numpy(np.asarray(stored["x"], dtype=np.float32))
        edge_index = torch.from_numpy(np.asarray(stored["edge_index"], dtype=np.int64))
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        node_ids = (
            np.asarray(stored["node_id"], dtype=np.int64)
            if "node_id" in stored
            else np.arange(features.shape[0], dtype=np.int64)
        )
        labels = None
        if include_labels:
            if "y" not in stored:
                raise ValueError("evaluation requested labels but y is absent")
            labels = np.asarray(stored["y"], dtype=np.int64)
    return GraphDataset(features, edge_index, node_ids, labels)
