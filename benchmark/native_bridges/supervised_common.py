"""Canonical graph plus explicit supervision loader for external references."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dgl
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from benchmark.native_bridges.common import load_label_free_bundle


def load_supervised_reference(
    bundle_path: str | Path,
    label_source: str | Path,
    *,
    trial: int = 0,
) -> dict[str, Any]:
    payload = load_label_free_bundle(bundle_path)
    source = Path(label_source).resolve()
    raw = None
    raw_format = "dgl_graph"
    arrays: dict[str, Any] = {}
    try:
        graphs, _ = dgl.load_graphs(str(source))
        if len(graphs) != 1:
            raise ValueError("canonical supervised label source must contain exactly one DGL graph")
        raw = graphs[0]
        feature_key = "feature" if "feature" in raw.ndata else "feat"
        label_key = "label" if "label" in raw.ndata else "labels"
        raw_features = torch.as_tensor(raw.ndata[feature_key], dtype=torch.float32).cpu()
        labels = torch.as_tensor(raw.ndata[label_key]).long().squeeze().cpu()
    except dgl.DGLError:
        if source.suffix.casefold() == ".npz":
            raw_format = "numpy_npz"
            with np.load(source, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            raw_features = torch.as_tensor(arrays["node_features"], dtype=torch.float32)
            labels = torch.as_tensor(arrays["node_labels"]).long().squeeze()
        elif source.suffix.casefold() == ".mat":
            raw_format = "scipy_mat"
            import scipy.io
            import scipy.sparse

            arrays = scipy.io.loadmat(source)
            feature_value = arrays.get("features", arrays.get("Attributes"))
            label_value = arrays.get("label", arrays.get("Label"))
            if feature_value is None or label_value is None:
                raise ValueError("MAT supervision source has no recognized features/label arrays")
            if scipy.sparse.issparse(feature_value):
                feature_value = feature_value.toarray()
            raw_features = torch.as_tensor(feature_value, dtype=torch.float32)
            labels = torch.as_tensor(np.asarray(label_value).reshape(-1)).long()
        else:
            raise
    if raw_features.shape[0] != payload["x"].shape[0]:
        raise ValueError("supervision source node count differs from the canonical bundle")
    if raw_features.shape != payload["x"].shape or not torch.equal(raw_features, payload["x"]):
        raise ValueError("supervision source feature order differs from the canonical bundle")
    if labels.ndim == 2:
        labels = labels.argmax(dim=1)
    masks = {}
    mask_container = raw.ndata if raw is not None else arrays
    if any(f"{split}_mask" in mask_container or f"{split}_masks" in mask_container for split in ("train", "val", "test")):
        split_source = "source_provided_masks"
        for split in ("train", "val", "test"):
            plural = f"{split}_masks"
            singular = f"{split}_mask"
            value = mask_container[plural] if plural in mask_container else mask_container[singular]
            value = torch.as_tensor(value)
            if value.ndim == 2:
                value = value[:, trial] if value.shape[0] == labels.numel() else value[trial]
            masks[split] = value.bool().cpu()
    else:
        split_source = "deterministic_stratified_40_20_40_random_state_2_plus_trial"
        eligible = np.where(np.isin(labels.numpy(), (0, 1)))[0]
        train_nodes, remainder = train_test_split(
            eligible,
            train_size=0.4,
            random_state=2 + trial,
            shuffle=True,
            stratify=labels.numpy()[eligible],
        )
        validation_nodes, test_nodes = train_test_split(
            remainder,
            test_size=2.0 / 3.0,
            random_state=2 + trial,
            shuffle=True,
            stratify=labels.numpy()[remainder],
        )
        for split, nodes in (("train", train_nodes), ("val", validation_nodes), ("test", test_nodes)):
            masks[split] = torch.zeros_like(labels, dtype=torch.bool)
            masks[split][torch.as_tensor(nodes, dtype=torch.long)] = True
    if any(mask.shape != labels.shape for mask in masks.values()):
        raise ValueError("supervised masks do not align with canonical node order")
    if any((masks[a] & masks[b]).any() for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("supervised train/validation/test masks overlap")
    graph = dgl.graph(
        (payload["edge_index"][0], payload["edge_index"][1]),
        num_nodes=payload["x"].shape[0],
    )
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    graph.ndata["feature"] = payload["x"]
    graph.ndata["label"] = labels
    graph.ndata["train_mask"] = masks["train"]
    graph.ndata["val_mask"] = masks["val"]
    graph.ndata["test_mask"] = masks["test"]
    return {
        "payload": payload,
        "graph": graph,
        "labels": labels,
        "train_mask": masks["train"],
        "val_mask": masks["val"],
        "test_mask": masks["test"],
        "label_source": str(source),
        "raw_format": raw_format,
        "split_source": split_source,
        "trial": trial,
    }
