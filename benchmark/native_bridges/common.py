"""Shared I/O and evidence helpers for native baseline bridges.

This module intentionally has no evaluator and never opens a label array.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


TRAINING_KEYS = {"features", "edge_index", "node_id", "provenance"}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_label_free_bundle(path: str | Path) -> dict[str, Any]:
    """Load the frozen torch bundle and reject any label-like key."""
    source = Path(path).resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if set(payload) != TRAINING_KEYS:
        raise ValueError(f"training bundle keys are not label-free: {sorted(payload)}")
    forbidden = {key for key in payload if key.casefold() in {"y", "label", "labels", "target"}}
    if forbidden:
        raise ValueError(f"training bundle contains forbidden label keys: {sorted(forbidden)}")
    provenance = dict(payload["provenance"])
    if provenance.get("contains_y_or_labels") is not False:
        raise ValueError("training bundle provenance does not attest label isolation")
    x = torch.as_tensor(payload["features"], dtype=torch.float32).contiguous()
    edge_index = torch.as_tensor(payload["edge_index"], dtype=torch.long).contiguous()
    node_id = torch.as_tensor(payload["node_id"], dtype=torch.long).contiguous()
    if x.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("invalid canonical graph shapes")
    if not torch.equal(node_id, torch.arange(x.shape[0], dtype=torch.long)):
        raise ValueError("canonical node order changed")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]):
        raise ValueError("edge index is outside canonical node order")
    return {
        "x": x,
        "edge_index": edge_index,
        "node_id": node_id,
        "provenance": provenance,
        "bundle_path": source,
    }


def load_normal_support(path: str | Path, *, num_nodes: int) -> np.ndarray:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    values = raw.get("train_normal_node_ids", raw.get("normal_support_ids"))
    if values is None:
        raise ValueError("support manifest has no frozen normal-support IDs")
    support = np.asarray(values, dtype=np.int64)
    budget = int(raw.get("normal_label_budget", raw.get("support_budget", support.size)))
    if support.shape != (budget,) or np.unique(support).size != support.size:
        raise ValueError("support IDs do not exactly match the frozen budget")
    if support.size and (support.min() < 0 or support.max() >= num_nodes):
        raise ValueError("support ID outside canonical node order")
    return support


def load_score_nodes(path: str | Path, *, num_nodes: int) -> np.ndarray:
    source = Path(path)
    if source.suffix.casefold() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            field = "node_id" if "node_id" in (reader.fieldnames or []) else None
            if field is None:
                raise ValueError("score-node CSV must contain node_id")
            nodes = [int(row[field]) for row in reader]
    else:
        raw = json.loads(source.read_text(encoding="utf-8"))
        nodes = raw.get("evaluation_nodes", raw.get("unlabeled_nodes"))
        if nodes is None:
            raise ValueError("score-node manifest has no evaluation node IDs")
    result = np.asarray(nodes, dtype=np.int64)
    if np.unique(result).size != result.size:
        raise ValueError("score-node IDs are not unique")
    if result.size and (result.min() < 0 or result.max() >= num_nodes):
        raise ValueError("score-node ID outside canonical node order")
    return result


def write_scores(
    path: str | Path,
    node_id: Iterable[int] | np.ndarray,
    anomaly_score: Iterable[float] | np.ndarray,
) -> Path:
    nodes = np.asarray(node_id, dtype=np.int64)
    scores = np.asarray(anomaly_score, dtype=np.float64)
    if nodes.ndim != 1 or scores.shape != nodes.shape or np.unique(nodes).size != nodes.size:
        raise ValueError("score vectors must be aligned and node-unique")
    if not np.all(np.isfinite(scores)):
        raise ValueError("anomaly scores must be finite")
    order = np.argsort(nodes, kind="mergesort")
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite score file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("node_id", "anomaly_score"))
        writer.writerows(zip(nodes[order].tolist(), scores[order].tolist()))
    return output


def git_head(source: str | Path) -> str:
    command = ["git", "-C", str(Path(source).resolve()), "rev-parse", "HEAD"]
    server_git = Path("/data1/anaconda3/bin/git")
    if server_git.is_file():
        command[0] = str(server_git)
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def write_evidence(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
