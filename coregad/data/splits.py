from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coregad.models.structural_residualization import hash64


OUTER_FOLDS = 5


@dataclass(frozen=True)
class OuterFoldManifest:
    normal_nodes: np.ndarray
    unlabeled_nodes: np.ndarray
    ownership: np.ndarray
    folds: int = OUTER_FOLDS

    def fold(self, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 0 <= int(fold) < self.folds:
            raise ValueError("fold is outside the manifest")
        heldout = self.unlabeled_nodes[self.ownership == int(fold)]
        training_unlabeled = self.unlabeled_nodes[self.ownership != int(fold)]
        return self.normal_nodes.copy(), training_unlabeled, heldout


def build_outer_fold_manifest(
    normal_nodes: np.ndarray,
    unlabeled_nodes: np.ndarray,
    *,
    seed: int = 0,
) -> OuterFoldManifest:
    normal = np.unique(np.asarray(normal_nodes, dtype=np.int64))
    unlabeled = np.unique(np.asarray(unlabeled_nodes, dtype=np.int64))
    if np.intersect1d(normal, unlabeled).size:
        raise ValueError("normal and unlabeled sets must be disjoint")
    ownership = (
        hash64(unlabeled.astype(np.uint64), int(seed)) % np.uint64(OUTER_FOLDS)
    ).astype(np.int8)
    return OuterFoldManifest(normal, unlabeled, ownership)


def read_split_manifest(path: str | Path) -> OuterFoldManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = int(payload.get("folds", OUTER_FOLDS))
    if folds != OUTER_FOLDS:
        raise ValueError("released protocol requires five outer folds")
    normal = np.asarray(payload["normal_nodes"], dtype=np.int64)
    unlabeled = np.asarray(payload["unlabeled_nodes"], dtype=np.int64)
    ownership = np.asarray(payload["ownership"], dtype=np.int8)
    if ownership.shape != unlabeled.shape:
        raise ValueError("ownership must align with unlabeled nodes")
    return OuterFoldManifest(normal, unlabeled, ownership, folds)
