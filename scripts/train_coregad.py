from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coregad.data.datasets import load_graph_npz
from coregad.data.oof import assemble_oof_scores
from coregad.data.splits import read_split_manifest
from coregad.training.pipeline import train_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CoReGAD under the five-fold normal-only OOF protocol."
    )
    parser.add_argument("--data", type=Path, required=True, help="Canonical graph NPZ.")
    parser.add_argument("--split", type=Path, required=True, help="Normal/unlabeled split JSON.")
    parser.add_argument("--output", type=Path, default=Path("outputs/coregad"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one epoch per stage for an installation smoke test only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_graph_npz(args.data, include_labels=False)
    manifest = read_split_manifest(args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    fold_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    owner_outputs: list[np.ndarray] = []
    for fold in range(5):
        normal, training_unlabeled, heldout = manifest.fold(fold)
        result = train_fold(
            features=dataset.features,
            edge_index=dataset.edge_index,
            normal_nodes=normal,
            training_unlabeled_nodes=training_unlabeled,
            heldout_nodes=heldout,
            fold=fold,
            model_seed=args.seed,
            device=args.device,
            normality_epochs=1 if args.smoke else 200,
            context_epochs=1 if args.smoke else 200,
            residual_epochs=1 if args.smoke else 300,
        )
        fold_outputs.append((result.heldout_nodes, result.final_anomaly_score))
        owner_outputs.append(np.full(result.heldout_nodes.size, fold, dtype=np.int8))
        torch.save(result.artifacts, args.output / f"fold_{fold}.pt")
    scores = assemble_oof_scores(dataset.features.shape[0], fold_outputs)
    nodes = np.concatenate([nodes for nodes, _ in fold_outputs])
    values = np.concatenate([values for _, values in fold_outputs])
    owners = np.concatenate(owner_outputs)
    order = np.argsort(nodes, kind="mergesort")
    np.savez_compressed(
        args.output / "oof_scores.npz",
        node_id=nodes[order],
        score=values[order],
        owner_fold=owners[order],
    )
    if not np.isfinite(scores[nodes]).all():
        raise RuntimeError("non-finite OOF score produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
