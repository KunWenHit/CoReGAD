from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coregad.data.datasets import load_graph_npz
from coregad.data.splits import build_outer_fold_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical graph data and write a split manifest.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--normal-nodes", type=Path, required=True, help="Text file with one node id per line.")
    parser.add_argument("--unlabeled-nodes", type=Path, required=True, help="Text file with one node id per line.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_ids(path: Path) -> np.ndarray:
    return np.asarray(
        [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()],
        dtype=np.int64,
    )


def main() -> int:
    args = parse_args()
    load_graph_npz(args.data, include_labels=False)
    manifest = build_outer_fold_manifest(
        read_ids(args.normal_nodes), read_ids(args.unlabeled_nodes), seed=args.seed
    )
    payload = {
        "folds": 5,
        "seed": int(args.seed),
        "normal_nodes": manifest.normal_nodes.tolist(),
        "unlabeled_nodes": manifest.unlabeled_nodes.tolist(),
        "ownership": manifest.ownership.astype(int).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
