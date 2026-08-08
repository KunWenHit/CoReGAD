from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coregad.data.datasets import load_graph_npz
from coregad.evaluation.protocol import evaluate_oof_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate materialized CoReGAD OOF scores.")
    parser.add_argument("--data", type=Path, required=True, help="Canonical graph NPZ with y.")
    parser.add_argument("--scores", type=Path, required=True, help="OOF score NPZ.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_graph_npz(args.data, include_labels=True)
    if dataset.labels is None:
        raise RuntimeError("evaluation labels are unavailable")
    metrics = evaluate_oof_file(args.scores, dataset.labels)
    text = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
