#!/usr/bin/env python3
"""Export the public registry into the server-managed baseline workspace."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    return json.loads((REPO / "benchmark" / name).read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def export(root: Path) -> None:
    registry = _load("baseline_registry.yaml")
    environments = _load("baseline_env_manifest.json")
    excluded = _load("excluded_baselines.yaml")
    manifest_root = root / "manifests"
    _write_json(manifest_root / "baseline_registry.yaml", registry)
    _write_json(manifest_root / "baseline_env_manifest.json", environments)
    _write_json(manifest_root / "excluded_baselines.yaml", excluded)
    rows = registry["baselines"]
    _write_csv(
        manifest_root / "environment_matrix.csv",
        ["method", "environment_id", "path", "status", "protected_pfr_gad"],
        [
            [
                row["method"], row["environment_id"],
                environments["environments"][row["environment_id"]]["path"],
                environments["environments"][row["environment_id"]]["status"],
                True,
            ]
            for row in rows
        ],
    )
    _write_csv(
        manifest_root / "protocol_matrix.csv",
        [
            "method", "protocol_class", "main_protocol", "native_training_preserved",
            "ground_truth_anomaly_labels_used_for_training", "strict_oof_available",
            "fair_ranking", "adapter_status", "runnable_status",
        ],
        [
            [
                row["method"], row["protocol_class"], row["main_protocol"],
                row["native_training_preserved"],
                row["ground_truth_anomaly_labels_used_for_training"],
                row["strict_oof_available"], row["fair_ranking"],
                row["adapter_status"], row["runnable_status"],
            ]
            for row in rows
        ],
    )
    _write_csv(
        manifest_root / "dataset_support_matrix.csv",
        [
            "method", "dataset", "benchmark_target", "native_supported",
            "validated", "resource_status", "preflight_status",
            "runnable_status", "no_subsampling",
        ],
        [
            [
                row["method"], dataset, True,
                dataset in row["native_supported_datasets"],
                dataset in row["validated_datasets"],
                row["resource_status_by_dataset"][dataset],
                row.get("resource_preflight_by_dataset", {}).get(dataset, "NOT_ASSESSED"),
                row["runnable_status"], True,
            ]
            for row in rows
            for dataset in row["benchmark_targets"]
        ],
    )
    source_rows = []
    for row in rows:
        checkout = row.get("source_checkout")
        if checkout and checkout.startswith("baselines/"):
            source_path = root.parent / checkout
        elif checkout and checkout.startswith("benchmark/"):
            source_path = REPO / checkout
        else:
            source_path = None
        source_rows.append(
            {
                "method": row["method"], "source_url": row["source_url"],
                "source_commit": row["source_commit"], "source_kind": row["source_kind"],
                "license": row["license"], "checkout": checkout,
                "checkout_present": bool(source_path and source_path.exists()),
                "reproduction_status": row["reproduction_status"],
            }
        )
    _write_json(
        manifest_root / "source_recovery_manifest.json",
        {
            "schema_version": 2,
            "generated_from": "coregad/benchmark/baseline_registry.yaml",
            "active_total": len(rows),
            "benchmark_target_cells": sum(len(row["benchmark_targets"]) for row in rows),
            "sources": source_rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True, type=Path)
    args = parser.parse_args()
    export(args.baseline_root.resolve())


if __name__ == "__main__":
    main()
