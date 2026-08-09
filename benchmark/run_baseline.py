#!/usr/bin/env python3
"""Preparation-first launcher for the 22-method CoReGAD baseline inventory."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).with_name("baseline_registry.yaml")
DATASETS = (
    "Amazon", "Weibo", "YelpChi", "Tolokers", "T-Finance",
    "Elliptic", "T-Social", "DGraph-Fin",
)


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def resolve_method(query: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    lowered = query.casefold().replace("-", "_").replace(" ", "_")
    for row in rows:
        aliases = {
            row["method"].casefold().replace("-", "_").replace(" ", "_"),
            row["method_id"].casefold(),
        }
        if lowered in aliases:
            return row
    raise SystemExit(f"UNKNOWN_METHOD:{query}")


def baseline_root() -> Path | None:
    import os

    configured = os.environ.get("COREGAD_BASELINE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    server_default = REPO.parent / "baselines"
    return server_default if server_default.is_dir() else None


def _source_state(row: dict[str, Any], root: Path | None) -> str:
    checkout = row.get("source_checkout")
    if not checkout:
        return "BLOCKED_NO_CHECKOUT"
    if checkout.startswith("benchmark/"):
        return "PRESENT" if (REPO / checkout).is_file() else "MISSING"
    if root is None:
        return "SERVER_CHECK_SKIPPED"
    relative = checkout.removeprefix("baselines/")
    path = root / relative
    return "PRESENT" if path.exists() else "MISSING"


def validate_registry() -> dict[str, Any]:
    registry = load_registry()
    rows = registry["baselines"]
    names = [row["method"] for row in rows]
    expected_classes = {
        "PRIMARY_TRANSDUCTIVE_NORMAL_ONLY",
        "PU_AUXILIARY",
        "EXTERNAL_SUPERVISED_REFERENCE",
    }
    excluded = {"PAGE", "TAQ", "TAQ-GAD"}
    errors: list[str] = []
    if registry["active_count"] != 22 or len(rows) != 22:
        errors.append("active inventory is not exactly 22")
    if len(set(names)) != len(names):
        errors.append("duplicate baseline names")
    if excluded & set(names):
        errors.append("excluded PAGE/TAQ method is active")
    if {row["protocol_class"] for row in rows} - expected_classes:
        errors.append("unknown protocol class")
    required = {
        "method", "method_id", "main_protocol", "native_training_preserved",
        "ground_truth_anomaly_labels_used_for_training", "source_kind",
        "source_commit", "adapter_status", "reproduction_status",
        "datasets_supported", "strict_oof_available", "fair_ranking",
        "execute_enabled", "environment_id", "source_url", "license",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            errors.append(f"{row.get('method', '?')}:missing:{sorted(missing)}")
        if row.get("datasets_supported") != list(DATASETS):
            errors.append(f"{row['method']}:dataset contract differs")
        if row.get("protocol_class") == "PRIMARY_TRANSDUCTIVE_NORMAL_ONLY":
            if row.get("ground_truth_anomaly_labels_used_for_training"):
                errors.append(f"{row['method']}:primary method uses anomaly labels")
            if not row.get("fair_ranking"):
                errors.append(f"{row['method']}:primary method not fair-ranked")
        if row.get("protocol_class") == "EXTERNAL_SUPERVISED_REFERENCE" and row.get("fair_ranking"):
            errors.append(f"{row['method']}:supervised reference marked fair")
        if row.get("main_protocol") != "STANDARD_TRANSDUCTIVE_NORMAL_ONLY":
            errors.append(f"{row['method']}:wrong main protocol")
    root = baseline_root()
    sources = {row["method"]: _source_state(row, root) for row in rows}
    return {"valid": not errors, "active_count": len(rows), "errors": errors, "sources": sources}


def smoke() -> dict[str, Any]:
    import numpy as np

    adapter_path = REPO / "benchmark" / "adapters" / "transductive_contract.py"
    spec = importlib.util.spec_from_file_location("transductive_contract", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import transductive contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="coregad_baseline_smoke_") as directory:
        root = Path(directory)
        graph = root / "graph.npz"
        support = root / "support.json"
        evaluation = root / "evaluation.json"
        bundle = root / "bundle.npz"
        scores = root / "scores.csv"
        np.savez_compressed(
            graph,
            x=np.arange(18, dtype=np.float32).reshape(6, 3),
            edge_index=np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
            node_id=np.arange(6, dtype=np.int64),
            y=np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int64),
        )
        support.write_text(json.dumps({"normal_label_budget": 2, "normal_support_ids": [0, 1]}), encoding="utf-8")
        evaluation.write_text(json.dumps({"evaluation_nodes": [2, 3, 4, 5]}), encoding="utf-8")
        module.prepare_transductive_bundle(graph, support, evaluation, bundle, uses_normal_support=True)
        with np.load(bundle, allow_pickle=False) as stored:
            fields = set(stored.files)
            if "y" in fields or "owner_fold" in fields or stored["training_nodes"].size != 6:
                raise RuntimeError("transductive training bundle violated label/full-graph contract")
        module.write_scores(scores, [2, 3, 4, 5], [0.1, 0.9, 0.2, 0.8], expected_nodes=[2, 3, 4, 5])
        metrics = module.evaluate_scores(graph, scores, evaluation)
    return {"passed": True, "auprc": metrics["auprc"], "full_graph_training": True, "label_free_bundle": True}


def command_plan(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": row["method"], "method_id": row["method_id"],
        "dataset": args.dataset, "seed": args.seed, "protocol": args.protocol,
        "device": args.device, "main_protocol": row["main_protocol"],
        "native_training_preserved": row["native_training_preserved"],
        "training_label_access": row["training_label_access"],
        "fair_ranking": row["fair_ranking"], "environment_id": row["environment_id"],
        "adapter_status": row["adapter_status"], "runnable_status": row["runnable_status"],
        "execute": bool(args.execute),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="CoReGAD 22-method baseline launcher")
    result.add_argument("--method")
    result.add_argument("--dataset", choices=DATASETS)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--protocol", choices=("transductive", "strict_oof"), default="transductive")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--list-methods", action="store_true")
    result.add_argument("--validate", action="store_true")
    result.add_argument("--smoke", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    registry = load_registry()
    rows = registry["baselines"]
    if args.list_methods:
        for row in rows:
            print(f"{row['method']}\t{row['protocol_class']}\t{row['runnable_status']}")
        return 0
    if args.validate:
        result = validate_registry()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 2
    if args.smoke:
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return 0
    if not args.method or not args.dataset:
        raise SystemExit("--method and --dataset are required unless a preparation action is selected")
    row = resolve_method(args.method, rows)
    if args.protocol == "strict_oof" and not row["strict_oof_available"]:
        print(f"STRICT_OOF_NOT_IMPLEMENTED:{row['method']}", file=sys.stderr)
        return 3
    plan = command_plan(row, args)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not row["execute_enabled"]:
        print(
            f"EXECUTION_BLOCKED:{row['method']}:{row['runnable_status']}:"
            "native bridge/sanity gate has not passed",
            file=sys.stderr,
        )
        return 4
    native_command = row.get("native_command")
    if not native_command:
        print(f"EXECUTION_BLOCKED:{row['method']}:NO_NATIVE_COMMAND", file=sys.stderr)
        return 5
    return subprocess.run(native_command, shell=False, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
