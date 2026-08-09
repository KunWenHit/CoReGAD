#!/usr/bin/env python3
"""Preparation-first launcher for the 21-method CoReGAD baseline inventory."""

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
RESOURCE_STATUSES = {"NOT_RUN", "PASS", "OOM", "OOT", "UNSUPPORTED", "ERROR"}
PRIMARY_GATES = {
    "SOURCE_PASS", "ENV_PASS", "NATIVE_SANITY_PASS", "LABEL_AUDIT_PASS",
    "CANONICAL_SMOKE_PASS", "SCORE_CONTRACT_PASS",
}


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def load_environments() -> dict[str, Any]:
    path = REGISTRY.with_name("baseline_env_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


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
    excluded = {"PAGE", "TAQ", "TAQ-GAD", "BMP"}
    errors: list[str] = []
    if registry["active_count"] != 21 or len(rows) != 21:
        errors.append("active inventory is not exactly 21")
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
        "benchmark_targets", "native_supported_datasets", "validated_datasets",
        "resource_status_by_dataset", "strict_oof_available", "fair_ranking",
        "execute_enabled", "environment_id", "source_url", "license",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            errors.append(f"{row.get('method', '?')}:missing:{sorted(missing)}")
        if row.get("benchmark_targets") != list(DATASETS):
            errors.append(f"{row['method']}:benchmark target contract differs")
        resource = row.get("resource_status_by_dataset", {})
        if set(resource) != set(DATASETS):
            errors.append(f"{row['method']}:resource matrix is incomplete")
        if set(resource.values()) - RESOURCE_STATUSES:
            errors.append(f"{row['method']}:invalid resource status")
        validated = row.get("validated_datasets", [])
        if not set(validated) <= set(DATASETS):
            errors.append(f"{row['method']}:validated dataset is not a benchmark target")
        evidence_by_dataset = row.get("validation_evidence_by_dataset", {})
        for dataset in validated:
            evidence = evidence_by_dataset.get(dataset)
            if not evidence or not (REPO / evidence).is_file():
                errors.append(f"{row['method']}:{dataset}:validated without evidence")
            if resource.get(dataset) != "PASS":
                errors.append(f"{row['method']}:{dataset}:validated without PASS status")
        if row.get("protocol_class") == "PRIMARY_TRANSDUCTIVE_NORMAL_ONLY":
            if row.get("ground_truth_anomaly_labels_used_for_training"):
                errors.append(f"{row['method']}:primary method uses anomaly labels")
            if not row.get("fair_ranking"):
                errors.append(f"{row['method']}:primary method not fair-ranked")
            gate_file = row.get("gate_file")
            if not gate_file or not (REPO / gate_file).is_file():
                errors.append(f"{row['method']}:primary gate file missing")
            else:
                gate = json.loads((REPO / gate_file).read_text(encoding="utf-8"))
                if gate.get("method") != row["method"] or set(gate.get("gates", {})) != PRIMARY_GATES:
                    errors.append(f"{row['method']}:invalid six-gate record")
                if row.get("execute_enabled"):
                    failed = []
                    for name, value in gate["gates"].items():
                        evidence = value.get("evidence")
                        evidence_paths = evidence if isinstance(evidence, list) else [evidence]
                        evidence_ok = bool(evidence_paths) and all(
                            isinstance(path, str) and (REPO / path).is_file()
                            for path in evidence_paths
                        )
                        if value.get("status") != "PASS" or not evidence_ok:
                            failed.append(name)
                    if failed:
                        errors.append(f"{row['method']}:execute enabled before gates:{failed}")
                    if not row.get("native_command"):
                        errors.append(f"{row['method']}:execute enabled without native command")
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
        "native_supported_datasets": row["native_supported_datasets"],
        "validated_datasets": row["validated_datasets"],
        "resource_status": row["resource_status_by_dataset"][args.dataset],
        "resource_preflight": row.get("resource_preflight_by_dataset", {}).get(args.dataset, "NOT_ASSESSED"),
        "execute": bool(args.execute),
    }


def render_native_command(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    release = REPO.parent
    environments = load_environments()
    environment = environments["environments"][row["environment_id"]]
    environment_path = environment.get("path")
    if not environment_path:
        raise RuntimeError(f"{row['method']}: environment path is unresolved")
    method_directory = row["method_id"].replace("/", "_")
    output = release / "outputs" / "baselines" / "seed0" / method_directory / args.dataset / f"seed_{args.seed}"
    values = {
        "environment_python": str(Path(environment_path) / "bin" / "python"),
        "bundle": str(release / "outputs" / "oof_anchor_audit" / "training_bundles" / args.dataset / "graph.pt"),
        "score_nodes": str(
            release / "datasets" / "splits" / f"{args.dataset}_seed0_folds.csv"
        ),
        "support": str(
            release / "datasets" / "supports" / f"{args.dataset}_seed0.json"
        ),
        "scores": str(output / "scores.csv"),
        "run_evidence": str(output / "run_evidence.json"),
        "source_checkout": str(release / row["source_checkout"]),
        "seed": str(args.seed),
        "device": args.device,
    }
    command = row.get("native_command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise RuntimeError(f"{row['method']}: native command must be an argv list")
    rendered = [item.format(**values) for item in command]
    forbidden = {"--y", "--label", "--labels", "--target"}
    if forbidden & set(rendered):
        raise RuntimeError(f"{row['method']}: native command exposes a forbidden label argument")
    return rendered


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="CoReGAD 21-method baseline launcher")
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
    try:
        native_command = render_native_command(row, args)
    except RuntimeError as error:
        print(f"EXECUTION_BLOCKED:{error}", file=sys.stderr)
        return 5
    return subprocess.run(native_command, shell=False, check=False, cwd=REPO).returncode


if __name__ == "__main__":
    raise SystemExit(main())
