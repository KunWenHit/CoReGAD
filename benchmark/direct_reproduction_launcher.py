"""One-command dispatcher for the frozen 21-method baseline inventory.

Training is always a child process.  Normal-only child processes receive only
the label-free graph bundle (and, where native semantics require it, the frozen
normal-support manifest).  Evaluation is a second child process started only
after the score contract has been validated.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent
RUNTIME_REGISTRY = ROOT / "benchmark" / "baseline_runtime_registry.json"
BASELINE_REGISTRY = ROOT / "benchmark" / "baseline_registry.yaml"
DATASETS = (
    "Amazon", "Weibo", "YelpChi", "Tolokers", "T-Finance",
    "Elliptic", "T-Social", "DGraph-Fin",
)
REQUIRED_GATES = (
    "SOURCE_PASS",
    "ENV_PASS",
    "NATIVE_SANITY_PASS",
    "LABEL_OR_SUPERVISION_AUDIT_PASS",
    "CANONICAL_BRIDGE_PASS",
    "SCORE_CONTRACT_PASS",
    "LAUNCHER_PASS",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _normalise_method(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _entries() -> list[dict[str, Any]]:
    registry = _read_json(RUNTIME_REGISTRY)
    if registry.get("active_total") != 21 or len(registry.get("methods", [])) != 21:
        raise RuntimeError("runtime registry is not the frozen 21-method inventory")
    return registry["methods"]


def _resolve_method(value: str) -> dict[str, Any]:
    key = _normalise_method(value)
    matches = [row for row in _entries() if key in {_normalise_method(row["method"]), _normalise_method(row["method_id"])}]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous baseline method: {value}")
    return matches[0]


def _gate_failures(row: dict[str, Any]) -> list[str]:
    gates = row.get("gates", {})
    return [name for name in REQUIRED_GATES if gates.get(name, {}).get("status") != "PASS"]


def _git(source: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    candidates = (
        Path("/data1/anaconda3/bin/git"),
        Path("/data1/wk/conda_envs/baseline_primary_torch24/bin/git"),
    )
    executable = next((str(path) for path in candidates if path.is_file()), "git")
    return subprocess.run(
        [executable, "-C", str(source), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _verify_source(row: dict[str, Any]) -> dict[str, Any]:
    source = RELEASE / row["native_source"]
    entrypoint = source / row["native_entrypoint"]
    if not source.exists() or not entrypoint.is_file():
        raise FileNotFoundError(f"missing native source/entrypoint: {entrypoint}")
    frozen = row["source_commit"]
    if frozen.startswith("paper:"):
        return {"source": str(source), "source_commit": frozen, "checkout_head": None}
    head = _git(source, "rev-parse", "HEAD").stdout.strip()
    ancestor = _git(source, "merge-base", "--is-ancestor", frozen, head, check=False)
    if ancestor.returncode != 0:
        raise RuntimeError(f"{row['method']}: frozen source commit is not in checkout HEAD")
    return {"source": str(source), "source_commit": frozen, "checkout_head": head}


def _verify_environment(row: dict[str, Any]) -> dict[str, Any]:
    environment = Path(row["environment_path"])
    python = Path(row["python_executable"])
    if not environment.is_dir() or not python.is_file():
        raise FileNotFoundError(f"missing runtime environment/python: {environment} / {python}")
    probe = subprocess.run(
        [str(python), "-c", "import json,platform; print(json.dumps({'python':platform.python_version()}))"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"environment": str(environment), **json.loads(probe.stdout)}


def _bundle_paths(dataset: str, seed: int) -> dict[str, Path]:
    return {
        "bundle": RELEASE / "outputs" / "oof_anchor_audit" / "training_bundles" / dataset / "graph.pt",
        "manifest": RELEASE / "datasets" / "manifests" / f"{dataset}.json",
        "support": RELEASE / "datasets" / "supports" / f"{dataset}_seed{seed}.json",
        "score_nodes": RELEASE / "datasets" / "splits" / f"{dataset}_seed{seed}_folds.csv",
    }


def _verify_bundle(dataset: str, seed: int) -> dict[str, Any]:
    paths = _bundle_paths(dataset, seed)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical contract file: {path}")
    manifest = _read_json(paths["manifest"])
    try:
        bundle = torch.load(paths["bundle"], map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(paths["bundle"], map_location="cpu")
    if set(bundle) != {"features", "edge_index", "node_id", "provenance"}:
        raise RuntimeError("canonical training bundle is not label-free")
    if bundle["provenance"].get("contains_y_or_labels") is not False:
        raise RuntimeError("canonical bundle lacks its label-isolation attestation")
    observed = {
        "num_nodes": int(bundle["features"].shape[0]),
        "num_features": int(bundle["features"].shape[1]),
        "num_edges": int(bundle["edge_index"].shape[1]),
    }
    expected = {name: int(manifest[name]) for name in observed}
    if observed != expected:
        raise RuntimeError(f"canonical bundle identity mismatch: {observed} != {expected}")
    provenance = bundle["provenance"]
    for name in ("raw_sha256", "node_order_sha256", "edge_order_sha256", "feature_sha256"):
        if provenance.get(name) != manifest.get(name):
            raise RuntimeError(f"canonical provenance mismatch: {name}")
    return {"dataset": dataset, **observed, "edge_count_convention": "directed_edge_index_entries"}


def _gpu_rows() -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in query.stdout.splitlines():
        index, name, total, used, utilization = [part.strip() for part in line.split(",", 4)]
        rows.append(
            {
                "index": int(index), "name": name, "memory_total_mib": int(total),
                "memory_used_mib": int(used), "utilization_percent": int(utilization),
            }
        )
    return rows


def _device(value: str, supports_gpu: bool) -> tuple[str, dict[str, Any] | None]:
    if value == "auto":
        if not supports_gpu:
            return "cpu", None
        candidates = [row for row in _gpu_rows() if row["utilization_percent"] <= 5 and row["memory_used_mib"] <= 1024]
        if not candidates:
            raise RuntimeError("WAITING_FREE_GPU:no GPU meets utilization<=5% and used_memory<=1024MiB")
        selected = min(candidates, key=lambda row: (row["memory_used_mib"], row["utilization_percent"], row["index"]))
        return f"cuda:{selected['index']}", selected
    if value == "cpu":
        return value, None
    match = re.fullmatch(r"cuda:(\d+)", value)
    if not match or not supports_gpu:
        raise ValueError(f"unsupported device for {value=}")
    rows = {row["index"]: row for row in _gpu_rows()}
    index = int(match.group(1))
    if index not in rows:
        raise ValueError(f"GPU index does not exist: {index}")
    return value, rows[index]


def _render_command(row: dict[str, Any], values: dict[str, str]) -> list[str]:
    template = row.get("canonical_command")
    if not isinstance(template, list) or not all(isinstance(item, str) for item in template):
        raise RuntimeError(f"{row['method']}: canonical command is not implemented")
    command = [item.format(**values) for item in template]
    forbidden = {"--y", "--label", "--labels", "--target"}
    if row["protocol_class"] != "EXTERNAL_SUPERVISED_REFERENCE" and forbidden & set(command):
        raise RuntimeError(f"{row['method']}: normal-only command exposes a label argument")
    return command


def _load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    nodes: list[int] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["node_id", "anomaly_score"]:
            raise RuntimeError("native score CSV violates the exact field contract")
        for row in reader:
            nodes.append(int(row["node_id"]))
            scores.append(float(row["anomaly_score"]))
    node_id = np.asarray(nodes, dtype=np.int64)
    anomaly_score = np.asarray(scores, dtype=np.float64)
    if node_id.shape != anomaly_score.shape or np.unique(node_id).size != node_id.size:
        raise RuntimeError("score vectors are not aligned/node-unique")
    if not np.isfinite(anomaly_score).all():
        raise RuntimeError("score vector contains non-finite values")
    if node_id.size and not np.array_equal(node_id, np.sort(node_id)):
        raise RuntimeError("score nodes are not in canonical sorted order")
    return node_id, anomaly_score


def _resource_snapshot(device: str, gpu: dict[str, Any] | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "timestamp": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "device": device,
        "gpu": gpu,
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        snapshot["cpu_ram_total_bytes"] = int(memory.total)
        snapshot["cpu_ram_available_bytes"] = int(memory.available)
    except ImportError:
        snapshot["cpu_ram_total_bytes"] = None
        snapshot["cpu_ram_available_bytes"] = None
    return snapshot


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dry_run(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    failures = _gate_failures(row)
    return {
        "method": row["method"],
        "method_id": row["method_id"],
        "dataset": args.dataset,
        "seed": args.seed,
        "device": args.device,
        "environment_path": row["environment_path"],
        "python_executable": row["python_executable"],
        "native_source": row["native_source"],
        "native_entrypoint": row["native_entrypoint"],
        "canonical_entrypoint": row.get("canonical_entrypoint"),
        "execute_enabled": bool(row["execute_enabled"]),
        "failed_or_pending_gates": failures,
        "blocker": row.get("blocker"),
        "output_directory": str(RELEASE / "outputs" / "baselines" / row["output_name"] / args.dataset / f"seed_{args.seed}"),
    }


def _execute(row: dict[str, Any], args: argparse.Namespace) -> int:
    failures = _gate_failures(row)
    if failures or not row.get("execute_enabled"):
        print(
            json.dumps(
                {"status": "EXECUTION_BLOCKED", "method": row["method"], "failed_or_pending_gates": failures, "blocker": row.get("blocker")},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 4
    source_identity = _verify_source(row)
    environment = _verify_environment(row)
    bundle_identity = _verify_bundle(args.dataset, args.seed)
    device, launch_gpu = _device(args.device, bool(row["supports_gpu"]))
    paths = _bundle_paths(args.dataset, args.seed)
    canonical_manifest = _read_json(paths["manifest"])
    label_source = Path(canonical_manifest.get("raw_source_path", ""))
    if row["protocol_class"] == "EXTERNAL_SUPERVISED_REFERENCE" and not label_source.is_file():
        raise FileNotFoundError(f"missing supervised reference label source: {label_source}")
    output = RELEASE / "outputs" / "baselines" / row["output_name"] / args.dataset / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing formal run: {output}")
    output.mkdir(parents=True)
    native_scores = output / "native_scores.csv"
    native_evidence = output / "native_evidence.json"
    values = {
        "python": row["python_executable"],
        "repo_root": str(ROOT),
        "release_root": str(RELEASE),
        "source": str(RELEASE / row["native_source"]),
        "bundle": str(paths["bundle"]),
        "support": str(paths["support"]),
        "score_nodes": str(paths["score_nodes"]),
        "scores": str(native_scores),
        "evidence": str(native_evidence),
        "seed": str(args.seed),
        "device": device,
        "label_source": str(label_source) if row["protocol_class"] == "EXTERNAL_SUPERVISED_REFERENCE" else "",
    }
    command = _render_command(row, values)
    started = time.time()
    started_at = _utc_now()
    resource = {"launch": _resource_snapshot(device, launch_gpu), "command": command}
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    process_environment = os.environ.copy()
    process_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
            env=process_environment,
        )
    elapsed = time.time() - started
    resource["completion"] = _resource_snapshot(device, next((x for x in _gpu_rows() if f"cuda:{x['index']}" == device), None) if device.startswith("cuda:") else None)
    resource["elapsed_seconds"] = elapsed
    resource["returncode"] = process.returncode
    _write_json(output / "resource.json", resource)
    if process.returncode != 0:
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        joined = "\n".join(stderr_tail)
        status = "OOM" if re.search(r"out of memory|CUDA error: out of memory", joined, re.I) else "ERROR"
        _write_json(output / "metrics.json", {"resource_status": status, "metrics": None})
        _write_json(
            output / "resource_failure.json",
            {
                "resource_status": status, "method": row["method"], "dataset": args.dataset,
                "seed": args.seed, "source_commit": row["source_commit"], "adapter_commit": _git(ROOT, "rev-parse", "HEAD").stdout.strip(),
                "environment": row["environment_path"], "exact_command": command, "gpu": launch_gpu,
                "exception_type": "SUBPROCESS_NONZERO_EXIT", "last_log_lines": stderr_tail,
                "timestamp": _utc_now(), "elapsed_seconds": elapsed,
            },
        )
        _write_json(
            output / "run_manifest.json",
            {"status": status, "started_at": started_at, "completed_at": _utc_now(), **source_identity, **environment, **bundle_identity, "command": command},
        )
        return process.returncode
    node_id, anomaly_score = _load_scores(native_scores)
    np.savez_compressed(
        output / "score.npz",
        node_id=node_id,
        anomaly_score=anomaly_score,
        higher_is_more_anomalous=np.asarray(True),
    )
    evaluator = ROOT / "benchmark" / "evaluate_baseline_scores.py"
    evaluator_python = "/data1/wk/conda_envs/pfr_gad/bin/python" if Path("/data1/wk/conda_envs/pfr_gad/bin/python").is_file() else sys.executable
    evaluation_command = [
        evaluator_python, str(evaluator), "--dataset", args.dataset,
        "--scores", str(output / "score.npz"), "--output", str(output / "metrics.json"),
    ]
    evaluation = subprocess.run(evaluation_command, cwd=ROOT, capture_output=True, text=True, check=False)
    if evaluation.returncode != 0:
        with stderr_path.open("a", encoding="utf-8") as stderr:
            stderr.write("\n=== independent evaluator ===\n" + evaluation.stderr)
        raise RuntimeError(f"independent evaluator failed: {' '.join(evaluation_command)}")
    _write_json(
        output / "run_manifest.json",
        {
            "status": "PASS", "resource_status": "PASS", "method": row["method"], "method_id": row["method_id"],
            "protocol_class": row["protocol_class"], "dataset": args.dataset, "seed": args.seed,
            "started_at": started_at, "completed_at": _utc_now(), "elapsed_seconds": elapsed,
            **source_identity, **environment, **bundle_identity, "adapter_commit": _git(ROOT, "rev-parse", "HEAD").stdout.strip(),
            "native_entrypoint": row["native_entrypoint"], "canonical_entrypoint": row["canonical_entrypoint"],
            "command": command, "evaluation_command": evaluation_command, "device": device,
            "score_contract": {"fields": ["node_id", "anomaly_score"], "count": int(node_id.size), "finite": True, "higher_is_more_anomalous": True},
        },
    )
    print(output)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="CoReGAD direct baseline reproduction launcher")
    result.add_argument("--method")
    result.add_argument("--dataset", choices=DATASETS)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="auto")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--list-methods", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.list_methods:
        for row in _entries():
            print(f"{row['method']}\t{row['protocol_class']}\t{str(row['execute_enabled']).lower()}")
        return 0
    if not args.method or not args.dataset:
        raise SystemExit("--method and --dataset are required")
    if args.seed != 0:
        raise SystemExit("this closure launcher is frozen to seed 0")
    row = _resolve_method(args.method)
    if not args.execute:
        print(json.dumps(_dry_run(row, args), indent=2, sort_keys=True))
        return 0
    return _execute(row, args)


if __name__ == "__main__":
    raise SystemExit(main())
