"""Resource-status and OOM/OOT evidence contract for the 21×8 matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RESOURCE_STATUSES = {"NOT_RUN", "PASS", "OOM", "OOT", "UNSUPPORTED", "ERROR"}
FAILURE_STATUSES = {"OOM", "OOT"}
OOM_FIELDS = {
    "method", "dataset", "seed", "source_commit", "adapter_commit",
    "environment", "exact_command", "gpu_model", "gpu_total_memory",
    "last_log_lines", "exception_type", "cuda_oom_message",
    "peak_memory", "timestamp",
}
OOT_FIELDS = {
    "method", "dataset", "seed", "source_commit", "adapter_commit",
    "environment", "exact_command", "wall_clock_limit", "elapsed_time",
    "progress", "epoch_or_iteration", "hardware", "timestamp",
}


def validate_cell(cell: dict[str, Any], *, evidence_root: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    status = cell.get("resource_status")
    if status not in RESOURCE_STATUSES:
        errors.append(f"invalid resource_status:{status}")
    metric = cell.get("metrics")
    evidence = cell.get("resource_evidence")
    if status in FAILURE_STATUSES:
        if metric not in (None, {}):
            errors.append(f"{status} cell must have empty metrics")
        if not evidence:
            errors.append(f"{status} cell must point to resource evidence")
        elif evidence_root is not None:
            path = Path(evidence_root) / evidence
            if not path.is_file():
                errors.append(f"resource evidence does not exist:{path}")
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                required = OOM_FIELDS if status == "OOM" else OOT_FIELDS
                missing = required - payload.keys()
                if missing:
                    errors.append(f"{status} evidence missing:{sorted(missing)}")
    return errors


def matrix_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in registry["baselines"]:
        targets = method["benchmark_targets"]
        for dataset in targets:
            cell = method["resource_status_by_dataset"][dataset]
            if isinstance(cell, str):
                cell = {
                    "resource_status": cell,
                    "preflight_status": method.get("resource_preflight_by_dataset", {}).get(
                        dataset, "NOT_ASSESSED"
                    ),
                    "metrics": method.get("metrics_by_dataset", {}).get(dataset),
                    "resource_evidence": method.get("resource_evidence_by_dataset", {}).get(dataset),
                }
            rows.append(
                {
                    "method": method["method"],
                    "dataset": dataset,
                    "resource_status": cell["resource_status"],
                    "preflight_status": cell.get("preflight_status", "NOT_ASSESSED"),
                    "metrics": cell.get("metrics"),
                    "resource_evidence": cell.get("resource_evidence"),
                }
            )
    return rows
