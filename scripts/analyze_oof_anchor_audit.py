from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from scipy.stats import rankdata, spearmanr

from coregad.evaluation.metrics import ranking_metrics


DATASETS = (
    "Amazon",
    "Weibo",
    "YelpChi",
    "Tolokers",
    "T-Finance",
    "Elliptic",
    "T-Social",
    "DGraph-Fin",
)


def _root() -> Path:
    override = os.environ.get("COREGAD_RELEASE_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[2]


def _manifest(dataset: str) -> dict[str, Any]:
    return json.loads((_root() / "datasets/manifests" / f"{dataset}.json").read_text(encoding="utf-8"))


def _label_cache(dataset: str) -> Path:
    return _root() / "outputs/oof_anchor_audit/evaluation_labels" / f"{dataset}.npz"


def _prepare_labels(dataset: str) -> Path:
    """Evaluation-only label extraction. This action is never imported by training."""

    manifest = _manifest(dataset)
    source = Path(manifest["raw_source_path"])
    evaluation_mask: np.ndarray | None = None
    if dataset == "Amazon":
        labels = np.asarray(sio.loadmat(source, variable_names=["label"])["label"]).reshape(-1)
    elif dataset == "Elliptic":
        labels = np.asarray(sio.loadmat(source, variable_names=["Label"])["Label"]).reshape(-1)
    elif dataset == "Tolokers":
        with np.load(source, allow_pickle=False) as stored:
            labels = np.asarray(stored["node_labels"]).reshape(-1)
    else:
        import dgl

        graphs, _ = dgl.load_graphs(str(source))
        graph = graphs[0]
        labels = graph.ndata["label"].detach().cpu().numpy().reshape(-1)
        if dataset == "DGraph-Fin" and "mark" in graph.ndata:
            evaluation_mask = graph.ndata["mark"].detach().cpu().numpy().reshape(-1).astype(bool)
    if labels.ndim != 1 or labels.size != int(manifest["num_nodes"]):
        raise RuntimeError(f"{dataset}: label identity mismatch")
    if labels.dtype.kind == "f" and not np.isfinite(labels).all():
        raise RuntimeError(f"{dataset}: non-finite labels")
    labels = labels.astype(np.int64, copy=False)
    if evaluation_mask is not None:
        labels = labels.copy()
        labels[~evaluation_mask] = -1
    target = _label_cache(dataset)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite evaluation label cache: {target}")
    np.savez_compressed(
        target,
        node_id=np.arange(labels.size, dtype=np.int64),
        y=labels,
        evaluation_only=np.asarray(True),
    )
    return target


def _scores(version: str, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    path = _root() / "outputs/oof_anchor_audit" / f"{version}_seed0" / dataset / "seed_0/oof_scores.npz"
    with np.load(path, allow_pickle=False) as stored:
        return np.asarray(stored["node_id"], dtype=np.int64), np.asarray(stored["score"], dtype=np.float64)


def _diagnostics(version: str, dataset: str) -> dict[str, np.ndarray]:
    run = _root() / "outputs/oof_anchor_audit" / f"{version}_seed0" / dataset / "seed_0"
    rows: dict[str, list[np.ndarray]] = {}
    for fold in range(5):
        try:
            artifact = torch.load(run / f"fold_{fold}.pt", map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(run / f"fold_{fold}.pt", map_location="cpu")
        diagnostics = artifact["diagnostics"]
        if diagnostics["used_for_training_or_selection"] is not False:
            raise RuntimeError("diagnostics were not evaluation-only")
        for key, value in diagnostics.items():
            if key == "used_for_training_or_selection":
                continue
            rows.setdefault(key, []).append(np.asarray(value))
    combined = {key: np.concatenate(values, axis=0) for key, values in rows.items()}
    order = np.argsort(combined["node_id"], kind="mergesort")
    return {key: value[order] for key, value in combined.items()}


def _aligned_labels(dataset: str, node_id: np.ndarray) -> np.ndarray:
    with np.load(_label_cache(dataset), allow_pickle=False) as stored:
        all_nodes = np.asarray(stored["node_id"], dtype=np.int64)
        labels = np.asarray(stored["y"], dtype=np.int64)
        if not bool(np.asarray(stored["evaluation_only"]).item()):
            raise RuntimeError("label cache is not evaluation-only")
    if not np.array_equal(all_nodes, np.arange(labels.size, dtype=np.int64)):
        raise RuntimeError("label-cache node order mismatch")
    return labels[node_id]


def _metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return ranking_metrics(labels, scores)


def _rank_change(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_rank = rankdata(-left, method="average")
    right_rank = rankdata(-right, method="average")
    n = max(1, left.size)
    anomaly_k = max(1, int(np.ceil(0.01 * n)))
    left_top = set(np.argsort(-left, kind="mergesort")[:anomaly_k].tolist())
    right_top = set(np.argsort(-right, kind="mergesort")[:anomaly_k].tolist())
    return {
        "mean_absolute_rank_change": float(np.mean(np.abs(left_rank - right_rank))),
        "normalized_mean_absolute_rank_change": float(np.mean(np.abs(left_rank - right_rank)) / n),
        "top_1pct_jaccard": float(len(left_top & right_top) / max(1, len(left_top | right_top))),
    }


def _analyze_dataset(dataset: str) -> dict[str, Any]:
    legacy_nodes, legacy_final = _scores("legacy", dataset)
    strict_nodes, strict_final = _scores("strict", dataset)
    if not np.array_equal(legacy_nodes, strict_nodes):
        raise RuntimeError(f"{dataset}: legacy/strict OOF nodes differ")
    labels = _aligned_labels(dataset, strict_nodes)
    legacy_diag = _diagnostics("legacy", dataset)
    strict_diag = _diagnostics("strict", dataset)
    for diagnostics in (legacy_diag, strict_diag):
        if not np.array_equal(diagnostics["node_id"], strict_nodes):
            raise RuntimeError(f"{dataset}: diagnostic node order differs from OOF order")
    if not np.array_equal(legacy_diag["final_anomaly_score"], legacy_final):
        raise RuntimeError(f"{dataset}: legacy diagnostic final score mismatch")
    if not np.array_equal(strict_diag["final_anomaly_score"], strict_final):
        raise RuntimeError(f"{dataset}: strict diagnostic final score mismatch")

    legacy_pre = _metric(labels, legacy_diag["pre_context_teacher_base_anomaly_score"])
    legacy_post = _metric(labels, legacy_diag["post_context_frozen_normality_score"])
    legacy_metrics = _metric(labels, legacy_final)
    strict_pre = _metric(labels, strict_diag["pre_context_teacher_base_anomaly_score"])
    strict_post = _metric(labels, strict_diag["post_context_frozen_normality_score"])
    strict_metrics = _metric(labels, strict_final)
    pearson = float(np.corrcoef(legacy_final, strict_final)[0, 1])
    spearman = float(spearmanr(legacy_final, strict_final).statistic)
    legacy_correction = legacy_diag["graph_correction"].astype(np.float64)
    strict_correction = strict_diag["graph_correction"].astype(np.float64)
    legacy_s = np.linalg.vector_norm(legacy_diag["spectral_discrepancy"].astype(np.float64), axis=1)
    strict_s = np.linalg.vector_norm(strict_diag["spectral_discrepancy"].astype(np.float64), axis=1)
    legacy_scr = np.linalg.vector_norm(legacy_diag["controlled_spectral_residual"].astype(np.float64), axis=1)
    strict_scr = np.linalg.vector_norm(strict_diag["controlled_spectral_residual"].astype(np.float64), axis=1)
    return {
        "dataset": dataset,
        "legacy_auprc": legacy_metrics["auprc"],
        "strict_auprc": strict_metrics["auprc"],
        "delta_auprc": strict_metrics["auprc"] - legacy_metrics["auprc"],
        "legacy_auroc": legacy_metrics["auroc"],
        "strict_auroc": strict_metrics["auroc"],
        "delta_auroc": strict_metrics["auroc"] - legacy_metrics["auroc"],
        "legacy_pre_context_auprc": legacy_pre["auprc"],
        "legacy_post_context_auprc": legacy_post["auprc"],
        "legacy_context_delta_auprc": legacy_post["auprc"] - legacy_pre["auprc"],
        "legacy_final_stage_delta_auprc": legacy_metrics["auprc"] - legacy_post["auprc"],
        "strict_pre_context_auprc": strict_pre["auprc"],
        "strict_post_context_auprc": strict_post["auprc"],
        "strict_context_delta_auprc": strict_post["auprc"] - strict_pre["auprc"],
        "strict_final_stage_delta_auprc": strict_metrics["auprc"] - strict_post["auprc"],
        "strict_gain_over_attribute_base_auprc": strict_metrics["auprc"] - strict_pre["auprc"],
        "legacy_graph_correction_mean": float(legacy_correction.mean()),
        "legacy_graph_correction_std": float(legacy_correction.std()),
        "strict_graph_correction_mean": float(strict_correction.mean()),
        "strict_graph_correction_std": float(strict_correction.std()),
        "legacy_spectral_discrepancy_norm_mean": float(legacy_s.mean()),
        "strict_spectral_discrepancy_norm_mean": float(strict_s.mean()),
        "legacy_controlled_residual_norm_mean": float(legacy_scr.mean()),
        "strict_controlled_residual_norm_mean": float(strict_scr.mean()),
        "score_pearson": pearson,
        "score_spearman": spearman,
        **_rank_change(legacy_final, strict_final),
    }


def _write_results(rows: list[dict[str, Any]]) -> None:
    report_root = _root() / "reports/oof_anchor_audit"
    report_root.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    import csv

    with (report_root / "OOF_ANCHOR_SEED0_COMPARISON.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    delta = np.asarray([row["delta_auprc"] for row in rows], dtype=np.float64)
    gains = np.asarray([row["strict_gain_over_attribute_base_auprc"] for row in rows], dtype=np.float64)
    worst = rows[int(np.argmin(delta))]
    payload = {
        "datasets": rows,
        "summary": {
            "mean_delta_auprc": float(delta.mean()),
            "median_delta_auprc": float(np.median(delta)),
            "worst_dataset": worst["dataset"],
            "worst_delta_auprc": worst["delta_auprc"],
            "mean_strict_gain_over_attribute_base_auprc": float(gains.mean()),
            "median_strict_gain_over_attribute_base_auprc": float(np.median(gains)),
        },
        "labels_used_for_training_or_version_selection": False,
        "posthoc_evaluation_only": True,
    }
    (report_root / "OOF_ANCHOR_SEED0_COMPARISON.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Posthoc legacy-vs-strict seed0 anchor analysis.")
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare-labels")
    prepare.add_argument("--dataset", choices=DATASETS)
    prepare.add_argument("--all", action="store_true")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--dataset", choices=DATASETS)
    analyze.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.dataset) == bool(args.all):
        raise SystemExit("choose exactly one of --dataset or --all")
    targets = DATASETS if args.all else (args.dataset,)
    if args.action == "prepare-labels":
        for dataset in targets:
            print(_prepare_labels(dataset))
    else:
        _write_results([_analyze_dataset(dataset) for dataset in targets])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
