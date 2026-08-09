from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch


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
LARGE_DATASETS = {"T-Social", "DGraph-Fin"}
LEGACY_REFERENCE_HEAD = "72ab80a3f2c2dc3e167cbe3d5cd0b931b457f961"
MODEL_SEED = 0
OUTER_FOLDS = 5
NORMALITY_EPOCHS = 200
CONTEXT_EPOCHS = 200
RESIDUAL_EPOCHS = 300


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(source: Path, *args: str) -> str:
    git_binary = os.environ.get("COREGAD_GIT")
    if not git_binary:
        server_git = Path("/data1/anaconda3/bin/git")
        git_binary = str(server_git) if server_git.is_file() else shutil.which("git")
    if not git_binary:
        raise RuntimeError("git executable not found")
    result = subprocess.run(
        [git_binary, "-C", str(source), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _release_root() -> Path:
    override = os.environ.get("COREGAD_RELEASE_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[2]


def _paths() -> dict[str, Path]:
    root = _release_root()
    return {
        "root": root,
        "strict": root / "coregad",
        "legacy_reference": root / "worktrees/LEGACY_REFERENCE",
        "legacy_executable": root / "worktrees/LEGACY_EXECUTABLE",
        "manifests": root / "datasets/manifests",
        "splits": root / "datasets/splits",
        "supports": root / "datasets/supports",
        "outputs": root / "outputs/oof_anchor_audit",
    }


def _dataset_contract(dataset: str, *, verify_raw_hash: bool = False) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    paths = _paths()
    manifest_path = paths["manifests"] / f"{dataset}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_path = paths["root"] / manifest["split_manifest"]
    support_path = paths["root"] / manifest["support_manifest"]
    raw_path = Path(manifest["raw_source_path"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw source is not a regular file: {raw_path}")
    if _sha256(split_path) != manifest["split_sha256"]:
        raise RuntimeError(f"{dataset}: split manifest hash mismatch")
    if _sha256(support_path) != manifest["support_sha256"]:
        raise RuntimeError(f"{dataset}: support manifest hash mismatch")
    if verify_raw_hash and _sha256(raw_path) != manifest["raw_sha256"]:
        raise RuntimeError(f"{dataset}: raw source hash mismatch")

    support = json.loads(support_path.read_text(encoding="utf-8"))
    normal = np.asarray(support["train_normal_node_ids"], dtype=np.int64)
    node_ids: list[int] = []
    ownership: list[int] = []
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != dataset or int(row["seed"]) != MODEL_SEED:
                raise RuntimeError(f"{dataset}: split row identity mismatch")
            if int(row["label_used_for_fold"]) != 0:
                raise RuntimeError(f"{dataset}: fold ownership used a label")
            node_ids.append(int(row["node_id"]))
            ownership.append(int(row["fold"]))
    unlabeled = np.asarray(node_ids, dtype=np.int64)
    owner = np.asarray(ownership, dtype=np.int8)
    if np.unique(normal).size != normal.size or np.unique(unlabeled).size != unlabeled.size:
        raise RuntimeError(f"{dataset}: duplicate support or OOF node IDs")
    if np.intersect1d(normal, unlabeled).size:
        raise RuntimeError(f"{dataset}: support and OOF nodes overlap")
    if owner.shape != unlabeled.shape or np.any((owner < 0) | (owner >= OUTER_FOLDS)):
        raise RuntimeError(f"{dataset}: invalid five-fold ownership")
    if normal.size != int(manifest["normal_label_budget"]):
        raise RuntimeError(f"{dataset}: support budget mismatch")
    if unlabeled.size != int(manifest["evaluation_node_count"]):
        raise RuntimeError(f"{dataset}: evaluation-node count mismatch")
    if normal.size and (normal.min() < 0 or normal.max() >= int(manifest["num_nodes"])):
        raise RuntimeError(f"{dataset}: support node out of bounds")
    if unlabeled.size and (unlabeled.min() < 0 or unlabeled.max() >= int(manifest["num_nodes"])):
        raise RuntimeError(f"{dataset}: OOF node out of bounds")
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "split_path": split_path,
        "support_path": support_path,
        "raw_path": raw_path,
        "normal_nodes": normal,
        "unlabeled_nodes": unlabeled,
        "ownership": owner,
        "raw_hash_verified": bool(verify_raw_hash),
    }


def _source_contract() -> dict[str, Any]:
    paths = _paths()
    for key in ("strict", "legacy_reference", "legacy_executable"):
        if not (paths[key] / "pyproject.toml").is_file():
            raise FileNotFoundError(f"missing source tree: {paths[key]}")
    reference_head = _git(paths["legacy_reference"], "rev-parse", "HEAD")
    if reference_head != LEGACY_REFERENCE_HEAD:
        raise RuntimeError("legacy reference HEAD changed")
    if _git(paths["legacy_reference"], "status", "--porcelain=v1"):
        raise RuntimeError("legacy reference worktree is not clean")
    strict_head = _git(paths["strict"], "rev-parse", "HEAD")
    legacy_executable_head = _git(paths["legacy_executable"], "rev-parse", "HEAD")
    changed_model_paths = _git(
        paths["strict"],
        "diff",
        "--name-only",
        legacy_executable_head,
        strict_head,
        "--",
        "coregad",
    ).splitlines()
    if changed_model_paths != ["coregad/training/normality.py"]:
        raise RuntimeError(
            "executable legacy/strict model trees differ outside the anchor file: "
            + repr(changed_model_paths)
        )
    strict_source = (paths["strict"] / "coregad/training/normality.py").read_text(encoding="utf-8")
    legacy_source = (paths["legacy_executable"] / "coregad/training/normality.py").read_text(encoding="utf-8")
    if 'output["normality_logit"][visible_index]' not in strict_source:
        raise RuntimeError("strict source does not contain visible-only anchor")
    if 'output["normality_logit"], teacher_output["normality_logit"]' not in legacy_source.replace("\n", " "):
        raise RuntimeError("legacy executable does not contain all-node anchor")
    return {
        "legacy_reference_head": reference_head,
        "legacy_executable_head": legacy_executable_head,
        "strict_head": strict_head,
        "only_model_source_difference": "coregad/training/normality.py",
        "only_behavioral_difference": "anchor_scope_all_nodes_vs_visible_index",
    }


def _dense_features(value: Any) -> torch.Tensor:
    if sp.issparse(value):
        value = value.toarray()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32).contiguous()


def _without_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    edge_index = edge_index.long().contiguous()
    return edge_index[:, edge_index[0] != edge_index[1]].contiguous()


def _undirected_edges(edges: np.ndarray, num_nodes: int) -> torch.Tensor:
    edge_index = _without_self_loops(torch.as_tensor(np.asarray(edges).T, dtype=torch.long))
    both = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    flat = torch.unique(both[0] * int(num_nodes) + both[1])
    return torch.stack([flat // int(num_nodes), flat % int(num_nodes)], dim=0).long()


def _load_training_graph(dataset: str, contract: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load features and edges without requesting or indexing a label tensor."""

    source = contract["raw_path"]
    label_tensor_accessed = False
    if dataset == "Amazon":
        payload = sio.loadmat(source, variable_names=["features", "homo"])
        features = _dense_features(payload["features"])
        adjacency = sp.coo_matrix(payload["homo"])
        keep = adjacency.row != adjacency.col
        edge_index = torch.from_numpy(
            np.stack([adjacency.row[keep], adjacency.col[keep]], axis=0).astype(np.int64, copy=False)
        ).long()
        loader = "scipy.io.loadmat(variable_names=features,homo)"
    elif dataset == "Elliptic":
        payload = sio.loadmat(source, variable_names=["Attributes", "Network"])
        features = _dense_features(payload["Attributes"])
        adjacency = sp.coo_matrix(payload["Network"])
        keep = adjacency.row != adjacency.col
        edge_index = torch.from_numpy(
            np.stack([adjacency.row[keep], adjacency.col[keep]], axis=0).astype(np.int64, copy=False)
        ).long()
        loader = "scipy.io.loadmat(variable_names=Attributes,Network)"
    elif dataset == "Tolokers":
        with np.load(source, allow_pickle=False) as payload:
            features = _dense_features(payload["node_features"])
            edge_index = _undirected_edges(payload["edges"], int(features.shape[0]))
        loader = "numpy.load(node_features,edges)"
    else:
        import dgl

        graphs, _ = dgl.load_graphs(str(source))
        if not graphs:
            raise RuntimeError(f"{dataset}: no DGL graph in source")
        graph = graphs[0]
        features = graph.ndata["feature"].detach().cpu().float().contiguous()
        source_nodes, target_nodes = graph.edges(order="eid")
        edge_index = _without_self_loops(torch.stack([source_nodes, target_nodes], dim=0).cpu())
        loader = "dgl.load_graphs(feature,edges_only_in_training_code)"
    manifest = contract["manifest"]
    observed = {
        "num_nodes": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "num_edges": int(edge_index.shape[1]),
    }
    expected = {key: int(manifest[key]) for key in observed}
    if observed != expected:
        raise RuntimeError(f"{dataset}: canonical graph identity mismatch: {observed} != {expected}")
    return features, edge_index, {
        "loader": loader,
        "label_tensor_accessed": label_tensor_accessed,
        **observed,
    }


def _activate_source(source: Path) -> Path:
    source = source.resolve()
    sys.path.insert(0, str(source))
    import coregad.training.normality as normality

    imported = Path(normality.__file__).resolve()
    expected = source / "coregad/training/normality.py"
    if imported != expected:
        raise RuntimeError(f"source import pollution: imported {imported}, expected {expected}")
    return imported


def _train_version(version: str, dataset: str, device: str) -> None:
    paths = _paths()
    source = paths["strict"] if version == "strict" else paths["legacy_executable"]
    source_state = _source_contract()
    contract = _dataset_contract(dataset, verify_raw_hash=True)
    imported = _activate_source(source)
    from coregad.data.oof import assemble_oof_scores
    from coregad.training.pipeline import train_fold

    features, edge_index, loader_audit = _load_training_graph(dataset, contract)
    output = paths["outputs"] / f"{version}_seed0" / dataset / "seed_0"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing audit output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    engine = "scalable" if dataset in LARGE_DATASETS else "standard"
    fold_rows: list[tuple[np.ndarray, np.ndarray]] = []
    owners: list[np.ndarray] = []
    for fold in range(OUTER_FOLDS):
        heldout = contract["unlabeled_nodes"][contract["ownership"] == fold]
        training_unlabeled = contract["unlabeled_nodes"][contract["ownership"] != fold]
        result = train_fold(
            features=features,
            edge_index=edge_index,
            normal_nodes=contract["normal_nodes"],
            training_unlabeled_nodes=training_unlabeled,
            heldout_nodes=heldout,
            fold=fold,
            model_seed=MODEL_SEED,
            device=device,
            normality_epochs=NORMALITY_EPOCHS,
            context_epochs=CONTEXT_EPOCHS,
            residual_epochs=RESIDUAL_EPOCHS,
            spectral_engine=engine,
            scalable_cache_dir=output / "scalable_cache" if engine == "scalable" else None,
        )
        torch.save(result.artifacts, output / f"fold_{fold}.pt")
        fold_rows.append((result.heldout_nodes, result.final_anomaly_score))
        owners.append(np.full(result.heldout_nodes.size, fold, dtype=np.int8))
    assembled = assemble_oof_scores(int(features.shape[0]), fold_rows)
    nodes = np.concatenate([row[0] for row in fold_rows])
    scores = np.concatenate([row[1] for row in fold_rows])
    owner = np.concatenate(owners)
    order = np.argsort(nodes, kind="mergesort")
    np.savez_compressed(output / "oof_scores.npz", node_id=nodes[order], score=scores[order], owner_fold=owner[order])
    if not np.isfinite(assembled[nodes]).all():
        raise RuntimeError("non-finite OOF scores")
    run_manifest = {
        "dataset": dataset,
        "version": version,
        "anchor_scope": "visible_index" if version == "strict" else "all_nodes",
        "model_seed": MODEL_SEED,
        "outer_folds": OUTER_FOLDS,
        "normality_epochs": NORMALITY_EPOCHS,
        "context_epochs": CONTEXT_EPOCHS,
        "residual_epochs": RESIDUAL_EPOCHS,
        "engine": engine.upper(),
        "device": device,
        "source_root": str(source),
        "source_head": _git(source, "rev-parse", "HEAD"),
        "imported_normality": str(imported),
        "legacy_reference_head": source_state["legacy_reference_head"],
        "raw_sha256": contract["manifest"]["raw_sha256"],
        "node_order_sha256": contract["manifest"]["node_order_sha256"],
        "split_sha256": contract["manifest"]["split_sha256"],
        "support_sha256": contract["manifest"]["support_sha256"],
        "label_tensor_accessed_during_training": False,
        "loader_audit": loader_audit,
        "diagnostics_used_for_training_or_selection": False,
        "formal_historical_outputs_overwritten": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")


def _pair_plan(dataset: str, device: str) -> dict[str, Any]:
    paths = _paths()
    source = _source_contract()
    data = _dataset_contract(dataset, verify_raw_hash=False)
    return {
        "dataset": dataset,
        "model_seed": MODEL_SEED,
        "device": device,
        "engine": "SCALABLE" if dataset in LARGE_DATASETS else "STANDARD",
        "epochs": {"base": NORMALITY_EPOCHS, "context": CONTEXT_EPOCHS, "residual": RESIDUAL_EPOCHS},
        "legacy_output": str(paths["outputs"] / "legacy_seed0" / dataset / "seed_0"),
        "strict_output": str(paths["outputs"] / "strict_seed0" / dataset / "seed_0"),
        "split_sha256": data["manifest"]["split_sha256"],
        "support_sha256": data["manifest"]["support_sha256"],
        "node_order_sha256": data["manifest"]["node_order_sha256"],
        "source_contract": source,
        "unique_causal_variable": "anchor_scope_all_nodes_vs_visible_index",
        "formal_training_executed_by_plan": False,
    }


def _run_pair(dataset: str, device: str, execute: bool) -> None:
    plan = _pair_plan(dataset, device)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not execute:
        print("PREPARATION_ONLY: add --execute for a future user-authorized seed0 pair run")
        return
    for version in ("legacy", "strict"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "train-version",
            "--version",
            version,
            "--dataset",
            dataset,
            "--device",
            device,
            "--execute-internal",
        ]
        subprocess.run(command, check=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run the seed0 strict-OOF anchor causal pair.")
    sub = parser.add_subparsers(dest="action", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--dataset", choices=DATASETS)
    preflight.add_argument("--all", action="store_true")
    preflight.add_argument("--verify-raw-hash", action="store_true")
    pair = sub.add_parser("run-pair")
    pair.add_argument("--dataset", required=True, choices=DATASETS)
    pair.add_argument("--device", required=True)
    pair.add_argument("--execute", action="store_true")
    internal = sub.add_parser("train-version", help=argparse.SUPPRESS)
    internal.add_argument("--version", required=True, choices=("legacy", "strict"))
    internal.add_argument("--dataset", required=True, choices=DATASETS)
    internal.add_argument("--device", required=True)
    internal.add_argument("--execute-internal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "preflight":
        if bool(args.dataset) == bool(args.all):
            raise SystemExit("choose exactly one of --dataset or --all")
        source = _source_contract()
        targets = DATASETS if args.all else (args.dataset,)
        result = {
            "source_contract": source,
            "datasets": {
                name: {
                    "valid": True,
                    "raw_hash_verified": _dataset_contract(name, verify_raw_hash=args.verify_raw_hash)["raw_hash_verified"],
                }
                for name in targets
            },
            "formal_training_executed": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.action == "run-pair":
        _run_pair(args.dataset, args.device, args.execute)
    elif args.action == "train-version":
        if not args.execute_internal:
            raise SystemExit("internal training action requires --execute-internal")
        _train_version(args.version, args.dataset, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
