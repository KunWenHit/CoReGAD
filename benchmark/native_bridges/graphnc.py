"""Label-isolated GraphNC bridge with an audited GGAD teacher stage."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    git_head,
    load_label_free_bundle,
    load_normal_support,
    load_score_nodes,
    seed_everything,
    sha256,
    write_evidence,
    write_scores,
)


GRAPHNC_COMMIT = "4985138639d96c43b85001eefa3a54eee1caebf1"
GGAD_COMMIT = "358fb4d4b4ee5b8b195445791a9e2b6d52487f2b"
STUDENT_EPOCHS = {"Amazon": 1300, "T-Finance": 2500, "Elliptic": 2000, "Tolokers": 1500, "YelpChi": 1500}
TEACHER_EPOCHS = {"Amazon": 800, "T-Finance": 500, "Elliptic": 150}


def _git(source: Path, *args: str) -> subprocess.CompletedProcess[str]:
    executable = "/data1/anaconda3/bin/git" if Path("/data1/anaconda3/bin/git").is_file() else "git"
    return subprocess.run([executable, "-C", str(source), *args], check=False, capture_output=True, text=True)


def _assert_ancestor(source: Path, commit: str, method: str) -> None:
    if _git(source, "merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise RuntimeError(f"{method} checkout does not contain its frozen source commit")


def _import_models(graphnc_source: Path, ggad_source: Path):
    _assert_ancestor(graphnc_source, GRAPHNC_COMMIT, "GraphNC")
    _assert_ancestor(ggad_source, GGAD_COMMIT, "GGAD teacher")
    sys.path.insert(0, str(graphnc_source))
    try:
        from model import Model_ocgnn  # type: ignore
    finally:
        sys.path.pop(0)
        sys.modules.pop("model", None)
    sys.path.insert(0, str(ggad_source))
    try:
        from model import Model as GGADModel  # type: ignore
        from utils import normalize_adj  # type: ignore
    finally:
        sys.path.pop(0)
        sys.modules.pop("model", None)
        sys.modules.pop("utils", None)
    return Model_ocgnn, GGADModel, normalize_adj


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device.index or 0)
    return device


def _adjacency(edge_index: torch.Tensor, num_nodes: int, normalize_adj) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = edge_index.numpy()
    adjacency = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, columns)), shape=(num_nodes, num_nodes))
    adjacency.data[:] = 1.0
    normalized = (normalize_adj(adjacency) + sp.eye(num_nodes, dtype=np.float32, format="coo")).toarray().astype(np.float32, copy=False)
    raw = (adjacency + sp.eye(num_nodes, dtype=np.float32, format="csr")).toarray().astype(np.float32, copy=False)
    return torch.from_numpy(normalized).unsqueeze(0), torch.from_numpy(raw)


def _minmax(value: torch.Tensor) -> torch.Tensor:
    minimum, maximum = value.min(), value.max()
    return torch.zeros_like(value) if bool(maximum == minimum) else (value - minimum) / (maximum - minimum)


def _train_teacher(
    model,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    raw_adjacency: torch.Tensor,
    normal_index: torch.Tensor,
    pseudo_index: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[list[float], torch.Tensor, torch.Tensor]:
    native_args = SimpleNamespace(mean=0.0, var=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([1.0], device=device))
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embedding, _, logits, generated, sampled = model(features, adjacency, pseudo_index, normal_index, True, native_args)
        target = torch.cat([torch.zeros(normal_index.numel(), device=device), torch.ones(generated.shape[0], device=device)]).view(1, -1, 1)
        loss_bce = bce(logits, target).mean()
        embedding = embedding.squeeze(0)
        normalized_embedding = embedding / torch.linalg.vector_norm(embedding, dim=-1, keepdim=True).clamp_min(1e-12)
        affinity = (torch.mm(normalized_embedding, normalized_embedding.T) * raw_adjacency).sum(0) / raw_adjacency.sum(0).clamp_min(1.0)
        margin = (0.7 - (affinity[normal_index].mean() - affinity[pseudo_index].mean())).clamp_min(0)
        reconstruction = torch.sqrt(torch.sum((generated - sampled) ** 2, dim=1)).mean()
        loss = loss_bce + margin + reconstruction
        if not torch.isfinite(loss):
            raise FloatingPointError("GraphNC GGAD teacher produced a non-finite loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        _, _, teacher_logits, _, _ = model(features, adjacency, pseudo_index, normal_index, False, native_args)
        _, _, _, _, pseudo_embedding = model(features, adjacency, pseudo_index, normal_index, True, native_args)
    return losses, teacher_logits.squeeze(0).squeeze(-1), pseudo_embedding.squeeze(0)


def preflight(bundle_path: str | Path) -> dict[str, object]:
    bundle = load_label_free_bundle(bundle_path)
    n = int(bundle["x"].shape[0])
    dense_bytes = n * n * 4
    return {
        "method": "GraphNC",
        "num_nodes": n,
        "num_edges": int(bundle["edge_index"].shape[1]),
        "dense_n_by_n_path": True,
        "teacher_is_trained_in_process": True,
        "minimum_named_dense_bytes": 3 * dense_bytes,
        "preflight_status": "NOT_RUN_RESOURCE_RISK" if 3 * dense_bytes >= 12 * 1024**3 else "PREFLIGHT_WITHIN_SINGLE_GPU_CAPACITY",
        "edge_count_convention": "directed_edge_index_entries",
    }


def run(args: argparse.Namespace) -> int:
    started = time.time()
    seed_everything(args.seed)
    graphnc_source = Path(args.source).resolve()
    ggad_source = Path(args.ggad_source).resolve()
    ModelOCGNN, GGADModel, normalize_adj = _import_models(graphnc_source, ggad_source)
    bundle = load_label_free_bundle(args.bundle)
    dataset = str(bundle["provenance"].get("dataset"))
    teacher_epochs = args.teacher_epochs or TEACHER_EPOCHS.get(dataset, 300)
    student_epochs = args.student_epochs or STUDENT_EPOCHS.get(dataset, 1000)
    lr = args.lr if args.lr > 0 else ({"Amazon": 5e-4, "T-Finance": 5e-4, "Elliptic": 1e-3, "Tolokers": 5e-3, "YelpChi": 5e-4}.get(dataset, 5e-3))
    support = load_normal_support(args.support, num_nodes=bundle["x"].shape[0])
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=bundle["x"].shape[0])
    normal_nodes = support.tolist()
    random.shuffle(normal_nodes)
    pseudo_count = max(1, int(len(normal_nodes) * (0.05 if dataset == "Amazon" else 0.15)))
    pseudo_nodes = normal_nodes[:pseudo_count]
    device = _device(args.device)
    adjacency_cpu, raw_cpu = _adjacency(bundle["edge_index"], bundle["x"].shape[0], normalize_adj)
    previous_default = torch.get_default_device()
    torch.set_default_device(device)
    try:
        features = bundle["x"].unsqueeze(0).to(device)
        adjacency = adjacency_cpu.to(device)
        raw_adjacency = raw_cpu.to(device)
        normal_index = torch.as_tensor(normal_nodes, dtype=torch.long, device=device)
        pseudo_index = torch.as_tensor(pseudo_nodes, dtype=torch.long, device=device)
        teacher = GGADModel(features.shape[-1], args.embedding_dim, "prelu", 1, "avg").to(device)
        teacher_losses, teacher_score_raw, pseudo_embedding = _train_teacher(
            teacher, features, adjacency, raw_adjacency, normal_index, pseudo_index, teacher_epochs, 1e-3, device
        )
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher_score = _minmax(teacher_score_raw).detach()
        teacher_concat = torch.cat([teacher_score, teacher_score[pseudo_index]], dim=0)
        pseudo_embedding = pseudo_embedding.detach()
        student = ModelOCGNN(features.shape[-1], args.embedding_dim, "prelu", 1, "avg").to(device)
        score_head = nn.Linear(args.embedding_dim, 1).to(device)
        pseudo_projection = nn.Linear(args.embedding_dim, args.embedding_dim).to(device)
        optimizers = [
            torch.optim.Adam(student.parameters(), lr=lr),
            torch.optim.Adam(score_head.parameters(), lr=lr),
            torch.optim.Adam(pseudo_projection.parameters(), lr=lr),
        ]
        student_losses: list[float] = []
        for _ in range(student_epochs):
            student.train()
            score_head.train()
            pseudo_projection.train()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            _, student_embedding_raw = student(features, adjacency)
            student_embedding = student_embedding_raw.squeeze(0)
            projected_pseudo = pseudo_projection(pseudo_embedding)
            concatenated = torch.cat([student_embedding, projected_pseudo], dim=0)
            student_concat = _minmax(score_head(concatenated).squeeze(-1))
            normal_features = features.squeeze(0)[normal_index]
            masked_normal = normal_features * (torch.rand_like(normal_features) > 0.3)
            masked_features = features.squeeze(0).clone()
            masked_features[normal_index] = masked_normal
            _, augmented_raw = student(masked_features.unsqueeze(0), adjacency)
            augmented = augmented_raw.squeeze(0)
            regularization = F.mse_loss(augmented[normal_index], student_embedding[normal_index], reduction="mean")
            score_alignment = F.mse_loss(student_concat, teacher_concat, reduction="mean")
            loss = score_alignment + 0.01 * regularization
            if not torch.isfinite(loss):
                raise FloatingPointError("GraphNC student produced a non-finite loss")
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            student_losses.append(float(loss.detach().cpu()))
        student.eval()
        score_head.eval()
        with torch.no_grad():
            _, final_embedding = student(features, adjacency)
            all_scores = torch.sigmoid(score_head(final_embedding.squeeze(0)).squeeze(-1))
            scores = all_scores[torch.as_tensor(score_nodes, dtype=torch.long, device=device)].cpu().numpy()
    finally:
        torch.set_default_device(previous_default)
    score_path = write_scores(args.scores, score_nodes, scores)
    evidence = {
        "method": "GraphNC",
        "source_kind": "AUTHOR_OFFICIAL",
        "source_url": "https://github.com/mala-lab/GraphNC.git",
        "source_commit": GRAPHNC_COMMIT,
        "checkout_head": git_head(graphnc_source),
        "native_student_import": str(graphnc_source / "model.py") + ":Model_ocgnn",
        "teacher_source_url": "https://github.com/mala-lab/GGAD.git",
        "teacher_source_commit": GGAD_COMMIT,
        "teacher_checkout_head": git_head(ggad_source),
        "teacher_training": "fresh in-process GGAD teacher on the same label-free canonical bundle and frozen normal support",
        "bundled_pretrained_teacher_used": False,
        "bundled_pretrained_teacher_exclusion_reason": "checkpoint support IDs and label-based selection provenance cannot be verified against the frozen canonical protocol",
        "teacher_input_contains_y": False,
        "teacher_early_stopping_or_metric_selection": False,
        "student_early_stopping_or_metric_selection": False,
        "test_metrics_or_labels_called_during_training": False,
        "ground_truth_anomaly_label_present_or_accessed": False,
        "coReGAD_cross_fitting_injected": False,
        "bundle": str(bundle["bundle_path"]),
        "bundle_sha256": sha256(bundle["bundle_path"]),
        "bundle_keys": ["edge_index", "features", "node_id", "provenance"],
        "normal_support_manifest": str(Path(args.support).resolve()),
        "normal_support_sha256": sha256(args.support),
        "normal_support_count": len(normal_nodes),
        "pseudo_anomaly_count": len(pseudo_nodes),
        "dataset": dataset,
        "seed": args.seed,
        "teacher_epochs": teacher_epochs,
        "student_epochs": student_epochs,
        "schedule_source": "upstream dataset default" if dataset in STUDENT_EPOCHS else "fixed upstream Reddit defaults",
        "teacher_loss_trace": teacher_losses,
        "student_loss_trace": student_losses,
        "hyperparameters": {"embedding_dim": args.embedding_dim, "student_lr": lr, "teacher_lr": 1e-3, "normal_feature_mask_probability": 0.3, "normality_regularization_weight": 0.01},
        "device": str(device),
        "score_file": str(score_path.resolve()),
        "score_count": int(score_nodes.size),
        "score_direction": "higher_is_more_anomalous",
        "scores_finite": bool(np.isfinite(scores).all()),
        "elapsed_seconds": time.time() - started,
        "preflight": preflight(args.bundle),
    }
    write_evidence(args.evidence, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="GraphNC canonical native bridge")
    result.add_argument("--bundle", required=True)
    result.add_argument("--support", required=True)
    result.add_argument("--score-nodes", required=True)
    result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--ggad-source", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--teacher-epochs", type=int, default=0)
    result.add_argument("--student-epochs", type=int, default=0)
    result.add_argument("--embedding-dim", type=int, default=300)
    result.add_argument("--lr", type=float, default=0.0)
    result.add_argument("--preflight", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.bundle), indent=2, sort_keys=True))
        return 0
    if args.teacher_epochs < 0 or args.student_epochs < 0 or args.lr < 0:
        raise SystemExit("invalid schedule override")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
