"""Direct canonical trainer for the author-official BWGNN implementation."""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F

from benchmark.native_bridges.common import git_head, load_score_nodes, seed_everything, sha256, write_evidence, write_scores
from benchmark.native_bridges.supervised_common import load_supervised_reference


def load_model_class(source: Path):
    spec = importlib.util.spec_from_file_location("coregad_upstream_bwgnn", source / "BWGNN.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load official BWGNN.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BWGNN


def validation_f1(labels: torch.Tensor, probabilities: torch.Tensor) -> float:
    target = labels.detach().cpu().numpy()
    score = probabilities.detach().cpu().numpy()
    return max(f1_score(target, score > threshold, average="macro") for threshold in np.linspace(0.05, 0.95, 19))


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    contract = load_supervised_reference(args.bundle, args.label_source, trial=args.trial)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=contract["payload"]["x"].shape[0])
    source = Path(args.source).resolve()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_default_device(device)
    graph = contract["graph"].to(device)
    features = graph.ndata["feature"]
    labels = graph.ndata["label"].long()
    train_mask = graph.ndata["train_mask"].bool()
    val_mask = graph.ndata["val_mask"].bool()
    BWGNN = load_model_class(source)
    model = BWGNN(features.shape[1], args.hidden_dimension, 2, graph, d=args.order).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    anomaly_weight = float((labels[train_mask] == 0).sum() / (labels[train_mask] == 1).sum())
    class_weight = torch.tensor([1.0, anomaly_weight], device=device)
    best_f1 = -1.0
    selected_probability = None
    losses: list[float] = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = model(features).softmax(1)[:, 1]
        score = validation_f1(labels[val_mask], probability[val_mask])
        if score > best_f1:
            best_f1 = score
            selected_probability = probability.detach().clone()
        losses.append(float(loss.detach().cpu()))
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} loss={losses[-1]:.10f} val_macro_f1={score:.10f}", flush=True)
    if selected_probability is None:
        raise RuntimeError("BWGNN did not produce validation-selected probabilities")
    scores = selected_probability.detach().cpu().numpy()
    write_scores(args.scores, score_nodes, scores[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method":"BWGNN","source_kind":"AUTHOR_OFFICIAL",
            "official_commit":"de0631f039bbd19c1890b483cc01f1007f596af7","checkout_head":git_head(source),
            "dataset_bundle_sha256":sha256(args.bundle),"label_source":contract["label_source"],
            "supervision":{"trial":args.trial,"train_nodes":int(train_mask.sum()),"validation_nodes":int(val_mask.sum()),"checkpoint_metric":"validation macro-F1"},
            "fair_ranking":False,"epochs":args.epochs,"real_optimizer_steps":args.epochs,
            "first_loss":losses[0],"last_loss":losses[-1],"best_validation_macro_f1":best_f1,
            "score_count":int(score_nodes.size),"finite_scores":bool(np.isfinite(scores).all()),
            "higher_is_more_anomalous":True,"seed":args.seed,"device":str(device),"elapsed_seconds":time.time()-started,
        },
    )
    if device.type == "cuda":
        torch.set_default_device("cpu")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--bundle", required=True)
    result.add_argument("--label-source", required=True)
    result.add_argument("--score-nodes", required=True)
    result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cpu")
    result.add_argument("--trial", type=int, default=0)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--lr", type=float, default=0.01)
    result.add_argument("--hidden-dimension", type=int, default=64)
    result.add_argument("--order", type=int, default=2)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
