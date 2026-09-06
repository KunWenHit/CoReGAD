"""Direct canonical trainer for the author-official DSGAD implementation."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from benchmark.native_bridges.common import git_head, load_score_nodes, seed_everything, sha256, write_evidence, write_scores
from benchmark.native_bridges.supervised_common import load_supervised_reference


def load_model_class(source: Path):
    sys.path.insert(0, str(source))
    try:
        return importlib.import_module("models.DSGAD").DSGAD
    finally:
        sys.path.remove(str(source))


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    contract = load_supervised_reference(args.bundle, args.label_source, trial=args.trial)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=contract["payload"]["x"].shape[0])
    source = Path(args.source).resolve()
    device = torch.device(args.device)
    graph = contract["graph"].to(device)
    features = graph.ndata["feature"]
    labels = graph.ndata["label"].long()
    train_mask = graph.ndata["train_mask"].bool()
    validation_mask = graph.ndata["val_mask"].bool()
    DSGAD = load_model_class(source)
    model = DSGAD(
        features.shape[0],
        features.shape[1],
        h_feats=args.hidden_dimension,
        num_classes=2,
        d=args.order,
        mix_beta=args.mixed_filters,
    ).to(device)
    parameter_groups = []
    for name, parameter in model.named_parameters():
        parameter_groups.append({"params": parameter, "lr": args.mixture_lr if name == "weights" else args.lr})
    optimizer = torch.optim.Adam(parameter_groups)
    anomaly_weight = float((labels[train_mask] == 0).sum() / (labels[train_mask] == 1).sum())
    class_weight = torch.tensor([1.0, anomaly_weight], device=device)
    losses: list[float] = []
    started = time.time()
    probability = None
    for epoch in range(args.epochs):
        model.train()
        logits = model(graph, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = model(graph, features).softmax(1)[:, 1]
        losses.append(float(loss.detach().cpu()))
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 10 == 0:
            validation_mean = float(probability[validation_mask].mean().detach().cpu())
            print(f"epoch={epoch + 1} loss={losses[-1]:.10f} validation_probability_mean={validation_mean:.10f}", flush=True)
    if probability is None:
        raise RuntimeError("DSGAD did not produce anomaly probabilities")
    scores = probability.detach().cpu().numpy()
    write_scores(args.scores, score_nodes, scores[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method":"DSGAD","source_kind":"AUTHOR_OFFICIAL",
            "official_commit":"bf7e0ac31a4d28c82c796b339d06b4a1a5308158","checkout_head":git_head(source),
            "dataset_bundle_sha256":sha256(args.bundle),"label_source":contract["label_source"],"fair_ranking":False,
            "supervision":{"trial":args.trial,"train_nodes":int(train_mask.sum()),"validation_nodes":int(validation_mask.sum()),"class_weight":anomaly_weight},
            "native_contract":{"model":"models.DSGAD.DSGAD","hidden_dimension":args.hidden_dimension,"polynomial_order":args.order,"mixed_filters":args.mixed_filters,"mixture_lr":args.mixture_lr,"other_lr":args.lr},
            "epochs":args.epochs,"real_optimizer_steps":args.epochs,"first_loss":losses[0],"last_loss":losses[-1],
            "score_count":int(score_nodes.size),"finite_scores":bool(np.isfinite(scores).all()),"higher_is_more_anomalous":True,
            "seed":args.seed,"device":str(device),"elapsed_seconds":time.time()-started,
        },
    )


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
    result.add_argument("--lr", type=float, default=0.001)
    result.add_argument("--mixture-lr", type=float, default=0.1)
    result.add_argument("--hidden-dimension", type=int, default=64)
    result.add_argument("--order", type=int, default=2)
    result.add_argument("--mixed-filters", type=int, default=2)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
