"""Direct canonical trainer for the author-official HSMAD implementation."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from benchmark.native_bridges.common import git_head, load_score_nodes, seed_everything, sha256, write_evidence, write_scores
from benchmark.native_bridges.supervised_common import load_supervised_reference


def load_model_class(source: Path):
    sys.path.insert(0, str(source))
    try:
        return importlib.import_module("model").CombinedModel
    finally:
        sys.path.remove(str(source))


def best_macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return max(f1_score(labels, probabilities > threshold, average="macro") for threshold in np.linspace(0.05, 0.95, 19))


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
    edge_labels = {}
    for edge_type in graph.etypes:
        source_nodes, destination_nodes = graph.edges(etype=edge_type)
        edge_labels[edge_type] = (labels[source_nodes] != labels[destination_nodes]).long()
    CombinedModel = load_model_class(source)
    model = CombinedModel(graph, features.shape[1], args.hidden_dimension, 2, d=args.order, quantile=args.quantile).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_f1 = -1.0
    best_epoch = 0
    selected_scores = None
    epochs_without_improvement = 0
    losses: list[float] = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        output, edge_predictions = model(features)
        loss = model.compute_loss(output, labels, edge_labels, edge_predictions, train_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (epoch + 1) % args.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                output, _ = model(features)
                probability = output.exp()[:, 1]
            validation_f1 = best_macro_f1(labels[validation_mask].cpu().numpy(), probability[validation_mask].cpu().numpy())
            if validation_f1 > best_f1:
                best_f1 = validation_f1
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                selected_scores = probability.detach().cpu().numpy().copy()
            else:
                epochs_without_improvement += args.validation_interval
            print(f"epoch={epoch + 1} loss={losses[-1]:.10f} best_validation_macro_f1={best_f1:.10f}", flush=True)
            if epochs_without_improvement >= args.patience:
                print(f"early_stop_epoch={epoch + 1}", flush=True)
                break
    if selected_scores is None:
        model.eval()
        with torch.no_grad():
            selected_scores = model(features)[0].exp()[:, 1].cpu().numpy()
        best_epoch = len(losses)
    write_scores(args.scores, score_nodes, selected_scores[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method":"HSMAD","source_kind":"AUTHOR_OFFICIAL",
            "official_commit":"7810e8e7dfe2143c4e4aa2ba804a0bfdc6543a5a","checkout_head":git_head(source),
            "dataset_bundle_sha256":sha256(args.bundle),"label_source":contract["label_source"],"fair_ranking":False,
            "supervision":{"trial":args.trial,"train_nodes":int(train_mask.sum()),"validation_nodes":int(validation_mask.sum()),"node_labels":"training-mask node loss","edge_labels":"labels materialized for all canonical edges; loss restricted to edges whose endpoints are in the active node mask","checkpoint_metric":"validation macro-F1"},
            "native_contract":{"model":"model.CombinedModel","epochs":args.epochs,"patience":args.patience,"validation_interval":args.validation_interval,"hidden_dimension":args.hidden_dimension,"order":args.order,"quantile":args.quantile,"lr":args.lr,"weight_decay":args.weight_decay},
            "real_optimizer_steps":len(losses),"first_loss":losses[0],"last_loss":losses[-1],"best_epoch":best_epoch,"best_validation_macro_f1":best_f1,
            "score_count":int(score_nodes.size),"finite_scores":bool(np.isfinite(selected_scores).all()),"higher_is_more_anomalous":True,
            "seed":args.seed,"device":str(device),"elapsed_seconds":time.time()-started,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--bundle", required=True); result.add_argument("--label-source", required=True)
    result.add_argument("--score-nodes", required=True); result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True); result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0); result.add_argument("--device", default="cpu")
    result.add_argument("--trial", type=int, default=0); result.add_argument("--epochs", type=int, default=1000)
    result.add_argument("--patience", type=int, default=100); result.add_argument("--validation-interval", type=int, default=10)
    result.add_argument("--hidden-dimension", type=int, default=64); result.add_argument("--order", type=int, default=2)
    result.add_argument("--quantile", type=float, default=0.5); result.add_argument("--lr", type=float, default=0.01)
    result.add_argument("--weight-decay", type=float, default=0.00001)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
