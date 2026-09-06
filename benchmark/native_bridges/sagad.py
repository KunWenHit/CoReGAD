"""Direct canonical trainer for the author-official SAGAD implementation."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from benchmark.native_bridges.common import git_head, load_score_nodes, seed_everything, sha256, write_evidence, write_scores
from benchmark.native_bridges.supervised_common import load_supervised_reference


def load_upstream(source: Path):
    sys.path.insert(0, str(source))
    try:
        return importlib.import_module("model"), importlib.import_module("mrqsampler")
    finally:
        sys.path.remove(str(source))


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    contract = load_supervised_reference(args.bundle, args.label_source, trial=args.trial)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=contract["payload"]["x"].shape[0])
    source = Path(args.source).resolve()
    device = torch.device(args.device)
    upstream_model, upstream_mrq = load_upstream(source)
    payload = contract["payload"]
    graph = dgl.graph((payload["edge_index"][0], payload["edge_index"][1]), num_nodes=payload["x"].shape[0])
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(graph)))
    features = F.normalize(payload["x"].float(), dim=1, p=1)
    labels = contract["labels"].long()
    tx_list = upstream_model.precompute_Tx(graph, features, args.order, conv_norm=args.convolution_normalization)
    cache_dir = Path(args.evidence).resolve().parent / "upstream_cache"
    (cache_dir / "mrqs").mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(cache_dir)
        mrq_graph = upstream_mrq.mrqsample("weibo", graph, features, one_hop=True)
    finally:
        os.chdir(old_cwd)
    neighbor_features = dgl.ops.copy_u_mean(mrq_graph, features)
    model_features = torch.concat([features, neighbor_features], dim=1).to(device)
    tx_list = [value.to(device) for value in tx_list]
    labels = labels.to(device)
    train_mask = contract["train_mask"].to(device)
    validation_mask = contract["val_mask"].to(device)
    in_dimension = model_features.shape[1]
    embedding_dimension = in_dimension // 2
    fusion = upstream_model.NodeDimGatedFusion(in_dimension, embedding_dimension, args.gate_layers)
    model = upstream_model.SAGAD(
        in_dimension,
        embedding_dimension,
        hid_dims=[args.hidden_dimension, args.hidden_dimension, 2],
        K=args.order,
        dropout=args.dropout,
        activation="ReLU",
        mlp_norm="none",
        fusion=fusion,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_labels = labels[train_mask]
    anomaly_weight = float((train_labels == 0).sum() / (train_labels == 1).sum())
    class_weight = torch.tensor([1.0, anomaly_weight], device=device)
    regularization_weight = torch.ones_like(train_labels, dtype=torch.float32)
    regularization_weight[train_labels == 1] = anomaly_weight
    target = train_labels.float().clone()
    target[train_labels == 1] = args.anomaly_target
    target[train_labels == 0] = args.normal_target
    best_auc = -1.0
    best_epoch = 0
    selected_scores = None
    losses: list[float] = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        logits, coefficients = model([value[train_mask] for value in tx_list], model_features[train_mask], return_coefs=True)
        classification_loss = F.cross_entropy(logits, train_labels, weight=class_weight)
        regularization_loss = F.binary_cross_entropy(coefficients.mean(1), target, weight=regularization_weight)
        loss = classification_loss + regularization_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            full_probability = model(tx_list, model_features).softmax(1)[:, 1]
            validation_auc = float(roc_auc_score(labels[validation_mask].cpu().numpy(), full_probability[validation_mask].cpu().numpy()))
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch + 1
            selected_scores = full_probability.detach().cpu().numpy().copy()
        elif epoch + 1 - best_epoch > args.patience:
            print(f"early_stop_epoch={epoch + 1}", flush=True)
            break
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 25 == 0:
            print(f"epoch={epoch + 1} loss={losses[-1]:.10f} best_validation_auroc={best_auc:.10f}", flush=True)
    if selected_scores is None:
        raise RuntimeError("SAGAD did not produce a validation-selected checkpoint")
    write_scores(args.scores, score_nodes, selected_scores[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method":"SAGAD","source_kind":"AUTHOR_OFFICIAL",
            "official_commit":"faddcbca117b475c8468e87a65c8428b616b694b","checkout_head":git_head(source),
            "dataset_bundle_sha256":sha256(args.bundle),"label_source":contract["label_source"],"fair_ranking":False,
            "supervision":{"trial":args.trial,"train_nodes":int(train_mask.sum()),"validation_nodes":int(validation_mask.sum()),"checkpoint_metric":"validation AUROC","class_weight":anomaly_weight},
            "native_contract":{"model":"model.SAGAD","fusion":"NodeDimGated","epochs":args.epochs,"patience":args.patience,"order":args.order,"feature_transform":"l1","hidden_dimensions":[args.hidden_dimension,args.hidden_dimension],"mrq_one_hop":True},
            "real_optimizer_steps":len(losses),"first_loss":losses[0],"last_loss":losses[-1],"best_epoch":best_epoch,"best_validation_auroc":best_auc,
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
    result.add_argument("--trial", type=int, default=0); result.add_argument("--epochs", type=int, default=500)
    result.add_argument("--patience", type=int, default=50); result.add_argument("--lr", type=float, default=0.1)
    result.add_argument("--weight-decay", type=float, default=0.0); result.add_argument("--dropout", type=float, default=0.0)
    result.add_argument("--hidden-dimension", type=int, default=128); result.add_argument("--gate-layers", type=int, default=2)
    result.add_argument("--order", type=int, default=5); result.add_argument("--convolution-normalization", default="none")
    result.add_argument("--anomaly-target", type=float, default=1.0); result.add_argument("--normal-target", type=float, default=1.0)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
