"""Direct canonical trainer for the author-official APF implementation."""

from __future__ import annotations

import argparse
import copy
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
        return (
            importlib.import_module("model"),
            importlib.import_module("finetune"),
            importlib.import_module("mrqsampler"),
        )
    finally:
        sys.path.remove(str(source))


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    contract = load_supervised_reference(args.bundle, args.label_source, trial=args.trial)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=contract["payload"]["x"].shape[0])
    source = Path(args.source).resolve()
    device = torch.device(args.device)
    upstream_model, upstream_finetune, upstream_mrq = load_upstream(source)
    payload = contract["payload"]
    graph = dgl.graph((payload["edge_index"][0], payload["edge_index"][1]), num_nodes=payload["x"].shape[0])
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(graph)))
    features = F.normalize(payload["x"].float(), dim=1, p=2)
    cache_dir = Path(args.evidence).resolve().parent / "upstream_cache"
    (cache_dir / "mrqs").mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(cache_dir)
        mrq_graph = upstream_mrq.mrqsample("weibo", graph, features, one_hop=True)
    finally:
        os.chdir(old_cwd)
    graph = graph.to(device)
    mrq_graph = mrq_graph.to(device)
    features = features.to(device)
    labels = contract["labels"].long().to(device)
    train_mask = contract["train_mask"].to(device)
    validation_mask = contract["val_mask"].to(device)
    encoder = upstream_model.ChebNetII(
        features.shape[1], args.hidden_dimension, args.order,
        args.activation, args.convolution_normalization, True,
    )
    model = upstream_model.DualDGI(args.hidden_dimension, encoder, dual_disc=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_weight_decay)
    best_pretrain_loss = float("inf")
    best_pretrain_epoch = 0
    best_pretrain_state = None
    pretrain_losses: list[float] = []
    started = time.time()
    for epoch in range(args.pretrain_epochs):
        model.train()
        low_loss = model(graph, mrq_graph, features, lowpass=True)
        optimizer.zero_grad(set_to_none=True); low_loss.backward(); optimizer.step()
        high_loss = model(graph, mrq_graph, features, lowpass=False)
        optimizer.zero_grad(set_to_none=True); high_loss.backward(); optimizer.step()
        loss = float(low_loss.detach().cpu() + high_loss.detach().cpu())
        pretrain_losses.append(loss)
        if loss < best_pretrain_loss:
            best_pretrain_loss = loss
            best_pretrain_epoch = epoch + 1
            best_pretrain_state = copy.deepcopy(model.state_dict())
        elif epoch + 1 - best_pretrain_epoch > args.pretrain_patience:
            print(f"pretrain_early_stop_epoch={epoch + 1}", flush=True)
            break
        if epoch == 0 or epoch + 1 == args.pretrain_epochs or (epoch + 1) % 10 == 0:
            print(f"pretrain_epoch={epoch + 1} loss={loss:.10f} best_loss={best_pretrain_loss:.10f}", flush=True)
    if best_pretrain_state is None:
        raise RuntimeError("APF pretraining did not produce a checkpoint")
    model.load_state_dict(best_pretrain_state)
    model.eval()
    with torch.no_grad():
        embedding_low = model.encoder(graph, features, lowpass=True)
        embedding_high = model.encoder(graph, features, lowpass=False)
    classifier = upstream_finetune.NDAdaptiveFineTune(
        raw_dim=features.shape[1], emb_dim=args.hidden_dimension,
        hid_dims=[args.hidden_dimension, 2], dropout=args.finetune_dropout,
        activation="ReLU", bias="rand", target_p=(args.anomaly_target, args.normal_target),
    ).to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.finetune_lr, weight_decay=args.finetune_weight_decay)
    train_labels = labels[train_mask]
    anomaly_weight = float((train_labels == 0).sum() / (train_labels == 1).sum())
    regularization_weight = torch.ones_like(train_labels, dtype=torch.float32)
    regularization_weight[train_labels == 1] = anomaly_weight
    target = train_labels.float().clone()
    target[train_labels == 1] = args.anomaly_target
    target[train_labels == 0] = args.normal_target
    best_validation_auc = -1.0
    best_finetune_epoch = 0
    selected_scores = None
    finetune_losses: list[float] = []
    for epoch in range(args.finetune_epochs):
        classifier.train()
        logits, coefficients = classifier(embedding_low[train_mask], embedding_high[train_mask], features[train_mask], return_coefs=True)
        classification_loss = F.nll_loss(F.log_softmax(logits, dim=-1), train_labels)
        regularization_loss = F.binary_cross_entropy(coefficients.mean(1), target, weight=regularization_weight)
        loss = classification_loss + regularization_loss
        classifier_optimizer.zero_grad(set_to_none=True); loss.backward(); classifier_optimizer.step()
        finetune_losses.append(float(loss.detach().cpu()))
        if (epoch + 1) % args.validation_interval == 0:
            classifier.eval()
            with torch.no_grad():
                full_probability = classifier(embedding_low, embedding_high, features).softmax(1)[:, 1]
            validation_auc = float(roc_auc_score(labels[validation_mask].cpu().numpy(), full_probability[validation_mask].cpu().numpy()))
            if validation_auc > best_validation_auc:
                best_validation_auc = validation_auc
                best_finetune_epoch = epoch + 1
                selected_scores = full_probability.detach().cpu().numpy().copy()
            if epoch == args.validation_interval - 1 or epoch + 1 == args.finetune_epochs or (epoch + 1) % 50 == 0:
                print(f"finetune_epoch={epoch + 1} loss={finetune_losses[-1]:.10f} best_validation_auroc={best_validation_auc:.10f}", flush=True)
    if selected_scores is None:
        raise RuntimeError("APF fine-tuning did not produce a validation-selected checkpoint")
    write_scores(args.scores, score_nodes, selected_scores[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method":"APF","source_kind":"AUTHOR_OFFICIAL",
            "official_commit":"0f42fd9344fa4c54e175fae45cb40e6981e6b54a","checkout_head":git_head(source),
            "dataset_bundle_sha256":sha256(args.bundle),"label_source":contract["label_source"],"fair_ranking":False,
            "supervision":{"trial":args.trial,"train_nodes":int(train_mask.sum()),"validation_nodes":int(validation_mask.sum()),"fine_tune_only":True,"checkpoint_metric":"validation AUROC"},
            "native_contract":{"encoder":"model.ChebNetII + model.DualDGI","fine_tuner":"finetune.NDAdaptiveFineTune","feature_transform":"l2","mrq_one_hop":True,"pretrain_epochs":args.pretrain_epochs,"pretrain_patience":args.pretrain_patience,"fine_tune_epochs":args.finetune_epochs,"order":args.order,"hidden_dimension":args.hidden_dimension},
            "pretrain_optimizer_steps":2*len(pretrain_losses),"fine_tune_optimizer_steps":len(finetune_losses),
            "first_pretrain_loss":pretrain_losses[0],"last_pretrain_loss":pretrain_losses[-1],"best_pretrain_epoch":best_pretrain_epoch,
            "first_finetune_loss":finetune_losses[0],"last_finetune_loss":finetune_losses[-1],"best_finetune_epoch":best_finetune_epoch,"best_validation_auroc":best_validation_auc,
            "score_count":int(score_nodes.size),"finite_scores":bool(np.isfinite(selected_scores).all()),"higher_is_more_anomalous":True,
            "seed":args.seed,"device":str(device),"elapsed_seconds":time.time()-started,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--bundle", required=True); result.add_argument("--label-source", required=True)
    result.add_argument("--score-nodes", required=True); result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True); result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0); result.add_argument("--device", default="cpu"); result.add_argument("--trial", type=int, default=0)
    result.add_argument("--pretrain-epochs", type=int, default=300); result.add_argument("--pretrain-patience", type=int, default=20)
    result.add_argument("--pretrain-lr", type=float, default=0.001); result.add_argument("--pretrain-weight-decay", type=float, default=0.0001)
    result.add_argument("--finetune-epochs", type=int, default=500); result.add_argument("--finetune-lr", type=float, default=0.01)
    result.add_argument("--finetune-weight-decay", type=float, default=0.0001); result.add_argument("--finetune-dropout", type=float, default=0.0)
    result.add_argument("--validation-interval", type=int, default=5); result.add_argument("--hidden-dimension", type=int, default=32)
    result.add_argument("--order", type=int, default=3); result.add_argument("--activation", default="ELU")
    result.add_argument("--convolution-normalization", default="batch")
    result.add_argument("--anomaly-target", type=float, default=1.0); result.add_argument("--normal-target", type=float, default=1.0)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
