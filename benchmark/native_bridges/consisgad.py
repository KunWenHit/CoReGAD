"""Direct canonical trainer for the author-official ConsisGAD implementation."""

from __future__ import annotations

import argparse
import copy
import importlib
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import scipy
import torch
import types
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from benchmark.native_bridges.common import git_head, load_score_nodes, seed_everything, sha256, write_evidence, write_scores
from benchmark.native_bridges.supervised_common import load_supervised_reference


def load_upstream(source: Path):
    # scikit-plot 0.3.7 imports this historical SciPy alias used by the frozen source.
    if not hasattr(scipy, "interp"):
        scipy.interp = np.interp
    # wandb 0.13.7, imported but unused by the trainer, expects NumPy 1.x's alias.
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128
    # The upstream trainer imports wandb but never references it; avoid importing
    # an obsolete telemetry stack into the isolated, offline reproduction path.
    sys.modules["wandb"] = types.ModuleType("wandb")
    sys.path.insert(0, str(source))
    try:
        return importlib.import_module("main")
    finally:
        sys.path.remove(str(source))


def predict(upstream, model, graph, sampler, nodes, batch_size, device):
    loader = DataLoader(nodes.cpu(), batch_size=batch_size, shuffle=False, drop_last=False)
    probability, _ = upstream.get_model_pred(model, graph, loader, sampler, {"device": device})
    return probability.detach().cpu()


def config(args: argparse.Namespace, device: torch.device, feature_dimension: int) -> dict:
    return {
        "data-set":"weibo","to-homo":False,"shuffle-train":True,"model":"backbone",
        "hidden-dim":args.hidden_dimension,"num-layers":1,"epochs":args.epochs,"lr":args.lr,
        "weight-decay":0.00001,"device":device,"training-ratio":1,"train-procedure":"CT",
        "mlp-drop":0.4,"input-drop":0.0,"hidden-drop":0.0,"mlp12-dim":128,"mlp3-dim":128,
        "bn-type":2,"optim":"adam","store-model":False,"trainable-consis-weight":1.5,
        "trainable-temp":0.0001,"trainable-eps":1e-12,"trainable-drop-rate":0.2,
        "trainable-warm-up":-1,"trainable-model":"mlp","trainable-optim":"adam",
        "trainable-lr":0.005,"trainable-weight-decay":0.00001,"topk-mode":4,
        "diversity-type":"cos","unlabel-ratio":4,"normal-th":7,"fraud-th":88,
        "trainable-detach-y":True,"trainable-div-eps":True,"trainable-detach-mask":False,
        "batch-size":args.batch_size,"train-iterations":args.train_iterations,
        "node-in-dim":feature_dimension,"node-out-dim":2,
    }


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    contract = load_supervised_reference(args.bundle, args.label_source, trial=args.trial)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=contract["payload"]["x"].shape[0])
    source = Path(args.source).resolve()
    device = torch.device(args.device)
    upstream = load_upstream(source)
    graph = contract["graph"].to(device)
    labels = graph.ndata["label"].long()
    train_nodes = torch.where(graph.ndata["train_mask"].bool())[0]
    validation_nodes = torch.where(graph.ndata["val_mask"].bool())[0]
    all_nodes = torch.arange(graph.num_nodes(), device=device)
    settings = config(args, device, graph.ndata["feature"].shape[1])
    upstream.args = settings
    model = upstream.create_model(settings, graph.etypes)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=0.0)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(settings["num-layers"])
    attention_drop = upstream.SoftAttentionDrop(settings).to(device)
    augmentation_optimizer = torch.optim.Adam(attention_drop.parameters(), lr=settings["trainable-lr"], weight_decay=0.0)
    augmentor = (sampler, attention_drop, augmentation_optimizer)
    labeled_loader = DataLoader(train_nodes.cpu(), batch_size=settings["batch-size"], shuffle=True, drop_last=True)
    unlabeled_loader = DataLoader(all_nodes.cpu(), batch_size=settings["batch-size"] * settings["unlabel-ratio"], shuffle=True, drop_last=True)
    if len(labeled_loader) == 0 or len(unlabeled_loader) == 0:
        raise RuntimeError("ConsisGAD canonical split is smaller than the author-official drop-last batch")
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    started = time.time()
    for epoch in range(args.epochs):
        upstream.UDA_train_epoch(epoch, model, upstream.nll_loss, graph, labeled_loader, unlabeled_loader, optimizer, augmentor, settings)
        validation_probability = predict(upstream, model, graph, sampler, validation_nodes, args.inference_batch_size, device)
        validation_auc = float(roc_auc_score(labels[validation_nodes].cpu().numpy(), validation_probability.numpy()))
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} best_validation_auroc={best_auc:.10f}", flush=True)
    if best_state is None:
        raise RuntimeError("ConsisGAD did not produce a validation-selected checkpoint")
    model.load_state_dict(best_state)
    scores = predict(upstream, model, graph, sampler, all_nodes, args.inference_batch_size, device).numpy()
    write_scores(args.scores, score_nodes, scores[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method":"ConsisGAD","source_kind":"AUTHOR_OFFICIAL",
            "official_commit":"36811c5bc79be49c9740f25a1f260496bb4736af","checkout_head":git_head(source),
            "dataset_bundle_sha256":sha256(args.bundle),"label_source":contract["label_source"],"fair_ranking":False,
            "supervision":{"trial":args.trial,"train_nodes":int(train_nodes.numel()),"validation_nodes":int(validation_nodes.numel()),"unlabeled_pool":"all canonical nodes","checkpoint_metric":"validation AUROC"},
            "native_contract":{"model":"models.simpleGNN_MR","trainer":"main.UDA_train_epoch","augmentor":"main.SoftAttentionDrop","schedule_source":"upstream yelp.yml fixed fallback because upstream publishes no Weibo configuration","epochs":args.epochs,"train_iterations_per_epoch":args.train_iterations,"batch_size":args.batch_size,"unlabeled_ratio":4,"normal_threshold_percent":7,"fraud_threshold_percent":88},
            "model_optimizer_steps":args.epochs*args.train_iterations,"augmentation_optimizer_steps":args.epochs*args.train_iterations,
            "best_epoch":best_epoch,"best_validation_auroc":best_auc,"score_count":int(score_nodes.size),
            "finite_scores":bool(np.isfinite(scores).all()),"higher_is_more_anomalous":True,
            "compatibility_shims":["scipy.interp=np.interp for frozen scikit-plot 0.3.7 import","NumPy 1.x float_/complex_ aliases","unused upstream wandb import replaced by an offline module stub"],
            "seed":args.seed,"device":str(device),"elapsed_seconds":time.time()-started,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--bundle", required=True); result.add_argument("--label-source", required=True)
    result.add_argument("--score-nodes", required=True); result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True); result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0); result.add_argument("--device", default="cpu")
    result.add_argument("--trial", type=int, default=0); result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--train-iterations", type=int, default=128); result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--hidden-dimension", type=int, default=64); result.add_argument("--lr", type=float, default=0.001)
    result.add_argument("--inference-batch-size", type=int, default=65536)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
