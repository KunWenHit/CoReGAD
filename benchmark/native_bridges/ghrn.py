"""Two-stage canonical bridge for author-official GHRN."""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn.pytorch.conv import EdgeWeightNorm
from torch.nn import functional as F

from benchmark.native_bridges.common import git_head, load_score_nodes, seed_everything, sha256, write_evidence, write_scores
from benchmark.native_bridges.supervised_common import load_supervised_reference


def model_class(source: Path):
    spec = importlib.util.spec_from_file_location("coregad_upstream_ghrn_bwgnn", source / "BWGNN.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load official GHRN BWGNN.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BWGNN


def train_stage(graph, BWGNN, args, device: torch.device):
    features = graph.ndata["feature"]
    labels = graph.ndata["label"].long()
    train_mask = graph.ndata["train_mask"].bool()
    val_mask = graph.ndata["val_mask"].bool()
    model = BWGNN(features.shape[1], args.hidden_dimension, 2, graph, d=args.order).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    anomaly_weight = float((labels[train_mask] == 0).sum() / (labels[train_mask] == 1).sum())
    class_weight = torch.tensor([1.0, anomaly_weight], device=device)
    best_validation_loss = float("inf")
    selected_probability = None
    losses = []
    for epoch in range(args.epochs):
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(features)
            validation_loss = F.cross_entropy(logits[val_mask], labels[val_mask], weight=class_weight)
            probability = logits.softmax(1)
        if float(validation_loss) <= best_validation_loss:
            best_validation_loss = float(validation_loss)
            selected_probability = probability.detach().clone()
        losses.append(float(loss.detach().cpu()))
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 20 == 0:
            print(f"epoch={epoch + 1} loss={losses[-1]:.10f} validation_loss={float(validation_loss):.10f}", flush=True)
    if selected_probability is None:
        raise RuntimeError("GHRN stage did not produce validation-selected probabilities")
    return selected_probability, losses, best_validation_loss


def high_frequency_prune(graph, probability: torch.Tensor, delete_ratio: float):
    with graph.local_scope():
        edge_weight = torch.ones(graph.num_edges(), device=graph.device)
        graph.edata["w"] = EdgeWeightNorm(norm="both")(graph, edge_weight)
        graph.ndata["h"] = probability
        graph.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "ay"))
        graph.ndata["ly"] = probability - graph.ndata["ay"]
        graph.apply_edges(lambda edges: {"inner_black": (edges.src["ly"] * edges.dst["ly"]).sum(axis=1)})
        remove_count = int(delete_ratio * graph.num_edges())
        edge_ids = graph.edata["inner_black"].argsort()[:remove_count]
    return dgl.remove_edges(graph, edge_ids), remove_count


def run(args: argparse.Namespace) -> None:
    if not 0.0 < args.delete_ratio < 1.0:
        raise ValueError("GHRN direct closure requires a positive frozen deletion ratio")
    seed_everything(args.seed)
    contract = load_supervised_reference(args.bundle, args.label_source, trial=args.trial)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=contract["payload"]["x"].shape[0])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_default_device(device)
    graph = contract["graph"].to(device)
    source = Path(args.source).resolve()
    BWGNN = model_class(source)
    started = time.time()
    print("stage=1 base_prediction", flush=True)
    stage1_probability, stage1_losses, stage1_best = train_stage(graph, BWGNN, args, device)
    pruned_graph, removed_edges = high_frequency_prune(graph, stage1_probability, args.delete_ratio)
    print(f"stage=prune removed_edges={removed_edges}", flush=True)
    print("stage=2 retrain_pruned_graph", flush=True)
    stage2_probability, stage2_losses, stage2_best = train_stage(pruned_graph, BWGNN, args, device)
    scores = stage2_probability[:, 1].cpu().numpy()
    write_scores(args.scores, score_nodes, scores[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method":"GHRN","source_kind":"AUTHOR_OFFICIAL",
            "official_commit":"d47e047b76df429c0c0a8444f5d9a5dea57f6801","checkout_head":git_head(source),
            "dataset_bundle_sha256":sha256(args.bundle),"label_source":contract["label_source"],"fair_ranking":False,
            "supervision":{"trial":args.trial,"train_nodes":int(graph.ndata["train_mask"].sum()),"validation_nodes":int(graph.ndata["val_mask"].sum()),"checkpoint_metric":"validation weighted cross-entropy"},
            "two_stage_contract":{"stage1_epochs":args.epochs,"delete_ratio":args.delete_ratio,"removed_edges":removed_edges,"stage2_epochs":args.epochs},
            "real_optimizer_steps":2*args.epochs,"stage1_first_loss":stage1_losses[0],"stage1_last_loss":stage1_losses[-1],"stage1_best_validation_loss":stage1_best,
            "stage2_first_loss":stage2_losses[0],"stage2_last_loss":stage2_losses[-1],"stage2_best_validation_loss":stage2_best,
            "score_count":int(score_nodes.size),"finite_scores":bool(np.isfinite(scores).all()),"higher_is_more_anomalous":True,
            "seed":args.seed,"device":str(device),"elapsed_seconds":time.time()-started,
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
    result.add_argument("--delete-ratio", type=float, default=0.015)
    result.add_argument("--lr", type=float, default=0.01)
    result.add_argument("--hidden-dimension", type=int, default=64)
    result.add_argument("--order", type=int, default=2)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
