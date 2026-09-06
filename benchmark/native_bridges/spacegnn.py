"""Direct canonical trainer for the author-official SpaceGNN model."""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import dgl
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from benchmark.native_bridges.common import git_head, load_score_nodes, seed_everything, sha256, write_evidence, write_scores
from benchmark.native_bridges.supervised_common import load_supervised_reference


def load_model_class(source: Path):
    spec = importlib.util.spec_from_file_location("coregad_upstream_spacegnn_model", source / "model.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load official SpaceGNN model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpaceGNN


def propagate(graph: dgl.DGLGraph, value: torch.Tensor) -> torch.Tensor:
    with graph.local_scope():
        graph.ndata["h"] = value
        graph.update_all(dgl.function.copy_u("h", "m"), dgl.function.mean("m", "h"))
        return graph.ndata["h"]


def predict(model, graph, sampler, nodes, batch_size, alpha, beta, device):
    loader = DataLoader(nodes, batch_size=batch_size, shuffle=False, drop_last=False)
    values = []
    model.eval()
    with torch.no_grad():
        for output_nodes in loader:
            output_nodes = output_nodes.to(device)
            _, _, blocks = sampler.sample_blocks(graph, output_nodes)
            first, second, third = model(blocks)
            probability = (1 - beta) * ((1 - alpha) * first.exp()[:, 1] + alpha * second.exp()[:, 1]) + beta * third.exp()[:, 1]
            values.append(probability.detach().cpu())
    return torch.cat(values)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    contract = load_supervised_reference(args.bundle, args.label_source, trial=args.trial)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=contract["payload"]["x"].shape[0])
    source = Path(args.source).resolve()
    device = torch.device(args.device)
    graph = contract["graph"]
    graph.ndata["feature_0"] = graph.ndata["feature"].float()
    propagated = graph.ndata["feature_0"]
    for layer in range(1, args.layers):
        propagated = propagate(graph, propagated)
        graph.ndata[f"feature_{layer}"] = propagated
    graph = graph.to(device)
    labels = graph.ndata["label"].long()
    train_nodes = torch.where(graph.ndata["train_mask"].bool())[0].cpu()
    SpaceGNN = load_model_class(source)
    torch.manual_seed(args.seed)
    cneg = torch.empty(args.layers).normal_(-0.1, args.negative_std)
    cpos = torch.empty(args.layers).normal_(0.1, args.positive_std)
    seed_everything(args.seed)
    model = SpaceGNN(features_dim := graph.ndata["feature"].shape[1], args.hidden_dimension, 2, args.layers, args.dropout, cneg, cpos).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    loader = DataLoader(train_nodes, batch_size=args.batch_size, shuffle=True, drop_last=True)
    losses: list[float] = []
    steps = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for output_nodes in loader:
            output_nodes = output_nodes.to(device)
            _, _, blocks = sampler.sample_blocks(graph, output_nodes)
            first, second, third = model(blocks)
            log_probability = (1 - args.beta) * ((1 - args.alpha) * first + args.alpha * second) + args.beta * third
            target = blocks[-1].dstdata["label"].long()
            loss = F.nll_loss(log_probability, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            epoch_losses.append(float(loss.detach().cpu()))
        if not epoch_losses:
            raise RuntimeError("SpaceGNN training split is smaller than the author-official drop-last batch")
        losses.append(float(np.mean(epoch_losses)))
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 5 == 0:
            print(f"epoch={epoch + 1} loss={losses[-1]:.10f} optimizer_steps={steps}", flush=True)
    all_nodes = torch.arange(graph.num_nodes())
    scores = predict(model, graph, sampler, all_nodes, args.inference_batch_size, args.alpha, args.beta, device).numpy()
    write_scores(args.scores, score_nodes, scores[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method":"SpaceGNN","source_kind":"AUTHOR_OFFICIAL",
            "official_commit":"921c03ff879b239dab9b319b296fef3bc3bda2d2","checkout_head":git_head(source),
            "dataset_bundle_sha256":sha256(args.bundle),"label_source":contract["label_source"],"fair_ranking":False,
            "supervision":{"trial":args.trial,"train_nodes":int(train_nodes.numel())},
            "native_contract":{"model":"model.SpaceGNN","features":features_dim,"hidden_dimension":args.hidden_dimension,"layers":args.layers,"dropout":args.dropout,"batch_size":args.batch_size,"alpha":args.alpha,"beta":args.beta,"curvature_std_negative":args.negative_std,"curvature_std_positive":args.positive_std},
            "epochs":args.epochs,"real_optimizer_steps":steps,"first_epoch_loss":losses[0],"last_epoch_loss":losses[-1],
            "score_count":int(score_nodes.size),"finite_scores":bool(np.isfinite(scores).all()),"higher_is_more_anomalous":True,
            "seed":args.seed,"device":str(device),"elapsed_seconds":time.time()-started,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--bundle", required=True); result.add_argument("--label-source", required=True)
    result.add_argument("--score-nodes", required=True); result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True); result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0); result.add_argument("--device", default="cpu")
    result.add_argument("--trial", type=int, default=0); result.add_argument("--epochs", type=int, default=25)
    result.add_argument("--lr", type=float, default=0.001); result.add_argument("--hidden-dimension", type=int, default=128)
    result.add_argument("--layers", type=int, default=6); result.add_argument("--dropout", type=float, default=0.0)
    result.add_argument("--batch-size", type=int, default=50); result.add_argument("--inference-batch-size", type=int, default=4096)
    result.add_argument("--alpha", type=float, default=0.5); result.add_argument("--beta", type=float, default=1.0)
    result.add_argument("--negative-std", type=float, default=0.011343642441124013)
    result.add_argument("--positive-std", type=float, default=0.018474337369372326)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
