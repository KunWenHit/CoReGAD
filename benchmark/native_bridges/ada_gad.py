"""Label-isolated canonical bridge for the official ADA-GAD implementation."""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj

from benchmark.native_bridges.common import (
    git_head,
    load_label_free_bundle,
    load_score_nodes,
    seed_everything,
    sha256,
    write_evidence,
    write_scores,
)


def upstream_arguments(source: Path, dataset_config: str):
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "pygod"))
    from model.utils import build_args, load_best_configs

    original = sys.argv
    try:
        sys.argv = ["ada-gad-canonical"]
        args = build_args()
    finally:
        sys.argv = original
    args.dataset = dataset_config
    args = load_best_configs(args, str(source / "config_ada-gad.yml"))
    if args.alpha_f == "None":
        args.alpha_f = None
    if args.all_encoder_layers != 0:
        args.node_encoder_num_layers = args.all_encoder_layers
        args.edge_encoder_num_layers = args.all_encoder_layers
        args.subgraph_encoder_num_layers = args.all_encoder_layers
    return args


def pretrain_encoder(model, graph: Data, optimizer, epochs: int, scheduler) -> list[float]:
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        loss, _, _ = model(graph.x, graph.edge_index)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if epoch == 0 or epoch + 1 == epochs:
            print(f"pretrain_epoch={epoch + 1} loss={losses[-1]:.10f}", flush=True)
    return losses


def run(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    seed_everything(args.seed)
    payload = load_label_free_bundle(args.bundle)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=payload["x"].shape[0])
    config = upstream_arguments(source, args.upstream_dataset_config)
    if args.pretrain_epochs is not None:
        config.max_epoch = args.pretrain_epochs
    if args.detection_epochs is not None:
        config.max_epoch_f = args.detection_epochs
    config.num_features = int(payload["x"].shape[1])
    device = torch.device(args.device)
    edge_index = add_remaining_self_loops(payload["edge_index"])[0]
    graph = Data(x=payload["x"], edge_index=edge_index)
    graph.s = to_dense_adj(edge_index)[0]
    graph = graph.to(device)

    # PyG 2.2 exposed ``degree`` through a tiny submodule which was removed in
    # PyG 2.6.  Recreate only that import alias; the function and mathematics
    # remain PyG's current canonical implementation.
    from torch_geometric.utils import degree, sort_edge_index, subgraph

    for legacy_name, function in {
        "degree": degree,
        "sort_edge_index": sort_edge_index,
        "subgraph": subgraph,
    }.items():
        qualified_name = f"torch_geometric.utils.{legacy_name}"
        if qualified_name not in sys.modules:
            compatibility_module = types.ModuleType(qualified_name)
            setattr(compatibility_module, legacy_name, function)
            sys.modules[qualified_name] = compatibility_module

    from model.models import build_model
    from model.utils import create_optimizer
    from pygod.models import ADANET

    attr_model, struct_model, topology_model = build_model(config)
    models = [attr_model, struct_model, topology_model][: int(config.use_encoder_num)]
    pretrain_losses: list[list[float]] = []
    started = time.time()
    for model in models:
        model.to(device)
        optimizer = create_optimizer(config.optimizer, model, config.lr, config.weight_decay)
        scheduler = None
        if config.scheduler:
            schedule = lambda epoch: (1 + np.cos(epoch * np.pi / config.max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
        pretrain_losses.append(pretrain_encoder(model, graph, optimizer, config.max_epoch, scheduler))
        model.eval()

    detector = ADANET(
        epoch=config.max_epoch_f,
        aggr=config.aggr_f,
        hid_dim=config.num_hidden,
        alpha=config.alpha_f,
        dropout=config.dropout_f,
        lr=config.lr_f,
        loss_name=config.loss_f,
        loss_weight=config.loss_weight_f,
        T=config.T_f,
        gpu=device.index if device.type == "cuda" else -1,
        use_encoder_num=config.use_encoder_num,
        attention=config.attention,
        attr_encoder_name=config.attr_encoder,
        struct_encoder_name=config.struct_encoder,
        topology_encoder_name=config.topology_encoder,
        attr_decoder_name=config.attr_decoder,
        struct_decoder_name=config.struct_decoder,
        node_encoder_num_layers=config.node_encoder_num_layers,
        edge_encoder_num_layers=config.edge_encoder_num_layers,
        subgraph_encoder_num_layers=config.subgraph_encoder_num_layers,
        attr_decoder_num_layers=config.attr_decoder_num_layers,
        struct_decoder_num_layers=config.struct_decoder_num_layers,
        sparse_attention_weight=config.sparse_attention_weight,
        theta=config.theta,
        eta=config.eta,
        verbose=True,
    )
    encoders = [model.encoder for model in models]
    detector.fit(
        graph,
        y_true=None,
        pretrain_attr_encoder=encoders[0] if len(encoders) > 0 else None,
        pretrain_struct_encoder=encoders[1] if len(encoders) > 1 else None,
        pretrain_topology_encoder=encoders[2] if len(encoders) > 2 else None,
    )
    anomaly_score = detector.decision_function(graph)
    write_scores(args.scores, score_nodes, anomaly_score[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method": "ADA-GAD",
            "source_kind": "AUTHOR_OFFICIAL",
            "official_commit": "71a3aca936ccfe430e3809864f256d4bf80b22ee",
            "checkout_head": git_head(source),
            "dataset_bundle_sha256": sha256(args.bundle),
            "contains_ground_truth_labels": False,
            "upstream_dataset_config": args.upstream_dataset_config,
            "pretrain_epochs_per_encoder": config.max_epoch,
            "pretrained_encoder_count": len(models),
            "detection_epochs": config.max_epoch_f,
            "pretrain_first_losses": [row[0] for row in pretrain_losses],
            "pretrain_last_losses": [row[-1] for row in pretrain_losses],
            "real_optimizer_steps": len(models) * config.max_epoch + config.max_epoch_f,
            "score_count": int(score_nodes.size),
            "finite_scores": bool(np.isfinite(anomaly_score).all()),
            "higher_is_more_anomalous": True,
            "seed": args.seed,
            "device": str(device),
            "elapsed_seconds": time.time() - started,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--bundle", required=True)
    result.add_argument("--score-nodes", required=True)
    result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cpu")
    result.add_argument("--upstream-dataset-config", default="weibo")
    result.add_argument("--pretrain-epochs", type=int)
    result.add_argument("--detection-epochs", type=int)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
