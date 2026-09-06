"""Run the official notebook class and extracted CLI class on one graph slice."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, GraphSAGE, PNAConv, SAGEConv
from torch_geometric.utils import add_self_loops

from benchmark.native_bridges.common import load_label_free_bundle, seed_everything
from benchmark.native_bridges.gad_nr import GNNStructEncoder


def train_trace(model, edge_index: torch.Tensor, x: torch.Tensor, degree: torch.Tensor, epochs: int) -> list[float]:
    degree_ids = {id(parameter) for parameter in model.degree_decoder.parameters()}
    base = [parameter for parameter in model.parameters() if id(parameter) not in degree_ids]
    optimizer = torch.optim.Adam(
        [{"params": base}, {"params": model.degree_decoder.parameters(), "lr": 1e-2}],
        lr=0.01,
        weight_decay=0.0003,
    )
    neighbor_dict: dict[int, list[int]] = {node: [] for node in range(x.shape[0])}
    for source, target in edge_index.t().tolist():
        neighbor_dict[source].append(target)
    trace = []
    for _ in range(epochs):
        try:
            outputs = model(edge_index, x, degree, neighbor_dict, device=x.device)
        except TypeError:
            outputs = model(edge_index, x, degree)
        loss = outputs[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach().cpu()))
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    payload = load_label_free_bundle(args.bundle)
    node_count = min(args.nodes, payload["x"].shape[0])
    device = torch.device(args.device)
    x = payload["x"][:node_count].to(device)
    keep = (payload["edge_index"][0] < node_count) & (payload["edge_index"][1] < node_count)
    edge_index = add_self_loops(payload["edge_index"][:, keep], num_nodes=node_count)[0].to(device)
    degree = torch.bincount(edge_index[0], minlength=node_count).to(device)

    notebook = json.loads(Path(args.notebook).read_text(encoding="utf-8"))
    namespace = {
        "torch": torch, "nn": nn, "F": F, "random": random, "math": math, "mp": mp,
        "GCNConv": GCNConv, "GINConv": GINConv, "SAGEConv": SAGEConv,
        "GATConv": GATConv, "PNAConv": PNAConv, "GraphSAGE": GraphSAGE,
        "args": SimpleNamespace(aggregator="mean", neigh_loss="KL"), "device": device,
    }
    for cell_index in (5, 8):
        exec("".join(notebook["cells"][cell_index]["source"]), namespace)
    seed_everything(0)
    native_model = namespace["GNNStructEncoder"](
        x.shape[1], 16, 16, 2, 10, device, degree,
        GNN_name="GCN", lambda_loss1=0.01, lambda_loss2=0.5, lambda_loss3=0.8,
    ).to(device)
    native_trace = train_trace(native_model, edge_index, x, degree, args.epochs)
    native_model.pool.close()
    native_model.pool.join()

    seed_everything(0)
    cli_model = GNNStructEncoder(
        x.shape[1], 16, 10, device, degree, encoder="GCN", aggregator="mean",
        lambda_loss1=0.01, lambda_loss2=0.5, lambda_loss3=0.8,
    ).to(device)
    cli_trace = train_trace(cli_model, edge_index, x, degree, args.epochs)
    result = {
        "status": "PASS" if native_trace[-1] < native_trace[0] and cli_trace[-1] < cli_trace[0] else "FAIL",
        "dataset": "Weibo canonical induced prefix",
        "nodes": node_count,
        "edges_with_self_loops": int(edge_index.shape[1]),
        "epochs": args.epochs,
        "official_notebook_cells": [5, 8],
        "native_notebook_loss_trace": native_trace,
        "extracted_cli_loss_trace": cli_trace,
        "trend_contract": "both first-to-last losses decrease; exact equality is not expected because dead notebook parameters consume RNG state",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
