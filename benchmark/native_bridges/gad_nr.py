"""Mechanical CLI extraction of the official GAD-NR Weibo notebook.

Model classes, optimizer groups, loss schedule, reconstruction sampling and
score aggregation map to notebook cells 5, 7, 8 and 10.  Ground-truth metric
logging and synthetic-label generation are deliberately outside this child
process; neither affects the native optimizer in the notebook.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, PNAConv, SAGEConv
from torch_geometric.utils import add_self_loops

from benchmark.native_bridges.common import (
    git_head,
    load_label_free_bundle,
    load_score_nodes,
    seed_everything,
    sha256,
    write_evidence,
    write_scores,
)


class MLP(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear_or_not = num_layers == 1
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError("number of layers should be positive")
        if self.linear_or_not:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            self.linears.extend(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            self.batch_norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear_or_not:
            return self.linear(x)
        h = x
        for layer in range(self.num_layers - 1):
            h = self.linears[layer](h)
            if h.ndim > 2:
                h = h.transpose(0, 1).transpose(1, 2)
            h = self.batch_norms[layer](h)
            if h.ndim > 2:
                h = h.transpose(1, 2).transpose(0, 1)
            h = F.relu(h)
        return self.linears[-1](h)


class MLPGenerator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, output_dim)] + [nn.Linear(output_dim, output_dim) for _ in range(3)]
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        value = embedding
        for layer in self.layers[:-1]:
            value = F.relu(layer(value))
        return self.layers[-1](value)


class FNN(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, layer_num: int) -> None:
        super().__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear2(F.relu(self.linear1(embedding))))


class GNNStructEncoder(nn.Module):
    def __init__(
        self,
        in_dim0: int,
        hidden_dim: int,
        sample_size: int,
        device: torch.device,
        neighbor_num_list: torch.Tensor,
        *,
        encoder: str,
        aggregator: str,
        lambda_loss1: float,
        lambda_loss2: float,
        lambda_loss3: float,
    ) -> None:
        super().__init__()
        self.mlp0 = nn.Linear(in_dim0, hidden_dim)
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3
        if encoder == "GIN":
            self.graphconv1 = GINConv(MLP(2, hidden_dim, hidden_dim, hidden_dim))
        elif encoder == "GCN":
            self.graphconv1 = GCNConv(hidden_dim, hidden_dim)
        elif encoder == "GAT":
            self.graphconv1 = GATConv(hidden_dim, hidden_dim)
        else:
            self.graphconv1 = SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
        self.tot_node = int(neighbor_num_list.numel())
        self.sample_size = sample_size
        self.device = device
        self.m_batched = torch.distributions.Normal(
            torch.zeros(sample_size, self.tot_node, hidden_dim),
            torch.ones(sample_size, self.tot_node, hidden_dim),
        )
        self.mlp_mean = FNN(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mlp_sigma = FNN(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mean_agg = SAGEConv(hidden_dim, hidden_dim, aggr=aggregator, normalize=False)
        self.std_agg = PNAConv(
            hidden_dim,
            hidden_dim,
            aggregators=["std"],
            scalers=["identity"],
            deg=neighbor_num_list,
        )
        self.layer1_generator = MLPGenerator(hidden_dim, hidden_dim)
        self.degree_decoder = FNN(hidden_dim, hidden_dim, 1, 4)
        self.feature_decoder = FNN(hidden_dim, hidden_dim, hidden_dim, 3)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()

    def reconstruction_neighbors(self, embedding: torch.Tensor, h0: torch.Tensor, edge_index: torch.Tensor):
        target_mean = self.mean_agg(h0, edge_index).detach()
        target_std = self.std_agg(h0, edge_index).detach()
        target_cov = torch.bmm(target_std.unsqueeze(-1), target_std.unsqueeze(1))
        repeated = embedding.unsqueeze(0).repeat(self.sample_size, 1, 1)
        generated_mean = self.mlp_mean(repeated)
        generated_sigma = self.mlp_sigma(repeated)
        noise = self.m_batched.sample().to(self.device)
        generated = self.layer1_generator(generated_mean + generated_sigma.exp() * noise)
        generated_mean = generated.mean(dim=0)
        generated_std = generated.std(dim=0)
        generated_cov = torch.bmm(generated_std.unsqueeze(-1), generated_std.unsqueeze(1)) / self.sample_size
        identity = torch.eye(embedding.shape[1], device=self.device).unsqueeze(0).repeat(self.tot_node, 1, 1)
        target_cov = target_cov + identity
        generated_cov = generated_cov + identity
        inverse_generated = torch.inverse(generated_cov)
        determinant_target = torch.linalg.det(target_cov)
        determinant_generated = torch.linalg.det(generated_cov)
        trace_matrix = torch.matmul(inverse_generated, target_cov)
        delta = generated_mean - target_mean
        quadratic = torch.bmm(torch.bmm(delta.unsqueeze(1), inverse_generated), delta.unsqueeze(-1)).squeeze()
        per_node = 0.5 * (
            torch.log(determinant_target / determinant_generated)
            - embedding.shape[1]
            + trace_matrix.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
            + quadratic
        )
        return per_node.mean(), per_node

    def forward(self, edge_index: torch.Tensor, x: torch.Tensor, degree: torch.Tensor):
        h0 = self.mlp0(x)
        embedding = self.graphconv1(h0, edge_index)
        degree_logits = F.relu(self.degree_decoder(embedding))
        degree_target = degree.unsqueeze(1).float()
        degree_loss = self.degree_loss_func(degree_logits, degree_target)
        degree_per_node = (degree_logits - degree_target).square().reshape(self.tot_node, 1)
        neighbor_losses = []
        neighbor_per_node = []
        feature_per_node = []
        for _ in range(3):
            reconstructed_x = self.feature_decoder(embedding)
            feature_per_node.append((h0 - reconstructed_x).square().mean(1))
            local_loss, local_per_node = self.reconstruction_neighbors(embedding, h0, edge_index)
            neighbor_losses.append(local_loss)
            neighbor_per_node.append(local_per_node)
        neighbor_loss = torch.stack(neighbor_losses).mean()
        neighbor_per_node_tensor = torch.stack(neighbor_per_node).mean(dim=0).reshape(self.tot_node, 1)
        feature_per_node_tensor = torch.stack(feature_per_node).mean(dim=0).reshape(self.tot_node, 1)
        feature_loss = feature_per_node_tensor.mean()
        loss = self.lambda_loss1 * neighbor_loss + self.lambda_loss3 * degree_loss + self.lambda_loss2 * feature_loss
        loss_per_node = (
            self.lambda_loss1 * neighbor_per_node_tensor
            + self.lambda_loss3 * degree_per_node
            + self.lambda_loss2 * feature_per_node_tensor
        )
        return loss, loss_per_node, neighbor_per_node_tensor, degree_per_node, feature_per_node_tensor


def normalized_component(value: torch.Tensor) -> torch.Tensor:
    return value / (value.max() - value.min())


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    payload = load_label_free_bundle(args.bundle)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=payload["x"].shape[0])
    device = torch.device(args.device)
    x = payload["x"].to(device)
    edge_index = add_self_loops(payload["edge_index"], num_nodes=x.shape[0])[0].to(device)
    degree = torch.bincount(edge_index[0], minlength=x.shape[0]).to(device)
    model = GNNStructEncoder(
        x.shape[1],
        args.dimension,
        args.sample_size,
        device,
        degree,
        encoder=args.encoder,
        aggregator=args.aggregator,
        lambda_loss1=args.lambda_loss1,
        lambda_loss2=args.lambda_loss2,
        lambda_loss3=args.lambda_loss3,
    ).to(device)
    degree_parameter_ids = {id(parameter) for parameter in model.degree_decoder.parameters()}
    base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in degree_parameter_ids]
    optimizer = torch.optim.Adam(
        [{"params": base_parameters}, {"params": model.degree_decoder.parameters(), "lr": 1e-2}],
        lr=args.lr,
        weight_decay=0.0003,
    )
    minimum_loss = float("inf")
    selected_score = None
    losses: list[float] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch % args.loss_step == 0:
            model.lambda_loss2 += 0.5
            model.lambda_loss3 /= 2
        loss, loss_per_node, neighbor_loss, degree_loss, feature_loss = model(edge_index, x, degree)
        combined_score = (
            args.h_loss_weight * normalized_component(neighbor_loss.detach())
            + args.degree_loss_weight * normalized_component(degree_loss.detach())
            + args.feature_loss_weight * normalized_component(feature_loss.detach())
        ).squeeze(1)
        comparison_score = loss_per_node.detach().squeeze(1) if args.real_loss else combined_score
        loss_value = float(loss.detach().cpu())
        if loss_value < minimum_loss:
            minimum_loss = loss_value
            selected_score = comparison_score.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss_value)
        if epoch == 1 or epoch == args.epochs or epoch % 25 == 0:
            print(f"epoch={epoch} loss={loss_value:.10f}", flush=True)
    if selected_score is None:
        raise RuntimeError("training did not produce a selected score")
    selected_score = selected_score.cpu().numpy()
    write_scores(args.scores, score_nodes, selected_score[score_nodes])
    source = Path(args.source).resolve()
    write_evidence(
        args.evidence,
        {
            "method": "GAD-NR",
            "source_kind": "AUTHOR_OFFICIAL",
            "official_commit": "b0a9590b10d0b9490e6d6cdfa60e7c50eb232c0d",
            "checkout_head": git_head(source),
            "dataset_bundle_sha256": sha256(args.bundle),
            "contains_ground_truth_labels": False,
            "notebook_extraction": {"notebook":"GAD-NR_weibo.ipynb","cells":[5,7,8,10]},
            "epochs": args.epochs,
            "real_optimizer_steps": args.epochs,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "minimum_loss": min(losses),
            "selection_rule": "minimum unsupervised reconstruction objective",
            "score_count": int(score_nodes.size),
            "finite_scores": bool(np.isfinite(selected_score).all()),
            "score_direction": "higher_is_more_anomalous",
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
    result.add_argument("--epochs", type=int, default=500)
    result.add_argument("--lr", type=float, default=0.01)
    result.add_argument("--lambda-loss1", type=float, default=0.01)
    result.add_argument("--lambda-loss2", type=float, default=0.5)
    result.add_argument("--lambda-loss3", type=float, default=0.8)
    result.add_argument("--sample-size", type=int, default=10)
    result.add_argument("--dimension", type=int, default=16)
    result.add_argument("--encoder", default="GCN")
    result.add_argument("--aggregator", default="mean")
    result.add_argument("--loss-step", type=int, default=100)
    result.add_argument("--real-loss", action="store_true")
    result.add_argument("--h-loss-weight", type=float, default=1.0)
    result.add_argument("--feature-loss-weight", type=float, default=2.0)
    result.add_argument("--degree-loss-weight", type=float, default=1.0)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
