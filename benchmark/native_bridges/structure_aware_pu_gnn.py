"""Deterministic canonical trainer for the paper-derived PU auxiliary.

The frozen normal support is the PU positive class (normal nodes).  The child
process never opens ground-truth anomaly labels; anomaly score is one minus
the inferred probability of the normal-positive class.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import GCNConv

from benchmark.native_bridges.common import (
    load_label_free_bundle,
    load_normal_support,
    load_score_nodes,
    seed_everything,
    sha256,
    write_evidence,
    write_scores,
)
from benchmark.paper_derived.structure_aware_pu_gnn import (
    PAPER_DEFAULTS,
    distance_aware_pu_loss,
    distance_partition,
    sample_non_neighbors,
)


class TwoLayerGCN(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.convolution1 = GCNConv(input_dimension, hidden_dimension)
        self.convolution2 = GCNConv(hidden_dimension, hidden_dimension)
        self.classifier = nn.Linear(hidden_dimension, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.convolution1(x, edge_index).relu()
        embedding = self.convolution2(embedding, edge_index)
        positive_probability = torch.sigmoid(self.classifier(embedding).squeeze(1))
        return embedding, positive_probability


def structural_regularizer_chunked(
    embedding: torch.Tensor,
    edge_index: torch.Tensor,
    negative_nodes: torch.Tensor,
    *,
    chunk_size: int = 16384,
) -> torch.Tensor:
    """Equation 3 evaluated in edge chunks to bound temporary GPU memory."""
    source, target = edge_index.long()
    positive_sum = embedding.new_zeros(())
    negative_sum = embedding.new_zeros(())
    for start in range(0, source.numel(), chunk_size):
        stop = min(start + chunk_size, source.numel())
        source_embedding = embedding[source[start:stop]]
        target_embedding = embedding[target[start:stop]]
        positive = torch.sigmoid((source_embedding * target_embedding).sum(dim=1))
        negative_embedding = embedding[negative_nodes[start:stop]]
        negative = torch.sigmoid((source_embedding.unsqueeze(1) * negative_embedding).sum(dim=2))
        positive_sum = positive_sum + (positive - 1.0).square().sum()
        negative_sum = negative_sum + negative.square().sum()
    return positive_sum / source.numel() + negative_sum / negative_nodes.numel()


def run(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    seed_everything(args.seed)
    payload = load_label_free_bundle(args.bundle)
    num_nodes = int(payload["x"].shape[0])
    support = load_normal_support(args.support, num_nodes=num_nodes)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=num_nodes)
    device = torch.device(args.device)
    x = payload["x"].to(device)
    edge_index = payload["edge_index"].to(device)
    positive = torch.as_tensor(support, dtype=torch.long, device=device)
    near, far = distance_partition(
        edge_index, positive, num_nodes, threshold=args.distance_threshold
    )
    negative_nodes = sample_non_neighbors(
        edge_index,
        num_nodes,
        samples_per_edge=args.negative_samples,
        seed=args.seed,
    ).to(device)
    model = TwoLayerGCN(x.shape[1], args.hidden_dimension).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses: list[float] = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embedding, probability = model(x, edge_index)
        pu_loss = distance_aware_pu_loss(
            probability,
            positive,
            near,
            far,
            near_prior=args.near_prior,
            far_prior=args.far_prior,
        )
        structural_loss = structural_regularizer_chunked(embedding, edge_index, negative_nodes)
        loss = pu_loss + args.alpha * structural_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} loss={losses[-1]:.10f}", flush=True)
    model.eval()
    with torch.no_grad():
        _, normal_probability = model(x, edge_index)
        scores = 1.0 - normal_probability
    write_scores(args.scores, score_nodes, scores[torch.as_tensor(score_nodes, device=device)].cpu().numpy())
    write_evidence(
        args.evidence,
        {
            "method": "Structure-aware PU-GNN",
            "source_kind": "PAPER_DERIVED",
            "source_identity": "paper:arxiv:2310.13538v1",
            "dataset_bundle_sha256": sha256(args.bundle),
            "support_manifest_sha256": sha256(args.support),
            "contains_ground_truth_labels": False,
            "positive_class_semantics": "frozen normal support",
            "anomaly_score": "1 - normal-positive probability",
            "paper_defaults": PAPER_DEFAULTS,
            "epochs": args.epochs,
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "minimum_loss": min(losses),
            "score_count": int(score_nodes.size),
            "finite_scores": bool(torch.isfinite(scores).all()),
            "near_unlabeled_count": int(near.numel()),
            "far_unlabeled_count": int(far.numel()),
            "elapsed_seconds": time.time() - started,
            "seed": args.seed,
            "device": str(device),
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--bundle", required=True)
    result.add_argument("--support", required=True)
    result.add_argument("--score-nodes", required=True)
    result.add_argument("--scores", required=True)
    result.add_argument("--evidence", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cpu")
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--hidden-dimension", type=int, default=16)
    result.add_argument("--lr", type=float, default=0.01)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--alpha", type=float, default=0.01)
    result.add_argument("--distance-threshold", type=int, default=3)
    result.add_argument("--negative-samples", type=int, default=50)
    result.add_argument("--near-prior", type=float, default=0.6)
    result.add_argument("--far-prior", type=float, default=0.3)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
