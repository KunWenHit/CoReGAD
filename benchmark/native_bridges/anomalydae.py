"""CPU TensorFlow-v1 compatibility bridge for author-official AnomalyDAE."""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import scipy.sparse as sp
import tensorflow as tensorflow_v2
import torch

from benchmark.native_bridges.common import git_head, load_label_free_bundle, load_score_nodes, seed_everything, sha256, write_evidence, write_scores


def install_tensorflow_v1_alias():
    tensorflow_v2.compat.v1.disable_v2_behavior()
    tf = tensorflow_v2.compat.v1
    # Mechanical replacements for the three tf.contrib.layers helpers used by
    # the upstream 2019 implementation.
    def legacy_bias_add(value):
        dimension = value.get_shape().as_list()[-1]
        bias = tf.Variable(tf.zeros([dimension]), name="biases")
        return tf.nn.bias_add(value, bias)

    tf.contrib = types.SimpleNamespace(
        layers=types.SimpleNamespace(
            xavier_initializer=tf.glorot_uniform_initializer,
            l2_regularizer=lambda scale: (lambda value: scale * tf.nn.l2_loss(value)),
            bias_add=legacy_bias_add,
        )
    )
    sys.modules["tensorflow"] = tf
    return tf


def define_flag_once(flags, name: str, value, help_text: str) -> None:
    if name in flags.FLAGS:
        setattr(flags.FLAGS, name, value)
        return
    if isinstance(value, int):
        flags.DEFINE_integer(name, value, help_text)
    else:
        flags.DEFINE_float(name, value, help_text)


def run(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise ValueError("the isolated TensorFlow 1.x compatibility path is CPU-only")
    seed_everything(args.seed)
    tf = install_tensorflow_v1_alias()
    tf.reset_default_graph()
    flags = tf.app.flags
    for name, value in {
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "learning_rate": args.lr,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "features": 1,
        "alpha": args.alpha,
        "eta": args.eta,
        "theta": args.theta,
    }.items():
        define_flag_once(flags, name, value, name)
    if not flags.FLAGS.is_parsed():
        flags.FLAGS([sys.argv[0]])
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source / "src"))
    from constructor import get_placeholder, update
    from model import AnomalyDAE
    from optimizer import OptimizerDAE
    from preprocessing import preprocess_graph, sparse_to_tuple

    payload = load_label_free_bundle(args.bundle)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=payload["x"].shape[0])
    node_count = int(payload["x"].shape[0])
    row = payload["edge_index"][0].numpy()
    column = payload["edge_index"][1].numpy()
    adjacency = sp.coo_matrix((np.ones(row.size, dtype=np.float32), (row, column)), shape=(node_count, node_count)).tocsr()
    adjacency.data[:] = 1.0
    features_matrix = sp.csr_matrix(payload["x"].numpy())
    features = sparse_to_tuple(features_matrix.tocoo())
    adjacency_normalized = preprocess_graph(adjacency)
    adjacency_label = sparse_to_tuple((adjacency + sp.eye(node_count, dtype=np.float32)).tocoo())
    placeholders = get_placeholder()
    model = AnomalyDAE(
        placeholders,
        int(features_matrix.shape[1]),
        node_count,
        int(features[1].shape[0]),
        decoder_act=[tf.nn.sigmoid, lambda value: value],
    )
    optimizer = OptimizerDAE(
        preds_attribute=model.attribute_reconstructions,
        labels_attribute=tf.sparse_tensor_to_dense(placeholders["features"]),
        preds_structure=model.structure_reconstructions,
        labels_structure=tf.sparse_tensor_to_dense(placeholders["adj_orig"]),
        alpha=args.alpha,
        eta=args.eta,
        theta=args.theta,
    )
    session_config = tf.ConfigProto(device_count={"GPU": 0})
    session_config.intra_op_parallelism_threads = args.cpu_threads
    session_config.inter_op_parallelism_threads = max(1, args.cpu_threads // 4)
    losses: list[float] = []
    structure_losses: list[float] = []
    attribute_losses: list[float] = []
    reconstruction_error = None
    started = time.time()
    with tf.Session(config=session_config) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(args.iterations):
            train_loss, structure_loss, attribute_loss, reconstruction_error = update(
                model,
                optimizer,
                session,
                adjacency_normalized,
                adjacency_label,
                features,
                placeholders,
                adjacency,
            )
            losses.append(float(train_loss))
            structure_losses.append(float(structure_loss))
            attribute_losses.append(float(attribute_loss))
            if epoch == 0 or epoch + 1 == args.iterations or (epoch + 1) % 10 == 0:
                print(
                    f"epoch={epoch + 1} loss={losses[-1]:.10f} structure={structure_losses[-1]:.10f} attribute={attribute_losses[-1]:.10f}",
                    flush=True,
                )
    if reconstruction_error is None:
        raise RuntimeError("AnomalyDAE training did not produce reconstruction errors")
    scores = np.asarray(reconstruction_error, dtype=np.float64)
    write_scores(args.scores, score_nodes, scores[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method":"AnomalyDAE","source_kind":"AUTHOR_OFFICIAL",
            "official_commit":"1199bae1820cd923efb5a6ec2d32ab758968f465","checkout_head":git_head(source),
            "dataset_bundle_sha256":sha256(args.bundle),"contains_ground_truth_labels":False,
            "runtime":{"tensorflow":tensorflow_v2.__version__,"api":"tensorflow.compat.v1","device":"cpu"},
            "schedule_source":"upstream BlogCatalog dataset default used as fixed label-free fallback",
            "iterations":args.iterations,"real_optimizer_steps":args.iterations,
            "alpha":args.alpha,"eta":args.eta,"theta":args.theta,"hidden1":args.hidden1,"hidden2":args.hidden2,
            "first_loss":losses[0],"last_loss":losses[-1],"minimum_loss":min(losses),
            "score_count":int(score_nodes.size),"finite_scores":bool(np.isfinite(scores).all()),"higher_is_more_anomalous":True,
            "seed":args.seed,"elapsed_seconds":time.time()-started,
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
    result.add_argument("--iterations", type=int, default=180)
    result.add_argument("--hidden1", type=int, default=256)
    result.add_argument("--hidden2", type=int, default=128)
    result.add_argument("--lr", type=float, default=0.001)
    result.add_argument("--alpha", type=float, default=0.7)
    result.add_argument("--eta", type=float, default=5.0)
    result.add_argument("--theta", type=float, default=40.0)
    result.add_argument("--cpu-threads", type=int, default=16)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
