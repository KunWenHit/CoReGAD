"""Direct canonical score exporter for GADBench's XGBGraph detector."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from benchmark.native_bridges.common import git_head, load_score_nodes, seed_everything, sha256, write_evidence, write_scores
from benchmark.native_bridges.supervised_common import load_supervised_reference


def load_gin_noparam(source: Path):
    spec = importlib.util.spec_from_file_location("coregad_gadbench_gnn", source / "models" / "gnn.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load GADBench models/gnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GIN_noparam


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    contract = load_supervised_reference(args.bundle, args.label_source, trial=args.trial)
    score_nodes = load_score_nodes(args.score_nodes, num_nodes=contract["payload"]["x"].shape[0])
    source = Path(args.source).resolve()
    device = torch.device(args.device)
    graph = contract["graph"].to(device)
    started = time.time()
    GINNoParam = load_gin_noparam(source)
    aggregator = GINNoParam(num_layers=args.graph_layers, agg=args.aggregation, init_eps=-1).to(device)
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        graph_features = aggregator(graph).cpu().numpy()
    labels = contract["labels"].numpy()
    train_mask = contract["train_mask"].numpy()
    val_mask = contract["val_mask"].numpy()
    anomaly_weight = float((labels[train_mask] == 0).sum() / (labels[train_mask] == 1).sum())
    sample_weight = np.where(labels[train_mask] == 0, 1.0, anomaly_weight)
    classifier_args = {
        "n_estimators": args.n_estimators,
        "tree_method": "gpu_hist" if device.type == "cuda" else "hist",
        "eval_metric": "aucpr",
        "random_state": args.seed,
        "subsample": 1.0,
        "n_jobs": args.cpu_threads,
    }
    if device.type == "cuda":
        classifier_args["gpu_id"] = device.index
    classifier = xgb.XGBClassifier(**classifier_args)
    classifier.fit(
        graph_features[train_mask],
        labels[train_mask],
        sample_weight=sample_weight,
        eval_set=[(graph_features[val_mask], labels[val_mask])],
        verbose=False,
    )
    scores = classifier.predict_proba(graph_features)[:, 1]
    write_scores(args.scores, score_nodes, scores[score_nodes])
    write_evidence(
        args.evidence,
        {
            "method":"GADBench / XGBGraph","source_kind":"AUTHOR_OFFICIAL",
            "official_commit":"f9aa021ce9b6c6580427fb633b596843be76ddc6","checkout_head":git_head(source),
            "dataset_bundle_sha256":sha256(args.bundle),"label_source":contract["label_source"],"fair_ranking":False,
            "supervision":{"trial":args.trial,"train_nodes":int(train_mask.sum()),"validation_nodes":int(val_mask.sum()),"validation_metric":"AUPRC"},
            "native_contract":{"graph_aggregator":"models/gnn.py:GIN_noparam","graph_layers":args.graph_layers,"aggregation":args.aggregation,"classifier":"xgboost.XGBClassifier","n_estimators":args.n_estimators},
            "real_boosting_rounds":args.n_estimators,"expanded_feature_dimension":int(graph_features.shape[1]),
            "score_count":int(score_nodes.size),"finite_scores":bool(np.isfinite(scores).all()),"higher_is_more_anomalous":True,
            "seed":args.seed,"device":str(device),"elapsed_seconds":time.time()-started,
        },
    )


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
    result.add_argument("--graph-layers", type=int, default=2)
    result.add_argument("--aggregation", default="mean")
    result.add_argument("--n-estimators", type=int, default=100)
    result.add_argument("--cpu-threads", type=int, default=16)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
