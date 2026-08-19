# STANDARD_TRANSDUCTIVE_NORMAL_ONLY

## Main-table contract

The main benchmark uses shared graph and supervision with native optimization.
Every method receives the same canonical feature matrix, adjacency, node order,
model-seed schedule, and evaluator. Methods that need known-normal supervision
receive the exact frozen `normal_support_ids` and budget. Fully unsupervised
methods do not receive support labels merely for uniformity.

Complete graph covariates (`X` and `A`) are visible during training. This is the
standard transductive setting: using evaluation-node features or topology is not
label leakage. Ground-truth anomaly labels remain unavailable to normal-only
training, pseudo-anomaly generation, teacher construction, hyperparameter
selection, early stopping, checkpoint selection, and threshold selection.

Adapters may only perform canonical data conversion, preserve node ordering,
inject the frozen normal support when the native method requires it, set the
seed, normalize output orientation, and connect the common evaluator. They may
not change a native architecture, loss, reconstruction target, graph, or
message-passing scope.

## Native optimization

Baselines keep their published training semantics. No CoReGAD-style outer
ownership or cross-fitting wrapper is added. A native transductive method trains
once on the full canonical graph and produces one anomaly score for every
evaluation node. CoReGAD retains its own internal cross-fitting because that is
part of CoReGAD rather than a benchmark-side fairness device.

## Classes and ranking

- `PRIMARY_TRANSDUCTIVE_NORMAL_ONLY` methods may be included in the fair
  normal-only ranking only when anomaly labels are absent from training.
- `PU_AUXILIARY` is reported separately and is never placed in the primary fair
  ranking.
- `EXTERNAL_SUPERVISED_REFERENCE` keeps the supervision required by its paper
  and is always marked `fair_ranking: false`.

## Evaluation boundary

Training outputs exactly two aligned fields: `node_id` and `anomaly_score`, with
higher values meaning more anomalous. Labels are opened only by the independent
evaluator. AUPRC is primary; AUROC, Recall@K, Precision@K, and NDCG@K are
secondary, with K equal to the anomaly count in the fixed evaluation set.

## Secondary robustness protocol

`STRICT_OOF` remains available as a secondary robustness protocol. Existing
CoReGAD strict-isolation code and tests are preserved. It is not a requirement
for main-table baselines, and strict adapters are added only method by method
when explicitly implemented and audited.
