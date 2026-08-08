# CoReGAD Normal-only Five-fold OOF Protocol

## Scope

Fair baselines may train with labeled-normal nodes and visible unlabeled nodes only. Ground-truth anomaly labels are unavailable to training, model selection, early stopping, pseudo-anomaly generation, teachers, and hyperparameter selection. Unsupervised methods may ignore labeled-normal nodes.

External supervised or limited-label references keep their published training semantics. They use the common node order and evaluator, but are marked `requires_anomaly_labels: true` and `fair_ranking: false`.

## Ownership

The split manifest fixes five outer folds. For fold `f`, an unlabeled node owned by `f` is omitted from that fold's training ownership and receives its final score only from the fold-`f` model. Every evaluated unlabeled node must receive exactly one score. Labeled-normal nodes are disjoint from unlabeled nodes.

The support manifest fixes the same number of unique labeled-normal support nodes in every fold. Dataset version, node ordering, graph, fold ownership, support budget, evaluation mask, random seeds, and metric implementation must be identical across methods.

## Canonical files

The graph is a compressed NPZ with `x`, `edge_index`, optional `node_id`, and evaluation-only `y`. Training bundles intentionally omit `y` and contain `labeled_normal_nodes`, `support_normal_nodes`, `training_unlabeled_nodes`, `train_visible_nodes`, `score_nodes`, and `owner_fold`.

Each method writes `fold_0.npz` through `fold_4.npz`, containing aligned `node_id` and `score` arrays for exactly the nodes owned by that fold. The collector rejects missing, duplicated, extra, non-finite, or mis-owned scores.

## Metrics

AUPRC is primary. AUROC, Recall@K, Precision@K, and NDCG@K are secondary, with `K` equal to the number of anomalies in the fixed evaluation mask. Labels are loaded only by the final evaluator after OOF collection.

## Adapter boundary

`run_coregad_protocol.py` in each repository delegates to the shared adapter. It validates manifests, creates label-free fold bundles, validates OOF ownership, and evaluates scores. Method-specific loaders may translate a fold bundle into the upstream representation, but must not alter the upstream model or objective except for the explicitly documented Normal-only restriction.

The `command` action is dry-run by default. It executes training only when `--allow-training` is explicitly supplied; this prevents accidental local full-dataset runs during preparation.
