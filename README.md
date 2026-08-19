# CoReGAD

Controlled Partial Residualization of Structural–Spectral Evidence
for Normal-Only Graph Anomaly Detection

CoReGAD is a normal-only graph anomaly detection framework that separates
attribute-driven normality learning from graph-spectral anomaly evidence. It
learns a cross-fitted normality core and applies controlled structural
residualization to spectral discrepancies, suppressing low-order structural
shortcuts while retaining useful relation-dependent anomaly cues.

## Overview

- No labeled anomalies.
- No external teacher.
- No real anomaly labels during training.
- Five-fold out-of-fold evaluation support.
- Stage-wise learning with explicit freezing boundaries.
- Controlled residual strength fixed at `0.75` in the released model.

Training consumes only two declared sets: labeled normal nodes and unlabeled
visible nodes. The training command deliberately does not load the `y` array.
Labels are opened only by the separate evaluation command after OOF scores have
been materialized.

## Method

CoReGAD first trains an attribute-only normality core independently for every
outer fold. Graph Context Shaping affects this training stage, while inference
through the frozen core remains attribute-only. A shared global spectral
reference combines low-, band-, and high-frequency graph components. Two
node-varying discrepancies compare the frozen embedding and decision with that
reference.

Three low-order statistics—log degree, visible-neighborhood support ratio, and
local embedding variation—feed a five-fold cross-fitted nuisance estimator.
The released detector subtracts exactly `0.75` times the structure-predictable
spectral component. The residual energy head receives only the resulting two
spectral channels; the reliability gate receives only the three structural
statistics. Their bounded correction is added to the base anomaly logit with a
fixed coefficient of one.

See [docs/METHOD.md](docs/METHOD.md) for the equations and stage boundaries.

## Installation

```bash
python -m venv .venv
python -m pip install -e ".[test]"
```

For CUDA, install the appropriate PyTorch build for your system before the
editable install.

## Dataset preparation

The canonical input is an uncompressed or compressed NumPy archive containing:

- `x`: float node-feature matrix with shape `[N, D]`;
- `edge_index`: integer edge array with shape `[2, E]`;
- optional `node_id`: stable integer node IDs;
- `y`: evaluation-only labels, which the training loader never accesses.

Prepare a five-fold split from separate normal and unlabeled node-ID files:

```bash
python scripts/prepare_data.py \
  --data ./data/graph.npz \
  --normal-nodes ./data/normal_nodes.txt \
  --unlabeled-nodes ./data/unlabeled_nodes.txt \
  --output ./data/split.json
```

## Training

```bash
python scripts/train_coregad.py \
  --data ./data/graph.npz \
  --split ./data/split.json \
  --output ./outputs/coregad \
  --device cpu \
  --seed 0
```

For `T-Social` and `DGraph-Fin`, use the mathematically equivalent scalable
engine:

```bash
python scripts/train_coregad.py --engine scalable --scalable-cache ./cache ...
```

The normality core, nuisance estimator, residual energy head, and reliability
gate are optimized in separate stages. `--smoke` reduces every stage to one
epoch and is only an installation check; it is not a reproduction setting.

## Evaluation

```bash
python scripts/evaluate_coregad.py \
  --data ./data/graph.npz \
  --scores ./outputs/coregad/oof_scores.npz \
  --output ./outputs/coregad/metrics.json
```

The primary metric is AUPRC. AUROC, Recall@K, Precision@K, and NDCG@K are
reported as secondary metrics.

## Reproduction protocol

The frozen configuration is [configs/coregad.yaml](configs/coregad.yaml).
Reproduction requires the same dataset version, node order, normal support,
five-fold ownership, fixed final epochs, and metric implementation. Model
selection, early stopping, and hyperparameter changes based on held-out anomaly
labels are prohibited. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository structure

```text
coregad/          formal model, data, training, and evaluation modules
configs/          frozen released configuration
scripts/          data preparation, training, and evaluation entry points
tests/            contract, leakage, and behavior-parity tests
docs/             method, data, and reproduction documentation
benchmark/        baseline registry and protocol adapters
```

## Baseline protocol

Fair baselines use the same normal/unlabeled support and OOF ownership.
Methods whose original training contract requires anomaly labels are reported
only as external supervised references and are excluded from the fair ranking.
See [benchmark/README.md](benchmark/README.md).

Exact public dataset identities (hashes and fixed support/split budgets, without
restricted node-ID lists) are in [reproducibility/](reproducibility/README.md).

## Citation

Citation will be added after publication.

## Acknowledgements

CoReGAD builds on the graph anomaly detection and normal-unlabeled learning
literature. Third-party baseline repositories remain separate so their
licenses, histories, and upstream attribution are preserved.
