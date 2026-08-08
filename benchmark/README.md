# Baseline Protocol

Third-party source code is intentionally kept outside this repository so that upstream licenses, Git histories, and attribution remain intact. The portable registry in [baseline_registry.yaml](baseline_registry.yaml) records 24 requested methods, verified source URLs and commits, protocol classes, and preparation status.

## Fair Normal-only Baselines

The fair group contains unsupervised methods and methods that can retain their published mechanism while training only on labeled-normal and visible unlabeled nodes. All fair runs must use the same dataset version, row order, five-fold ownership, labeled-normal support budget, evaluation mask, seed schedule, and evaluator. Real anomaly labels are prohibited from training, early stopping, model selection, pseudo-anomaly generation, and teacher construction.

AEGIS and GAAN have no author standalone repository recorded here. Their prepared source is explicitly identified as the verified reimplementation in the official GGAD benchmark repository; it is not presented as author code. GGAD's native pseudo-anomaly generation remains permitted because it does not read ground-truth anomaly labels.

## External Supervised References

BWGNN, GHRN, XGBGraph/GADBench, ConsisGAD, SpaceGNN, DSGAD, APF, SAGAD, and HSMAD retain their original limited-label or supervised training semantics. The adapter normalizes data ordering, OOF score ownership, and evaluation only. These methods have `requires_anomaly_labels: true` and `fair_ranking: false`; they are contextual references and are excluded from the fair ranking.

## Local workspace

Set `COREGAD_BASELINE_ROOT` to a directory containing `repos/`, `protocol/`, `patches/`, and `manifests/`:

```bash
export COREGAD_BASELINE_ROOT=/path/to/CoReGAD_Baselines
python "$COREGAD_BASELINE_ROOT/repos/DOMINANT/run_coregad_protocol.py" --help
```

On PowerShell:

```powershell
$env:COREGAD_BASELINE_ROOT = (Resolve-Path ".\CoReGAD_Baselines").Path
python "$env:COREGAD_BASELINE_ROOT\repos\DOMINANT\run_coregad_protocol.py" --help
```

The path is never embedded in model code. The external preparation workspace preserves each upstream commit on its original branch, adds one local `coregad-protocol` commit, and stores a rebuildable patch.

## Protocol use

The canonical graph NPZ contains `x`, `edge_index`, optional `node_id`, and evaluation-only `y`. The split manifest contains `folds`, `normal_nodes`, `unlabeled_nodes`, and one ownership value per unlabeled node. A support manifest fixes one equal-size labeled-normal support list per fold:

```json
{
  "folds": 5,
  "support_budget": 2,
  "support_nodes_by_fold": [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
}
```

Validate and create a label-free training bundle:

```bash
python benchmark/adapters/coregad_protocol_runner.py validate \
  --dataset ./data/graph.npz \
  --split-manifest ./data/split.json \
  --support-manifest ./data/support.json

python benchmark/adapters/coregad_protocol_runner.py prepare \
  --dataset ./data/graph.npz \
  --split-manifest ./data/split.json \
  --support-manifest ./data/support.json \
  --fold 0 \
  --output ./outputs/baseline/fold_0_input.npz
```

After every baseline writes `fold_0.npz` through `fold_4.npz` with `node_id` and `score`, collect and evaluate:

```bash
python benchmark/adapters/coregad_protocol_runner.py collect \
  --dataset ./data/graph.npz \
  --split-manifest ./data/split.json \
  --score-dir ./outputs/baseline/folds \
  --output ./outputs/baseline/oof_scores.npz

python benchmark/adapters/coregad_protocol_runner.py evaluate \
  --dataset ./data/graph.npz \
  --oof-scores ./outputs/baseline/oof_scores.npz
```

AUPRC is primary; AUROC, Recall@K, Precision@K, and NDCG@K are secondary. See [NORMAL_ONLY_OOF_PROTOCOL.md](protocol/NORMAL_ONLY_OOF_PROTOCOL.md) for the normative contract.

No full baseline result is claimed by this release. `ADAPTED_STATIC_TESTED` means that source identity, portable wrapper behavior, label isolation, ownership checks, and evaluator wiring were tested locally.
