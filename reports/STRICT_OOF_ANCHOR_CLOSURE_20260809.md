# CoReGAD Strict-OOF Anchor Closure — 2026-08-09

## Outcome

The pre-fix source did contain a held-out optimization path. In commit
`72ab80a3f2c2dc3e167cbe3d5cd0b931b457f961`, the context-continuation anchor
compared student and frozen-teacher logits over all nodes. The student encoder
was trainable, so outer-held-out attributes could influence its gradients even
though held-out nodes were absent from the normal/unlabeled BCE and visible
graph-context rows. No anomaly label was read, but this behavior did not satisfy
the published strict-OOF statement that the held-out fold is excluded from
optimization.

The path is closed in the strict source snapshot
`da356a769c388ac0e80798b3132474e3d1a1b23d`. The anchor now reads only the
single canonical `visible_index = normal_nodes + training_unlabeled_nodes`, and
the same index supplies Graph Context Shaping. A runtime contract rejects any
intersection between that index and `heldout_nodes` before optimization starts.

## Active source and environment provenance

- Git repository root: `/data1/wk/codes/CoReGAD_RELEASE/coregad`
- Python project root: `/data1/wk/codes/CoReGAD_RELEASE/coregad`
- Project metadata: `/data1/wk/codes/CoReGAD_RELEASE/coregad/pyproject.toml`
- Strict branch: `codex/strict-oof-anchor-closure`
- Runtime Python: `/data1/wk/conda_envs/pfr_gad/bin/python`
- Imported package: `/data1/wk/codes/CoReGAD_RELEASE/coregad/coregad/__init__.py`
- Imported normality module: `/data1/wk/codes/CoReGAD_RELEASE/coregad/coregad/training/normality.py`
- Editable installation command: `python -m pip install -e ".[test]"`
- Import-path evidence: `/data1/wk/codes/CoReGAD_RELEASE/logs/strict_oof/strict_oof_import_path.log`
- Installation evidence: `/data1/wk/codes/CoReGAD_RELEASE/logs/strict_oof/strict_oof_editable_install.log`

Neither the pre-fix nor post-fix import resolved through `build/lib`, a wheel,
or site-packages. `build/lib` was not edited.

## Exact code changes

In `coregad/training/normality.py`:

1. Construct `visible_index` once after all ownership tensors are placed on the
   selected device.
2. Reject `visible_index ∩ heldout_nodes != ∅` with a non-optional runtime
   error.
3. Build graph-context visible rows from that same index.
4. Slice both student and frozen-teacher anchor logits by `visible_index`.
5. Retain the frozen pre-context teacher in the fold result solely so posthoc
   diagnostics can export its held-out inference score.

In `coregad/training/pipeline.py`, the code exports held-out-only diagnostics
after training: pre-context teacher/base score, post-context frozen normality
score, spectral discrepancy, controlled spectral residual, graph correction,
and final anomaly score. They are tagged
`used_for_training_or_selection=false`; no diagnostic is consumed by a loss,
checkpoint decision, or version selector.

No mathematical module was changed. Git blob identities are equal between the
legacy reference and strict snapshot for:

- `coregad/models/spectral_reference.py`
- `coregad/models/structural_residualization.py`
- `coregad/models/coregad.py`
- `coregad/models/reliability.py`
- `coregad/models/graph_context.py`, including the mathematical definition of
  `anchor_preservation_loss`
- `coregad/training/residual.py`

Accordingly, alpha remains `0.75`; spectral reference and discrepancy,
structural statistics, energy, reliability, gamma, anchor weight, context loss,
teacher/student initialization, optimizers, learning rates, weight decay, and
the `200 / 200 / 300` epoch schedule are unchanged.

## Strict-OOF causality tests

`tests/test_strict_oof_anchor.py` adds four tests:

1. The ownership/anchor-scope test records each anchor call and requires an
   input length of `len(normal_nodes) + len(training_unlabeled_nodes)`, not `N`;
   an overlapping held-out set must fail before training.
2. A deterministic CPU pair changes only held-out attributes to fixed extreme
   values and requires exact `torch.equal` identity for every trained student
   encoder, normality-head, and GraphContextShaping tensor.
3. A one-step context-continuation pair imposes the same exact state-identity
   requirement, directly testing the permitted gradient-isolation fallback.
4. Diagnostic arrays must be held-out-only, align with final OOF scores, and be
   marked as unavailable to training or selection.

No tolerance fallback was needed: CPU trainable states were bitwise identical.
The complete public suite passed without deleting or weakening an existing
test:

```text
.....................................                                    [100%]
37 passed in 3.27s
```

The test log is
`/data1/wk/codes/CoReGAD_RELEASE/logs/strict_oof/strict_oof_pytest.log`.

## Snapshot identities

### Immutable legacy reference

- Commit: `72ab80a3f2c2dc3e167cbe3d5cd0b931b457f961`
- Detached worktree: `/data1/wk/codes/CoReGAD_RELEASE/worktrees/LEGACY_REFERENCE`
- Archive: `/data1/wk/codes/CoReGAD_RELEASE/snapshots/LEGACY_REFERENCE_72ab80a.tar.gz`
- Archive SHA256: `bc7c76a2089189585ea79159fa90dbfd5dc72619a5b13833f4b5cef4ad6d9379`

### Instrumented legacy executable

For a causal comparison with identical diagnostic code, the executable legacy
tree is derived from the strict instrumentation and reverts only the two anchor
arguments to unsliced all-node logits.

- Commit: `f54678e532d872a52fc2958e40279d3c74bed9c4`
- Worktree: `/data1/wk/codes/CoReGAD_RELEASE/worktrees/LEGACY_EXECUTABLE`
- Archive: `/data1/wk/codes/CoReGAD_RELEASE/snapshots/LEGACY_EXECUTABLE_f54678e.tar.gz`
- Archive SHA256: `e1ebbc8b2a94c64b661cbd9003783308c3192377c43edd8a23fb4ee0dad7ad7a`
- Model-tree diff versus strict: only `coregad/training/normality.py`
- Behavioral difference: all-node anchor versus visible-only anchor

### Strict source snapshot

- Commit: `da356a769c388ac0e80798b3132474e3d1a1b23d`
- Worktree: `/data1/wk/codes/CoReGAD_RELEASE/coregad`
- Archive: `/data1/wk/codes/CoReGAD_RELEASE/snapshots/STRICT_OOF_da356a7.tar.gz`
- Archive SHA256: `a9589b23a1857cd9685756a87c8132eef6f6f98f32a53debf4f32ffc7ef8ab89`

The report itself may be committed after this code snapshot; that documentation
commit does not alter the snapshot used by tests and launcher preflight.

## Exact eight-dataset seed-0 audit launcher

The source contract and all eight exact split/support manifests passed
preflight. A read-only `--verify-raw-hash` pass also matched all eight raw
assets to their recovered SHA256 identities; its evidence is
`/data1/wk/codes/CoReGAD_RELEASE/logs/strict_oof/seed0_launcher_preflight_raw_verified.json`.
The launcher has no model-seed, epoch, optimizer, learning-rate,
support, split, evaluator, or node-order override. The model seed is fixed to
zero. Outputs are isolated under:

- `outputs/oof_anchor_audit/legacy_seed0/<dataset>/seed_0/`
- `outputs/oof_anchor_audit/strict_seed0/<dataset>/seed_0/`

The default is preparation-only and cannot train:

```bash
cd /data1/wk/codes/CoReGAD_RELEASE
bash coregad/scripts/run_oof_anchor_audit_seed0.sh preflight --all
bash coregad/scripts/run_oof_anchor_audit_seed0.sh run-pair --dataset Amazon --device cuda:0
```

Future user-authorized single-dataset execution requires the explicit gate:

```bash
bash coregad/scripts/run_oof_anchor_audit_seed0.sh \
  run-pair --dataset Amazon --device cuda:0 --execute
```

The prepared, but unexecuted, exact eight-dataset launcher is:

```bash
bash coregad/scripts/run_oof_anchor_audit_all_seed0.sh --device cuda:0 --execute
```

Before either model process starts, a separate data-preparation process verifies
the raw hash and writes a bundle whose only data fields are `features`,
`edge_index`, and `node_id`. It imports no model or optimizer. Both model
processes consume that same bundle and reject any `y`/label key. Labels are
extracted by `scripts/analyze_oof_anchor_audit.py prepare-labels` in a separate
evaluation-only action after OOF scores exist.

The launcher reuses the recovered dataset manifests, exact seed-0 folds,
support IDs, node order, standard/scalable engine rule, and public evaluator.
It refuses nonempty target output directories and does not touch historical
champion artifacts. Historical seed-0 outputs were discovered under the real
`RefugeGAD_H1_OPT/project/outputs` phase trees and are read-only provenance, not
launcher targets.

## Posthoc impact analysis

`scripts/analyze_oof_anchor_audit.py` is prepared but was not run. Once the
future pair outputs exist, it reports per dataset:

- legacy/strict AUPRC and AUROC with deltas;
- pre-context, post-context, and final AUPRC with stage deltas;
- graph-correction mean/std;
- spectral-discrepancy and controlled-residual norm diagnostics;
- Pearson, Spearman, normalized rank displacement, and top-1% Jaccard;
- strict final gain over its attribute/pre-context base.

The eight-dataset summary reports mean/median delta AUPRC, the worst dataset,
and mean/median strict gain over the attribute base. Labels are posthoc-only and
cannot feed training, tuning, rescue selection, or version choice.

## Ordered rescue designs — not implemented

R1 keeps five-fold strict OOF. If a label-free protocol decision finds visible
teacher preservation insufficient, it may test a minimal stronger preservation
term over visible representations or initialized/current encoder parameters.
Held-out covariates remain absent from every loss; no new detection module is
added.

R2 defines a separate ten-fold strict protocol so each fold trains on 90% of
unlabeled covariates and excludes its owned 10% from every optimization path.
Manifests must be frozen before evaluation, and every fair baseline must use the
same ten-fold ownership. R2 results cannot be mixed with five-fold results.

R3 is an explicitly transductive normal-only protocol. Full unlabeled
covariates may be used, but the strict-heldout-exclusion claim must be abandoned
and every fair baseline must receive the same access. The old all-node anchor
can never be called strict OOF.

The fixed order is R1, R2, R3. This task implements and selects none of them.
The detailed design is in `docs/STRICT_OOF_RESCUE_PROTOCOL.md`.

## Remaining leakage or protocol ambiguity

- No known held-out optimization path remains in the audited strict normality
  continuation, and the causality tests cover ownership, multi-step state
  invariance, and one-step gradient/update invariance.
- Held-out attributes legitimately affect their own inference scores; inference
  invariance is not claimed or required.
- DGL source containers may deserialize a label payload in the isolated data
  preparation process, but that process imports no training model and never
  indexes the label field. The serialized training bundle contains no label key,
  and the training process mechanically rejects one.
- The numerical legacy-vs-strict impact is unknown because no seed-0 pair, full
  eight-dataset training, three-seed run, or rescue experiment was executed.
- Large-graph resource feasibility for the future paired retraining has not been
  re-established by this preparation task; the existing scalable graph engine
  is selected for T-Social and DGraph-Fin, but formal execution remains a future
  user-authorized operation.

No GitHub remote was changed, no branch was pushed, no tag was created, and no
formal audit-training output was produced.
