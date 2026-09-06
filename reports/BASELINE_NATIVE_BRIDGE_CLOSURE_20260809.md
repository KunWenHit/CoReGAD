# CoReGAD Primary Baseline Native Bridge Closure

Date: 2026-08-09  
Branch: `codex/primary-native-bridge-closure`  
Server workspace: `/data1/wk/codes/CoReGAD_RELEASE`  
Protocol: `STANDARD_TRANSDUCTIVE_NORMAL_ONLY`

## Outcome

This round is an **infrastructure closure with two fully gate-closed primary methods**, not a claim that every baseline has been reproduced.

- `active_total = 21`
- `primary_total = 11`
- `pu_auxiliary_total = 1`
- `external_reference_total = 9`
- `benchmark_target_cells = 21 × 8 = 168`
- `primary_execute_enabled = 2`
- `primary_native_sanity_passed = 2`
- `primary_canonical_seed0_passed = 2`

DOMINANT and OCGNN passed all six gates and are executable. The other nine primary methods remain honestly disabled. No 21×8 batch, three-seed run, broad hyperparameter search, or GitHub push was performed.

BMP is not an active baseline. Its historical mathematical core and source-search evidence remain archived with reason `EXCLUDED_NO_RECOVERABLE_OFFICIAL_SOURCE`; it is absent from active registries, launchers, the 168-cell matrix, and paper-table candidates.

## Primary closure matrix

| Method | Source | Env | Native sanity | Label audit | Canonical smoke | Score contract | execute_enabled | Validated datasets | Remaining blocker |
|---|---|---|---|---|---|---|---:|---|---|
| DOMINANT | PASS, author PyTorch commit `83fa939` | PASS | PASS, BlogCatalog, 1 epoch | PASS | PASS, Amazon seed 0 | PASS | true | Amazon | Formal full-epoch runs and the other seven target cells are not run; dense N² risk on the three large-edge/large-node datasets |
| AnomalyDAE | PASS, author commit `1199bae` | PENDING | PENDING | PENDING | PENDING | PENDING | false | — | Reproduce the frozen TensorFlow 1.10/Python 3.6 family without changing the native method |
| OCGNN | PASS, author commit `ec7d11c` | PASS | PASS, Cora, 1 epoch | PASS | PASS, Amazon seed 0 | PASS | true | Amazon | Formal full-epoch runs and the other seven target cells are not run; sparse full-batch feasibility requires real runs |
| AEGIS | PASS, GGAD official benchmark reimplementation commit `358fb4d` | PENDING | PENDING | PENDING | PENDING | PENDING | false | — | External native bridge and removal of reporting-label dependence from the training process |
| GAAN | PASS, GGAD official benchmark reimplementation commit `358fb4d` | PENDING | PENDING | PENDING | PENDING | PENDING | false | — | External native bridge; official path constructs dense N² adjacency/reconstruction |
| TAM | PASS, author commit `e82ead0` | PENDING | PENDING | PENDING | PENDING | PENDING | false | — | Native bridge and label-isolated fixed-epoch sanity; official path is dense N² |
| GAD-NR | PASS, author commit `b0a9590` | PENDING | PENDING | PENDING | PENDING | PENDING | false | — | Mechanically extract the notebook training logic into a deterministic entrypoint |
| ADA-GAD | PASS, author commit `71a3aca` | PENDING | PENDING | PENDING | PENDING | PENDING | false | — | Upstream pins Torch 1.10.1/PyG 2.2.0; create a compatible isolated family and preserve dense reconstruction behavior |
| GGAD | PASS, author commit `358fb4d` | PENDING | PENDING | PENDING | PENDING | PENDING | false | — | Remove true-label metric/selection dependence while retaining pseudo-anomaly generation |
| RHO | PASS, frozen author commit `a394d55` | PENDING | PENDING | PENDING | PENDING | PENDING | false | — | Reuse existing sparse Laplacian and official batched-NCE settings in a label-free bridge |
| GraphNC | PASS, author commit `4985138` | PENDING | PENDING | PENDING | PENDING | PENDING | false | — | Audit teacher, checkpoint, and metric paths; native launcher currently materializes dense adjacency |

`ENV_PASS` is method-gated. The shared Torch 2.4 environment existing does not automatically pass a method whose exact dependency family and native entrypoint have not been exercised.

## Source identity and HSMAD recovery

HSMAD was recovered as an official external supervised reference:

- source: `https://github.com/cozy24/HSMAD.git`
- commit: `7810e8e7dfe2143c4e4aa2ba804a0bfdc6543a5a`
- source kind: `AUTHOR_OFFICIAL`
- `git ls-remote`: PASS
- complete-history bundle verification: PASS
- server checkout: clean at the exact commit

The normal server-side GitHub pack transfer timed out even though `ls-remote` worked. A complete official clone was therefore made locally, unshallowed, verified with `git fsck`/`git bundle verify`, transferred, and cloned from the complete bundle on the server. Failed/incomplete network attempts were retained as provenance rather than deleted.

DOMINANT provenance was not silently switched. The older `kaize0409/GCN_AnomalyDetection` release is Python 2/TensorFlow and its README points to the author's PyTorch successor. The selected `kaize0409/GCN_AnomalyDetection_pytorch` commit `83fa939` has a reproducible Python 3/PyTorch entrypoint and directly exposes the native model and loss needed by the canonical bridge. Its dense structure decoder (`x @ x.T`) is preserved.

Source evidence: `benchmark/evidence/source_identity_20260809.json`.

## Environment closure

Created without mutating `/data1/wk/conda_envs/pfr_gad`:

- path: `/data1/wk/conda_envs/baseline_primary_torch24`
- Python 3.10.20
- Torch 2.4.0+cu124; CUDA 12.4
- DGL 2.4.0+cu124
- PyG 2.6.1
- NumPy 2.1.3; SciPy 1.14.1; scikit-learn 1.7.2
- `pip check`: no broken requirements

The first Conda clone failed because the shared Conda package cache did not allow writing `.partial` files. That partial environment was renamed and retained. The successful environment was created from a read-only `conda-pack` of `pfr_gad`, relocated with `conda-unpack`, and extended only in the new environment. Environment evidence is in `benchmark/evidence/environments/primary_torch24.json`.

The user-provided proxy port was recorded as `7897`; `127.0.0.1:7897` was not listening on the server. DGL data download succeeded directly. GitHub source recovery used the verified local-to-server bundle route described above.

## Closed method evidence

### DOMINANT

Native sanity used the untouched author entrypoint on BlogCatalog for one epoch in the isolated environment. It returned code 0, finite loss `4.54940`, and AUROC `0.806621522001858`. The author entrypoint has no model-seed CLI; `PYTHONHASHSEED=0` does not claim full native determinism. AUROC was reporting-only after training; there was no early stopping, checkpoint selection, or tuning. There is no paper metric for this exact one-epoch protocol, so the formal delta is `N/A`.

The canonical Amazon seed-0 smoke imported the author model and loss, read only the four-key label-free graph bundle, trained one epoch with native defaults, and wrote 10,610 finite scores. The score node IDs exactly match the frozen evaluation set. The bridge retains dense structure reconstruction and refuses to weaken it into a sparse surrogate.

Evidence: `benchmark/evidence/primary/dominant/native_bridge_validation.json`.

### OCGNN

Native sanity used the official Cora path, seed 0, GraphSAGE, and one epoch. DGL 2.4 required a mechanical compatibility patch that removes a deleted unused import and adapts the built-in dataset object's renamed fields; it does not alter architecture, loss, masks, optimizer, or scoring. The run returned code 0 with finite training loss. Its one-epoch test AUROC/AUPRC were `0.2882/0.3825`; these are startup/sanity values, not paper reproduction claims, and there is no exact one-epoch paper comparator.

The canonical Amazon seed-0 bridge replaces upstream label-derived training masks with the frozen 1,334 normal-support IDs. Center initialization, one-class loss, radius update, AdamW optimizer, GraphSAGE architecture, and score formula are imported/preserved. It produced 10,610 unique, finite, correctly oriented scores matching the frozen evaluation node set.

Evidence: `benchmark/evidence/primary/ocgnn/native_bridge_validation.json`; compatibility patch: `benchmark/patches/ocgnn_dgl24_compat.patch`.

## Dataset/resource semantics

Every active baseline has all eight `benchmark_targets`, but `native_supported_datasets` and `validated_datasets` are separate. Only two method-dataset cells are validated (`DOMINANT/Amazon`, `OCGNN/Amazon`). This is not an 8/8 runnable claim.

The only resource-status values are:

`NOT_RUN`, `PASS`, `OOM`, `OOT`, `UNSUPPORTED`, `ERROR`.

`NOT_RUN_RESOURCE_RISK` is a separate preflight flag, never an execution result. No OOM or OOT is claimed in this round. A future OOM/OOT cell must have empty metrics and the complete machine-readable evidence contract in `benchmark/resource_failure.schema.json`.

Large-graph preflight covers T-Finance, T-Social, and DGraph-Fin for all 11 primary methods. It audits dense N×N adjacency, pairwise N² work, full-edge attention, dense reconstruction targets, full-node covariance, all-pairs distance, and dense spectral decomposition. No node/edge subsampling or weakened architecture was introduced.

T-Social uses 146,211,016 directed `edge_index` entries. Approximately 73,105,508 undirected unique edges can describe the same graph; canonical identity is governed by raw hash, node count, feature shape, and node ordering rather than the approximately 2× edge-count convention alone.

Full preflight: `benchmark/evidence/primary_large_graph_preflight_20260809.json`.

## PU auxiliary and external references

- Structure-aware PU-GNN remains `PU_AUXILIARY`, `fair_ranking=false`, paper-derived, and not execute-enabled. It is not counted among the 11 primary methods.
- The nine external supervised references remain `fair_ranking=false`. Their checkouts are present, but this round did not falsely mark their environments/native sanity/canonical bridges complete.
- HSMAD source recovery is PASS and official; its method environment/native bridge remain pending.

## Tests and reproducibility checks

The server regression suite passed: **54 passed**.

Additional enforced invariants include:

- active/class totals `21/11/1/9` and exactly 168 target cells;
- BMP, PAGE, TAQ, and TAQ-GAD absent from active registry and launchers;
- six evidenced PASS gates plus a native command before execution is enabled;
- no label or CoReGAD cross-fitting argument in native commands;
- validated datasets require repository evidence and `PASS` resource status;
- label-free four-key bundles, exact canonical node order, frozen support IDs, finite scores, and higher-is-more-anomalous orientation;
- OOM/OOT metrics must be empty and formal resource evidence must exist.

The public runner validates cleanly with `active_count=21`, and exported server manifests contain exactly 168 dataset rows with no BMP entry.

## Honest boundary of this closure

This report establishes two genuine executable primary bridges and the infrastructure needed to close the remaining nine. It does **not** claim 21 baseline reproductions, 8/8 runnable support, formal paper metrics, full-epoch Amazon results, large-graph OOM, or benchmark ranking results. All unexecuted method-dataset cells remain `NOT_RUN`.
