# CoReGAD Baseline Transductive Protocol Closure — 2026-08-09

> Historical report, superseded by the Primary Native Bridge Closure. Its
> 22/176 inventory describes the earlier closure state only. The active
> inventory is now 21 methods / 168 targets; BMP is excluded provenance.

## Executive status

This closure freezes the protocol, active inventory, source provenance, dataset
identity, label-isolated adapter boundary, evaluator, environment plan, and
preparation-only launchers. It does **not** claim that the 22 methods have been
natively reproduced. Formal seed-0 execution is intentionally blocked until
each native bridge and native sanity gate passes.

- Working tree: `/data1/wk/codes/CoReGAD_RELEASE/coregad`
- Baseline root: `/data1/wk/codes/CoReGAD_RELEASE/baselines`
- Branch: `codex/transductive-baseline-closure`
- Base HEAD: `665598a46849c5734399125c18acf826bb07c668`
- Active inventory: exactly 22
- Declared dataset contracts: 22 × 8 = 176
- Formal output files created by this task: 0
- Regression result: 45 tests passed
- CoReGAD model/config diff: empty
- GitHub push: not performed

## 1. Main-table protocol

The main-table protocol is `STANDARD_TRANSDUCTIVE_NORMAL_ONLY`: shared graph
and supervision, native optimization. Every fair normal-only method receives
the same canonical `X`, `A`, node order, fixed normal-support IDs when the
native method needs them, normal-label budget, seed schedule, and independent
evaluator. A natively unsupervised method is not forced to consume support
labels.

The complete unlabeled graph covariates may be visible during training. Seeing
evaluation-node features or topology in a native transductive method is not
label leakage. Ground-truth anomaly labels are forbidden from training,
pseudo-anomaly construction, tuning, early stopping, model selection, teacher
construction, and threshold selection. The training bundle contains `x`,
`edge_index`, `node_id`, optional frozen normal support, full-graph training
rows, and evaluation node IDs; it does not contain `y` or ownership/fold data.

Every adapter must emit exactly `node_id,anomaly_score`, with larger scores
meaning more anomalous. Only the separate evaluator opens `y` and computes
AUPRC (primary), AUROC, Recall@K, Precision@K, and NDCG@K.

## 2. Why CoReGAD cross-fitting is not imposed on baselines

CoReGAD's internal cross-fitting is part of CoReGAD's algorithm. Wrapping a
native baseline in CoReGAD-style five-fold ownership would change its training
graph, reconstruction target, message passing, optimization, or score
construction. That would no longer be a faithful baseline. The main table
therefore preserves each baseline's native transductive semantics and uses the
shared covariates/supervision/evaluator as the fairness boundary.

## 3. STRICT_OOF is secondary robustness

`STRICT_OOF` and all completed CoReGAD strict-isolation tests remain intact,
but strict OOF is now a secondary robustness protocol. It is not required for
main-table baselines. The launcher defaults to `--protocol transductive` and
rejects `--protocol strict_oof` for methods without a future explicit adapter.

## 4. Exact active inventory and provenance

Dataset code `8/8` means the exact ordered set: Amazon, Weibo, YelpChi,
Tolokers, T-Finance, Elliptic, T-Social, and DGraph-Fin. “Declared” is a data
contract, not proof that native execution has passed.

| Method | Class / fair | Source URL | Commit / identity | Source kind | License | Env | Adapter | Runnable | Data |
|---|---|---|---|---|---|---|---|---|---|
| DOMINANT | Primary / yes | [source](https://github.com/kaize0409/GCN_AnomalyDetection) | `04ebcf093ad7ecc28cd148c2ebea11e12f53a37c` | AUTHOR_OFFICIAL | no license file | `pygod_legacy` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| AnomalyDAE | Primary / yes | [source](https://github.com/haoyfan/AnomalyDAE) | `1199bae1820cd923efb5a6ec2d32ab758968f465` | AUTHOR_OFFICIAL | MIT | `pygod_legacy` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| OCGNN | Primary / yes | [source](https://github.com/WangXuhongCN/OCGNN) | `ec7d11c5ae2f91f4165e384131c6a8358836ff58` | AUTHOR_OFFICIAL | MIT | `ocgnn_legacy` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| AEGIS | Primary / yes | [GGAD benchmark](https://github.com/mala-lab/GGAD) | `358fb4d4b4ee5b8b195445791a9e2b6d52487f2b` | OFFICIAL_BENCHMARK_REIMPLEMENTATION | no license file | `ggad` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| GAAN | Primary / yes | [GGAD benchmark](https://github.com/mala-lab/GGAD) | `358fb4d4b4ee5b8b195445791a9e2b6d52487f2b` | OFFICIAL_BENCHMARK_REIMPLEMENTATION | no license file | `ggad` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| TAM | Primary / yes | [source](https://github.com/mala-lab/TAM-master) | `e82ead0584b9b807d4db56a9ffdf89ff94c74dfa` | AUTHOR_OFFICIAL | no license file | `tam` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| GAD-NR | Primary / yes | [source](https://github.com/Graph-COM/GAD-NR) | `b0a9590b10d0b9490e6d6cdfa60e7c50eb232c0d` | AUTHOR_OFFICIAL | no license file | `gad_nr` | notebook bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| ADA-GAD | Primary / yes | [source](https://github.com/jweihe/ADA-GAD) | `71a3aca936ccfe430e3809864f256d4bf80b22ee` | AUTHOR_OFFICIAL | Apache-2.0 | `ada_gad` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| GGAD | Primary / yes | [source](https://github.com/mala-lab/GGAD) | `358fb4d4b4ee5b8b195445791a9e2b6d52487f2b` | AUTHOR_OFFICIAL | no license file | `ggad` | label/metric path removal pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| RHO | Primary / yes | [source](https://github.com/mala-lab/RHO) | `a394d5575dea6745215b15e0453e1f925ffcc1f2` | AUTHOR_OFFICIAL | no license file | `rho` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| GraphNC | Primary / yes | [source](https://github.com/mala-lab/GraphNC) | `4985138639d96c43b85001eefa3a54eee1caebf1` | AUTHOR_OFFICIAL | no license file | `graphnc` | label/metric path removal pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| BMP | Primary / yes | [advertised, unavailable](https://github.com/Thankstaro/BMP) | `paper:doi:10.1609/aaai.v40i19.38671` | **PAPER_DERIVED** | no upstream code license | `paper_derived_torch` | math core ready; training bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| Structure-aware PU-GNN | PU auxiliary / no | [paper](https://arxiv.org/abs/2310.13538) | `paper:arxiv:2310.13538v1` | **PAPER_DERIVED** | no upstream code license | `paper_derived_torch` | math core ready; training bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| BWGNN | External supervised / no | [source](https://github.com/squareRoot3/Rethinking-Anomaly-Detection) | `de0631f039bbd19c1890b483cc01f1007f596af7` | AUTHOR_OFFICIAL | no license file | `gadbench_dgl` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| GHRN | External supervised / no | [source](https://github.com/blacksingular/GHRN) | `d47e047b76df429c0c0a8444f5d9a5dea57f6801` | AUTHOR_OFFICIAL | no license file | `gadbench_dgl` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| GADBench / XGBGraph | External supervised / no | [source](https://github.com/squareRoot3/GADBench) | `f9aa021ce9b6c6580427fb633b596843be76ddc6` | AUTHOR_OFFICIAL | no license file | `gadbench_dgl` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| ConsisGAD | External supervised / no | [source](https://github.com/Xtra-Computing/ConsisGAD) | `36811c5bc79be49c9740f25a1f260496bb4736af` | AUTHOR_OFFICIAL | MIT | `consisgad` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| SpaceGNN | External supervised / no | [source](https://github.com/xydong127/SpaceGNN) | `921c03ff879b239dab9b319b296fef3bc3bda2d2` | AUTHOR_OFFICIAL | no license file | `spacegnn` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| DSGAD | External supervised / no | [source](https://github.com/IWantBe/Dynamic-Spectral-Graph-Anomaly-Detection) | `bf7e0ac31a4d28c82c796b339d06b4a1a5308158` | AUTHOR_OFFICIAL | no license file | `dsgad` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| APF | External supervised / no | [source](https://github.com/Cloudy1225/APF) | `0f42fd9344fa4c54e175fae45cb40e6981e6b54a` | AUTHOR_OFFICIAL | no license file | `modern_dgl24` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| SAGAD | External supervised / no | [source](https://github.com/Cloudy1225/SAGAD) | `faddcbca117b475c8468e87a65c8428b616b694b` | AUTHOR_OFFICIAL | no license file | `modern_dgl24` | contract ready; native bridge pending | BLOCKED_NATIVE_SANITY | 8/8 declared |
| HSMAD | External supervised / no | [ICML 2026 listing](https://icml.cc/virtual/2026/papers.html) | `UNAVAILABLE_PUBLIC_SOURCE_2026-08-09` | PAPER_DERIVED metadata only; no implementation | unknown | `unresolved_hsmad` | public source unavailable | BLOCKED_SOURCE | 8/8 declared only |

The nine external supervised references are BWGNN, GHRN, GADBench/XGBGraph,
ConsisGAD, SpaceGNN, DSGAD, APF, SAGAD, and HSMAD. They are explicitly
non-fair and must not be merged into the fair normal-only ranking.

## 5. PAGE and TAQ removal evidence

PAGE, TAQ, and TAQ-GAD are absent from the active registry, target count,
active README list, seed-0 batch launcher, server list launcher, and required
tests. A grep over those active surfaces returned no match. Historical patches
and provenance remain under inactive/excluded records and were not destroyed.
No new PAGE/TAQ result was produced.

## 6. Dataset identity and coverage

The server canonical pointers confirm label-free training fields
`x,edge_index,node_id`; `y,evaluation_mask` exist only in the evaluation bundle.
The complete recovered identity remains at
`/data1/wk/codes/CoReGAD_RELEASE/datasets/hashes/recovered_dataset_identities.json`.

| Dataset | Nodes | Directed edges | Features | Eval nodes | Normal budget | Raw SHA256 (prefix) |
|---|---:|---:|---:|---:|---:|---|
| Amazon | 11,944 | 8,796,784 | 25 | 10,610 | 1,334 | `4b7e3f9cccc62b73` |
| Weibo | 8,405 | 407,963 | 400 | 7,501 | 904 | `111402a2cb3b6a10` |
| YelpChi | 45,954 | 7,693,958 | 32 | 41,242 | 4,712 | `04bd2dd061c67b43` |
| Tolokers | 11,758 | 1,038,000 | 10 | 10,656 | 1,102 | `dacf3ac94cec53d0` |
| T-Finance | 39,357 | 42,445,086 | 10 | 34,852 | 4,505 | `b7d853ec4079e9f7` |
| Elliptic | 46,564 | 73,248 | 93 | 41,523 | 5,041 | `2f502df4b87be8f8` |
| T-Social | 5,781,065 | 146,211,016 | 10 | 5,108,252 | 672,813 | `8d577114cff12f7d` |
| DGraph-Fin | 3,700,550 | 4,300,999 | 17 | 3,555,340 | 145,210 | `35aca506fdfbb3be` |

All 22 rows declare all eight canonical datasets, producing 176 matrix rows.
This means “must attempt without silent substitution/subsampling”; it does not
pre-claim feasibility. Any actual OOM must become `RESOURCE_INFEASIBLE` with
resource evidence rather than an invented or omitted score.

| Method group | Amazon | Weibo | YelpChi | Tolokers | T-Finance | Elliptic | T-Social | DGraph-Fin |
|---|---|---|---|---|---|---|---|---|
| Primary normal-only (12) | declared | declared | declared | declared | declared | declared | declared | declared |
| PU auxiliary (1) | declared | declared | declared | declared | declared | declared | declared | declared |
| External supervised (9) | declared | declared | declared | declared | declared | declared | declared | declared |

## 7. BMP source recovery and paper-derived implementation

The AAAI proceedings paper (DOI `10.1609/aaai.v40i19.38671`) advertises
`https://github.com/Thankstaro/BMP`. Git clone/ls-remote requested credentials,
the GitHub repository API returned 404, the author's public repository list did
not contain BMP, and exact-title/URL/fork/author searches found no public
checkout or trustworthy mirror. BMP is therefore explicitly `PAPER_DERIVED`,
never author official.

The local mathematical core implements probability ordering, normalized
upper/lower routing, independently parameterized BMP trees/forest, predictor,
mask consistency, supervised term, and mask regularization; recorded defaults
include forest order 3 and route-update interval 50. Shape and finite-loss unit
tests pass. A native training bridge and sanity comparison are still required.

## 8. Structure-aware PU-GNN recovery and paper-derived implementation

The CIKM 2023 paper is arXiv `2310.13538v1`, DOI
`10.1145/3583780.3615250`. The paper has no code URL; author pages/repositories,
exact-title/arXiv/equation searches, and Papers with Code yielded no public
implementation. The method is therefore `PAPER_DERIVED`, `PU_AUXILIARY`, and
`fair_ranking: false`.

The local core implements distance partitioning, equation-2 distance-aware PU
loss, equation-3 structural regularization with deterministic non-neighbor
sampling, and recorded defaults alpha 0.01, distance 3, K 50, priors 0.6/0.3,
and two-layer GCN hidden size 16. It includes no CoReGAD module. Unit tests pass;
the end-to-end training bridge and native/paper sanity remain blocked.

## 9. Environment isolation

No baseline package was installed into the protected
`/data1/wk/conda_envs/pfr_gad` environment. The only reuse is read-only unit
testing of the two paper-derived mathematical modules using Python 3.10.20,
PyTorch 2.4.0+cu124, NumPy 2.1.3, SciPy 1.14.1, scikit-learn 1.7.2, and DGL
2.4.0+cu124. PyG is absent. Incompatible families have separate planned paths
in `benchmark/baseline_env_manifest.json`; most installations remain pending.

## 10. Native reproduction sanity gate

No paper/native metric was generated in this task. For every method, the
record is currently: paper metric `NOT_RECORDED_FOR_GATE`, reproduced metric
`NOT_RUN`, absolute delta `N/A`, and protocol difference `N/A`. This is a
deliberate hard stop, not a passed sanity gate.

What did run:

- source checkout presence check: 21 present (including two local
  paper-derived cores), HSMAD blocked;
- launcher/data-contract synthetic smoke: passed, label-free full graph,
  AUPRC 1.0 on synthetic data only;
- Python static compilation of new launch/adapter/paper-derived code: passed;
- full repository test suite: `45 passed in 3.40s`;
- formal result directory: zero files.

The synthetic AUPRC is not a baseline result and must never enter a paper
table.

## 11. Unresolved blockers

1. All 22 entries have `execute_enabled: false`; no native bridge has passed
   its method-specific runtime sanity gate.
2. Planned isolated environments are not installed; `pfr_gad` was protected.
3. GGAD and GraphNC need native label/metric paths removed from training-time
   access; GAD-NR needs a notebook-to-entrypoint bridge.
4. BMP and Structure-aware PU-GNN have tested mathematical cores but no
   end-to-end canonical training bridge.
5. HSMAD has no recovered public source or implementation and remains
   `BLOCKED_SOURCE`.
6. The 176 method-dataset pairs have not undergone memory/complexity testing,
   especially T-Social and DGraph-Fin.
7. Several upstream repositories contain no license file and require reuse
   review before redistribution.
8. No native paper metric/reproduced metric/delta record exists yet; therefore
   the benchmark is not ready for formal seed-0 execution.

## 12. Exact seed-0 commands

Preparation and validation:

```bash
cd /data1/wk/codes/CoReGAD_RELEASE/coregad
/data1/wk/conda_envs/pfr_gad/bin/python benchmark/run_baseline.py --validate
/data1/wk/conda_envs/pfr_gad/bin/python benchmark/run_baseline.py --smoke
/data1/wk/codes/CoReGAD_RELEASE/scripts/list_baselines.sh
```

One exact dry-run plan (default protocol is transductive):

```bash
/data1/wk/codes/CoReGAD_RELEASE/scripts/run_baseline.sh \
  --method DOMINANT --dataset Amazon --seed 0 --protocol transductive \
  --device cuda:0
```

All 22 × 8 seed-0 plans, still dry-run:

```bash
/data1/wk/codes/CoReGAD_RELEASE/scripts/run_all_baselines_seed0.sh
```

After user review, environment installation, native bridge completion, and
per-method sanity approval, the intended explicit execution forms are:

```bash
/data1/wk/codes/CoReGAD_RELEASE/scripts/run_baseline.sh \
  --method DOMINANT --dataset Amazon --seed 0 --protocol transductive \
  --device cuda:0 --execute

/data1/wk/codes/CoReGAD_RELEASE/scripts/run_all_baselines_seed0.sh --execute
```

At this closure point both explicit forms remain blocked by the per-method
`execute_enabled` gate. The three-seed launcher returns exit code 4 by design.
Do not start formal seed-0 until this report and the blockers are reviewed.
