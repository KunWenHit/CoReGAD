# CoReGAD — All Active Baselines Direct Reproduction Closure

Date: 2026-08-10

Server workspace: `/data1/wk/codes/CoReGAD_RELEASE`

Main protocol: `STANDARD_TRANSDUCTIVE_NORMAL_ONLY`

## Closure summary

- `active_total = 21`
- `direct_reproduction_ready = 21/21`
- `primary_ready = 11/11`
- `pu_ready = 1/1`
- `external_ready = 9/9`
- `environments_reused = 4` distinct isolated runtime families
- `environments_created = 11` distinct isolated runtime families
- `native_sanity_pass = 21/21`
- `canonical_sanity_pass = 21/21`
- `still_blocked = []`
- `benchmark_targets = 21 × 8 = 168`; this closure did not start the prohibited 21×8 benchmark.

Every active runtime has all seven required gates at `PASS`: `SOURCE_PASS`, `ENV_PASS`, `NATIVE_SANITY_PASS`, `LABEL_OR_SUPERVISION_AUDIT_PASS`, `CANONICAL_BRIDGE_PASS`, `SCORE_CONTRACT_PASS`, and `LAUNCHER_PASS`. BMP, PAGE, TAQ, and TAQ-GAD are absent from both active registries and launchers. BMP remains only as excluded provenance with `EXCLUDED_NO_RECOVERABLE_OFFICIAL_SOURCE`.

## Direct reproduction command

```bash
cd /data1/wk/codes/CoReGAD_RELEASE/coregad
scripts/run_baseline.sh --method METHOD --dataset DATASET --seed 0 --device cuda:X --execute
```

For CPU-only AnomalyDAE, use `--device cpu`. The shell entrypoint performs registry lookup, absolute interpreter dispatch, source/environment/canonical-identity verification, training, score validation, and isolated evaluation. Users do not activate an environment, edit a path, copy a dataset, open a notebook, or manually convert scores.

Formal output contract:

```text
/data1/wk/codes/CoReGAD_RELEASE/outputs/baselines/METHOD/DATASET/seed_0/
├── run_manifest.json
├── score.npz
├── metrics.json
├── stdout.log
├── stderr.log
└── resource.json
```

An actual OOM/OOT additionally produces `resource_failure.json`; metrics remain empty. No OOM/OOT was inferred from complexity alone.

## Method closure matrix

| Method | Class | Source kind / commit | Environment | Runtime stack | Native sanity | Canonical sanity | Label/supervision audit | Launcher | Enabled | Validated | Resource risks / blocker |
|---|---|---|---|---|---|---|---|---|---:|---|---|
| DOMINANT | Primary | author official / `83fa939` | `baseline_primary_torch24` | Py3.10, Torch2.4, DGL2.4/PyG2.6 | PASS | Amazon seed0, 100 epochs | no anomaly labels | PASS | true | Amazon | dense reconstruction risk on large graphs; none blocking |
| AnomalyDAE | Primary | author official / `1199bae` | `coregad_anomalydae` | Py3.10, TF2.15 `compat.v1`, CPU | PASS | Weibo seed0, 180 iterations | no anomaly labels | PASS | true | Weibo | dense reconstruction risk; none blocking |
| OCGNN | Primary | author official / `ec7d11c` | `baseline_primary_torch24` | Py3.10, Torch2.4, DGL2.4/PyG2.6 | PASS | Amazon seed0, default 5000 epochs | normal support only; upstream full-graph center init retained | PASS | true | Amazon | full-graph propagation risk; none blocking |
| AEGIS | Primary | official benchmark implementation / `358fb4d` | `coregad_ggad` | Py3.10, Torch2.4, DGL2.4/PyG2.6 | PASS | Weibo seed0, full schedule | no anomaly labels | PASS | true | Weibo | reconstruction/full-graph risk; none blocking |
| GAAN | Primary | official benchmark implementation / `358fb4d` | `coregad_ggad` | Py3.10, Torch2.4, DGL2.4/PyG2.6 | PASS | Weibo seed0, full schedule | no anomaly labels | PASS | true | Weibo | adversarial full-graph risk; none blocking |
| TAM | Primary | author official / `e82ead0` | `coregad_tam` | Py3.10, Torch2.4, DGL2.4/PyG2.6 | PASS | Weibo seed0, 18×3×500 = 27,000 steps | no anomaly labels | PASS | true | Weibo | affinity/tree cost on large graphs; none blocking |
| GAD-NR | Primary | author official / `b0a9590` | `coregad_gadnr` | Py3.10, Torch2.4, DGL2.4/PyG2.6 | PASS | Weibo seed0, 500 epochs | no anomaly labels; notebook/CLI parity PASS | PASS | true | Weibo | reconstruction risk; none blocking |
| ADA-GAD | Primary | author official / `71a3aca` | `coregad_adagad` | Py3.10, Torch2.4, PyG2.6 | PASS | Weibo seed0, 3×20 pretrain + 20 detection | no anomaly labels | PASS | true | Weibo | full-graph encoder risk; none blocking |
| GGAD | Primary | author official / `358fb4d` | `coregad_ggad` | Py3.10, Torch2.4, DGL2.4/PyG2.6 | PASS | Weibo seed0, full schedule | pseudo anomalies only; no ground truth | PASS | true | Weibo | full-graph risk; none blocking |
| RHO | Primary | author official / `a394d55` | `coregad_rho` | Py3.10, Torch2.4, DGL2.4/PyG2.6 | PASS | Weibo seed0, sparse/batched path | normal support only; no anomaly labels | PASS | true | Weibo | sparse edge and NCE scaling risk; none blocking |
| GraphNC | Primary | author official / `4985138` | `coregad_graphnc` | Py3.10, Torch2.4, DGL2.4/PyG2.6 | PASS | Weibo seed0, native teacher/student schedule | no labels in training process; no metric-driven selection | PASS | true | Weibo | teacher/student full-graph risk; none blocking |
| Structure-aware PU-GNN | PU auxiliary | paper derived / arXiv `2310.13538v1` | `coregad_pu` | Py3.10, Torch2.4 | PASS | Elliptic seed0, 100 epochs | positive/normal support only; `fair_ranking=false` | PASS | true | Elliptic | negative sampling/edge scaling risk; none blocking |
| BWGNN | External supervised | author official / `de0631f` | `coregad_gadbench` | Py3.10, Torch2.4, DGL2.4 | PASS | Weibo seed0, 100 epochs | native supervised trial0; `fair_ranking=false` | PASS | true | Weibo | full-graph wavelet risk; none blocking |
| GHRN | External supervised | author official / `d47e047` | `coregad_gadbench` | Py3.10, Torch2.4, DGL2.4 | PASS | Weibo seed0, 100+100 epochs | native supervised trial0; `fair_ranking=false` | PASS | true | Weibo | two-stage full-graph risk; none blocking |
| GADBench / XGBGraph | External supervised | official benchmark / `f9aa021` | `coregad_gadbench` | Py3.10, Torch2.4, DGL2.4, XGBoost1.7 | PASS | Weibo seed0, 100 trees | native supervised trial0; `fair_ranking=false` | PASS | true | Weibo | expanded GIN features on T-Social; none blocking |
| ConsisGAD | External supervised | author official / `36811c5` | `coregad_consisgad` | Py3.10, Torch2.4, DGL2.4 | PASS | Weibo seed0, 100×128 iterations | native supervised + unlabeled pool; `fair_ranking=false` | PASS | true | Weibo | sampler cost; no official Weibo config, fixed upstream Yelp fallback recorded |
| SpaceGNN | External supervised | author official / `921c03f` | `coregad_spacegnn` | Py3.10, Torch2.4, DGL2.4 | PASS | Weibo seed0, 25 epochs, 1,675 steps | native supervised trial0; `fair_ranking=false` | PASS | true | Weibo | full-neighbor/multi-space cost; none blocking |
| DSGAD | External supervised | author official / `bf7e0ac` | `coregad_dsgad` | Py3.10, Torch2.4, DGL2.4 | PASS | Weibo seed0, 100 epochs | native supervised trial0; `fair_ranking=false` | PASS | true | Weibo | per-node filter weights on large graphs; none blocking |
| APF | External supervised | author official / `0f42fd9` | `coregad_modern_dgl` | Py3.10, Torch2.4, DGL2.4 | PASS | Weibo seed0, 300 pretrain + 500 fine-tune | labels only in fine-tune; `fair_ranking=false` | PASS | true | Weibo | MRQ preprocessing/full-graph spectra; none blocking |
| SAGAD | External supervised | author official / `faddcbc` | `coregad_modern_dgl` | Py3.10, Torch2.4, DGL2.4 | PASS | Weibo seed0, early stop 64/500 | native supervised trial0; `fair_ranking=false` | PASS | true | Weibo | MRQ preprocessing; none blocking |
| HSMAD | External supervised | author official / `7810e8e` | `coregad_hsmad` | Py3.10, Torch2.4, DGL2.4 | PASS | Weibo seed0, early stop 240/1000 | node + train-mask edge supervision; `fair_ranking=false` | PASS | true | Weibo | full-edge partitioning/attention; none blocking |

Full hashes, loss traces, optimizer-step counts, elapsed times, score counts, and supervision details are stored under `benchmark/evidence/{primary,pu,external}/METHOD/` and in the server `outputs/baselines/full_native_sanity/` tree.

## Dataset and supervision bridge audit

The eight canonical identities remain frozen: Amazon, Weibo, YelpChi, Tolokers, T-Finance, Elliptic, T-Social, and DGraph-Fin. The label-free training bundle is verified against node count, feature shape, node ordering, directed edge count, and provenance hashes before dispatch.

External references alone receive a separate raw supervision source. The bridge was read-only tested on all eight formats:

- DGL binaries: Weibo, YelpChi, T-Finance, T-Social, DGraph-Fin.
- SciPy MAT: Amazon, Elliptic.
- NumPy NPZ: Tolokers.
- Source-provided masks are retained where present. Where an official asset has labels but no masks, the adapter creates a deterministic stratified 40/20/40 split with `random_state = 2 + trial`; this is recorded as supervision provenance and never applies to primary or PU training.

T-Social identity uses `146,211,016` directed `edge_index` entries; reports using about `73,105,508` count undirected unique edges. No version mismatch is inferred from this factor-of-two convention alone.

## Compatibility and environment decisions

- `/data1/wk/conda_envs/pfr_gad` was not modified or used for baseline dispatch.
- AnomalyDAE uses TensorFlow 2.15.1's TensorFlow-1 compatibility API on CPU; only removed `tf.contrib.layers` initializer/regularizer helpers are shimmed, with unchanged equations.
- GAD-NR is a mechanical extraction of the official notebook; a same-subgraph three-epoch notebook/CLI parity audit passed.
- ConsisGAD uses mechanical SciPy/NumPy aliases and an offline stub for an imported-but-unused W&B telemetry module. Its graph model, augmentation, loss, thresholds, and optimizer schedule are author code.
- APF/SAGAD cache MRQ artifacts below their run output, never in the frozen source checkout.
- Existing GGAD, RHO, and GraphNC environments were reused; incompatible families were isolated instead of modifying historical environments.

## Verification

- Both registries parse as JSON and contain exactly 21 active methods.
- `execute_enabled=true` for 21/21 and every enabled method has a real canonical command and seven PASS gates.
- The complete server suite passes (`60 passed`), and all 21 public shell-launcher dry-run dispatches resolve on Weibo without training.
- Environment, source entrypoint, and canonical entrypoint existence checks pass on the server.
- Score evidence for every full canonical sanity contains unique node IDs, finite scores, and `higher_is_more_anomalous=true`.
- No source repository was pushed. The closure branch is local only.

## Result

The direct-reproduction closure is complete. The next authorized action is user acceptance or an explicitly requested benchmark run; the 21×8 production matrix remains unstarted.
