# CoReGAD baseline benchmark

The paper main table uses
[`STANDARD_TRANSDUCTIVE_NORMAL_ONLY`](protocol/STANDARD_TRANSDUCTIVE_NORMAL_ONLY.md):
shared canonical graph and supervision, native method optimization, and a
single label-isolated evaluator. Complete unlabeled graph covariates are visible
during training. CoReGAD's internal cross-fitting is not imposed on baselines.

The active inventory contains exactly 21 methods:

- Primary normal-only/unsupervised: DOMINANT, AnomalyDAE, OCGNN, AEGIS,
  GAAN, TAM, GAD-NR, ADA-GAD, GGAD, RHO, and GraphNC.
- PU auxiliary: Structure-aware PU-GNN.
- External supervised references: BWGNN, GHRN, GADBench / XGBGraph,
  ConsisGAD, SpaceGNN, DSGAD, APF, SAGAD, and HSMAD.

External supervised references keep their published anomaly supervision and
are excluded from the fair normal-only ranking. The registry distinguishes
author source, official benchmark reimplementation, and paper-derived code and
does not turn source recovery into a claim of successful reproduction.

PAGE, TAQ/TAQ-GAD, and BMP are inactive and absent from the registry and
launchers. BMP is excluded as `EXCLUDED_NO_RECOVERABLE_OFFICIAL_SOURCE`; its
paper-derived mathematical core and search evidence are provenance only and
are not an execution candidate. Historical source and patch provenance may
remain in the archive. HUGE is also outside the frozen 21-method inventory.

Every active method targets the same eight frozen datasets, but a target is not
a support or validation claim. The registry separately records
`benchmark_targets`, `native_supported_datasets`, `validated_datasets`, and a
per-cell resource status. The 21 × 8 matrix therefore has 168 targets, initially
`NOT_RUN`. OOM and OOT require actual execution evidence; complexity analysis
uses a separate `NOT_RUN_RESOURCE_RISK` preflight marker and never fabricates a
runtime failure.

`STRICT_OOF` remains a secondary robustness option and its existing CoReGAD
implementation and tests are preserved. It is not a mandatory baseline wrapper.

Use `python benchmark/run_baseline.py --list-methods`, `--validate`, or
`--smoke` for preparation. A training process can start only with the explicit
`--execute` flag and only after all six native-bridge gates have evidence.
The batch launcher is dry-run by default. No formal benchmark is run by these
preparation commands.
