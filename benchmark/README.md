# CoReGAD baseline benchmark

The paper main table uses
[`STANDARD_TRANSDUCTIVE_NORMAL_ONLY`](protocol/STANDARD_TRANSDUCTIVE_NORMAL_ONLY.md):
shared canonical graph and supervision, native method optimization, and a
single label-isolated evaluator. Complete unlabeled graph covariates are visible
during training. CoReGAD's internal cross-fitting is not imposed on baselines.

The active inventory contains exactly 22 methods:

- Primary normal-only/unsupervised: DOMINANT, AnomalyDAE, OCGNN, AEGIS,
  GAAN, TAM, GAD-NR, ADA-GAD, GGAD, RHO, GraphNC, and BMP.
- PU auxiliary: Structure-aware PU-GNN.
- External supervised references: BWGNN, GHRN, GADBench / XGBGraph,
  ConsisGAD, SpaceGNN, DSGAD, APF, SAGAD, and HSMAD.

External supervised references keep their published anomaly supervision and
are excluded from the fair normal-only ranking. The registry distinguishes
author source, official benchmark reimplementation, and paper-derived code and
does not turn source recovery into a claim of successful reproduction.

PAGE and TAQ/TAQ-GAD are inactive and absent from the registry and launchers.
Historical source and patch provenance may remain in the archive only. HUGE is
also archived because it is not part of the frozen 22-method inventory.

`STRICT_OOF` remains a secondary robustness option and its existing CoReGAD
implementation and tests are preserved. It is not a mandatory baseline wrapper.

Use `python benchmark/run_baseline.py --list-methods`, `--validate`, or
`--smoke` for preparation. A training process can start only with the explicit
`--execute` flag and only after the registry marks its native bridge executable.
The batch launcher is dry-run by default. No formal benchmark is run by these
preparation commands.
