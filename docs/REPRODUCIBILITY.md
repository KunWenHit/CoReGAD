# Reproducibility

## Canonical F2 release gate

The strict frozen historical parity test is a reproduction gate for the
canonical release environment, not a claim that every future numerical kernel
will produce elementwise-identical float32 results. It is executed in the
validated Python 3.10.20 / PyTorch 2.4.0+cu124 canonical environment, where the
complete suite has 79 passing tests and the F2 parity maximum absolute error is
zero.

The fixture and its strict `atol=1e-6`, `rtol=0` comparison remain unchanged.
The ordinary package constraint (`torch>=2.1`) continues to describe supported
public installations.

GitHub's hosted CPU runners validate package behavior, interfaces, safety,
isolation, routing, configuration, and ordinary numerical stability. They use
Python 3.10 and the official PyTorch 2.4.0+cu124 wheel, record the runner CPU,
platform, dependency versions, thread count, and PyTorch build configuration,
and run `scripts/validate_f2_parity.py` explicitly. They do not claim to execute
the canonical strict historical gate: the generic suite explicitly ignores
`tests/test_historical_reference_parity.py`, while retaining that test, its
fixture, and its tolerance unchanged in the repository.

CoReGAD v0.1.1 is a reproducibility-hardening release; it does not change the
paper method. The residual strength remains fixed at `0.75`, the energy head
receives exactly `(r_emb, r_dec)`, and the reliability gate receives exactly
`(log_degree, support_ratio, local_embedding_variation)`.

## Execution engines

`STANDARD` builds the fold-safe sparse PyTorch operator directly and is the
default for small and medium graphs. `SCALABLE` is the same mathematical model
with a CPU CSR graph operator, row-block sparse multiplication, and disk-backed
spectral caches. It never constructs a dense `N x N` matrix or a full-graph
device COO tensor.

```bash
python scripts/train_coregad.py --engine standard ...
python scripts/train_coregad.py --engine scalable --scalable-cache ./cache ...
```

The float32 scalable parity mode follows the standard arithmetic schedule. The
historical large-graph mode uses float16 basis caches while keeping arithmetic
and compact summaries in float32. `T-Social` and `DGraph-Fin` should use
`SCALABLE`; this is an engineering path, not another model version.

## Frozen identity

The eight dataset identities are recorded under
[`reproducibility/`](../reproducibility/README.md). Their support and five-fold
ownership were recovered from the final historical experiment products and
must never be resampled. Dataset licenses prevent publishing exact node-ID
lists, so the public repository contains hashes, budgets, source identity, and
generation rules; the server workspace retains the exact manifests.

Real anomaly labels are forbidden from training, early stopping, checkpoint
selection, hyperparameter search, pseudo-anomaly selection, teacher fitting,
score orientation, and threshold selection. The evaluator opens labels only
after the five owned fold-score files have been assembled.

The synthetic fixture
`tests/fixtures/frozen_release_reference.npz` contains no real dataset and
checks the historical spectral reference, discrepancies, structural channels,
OOF nuisance prediction, controlled residual, energy, reliability, correction,
and final score at `max_abs_diff <= 1e-6`.
