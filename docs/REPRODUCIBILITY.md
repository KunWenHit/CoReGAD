# Reproducibility

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
