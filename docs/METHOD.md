# Method

## Cross-Fitted Normality Core

Each of five outer folds trains on labeled normal nodes and the four owned
unlabeled folds. Its held-out unlabeled fold is excluded from optimization.
The attribute encoder has width 64 and dropout 0.2. Graph Context Shaping is a
training-time objective; the frozen inference path reads node attributes only.

## Global Spectral Reference

For frozen node embeddings `H` and the fold-safe normalized adjacency
`A_hat`, the three components are

```text
Z_low  = A_hat H
Z_band = A_hat H - A_hat^2 H
Z_high = H - A_hat H
```

A single shared softmax weight vector from logits `[1, 0, 0]` forms the
spectral reference. The node-level spectral evidence contains only embedding
and decision discrepancies.

## Controlled Structural Residualization

The nuisance model is five-fold cross-fitted RBF random-Fourier ridge
regression with 256 random features and ridge coefficient `1e-2`. It predicts
the two spectral discrepancies from log degree, support ratio, and local
embedding variation. The released strength is fixed:

```text
controlled_spectral_residual
  = spectral_discrepancy
  - 0.75 * structure_predictable_spectral_component
```

The training-fold controlled residual is median/MAD normalized using training
statistics only.

## Reliability-Constrained Detection

The energy network is `Linear(2,16) → GELU → LayerNorm(16) → Linear(16,1) →
Softplus`. The reliability network is `Linear(3,8) → GELU → Linear(8,1) →
Sigmoid`. The released correction is

```text
graph_correction = gamma * structural_reliability * tanh(spectral_residual_energy)
final_anomaly_logit = base_anomaly_logit + graph_correction
final_anomaly_score = sigmoid(final_anomaly_logit)
```

The base-logit coefficient is fixed to one.

## Stage-wise learning

1. Train the cross-fitted normality core with Graph Context Shaping.
2. Freeze the normality core and context predictor.
3. Fit the cross-fitted nuisance estimator.
4. Freeze its predictions.
5. Train the spectral residual energy and structural reliability modules.
