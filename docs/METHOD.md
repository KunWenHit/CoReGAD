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

## Find → Trust → Use

CoReGAD-F2 is one causal sequence rather than three unrelated modules. It
first **finds** graph evidence not explained by the stable attribute base,
then **trusts** that evidence according to node-wise structural reliability,
and finally **uses** normal-calibrated routes to select an evidence
representation and a finite correction bound.

## M1 — Complementary Graph Residual Evidence

**互补图残差证据模块**

The nuisance model is five-fold cross-fitted RBF random-Fourier ridge
regression with 256 random features and ridge coefficient `1e-2`. It predicts
the two spectral discrepancies from log degree, support ratio, and local
embedding variation. The released strength is fixed:

```text
D_raw  = normalized_spectral_discrepancy
D_ctrl = normalize(D_raw - 0.75 * structure_predictable_component)
```

The training-fold controlled residual is median/MAD normalized using training
statistics only. Both `D_raw` and `D_ctrl` are retained. Removing M1 removes
the hierarchical graph branch and leaves the post-GCS attribute base score.

## M2 — Node-wise Structural Reliability Control

**节点级结构可靠性控制模块**

The energy network is `Linear(2,16) → GELU → LayerNorm(16) → Linear(16,1) →
Softplus`. The same energy-head instance maps both paths:

```text
E_raw  = EnergyHead(D_raw)
E_ctrl = EnergyHead(D_ctrl)
```

The reliability network is `Linear(3,8) → GELU → Linear(8,1) → Sigmoid` and
produces `q`, the trustworthiness of graph evidence for each node. It is not an
anomaly score and not a router probability. Removing M2 sets `q = 1`, removes
the reliability parameters, and refits the normal-only route calibration.

## M3 — Normal-Calibrated Factorized Evidence–Bound Routing

**正常样本校准的因子化证据—边界路由模块**

For a normal-reference vector `v`, the reusable strict-lower percentile is

```text
F_N^-(x) = count(v < x) / len(v)
```

The strict `<` rule prevents tied normal values from mass-activating a route.
All references come only from declared outer-training normals. With
`rho = 0.80`, the tail map is

```text
T_rho(z) = clip((z - rho) / (1 - rho), 0, 1).
```

The evidence route `r` calibrates the geometric composite of reliability `q`,
joint channel normal support, and the relative suppression from `E_raw` to
`E_ctrl`. The bound route `b` calibrates the geometric composite of `q`, joint
channel support, and the normal-percentile extremeness of the maximum energy.
Both routes are detached and M3 adds zero trainable parameters.

The four finite endpoints are

```text
C_CT = gamma * q * tanh(E_ctrl)
C_CW = gamma * q * 2 * tanh(E_ctrl / 2)
C_RT = gamma * q * tanh(E_raw)
C_RW = gamma * q * 2 * tanh(E_raw / 2)
```

and the final factorized correction is

```text
Delta = (1-r)(1-b) C_CT + (1-r)b C_CW + r(1-b) C_RT + rb C_RW
final_anomaly_logit = base_anomaly_logit + Delta
final_anomaly_score = sigmoid(final_anomaly_logit)
```

The base-logit coefficient is fixed to one. Removing M3 fixes `r = b = 0`, so
`Delta = C_CT`; this is the reproducible legacy Controlled+Tight path.

## Stage-wise learning

1. Train the cross-fitted normality core with Graph Context Shaping.
2. Freeze the normality core and context predictor.
3. Fit the cross-fitted nuisance estimator.
4. Freeze its predictions.
5. Train the shared spectral energy and structural reliability modules while
   recomputing parameter-free normal calibration from the training reference.
6. Apply the frozen normal calibration to held-out nodes; evaluation labels are
   never available to the model or router.
