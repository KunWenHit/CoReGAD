# Strict-OOF rescue protocol (design only)

This document defines a fixed escalation order if the seed-0 causal audit later
shows a material loss after closing the all-node anchor path. None of these
rescue variants is implemented, selected, or trained by the closure task.
Ground-truth anomaly labels may be used only for the final posthoc report and
must never choose among R1, R2, and R3.

## R1 — stronger leakage-free preservation

Keep the five outer folds and visible-only optimization contract unchanged.
Replace neither the detector nor its graph residual equations. First audit
whether visible-only teacher preservation is under-constrained, then evaluate a
minimal preservation term that depends only on visible nodes or on parameters,
such as a fixed coefficient on the distance between the initialized and current
student encoder parameters, or visible-node representation preservation. The
held-out attributes, embeddings, logits, graph rows, and labels remain
unavailable to every loss. Freeze the exact coefficient and design using
training-only behavior or synthetic fixtures before any anomaly-label metric is
opened. All fair baselines retain the same five-fold ownership.

## R2 — ten-fold strict OOF

If R1 is rejected on label-free grounds, define a separate protocol version
with ten deterministic outer folds. Each model may use 90% of the unlabeled
nodes as visible training covariates while the owned 10% is excluded from every
optimization path. Support nodes, node ordering, evaluation mask, preprocessing,
model architecture, alpha, optimizer, epochs, and evaluator remain fixed. The
fold manifest must be generated and frozen before evaluation, and every fair
baseline must use the same ten-fold ownership. R2 results cannot be mixed with
the released five-fold table.

## R3 — explicitly transductive normal-only protocol

Only if the research setting expressly permits full-graph unlabeled covariates,
rename the protocol as transductive normal-only GAD. The all-node anchor may then
access held-out covariates but never their anomaly labels. Remove every claim
that held-out nodes are excluded from optimization, disclose full-covariate
access, and give every fair baseline the same transductive access. R3 is a
different comparison regime, not a repair of strict OOF, and cannot be reported
under the strict-OOF name.

The order is R1, then R2, then R3. Advancement requires a protocol decision made
without consulting evaluation anomaly labels; this closure task makes no such
decision.
