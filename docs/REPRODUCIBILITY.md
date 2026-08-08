# Reproducibility

The released model uses five outer OOF folds and five inner nuisance folds.
Training labels identify normal support only; real anomaly labels are not
available to any optimizer, normalizer, checkpoint selector, or early-stopping
rule.

For every outer fold:

1. exclude the owned unlabeled nodes from normality-core training;
2. construct graph context using visible training nodes only;
3. freeze the normality core before graph-spectral processing;
4. fit nuisance predictions with inner OOF ownership for training nodes;
5. fit a full-visible nuisance model only for the outer-held-out nodes;
6. train the residual detector for its fixed final epoch;
7. materialize scores before the evaluator opens anomaly labels.

The released validation seed policy is `[0, 1, 2]`. Dataset version, node
order, normal support, split ownership, optimizer, epoch budget, and all frozen
constants must match [the configuration](../configs/coregad.yaml).

Full benchmark training is intended for a server with the required datasets.
Local verification should use import tests, formula tests, OOF ownership tests,
and the tiny deterministic behavior-parity fixture.
