# Baseline execution plan

Do not execute this plan until every selected method is `READY` in the server
support matrix.

Method order:

1. Direct competitors: GGAD, RHO, GraphNC.
2. Recent direct competitors: PAGE, TAQ-GAD.
3. Mechanism references: TAM, HUGE.
4. Anchors: OCGNN, GAD-NR.

Dataset order:

1. Stage A: Amazon, Tolokers, Elliptic.
2. Stage B: YelpChi, Weibo, T-Finance.
3. Stage C: T-Social, DGraph-Fin.

Run the medium graphs first so protocol defects are discovered before the two
large scalable jobs. Model seeds are `0, 1, 2`; exact support and fold ownership
remain seed-independent.
