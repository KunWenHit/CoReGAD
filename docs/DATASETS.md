# Datasets

CoReGAD does not redistribute datasets. Prepare each graph as a canonical NPZ
with `x`, `edge_index`, optional `node_id`, and evaluation-only `y` arrays.

Normal support and unlabeled ownership are stored in a separate JSON manifest.
This separation is deliberate: the training loader opens features and edges but
does not access `y`. Preserve original node order and document any upstream
preprocessing outside the repository.

Large arrays, raw datasets, graph caches, checkpoints, and experiment outputs
are ignored by Git and must not be committed.
