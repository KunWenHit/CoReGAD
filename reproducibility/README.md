# Exact experiment identities

This directory publishes the immutable identity of the eight datasets used by
the final CoReGAD experiments: Amazon, Weibo, YelpChi, Tolokers, T-Finance,
Elliptic, T-Social, and DGraph-Fin.

The public files deliberately omit support and evaluation node IDs because the
underlying dataset licenses have not been cleared for republishing derived
node lists. Each dataset record instead publishes the raw/source identity,
tensor and ordering hashes, exact support budget, split/support file hashes,
five-fold counts, seeds, and the rule that the historical manifests must be
reused rather than regenerated. The controlled server release workspace keeps
the exact CSV/JSON manifests.

Training bundles expose only `x`, `edge_index`, and stable `node_id`. Labels and
the evaluation mask belong to the separate evaluation bundle. Model seeds may
change optimization randomness, but never dataset order, support, or fold
ownership.
