# Protocol patches

Each patch applies one repository-local protocol entrypoint on top of the
official upstream SHA recorded in `manifest.json`. The patches do not vendor
third-party source or change a baseline encoder, loss, graph operator,
pseudo-anomaly mechanism, teacher, or optimization logic.

The entrypoint provides the shared label-free fold-bundle, OOF collection, and
evaluation interface. A patch being present is not by itself evidence that the
native formal-training bridge is complete; consult `baseline_registry.yaml`.
