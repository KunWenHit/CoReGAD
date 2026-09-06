# Baseline adapters

`transductive_contract.py` is the main-table boundary. It writes one label-free
full-graph bundle, injects the exact frozen normal support only when required,
preserves node order, validates `node_id,anomaly_score`, and opens labels only
inside the independent evaluator. It contains no outer ownership or cross-fit
logic.

The older `coregad_protocol_runner.py` and upstream
`run_coregad_protocol.py` transplants remain only for the optional `STRICT_OOF`
robustness protocol. They are not active main-table launchers.

Source recovery, a present adapter contract, and a runnable native bridge are
separate registry states. No method is marked executable until the native
optimizer is connected without test-label tuning and passes its native sanity
gate.

Method-specific executable bridges live under `benchmark/native_bridges/`.
Their shared loader accepts only the frozen label-free torch bundle and rejects
label-like keys. `resource_contract.py` defines the independent NOT_RUN/PASS/
OOM/OOT/UNSUPPORTED/ERROR matrix and mandatory OOM/OOT evidence rules.
