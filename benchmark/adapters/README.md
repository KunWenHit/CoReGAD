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
separate registry states. The public snapshot exposes the adapter/protocol
contract only. The method-specific executable layer remains server-only
because it depends on private environments, gate evidence, outputs, and
external source checkouts.
