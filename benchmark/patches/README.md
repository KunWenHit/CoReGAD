# Third-party patches

Third-party patches are not vendored into the CoReGAD source release. They are generated in the separate baseline workspace and apply one portable `run_coregad_protocol.py` wrapper on top of the exact upstream commit recorded in `../baseline_registry.yaml`.

Set `COREGAD_BASELINE_ROOT` to the prepared external workspace to access `patches/<METHOD>_coregad_protocol.patch`.
