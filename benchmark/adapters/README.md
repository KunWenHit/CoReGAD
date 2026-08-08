# Adapter boundary

`coregad_protocol_runner.py` is dependency-light and method-agnostic. Repository-local wrappers delegate to it rather than modifying upstream model definitions. The training-bundle writer never loads `y`; only the separate `evaluate` action reads labels.

The `command` action renders a method invocation without running it unless `--allow-training` is passed explicitly. This guard prevents accidental full baseline runs during source preparation.
