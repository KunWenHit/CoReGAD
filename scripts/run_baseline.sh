#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${COREGAD_LAUNCHER_PYTHON:-/data1/wk/conda_envs/pfr_gad/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "launcher Python does not exist: $PYTHON" >&2
  exit 2
fi

exec "$PYTHON" "$ROOT/benchmark/direct_reproduction_launcher.py" "$@"
