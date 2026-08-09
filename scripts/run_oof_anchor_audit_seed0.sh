#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/wk/codes/CoReGAD_RELEASE
PYTHON=/data1/wk/conda_envs/pfr_gad/bin/python

cd "$ROOT"
exec "$PYTHON" "$ROOT/coregad/scripts/oof_anchor_audit_runner.py" "$@"
