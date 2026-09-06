#!/usr/bin/env bash
set -euo pipefail
exec /data1/wk/conda_envs/pfr_gad/bin/python \
  /data1/wk/codes/CoReGAD_RELEASE/coregad/benchmark/run_baseline.py --list-methods
