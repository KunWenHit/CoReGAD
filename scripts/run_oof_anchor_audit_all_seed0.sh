#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/wk/codes/CoReGAD_RELEASE
SINGLE="$ROOT/coregad/scripts/run_oof_anchor_audit_seed0.sh"
DEVICE=""
EXECUTE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="${2:?--device requires a value}"
      shift 2
      ;;
    --execute)
      EXECUTE=true
      shift
      ;;
    *)
      echo "usage: $0 --device DEVICE [--execute]" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DEVICE" ]]; then
  echo "usage: $0 --device DEVICE [--execute]" >&2
  exit 2
fi

datasets=(Amazon Weibo YelpChi Tolokers T-Finance Elliptic T-Social DGraph-Fin)
for dataset in "${datasets[@]}"; do
  command=("$SINGLE" run-pair --dataset "$dataset" --device "$DEVICE")
  if [[ "$EXECUTE" == true ]]; then
    command+=(--execute)
  fi
  "${command[@]}"
done
