#!/usr/bin/env bash
set -euo pipefail

execute_flag=()
if [[ "${1:-}" == "--execute" ]]; then
  execute_flag=(--execute)
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--execute]" >&2
  exit 2
fi

methods=(
  "DOMINANT" "AnomalyDAE" "OCGNN" "AEGIS" "GAAN" "TAM" "GAD-NR"
  "ADA-GAD" "GGAD" "RHO" "GraphNC" "BMP" "Structure-aware PU-GNN"
  "BWGNN" "GHRN" "GADBench / XGBGraph" "ConsisGAD" "SpaceGNN"
  "DSGAD" "APF" "SAGAD" "HSMAD"
)
datasets=(Amazon Weibo YelpChi Tolokers T-Finance Elliptic T-Social DGraph-Fin)

launcher="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_baseline.py"
for method in "${methods[@]}"; do
  for dataset in "${datasets[@]}"; do
    python "$launcher" --method "$method" --dataset "$dataset" --seed 0 \
      --protocol transductive "${execute_flag[@]}"
  done
done
