#!/bin/bash

set -euo pipefail

repo_root=$(realpath "$(dirname "$0")/../..")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "[cos_sim] Running calculation for DUET and RWKU bugfix runs"
python "${repo_root}/scripts/forget_metrics/calc_cos_sim.py" \
    --base-dir "${repo_root}" \
    --benchmarks "duet,rwku" \
    --gpu "${CUDA_VISIBLE_DEVICES}"

echo "[cos_sim] Running visualization..."
python "${repo_root}/notebooks/benchmarks_visualizations/cos_sim/cos_sim_v2.py"
