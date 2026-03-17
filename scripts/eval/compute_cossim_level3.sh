#!/bin/bash
# GPU 0: cos_sim re-run (RWKU) + level3 eval for missing NPO/Llama entry

set -euo pipefail
repo_root=$(realpath "$(dirname "$0")/../..")
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "================================================================"
echo " STEP 1 — cos_sim for RWKU (fills NPO/Llama and WGA/Gemma gaps)"
echo "================================================================"
python "${repo_root}/scripts/forget_metrics/calc_cos_sim.py" \
    --base-dir "${repo_root}" \
    --benchmarks "rwku" \
    --gpu "${GPU_ID}"

echo ""
echo "================================================================"
echo " STEP 2 — level3 eval for missing RWKU entries (NPO/Llama)"
echo "================================================================"
python "${repo_root}/scripts/analysis/eval_additional_subsets.py" \
    --gpu "${GPU_ID}"

echo ""
echo "All done."
