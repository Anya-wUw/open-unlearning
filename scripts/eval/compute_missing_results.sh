#!/bin/bash
# Compute all missing results:
#   1. cos_sim for RWKU/NPO/Llama and RWKU/WGA/Gemma  (re-run calc_cos_sim)
#   2. level3 ROUGE-L for RWKU/NPO/Llama              (eval_additional_subsets)
#   3. MMLU + HellaSwag for ALL Llama and Qwen unlearned checkpoints

set -euo pipefail
repo_root=$(realpath "$(dirname "$0")/../..")
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "================================================================"
echo " STEP 1 — cos_sim for RWKU (fills NPO/Llama and WGA/Gemma gaps)"
echo "================================================================"
conda run -n MU python "${repo_root}/scripts/forget_metrics/calc_cos_sim.py" \
    --base-dir "${repo_root}" \
    --benchmarks "rwku" \
    --gpu "${GPU_ID}"

echo ""
echo "================================================================"
echo " STEP 2 — level3 eval for missing RWKU entries (NPO/Llama)"
echo "================================================================"
conda run -n MU python "${repo_root}/scripts/analysis/eval_additional_subsets.py" \
    --gpu "${GPU_ID}"

echo ""
echo "================================================================"
echo " STEP 3 — MMLU + HellaSwag for Llama and Qwen unlearned models"
echo "================================================================"
conda run -n MU python "${repo_root}/Benchmark_Evaluation/run_benchmarks.py" --run

echo ""
echo "================================================================"
echo " STEP 4 — Regenerate summary table"
echo "================================================================"
conda run -n MU python "${repo_root}/Benchmark_Evaluation/run_benchmarks.py" --summarize

echo ""
echo "All done."
