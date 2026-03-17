#!/bin/bash
# GPU 1: MMLU + HellaSwag for all Llama and Qwen unlearned checkpoints

set -euo pipefail
repo_root=$(realpath "$(dirname "$0")/../..")
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "================================================================"
echo " STEP 1 — MMLU + HellaSwag for Llama and Qwen unlearned models"
echo "================================================================"
python "${repo_root}/Benchmark_Evaluation/run_benchmarks.py" --run

echo ""
echo "================================================================"
echo " STEP 2 — Regenerate summary table"
echo "================================================================"
python "${repo_root}/Benchmark_Evaluation/run_benchmarks.py" --summarize

echo ""
echo "All done."
