#!/bin/bash
# Compute forget metrics for RWKU NEW_BUGFIX_* runs at lr=1e-4 (forget_level2 split).

set -euo pipefail

repo_root=$(realpath "$(dirname "$0")/../..")

bench="rwku"
forget_split="${FORGET_SPLIT:-forget_level2}"
retain_split="${RETAIN_SPLIT:-neighbor_level2}"
lr_tag="lr1e-4"
BATCH_SIZE="${BATCH_SIZE:-16}"
AMP_MODE="${AMP_MODE:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
GPU_ID=${GPU_ID:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# base_model_tag | model_cfg | base_model_path
models=(
  "Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct-lora|meta-llama/Llama-3.1-8B-Instruct"
  "gemma-7b-it|gemma-7b-it-lora|google/gemma-7b-it"
  "Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct-lora|Qwen/Qwen2.5-7B-Instruct"
)

forget_tag="${forget_split//+/__}"
retain_tag="${retain_split//+/__}"

for model_row in "${models[@]}"; do
  IFS='|' read -r base_model model_cfg base_path <<< "${model_row}"
  echo "[metrics][rwku] model=${base_model}"

  for algo_dir in "${repo_root}/saves/unlearn/rwku"/NEW_BUGFIX_*; do
    [ -d "${algo_dir}" ] || continue
    algo=$(basename "${algo_dir}")
    for run_dir in "${algo_dir}"/*"${lr_tag}"*; do
      [ -d "${run_dir}" ] || continue
      [[ "${run_dir}" == *"${base_model}"* ]] || continue

      eval_dir="${run_dir}/evals/forget_metrics__f_${forget_tag}__r_${retain_tag}"
      summary_path="${eval_dir}/FORGET_METRICS_SUMMARY.json"
      if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "[metrics][rwku] skip (exists): $(basename "${run_dir}")"
        continue
      fi

      mkdir -p "${eval_dir}"
      echo "[metrics][rwku] ${algo} -> $(basename "${run_dir}")"
      python "${repo_root}/scripts/forget_metrics/forget_metrics.py" \
        --benchmark  "${bench}" \
        --forget-split "${forget_split}" \
        --retain-split "${retain_split}" \
        --model-config "${model_cfg}" \
        --base-model-path "${base_path}" \
        --adapter-path "${run_dir}" \
        --output-dir   "${eval_dir}" \
        --batch-size   "${BATCH_SIZE}" \
        --amp          "${AMP_MODE}" \
        --gpu          "${GPU_ID}" \
        --num-workers  "${NUM_WORKERS}" \
        --prefetch-factor "${PREFETCH_FACTOR}"
    done
  done
done
