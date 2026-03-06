#!/bin/bash

set -euo pipefail

repo_root=$(realpath "$(dirname "$0")/../..")

echo "Starting cleanup of forget internal metrics..."

# --- Duet ---
bench="duet"
forget_split_duet="${FORGET_SPLIT:-city_forget_rare_5+city_forget_popular_5}"
retain_split_duet="${RETAIN_SPLIT:-city_fast_retain_500}"
forget_tag_duet="${forget_split_duet//+/__}"
retain_tag_duet="${retain_split_duet//+/__}"
eval_dir_pattern_duet="forget_metrics__f_${forget_tag_duet}__r_${retain_tag_duet}"

echo "[cleanup] Cleaning Duet metrics: ${eval_dir_pattern_duet}"
find "${repo_root}/saves/unlearn/duet" -type d -name "${eval_dir_pattern_duet}" -exec rm -rf {} + 2>/dev/null || true

# --- RWKU ---
bench="rwku"
forget_split_rwku="${FORGET_SPLIT:-forget_level2}"
retain_split_rwku="${RETAIN_SPLIT:-neighbor_level2}"
forget_tag_rwku="${forget_split_rwku//+/__}"
retain_tag_rwku="${retain_split_rwku//+/__}"
eval_dir_pattern_rwku="forget_metrics__f_${forget_tag_rwku}__r_${retain_tag_rwku}"

echo "[cleanup] Cleaning RWKU metrics: ${eval_dir_pattern_rwku}"
find "${repo_root}/saves/unlearn/rwku" -type d -name "${eval_dir_pattern_rwku}" -exec rm -rf {} + 2>/dev/null || true

# --- PopQA (Optional but consistent) ---
echo "[cleanup] Cleaning PopQA metrics: forget_metrics__f_*"
find "${repo_root}/saves/unlearn/popqa" -type d -name "forget_metrics__f_*" -exec rm -rf {} + 2>/dev/null || true

echo "Cleanup complete. You can now run the metrics scripts to recalculate."
