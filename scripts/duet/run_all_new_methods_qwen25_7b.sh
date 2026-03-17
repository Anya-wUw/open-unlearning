#!/bin/bash
# Run UNDIAL, RMU, PDU, AltPO on Qwen2.5-7B-Instruct / DUET at lr=1e-4.

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LRS=${LRS:-"1e-4"}
export BASE_MODEL="Qwen2.5-7B-Instruct"
export MODEL_CONFIG="Qwen2.5-7B-Instruct-lora"
export HF_BASE_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export LOCAL_SFT_BASE="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/Qwen2.5-7B-Instruct_full_3ep_ft_tripunlamb"
export USE_SFT_BASE=1
export MERGE_POPULARITY_FORGET=1

echo "[duet][qwen2.5-7b] Running UNDIAL"
bash "${script_dir}/undial_duet.sh"

echo "[duet][qwen2.5-7b] Running RMU"
bash "${script_dir}/rmu_duet.sh"

echo "[duet][qwen2.5-7b] Running PDU"
bash "${script_dir}/pdu_duet.sh"

# echo "[duet][qwen2.5-7b] Running AltPO (requires ALTPO_DATA_DIR to be set with pre-generated data)"
# bash "${script_dir}/altpo_duet.sh"
