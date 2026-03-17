#!/bin/bash
# Run UNDIAL, RMU, PDU, AltPO on gemma-7b-it / DUET at lr=1e-4.
# AltPO requires pre-generated alternate data; set ALTPO_DATA_DIR or skip.

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LRS=${LRS:-"1e-4"}
export BASE_MODEL="gemma-7b-it"
export MODEL_CONFIG="gemma-7b-it-lora"
export HF_BASE_MODEL_PATH="google/gemma-7b-it"
export LOCAL_SFT_BASE="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/gemma-7b-it_full_3ep_ft_tripunlamb"
export USE_SFT_BASE=1
export MERGE_POPULARITY_FORGET=1

echo "[duet][gemma-7b] Running UNDIAL"
bash "${script_dir}/undial_duet.sh"

echo "[duet][gemma-7b] Running RMU"
bash "${script_dir}/rmu_duet.sh"

echo "[duet][gemma-7b] Running PDU"
bash "${script_dir}/pdu_duet.sh"

# echo "[duet][gemma-7b] Running AltPO (requires ALTPO_DATA_DIR to be set with pre-generated data)"
# bash "${script_dir}/altpo_duet.sh"
