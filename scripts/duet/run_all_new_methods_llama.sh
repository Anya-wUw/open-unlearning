#!/bin/bash
# Run UNDIAL, RMU, PDU, AltPO on Llama-3.1-8B-Instruct / DUET at lr=1e-4.

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LRS=${LRS:-"1e-4"}
export BASE_MODEL="Llama-3.1-8B-Instruct"
export MODEL_CONFIG="Llama-3.1-8B-Instruct-lora"
export HF_BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
export LOCAL_SFT_BASE="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb"
export USE_SFT_BASE=1
export MERGE_POPULARITY_FORGET=1

echo "[duet][llama-3.1-8b] Running UNDIAL"
bash "${script_dir}/undial_duet.sh"

echo "[duet][llama-3.1-8b] Running RMU"
bash "${script_dir}/rmu_duet.sh"

echo "[duet][llama-3.1-8b] Running PDU"
bash "${script_dir}/pdu_duet.sh"





# AltPO: run data generation first

#   # DUET (run once per model; use the SFT checkpoint for best alternate quality)
#   CUDA_VISIBLE_DEVICES=0 conda run -n MU python scripts/analysis/generate_altpo_data.py \
#       --dataset duet --split city_forget_5 \
#       --model_path /mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb \
#       --output_path data/altpo/duet_city_forget_5_alt.jsonl

#   # RWKU
#   CUDA_VISIBLE_DEVICES=0 conda run -n MU python scripts/analysis/generate_altpo_data.py \
#       --dataset rwku --split forget_level2 \
#       --model_path meta-llama/Llama-3.1-8B-Instruct \
#       --output_path data/altpo/rwku_forget_level2_alt.jsonl

# echo "[duet][llama-3.1-8b] Running AltPO (requires ALTPO_DATA_DIR to be set with pre-generated data)"
# bash "${script_dir}/altpo_duet.sh"
