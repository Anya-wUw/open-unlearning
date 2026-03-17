#!/bin/bash
# Run UNDIAL, RMU, PDU, AltPO on Llama-3.1-8B-Instruct / RWKU at lr=1e-4.

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LRS=${LRS:-"1e-4"}
export BASE_MODEL="Llama-3.1-8B-Instruct"
export MODEL_CONFIG="Llama-3.1-8B-Instruct-lora"
export HF_BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "[rwku][llama-3.1-8b] Running UNDIAL"
bash "${script_dir}/undial_rwku.sh"

echo "[rwku][llama-3.1-8b] Running RMU"
bash "${script_dir}/rmu_rwku.sh"

echo "[rwku][llama-3.1-8b] Running PDU"
bash "${script_dir}/pdu_rwku.sh"

# echo "[rwku][llama-3.1-8b] Running AltPO (requires ALTPO_DATA_DIR to be set with pre-generated data)"
# bash "${script_dir}/altpo_rwku.sh"
