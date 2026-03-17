#!/bin/bash
# AltPO (Alternate Preference Optimisation) for DUET.
#
# PREREQUISITE: alternate answers must be generated before training.
# Run once per forget split and model, then set ALTPO_DATA_DIR:
#
#   CUDA_VISIBLE_DEVICES=0 conda run -n MU python scripts/analysis/generate_altpo_data.py \
#       --dataset duet \
#       --split city_forget_5 \
#       --model_path /mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb \
#       --output_path data/altpo/duet_city_forget_5_alt.jsonl
#
# The script expects one JSONL file per forget split named
#   ${ALTPO_DATA_DIR}/duet_${forget_label}_alt.jsonl
# where each line is {"question": "...", "answer": "...", "alternate": "..."}.

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")
source "${script_dir}/_splits.sh"

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
hf_base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
local_sft_base="${LOCAL_SFT_BASE:-/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb}"

use_sft_base=${USE_SFT_BASE:-1}
if [[ "${use_sft_base}" == "1" ]]; then
    base_model_path="${local_sft_base}"
    echo "[duet][AltPO] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    echo "[duet][AltPO] Using Hugging Face base checkpoint ${base_model_path}"
fi

# Directory containing the pre-generated alternate data JSONL files.
ALTPO_DATA_DIR="${ALTPO_DATA_DIR:-${repo_root}/data/altpo}"

# AltPO is implemented as DPO in this codebase.
# The forget loss treats the alternate answer as the "preferred" response and
# the original answer as the "rejected" response, driving forgetting via DPO.
experiment="unlearn/duet/wga_lora.yaml"
trainer="DPO"

output_root="${repo_root}/saves/unlearn/duet/NEW_BUGFIX_altpo"
mkdir -p "${output_root}"

set_forget_retain_splits

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
gradient_accumulation_steps=${GRAD_ACCUM:-32}
num_train_epochs=${NUM_EPOCHS:-5}

raw_lrs="${LRS:-1e-4}"
raw_lrs="${raw_lrs//,/ }"; raw_lrs="${raw_lrs//\"/}"; raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

# beta: DPO temperature (recommended: 0.05–0.5; start at 0.5)
raw_betas="${BETAS:-0.5}"
raw_betas="${raw_betas//,/ }"; raw_betas="${raw_betas//\"/}"; raw_betas="${raw_betas//\'/}"
read -r -a betas <<< "${raw_betas}"

raw_alphas="${ALPHAS:-1.0}"
raw_alphas="${raw_alphas//,/ }"; raw_alphas="${raw_alphas//\"/}"; raw_alphas="${raw_alphas//\'/}"
read -r -a alphas <<< "${raw_alphas}"

raw_gammas="${GAMMAS:-1.0}"
raw_gammas="${raw_gammas//,/ }"; raw_gammas="${raw_gammas//\"/}"; raw_gammas="${raw_gammas//\'/}"
read -r -a gammas <<< "${raw_gammas}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for split in "${forget_retain_splits[@]}"; do
    read -r forget_split retain_split forget_label <<< "${split}"
    if [[ -z "${forget_label:-}" ]]; then
        forget_label="${forget_split}"
    fi

    alt_data_file="${ALTPO_DATA_DIR}/duet_${forget_label}_alt.jsonl"
    if [[ ! -f "${alt_data_file}" ]]; then
        echo "[duet][AltPO] ERROR: alternate data file not found: ${alt_data_file}"
        echo "  Run scripts/analysis/generate_altpo_data.py first (see script header for usage)."
        exit 1
    fi

    for lr in "${lrs[@]}"; do
        for beta in "${betas[@]}"; do
            beta_tag=${beta//./p}
            for alpha in "${alphas[@]}"; do
                alpha_tag=${alpha//./p}
                for gamma in "${gammas[@]}"; do
                    gamma_tag=${gamma//./p}
                    for lora_r in "${lora_rs[@]}"; do
                        for lora_alpha in "${lora_alphas[@]}"; do
                            for lora_dropout in "${lora_dropouts[@]}"; do
                                dropout_tag=${lora_dropout//./p}
                                task_name=duet_${base_model}_${forget_label}_altpo_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_beta${beta_tag}_alpha${alpha_tag}_gamma${gamma_tag}
                                run_dir=${output_root}/${task_name}
                                eval_dir=${run_dir}/evals
                                summary_path=${eval_dir}/DUET_SUMMARY.json

                                if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                    echo "[duet][AltPO] Skipping ${task_name}: found existing summary at ${summary_path}"
                                    continue
                                fi

                                echo "${task_name}: AltPO LoRA unlearning ${base_model_path} on ${forget_split}"

                                adapter_path=${run_dir}/adapter_model.safetensors
                                log_file=${run_dir}/altpo.log
                                if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
                                    mkdir -p "${run_dir}"
                                    echo "[TRAIN] $(date) task=${task_name}" | tee -a "${log_file}"
                                    python src/train.py --config-name=unlearn.yaml \
                                        experiment=${experiment} \
                                        trainer=${trainer} \
                                        task_name=${task_name} \
                                        model=${lora_model} \
                                        forget_split=${forget_split} \
                                        retain_split=${retain_split} \
                                        model.model_args.pretrained_model_name_or_path=${base_model_path} \
                                        model.model_args.device_map="auto" \
                                        model.model_args.low_cpu_mem_usage=true \
                                        model.lora_config.r=${lora_r} \
                                        model.lora_config.lora_alpha=${lora_alpha} \
                                        model.lora_config.lora_dropout=${lora_dropout} \
                                        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                                        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                                        trainer.args.num_train_epochs=${num_train_epochs} \
                                        trainer.args.learning_rate=${lr} \
                                        trainer.method_args.beta=${beta} \
                                        trainer.method_args.alpha=${alpha} \
                                        trainer.method_args.gamma=${gamma} \
                                        trainer.method_args.retain_loss_type=NLL \
                                        "data.forget.DUET_QA_forget.handler=QAwithAlternateDataset" \
                                        "data.forget.DUET_QA_forget.args.hf_args.path=json" \
                                        "data.forget.DUET_QA_forget.args.hf_args.split=train" \
                                        "+data.forget.DUET_QA_forget.args.hf_args.data_files=${alt_data_file}" \
                                        "+data.forget.DUET_QA_forget.args.alternate_key=alternate" \
                                        "+data.forget.DUET_QA_forget.args.return_original=True" \
                                        retain_logs_path=null \
                                        paths.output_dir=${run_dir} \
                                        |& tee -a "${log_file}"
                                fi

                                mkdir -p "${eval_dir}"
                                if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
                                    rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
                                fi

                                eval_cmd=(
                                    experiment=eval/duet/default.yaml
                                    model=${lora_model}
                                    forget_split=${forget_split}
                                    holdout_split=${retain_split}
                                    task_name=${task_name}
                                    model.model_args.pretrained_model_name_or_path=${run_dir}
                                    model.model_args.base_model_name_or_path=${base_model_path}
                                    model.model_args.device_map="auto"
                                    model.model_args.low_cpu_mem_usage=true
                                    model.lora_config.r=${lora_r}
                                    model.lora_config.lora_alpha=${lora_alpha}
                                    model.lora_config.lora_dropout=${lora_dropout}
                                    eval.duet.overwrite=true
                                    paths.output_dir=${eval_dir}
                                    retain_logs_path=null
                                )
                                echo "[EVAL] $(date) task=${task_name}" | tee -a "${log_file}"
                                python src/eval.py "${eval_cmd[@]}" |& tee -a "${log_file}"
                            done
                        done
                    done
                done
            done
        done
    done
done
