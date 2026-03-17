#!/bin/bash
# Run NPO, WGA, AdaPop at lr=1e-4 with seeds 1 and 219 across all 3 models × 2 benchmarks.
# Seed 42 (baseline) is already done; this adds 2 extra seeds for std calculation.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/eval/run_extra_seeds.sh

set -euo pipefail

REPO_ROOT=$(realpath "$(dirname "$0")/../..")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

LR="1e-4"
SEEDS=(1 219)

# ── Model configs ──────────────────────────────────────────────────────────────
declare -A MODEL_CONFIG=(
    [Llama-3.1-8B-Instruct]="Llama-3.1-8B-Instruct-lora"
    [gemma-7b-it]="gemma-7b-it-lora"
    [Qwen2.5-7B-Instruct]="Qwen2.5-7B-Instruct-lora"
)
declare -A HF_PATH=(
    [Llama-3.1-8B-Instruct]="meta-llama/Llama-3.1-8B-Instruct"
    [gemma-7b-it]="google/gemma-7b-it"
    [Qwen2.5-7B-Instruct]="Qwen/Qwen2.5-7B-Instruct"
)
declare -A SFT_PATH=(
    [Llama-3.1-8B-Instruct]="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb"
    [gemma-7b-it]="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/gemma-7b-it_full_3ep_ft_tripunlamb"
    [Qwen2.5-7B-Instruct]="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/Qwen2.5-7B-Instruct_full_3ep_ft_tripunlamb"
)

MODELS=(Llama-3.1-8B-Instruct gemma-7b-it Qwen2.5-7B-Instruct)

# ── Shared LoRA / training defaults ───────────────────────────────────────────
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.0
LORA_DROPOUT_TAG="0p0"
EPOCHS=5
BS=1
GRAD_ACCUM=32

# ── Helper: train + eval ───────────────────────────────────────────────────────
run_train() {
    local run_dir="$1"; shift
    python src/train.py --config-name=unlearn.yaml "$@" \
        model.model_args.device_map="auto" \
        model.model_args.low_cpu_mem_usage=true \
        model.lora_config.r=${LORA_R} \
        model.lora_config.lora_alpha=${LORA_ALPHA} \
        model.lora_config.lora_dropout=${LORA_DROPOUT} \
        trainer.args.per_device_train_batch_size=${BS} \
        trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
        trainer.args.num_train_epochs=${EPOCHS} \
        trainer.args.learning_rate=${LR} \
        paths.output_dir=${run_dir}
}

run_eval_duet() {
    local run_dir="$1"; local eval_dir="$2"; local base_model_path="$3"
    local lora_model="$4"; local forget_split="$5"; local retain_split="$6"; local task_name="$7"
    python src/eval.py \
        experiment=eval/duet/default.yaml \
        model=${lora_model} \
        forget_split=${forget_split} \
        holdout_split=${retain_split} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${run_dir} \
        model.model_args.base_model_name_or_path=${base_model_path} \
        model.model_args.device_map="auto" \
        model.model_args.low_cpu_mem_usage=true \
        model.lora_config.r=${LORA_R} \
        model.lora_config.lora_alpha=${LORA_ALPHA} \
        model.lora_config.lora_dropout=${LORA_DROPOUT} \
        eval.duet.overwrite=true \
        paths.output_dir=${eval_dir} \
        retain_logs_path=null
}

run_eval_rwku() {
    local run_dir="$1"; local eval_dir="$2"; local base_model_path="$3"
    local lora_model="$4"; local forget_split="$5"; local retain_split="$6"; local task_name="$7"
    python src/eval.py \
        experiment=eval/rwku/default.yaml \
        model=${lora_model} \
        forget_split=${forget_split} \
        holdout_split=${retain_split} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${run_dir} \
        model.model_args.base_model_name_or_path=${base_model_path} \
        model.model_args.device_map="auto" \
        model.model_args.low_cpu_mem_usage=true \
        model.lora_config.r=${LORA_R} \
        model.lora_config.lora_alpha=${LORA_ALPHA} \
        model.lora_config.lora_dropout=${LORA_DROPOUT} \
        eval.duet.overwrite=true \
        paths.output_dir=${eval_dir} \
        retain_logs_path=null
}

# ── Main loop ──────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

for seed in "${SEEDS[@]}"; do
for model in "${MODELS[@]}"; do
    lora_model="${MODEL_CONFIG[$model]}"
    hf_path="${HF_PATH[$model]}"
    sft_path="${SFT_PATH[$model]}"

    echo ""
    echo "======================================================================="
    echo "  seed=${seed}  model=${model}"
    echo "======================================================================="

    # ── DUET ─────────────────────────────────────────────────────────────────
    bench="duet"
    base_model_path="${sft_path}"
    forget_split="city_forget_rare_5+city_forget_popular_5"
    retain_split="city_fast_retain_500"
    forget_label="city_forget_5"

    # NPO / DUET
    algo="npo"
    task_name="${bench}_${model}_${forget_label}_${algo}_lora_r${LORA_R}_lalpha${LORA_ALPHA}_ldrop${LORA_DROPOUT_TAG}_lr${LR}_beta0p5_alpha1p0_gamma1p0_seed${seed}"
    run_dir="${REPO_ROOT}/saves/unlearn/${bench}/NEW_BUGFIX_${algo}/${task_name}"
    eval_dir="${run_dir}/evals"
    summary="${eval_dir}/DUET_SUMMARY.json"
    if [[ -f "${summary}" ]]; then
        echo "[SKIP] ${task_name}"
    else
        echo "[RUN]  ${task_name}"
        mkdir -p "${run_dir}"
        run_train "${run_dir}" \
            experiment=unlearn/duet/grad_ascent_lora.yaml \
            trainer=NPO \
            task_name=${task_name} \
            model=${lora_model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${base_model_path} \
            trainer.args.seed=${seed} \
            trainer.method_args.beta=0.5 \
            trainer.method_args.alpha=1.0 \
            trainer.method_args.gamma=1.0 \
            trainer.method_args.retain_loss_type=NLL \
            retain_logs_path=null
        mkdir -p "${eval_dir}"
        run_eval_duet "${run_dir}" "${eval_dir}" "${base_model_path}" "${lora_model}" \
            "${forget_split}" "${retain_split}" "${task_name}"
    fi

    # WGA / DUET
    algo="wga"
    task_name="${bench}_${model}_${forget_label}_${algo}_lora_r${LORA_R}_lalpha${LORA_ALPHA}_ldrop${LORA_DROPOUT_TAG}_lr${LR}_beta1p0_alpha1p0_gamma1p0_seed${seed}"
    run_dir="${REPO_ROOT}/saves/unlearn/${bench}/NEW_BUGFIX_${algo}/${task_name}"
    eval_dir="${run_dir}/evals"
    summary="${eval_dir}/DUET_SUMMARY.json"
    if [[ -f "${summary}" ]]; then
        echo "[SKIP] ${task_name}"
    else
        echo "[RUN]  ${task_name}"
        mkdir -p "${run_dir}"
        run_train "${run_dir}" \
            experiment=unlearn/duet/wga_lora.yaml \
            trainer=WGA \
            task_name=${task_name} \
            model=${lora_model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${base_model_path} \
            trainer.args.seed=${seed} \
            trainer.method_args.beta=1.0 \
            trainer.method_args.alpha=1.0 \
            trainer.method_args.gamma=1.0 \
            trainer.method_args.retain_loss_type=NLL \
            retain_logs_path=null
        mkdir -p "${eval_dir}"
        run_eval_duet "${run_dir}" "${eval_dir}" "${base_model_path}" "${lora_model}" \
            "${forget_split}" "${retain_split}" "${task_name}"
    fi

    # AdaPop / DUET
    algo="ada_pop"
    task_name="${bench}_${model}_${forget_label}_${algo}_lora_r${LORA_R}_lalpha${LORA_ALPHA}_ldrop${LORA_DROPOUT_TAG}_lr${LR}_adyn_bdyn_gamma1p0_seed${seed}"
    run_dir="${REPO_ROOT}/saves/unlearn/${bench}/NEW_BUGFIX_${algo}/${task_name}"
    eval_dir="${run_dir}/evals"
    summary="${eval_dir}/DUET_SUMMARY.json"
    if [[ -f "${summary}" ]]; then
        echo "[SKIP] ${task_name}"
    else
        echo "[RUN]  ${task_name}"
        mkdir -p "${run_dir}"
        run_train "${run_dir}" \
            experiment=unlearn/duet/wga_lora.yaml \
            trainer=AdaPop \
            task_name=${task_name} \
            model=${lora_model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${base_model_path} \
            trainer.args.seed=${seed} \
            trainer.method_args.gamma=1.0 \
            trainer.method_args.retain_loss_type=NLL \
            retain_logs_path=null
        mkdir -p "${eval_dir}"
        run_eval_duet "${run_dir}" "${eval_dir}" "${base_model_path}" "${lora_model}" \
            "${forget_split}" "${retain_split}" "${task_name}"
    fi

    # ── RWKU ─────────────────────────────────────────────────────────────────
    bench="rwku"
    base_model_path="${hf_path}"
    forget_split="forget_level2"
    retain_split="neighbor_level2"

    # NPO / RWKU
    algo="npo"
    task_name="${bench}_${model}_${forget_split}_${algo}_lora_r${LORA_R}_lalpha${LORA_ALPHA}_ldrop${LORA_DROPOUT_TAG}_lr${LR}_beta0p5_alpha1p0_gamma1p0_seed${seed}"
    run_dir="${REPO_ROOT}/saves/unlearn/${bench}/NEW_BUGFIX_${algo}/${task_name}"
    eval_dir="${run_dir}/evals"
    summary="${eval_dir}/DUET_SUMMARY.json"
    if [[ -f "${summary}" ]]; then
        echo "[SKIP] ${task_name}"
    else
        echo "[RUN]  ${task_name}"
        mkdir -p "${run_dir}"
        run_train "${run_dir}" \
            experiment=unlearn/rwku/wga_lora.yaml \
            trainer=NPO \
            task_name=${task_name} \
            model=${lora_model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${base_model_path} \
            trainer.args.seed=${seed} \
            trainer.method_args.beta=0.5 \
            trainer.method_args.alpha=1.0 \
            trainer.method_args.gamma=1.0 \
            trainer.method_args.retain_loss_type=NLL \
            retain_logs_path=null
        mkdir -p "${eval_dir}"
        run_eval_rwku "${run_dir}" "${eval_dir}" "${base_model_path}" "${lora_model}" \
            "${forget_split}" "${retain_split}" "${task_name}"
    fi

    # WGA / RWKU
    algo="wga"
    task_name="${bench}_${model}_${forget_split}_${algo}_lora_r${LORA_R}_lalpha${LORA_ALPHA}_ldrop${LORA_DROPOUT_TAG}_lr${LR}_beta1p0_alpha1p0_gamma1p0_seed${seed}"
    run_dir="${REPO_ROOT}/saves/unlearn/${bench}/NEW_BUGFIX_${algo}/${task_name}"
    eval_dir="${run_dir}/evals"
    summary="${eval_dir}/DUET_SUMMARY.json"
    if [[ -f "${summary}" ]]; then
        echo "[SKIP] ${task_name}"
    else
        echo "[RUN]  ${task_name}"
        mkdir -p "${run_dir}"
        run_train "${run_dir}" \
            experiment=unlearn/rwku/wga_lora.yaml \
            trainer=WGA \
            task_name=${task_name} \
            model=${lora_model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${base_model_path} \
            trainer.args.seed=${seed} \
            trainer.method_args.beta=1.0 \
            trainer.method_args.alpha=1.0 \
            trainer.method_args.gamma=1.0 \
            trainer.method_args.retain_loss_type=NLL \
            retain_logs_path=null
        mkdir -p "${eval_dir}"
        run_eval_rwku "${run_dir}" "${eval_dir}" "${base_model_path}" "${lora_model}" \
            "${forget_split}" "${retain_split}" "${task_name}"
    fi

    # AdaPop / RWKU
    algo="ada_pop"
    task_name="${bench}_${model}_${forget_split}_${algo}_lora_r${LORA_R}_lalpha${LORA_ALPHA}_ldrop${LORA_DROPOUT_TAG}_lr${LR}_adyn_bdyn_gamma1p0_seed${seed}"
    run_dir="${REPO_ROOT}/saves/unlearn/${bench}/NEW_BUGFIX_${algo}/${task_name}"
    eval_dir="${run_dir}/evals"
    summary="${eval_dir}/DUET_SUMMARY.json"
    if [[ -f "${summary}" ]]; then
        echo "[SKIP] ${task_name}"
    else
        echo "[RUN]  ${task_name}"
        mkdir -p "${run_dir}"
        run_train "${run_dir}" \
            experiment=unlearn/rwku/wga_lora.yaml \
            trainer=AdaPop \
            task_name=${task_name} \
            model=${lora_model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${base_model_path} \
            trainer.args.seed=${seed} \
            trainer.method_args.gamma=1.0 \
            trainer.method_args.retain_loss_type=NLL \
            retain_logs_path=null
        mkdir -p "${eval_dir}"
        run_eval_rwku "${run_dir}" "${eval_dir}" "${base_model_path}" "${lora_model}" \
            "${forget_split}" "${retain_split}" "${task_name}"
    fi

done
done

echo ""
echo "All extra-seed runs complete."
