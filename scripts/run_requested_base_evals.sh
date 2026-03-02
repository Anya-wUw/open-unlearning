#!/bin/bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES

# DUET Evaluations
DUET_MODELS=(
    "Llama-3.1-8B-Instruct /mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb"
    "Qwen2.5-7B-Instruct /mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/Qwen2.5-7B-Instruct_full_3ep_ft_tripunlamb"
    "gemma-7b-it /mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/gemma-7b-it_full_3ep_ft_tripunlamb"
)
DUET_SPLITS=("city_forget_rare_5 city_fast_retain_500" "city_forget_popular_5 city_fast_retain_500")

for model_info in "${DUET_MODELS[@]}"; do
    base_model=$(echo $model_info | cut -d' ' -f1)
    model_path=$(echo $model_info | cut -d' ' -f2)
    for split in "${DUET_SPLITS[@]}"; do
        forget_split=$(echo "$split" | cut -d' ' -f1)
        retain_split=$(echo "$split" | cut -d' ' -f2)
        task_name="duet_${base_model}_${forget_split}_base_eval"
        output_dir="saves/evals/duet_base/${task_name}"
        
        if [ -f "${output_dir}/DUET_SUMMARY.json" ]; then
            echo "[DUET] Skipping $base_model on $forget_split (already exists at $output_dir)"
            continue
        fi

        echo "[DUET] Evaluating $base_model on $forget_split..."
        conda run --no-capture-output -n MU python src/eval.py \
            experiment=eval/duet/default.yaml \
            model=${base_model} \
            forget_split=${forget_split} \
            holdout_split=${retain_split} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            ++model.model_args.device_map=auto \
            ++model.model_args.low_cpu_mem_usage=true \
            eval.duet.overwrite=true \
            paths.output_dir=${output_dir} \
            retain_logs_path=null
    done
done

# RWKU Evaluations
RWKU_MODELS=(
    "Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct"
    "Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct"
    "gemma-7b-it google/gemma-7b-it"
)
RWKU_SPLITS=("forget_level2 neighbor_level2")

for model_info in "${RWKU_MODELS[@]}"; do
    base_model=$(echo $model_info | cut -d' ' -f1)
    model_path=$(echo $model_info | cut -d' ' -f2)
    for split in "${RWKU_SPLITS[@]}"; do
        forget_split=$(echo "$split" | cut -d' ' -f1)
        retain_split=$(echo "$split" | cut -d' ' -f2)
        task_name="rwku_${base_model}_${forget_split}_base_eval"
        output_dir="saves/evals/rwku_base/${task_name}"

        if [ -f "${output_dir}/DUET_SUMMARY.json" ]; then
            echo "[RWKU] Skipping $base_model on $forget_split (already exists at $output_dir)"
            continue
        fi

        echo "[RWKU] Evaluating $base_model on $forget_split..."
        conda run --no-capture-output -n MU python src/eval.py \
            experiment=eval/rwku/default.yaml \
            model=${base_model} \
            forget_split=${forget_split} \
            holdout_split=${retain_split} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            ++model.model_args.device_map=auto \
            ++model.model_args.low_cpu_mem_usage=true \
            paths.output_dir=${output_dir} \
            retain_logs_path=null
    done
done

# PopQA Evaluations
POPQA_MODELS=(
    "Llama-3.1-8B-Instruct /mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/popqa/llama3.1-8b_full_5ep_ft_popqa"
    "Qwen2.5-7B-Instruct /mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/popqa/qwen2.5-7b_full_5ep_ft_popqa"
    "gemma-7b-it /mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/popqa/gemma-7b-it_full_5ep_ft_popqa"
)
POPQA_SPLITS=("rare_forget5_sum fast_retain_500" "popular_forget5_sum fast_retain_500")

for model_info in "${POPQA_MODELS[@]}"; do
    base_model=$(echo $model_info | cut -d' ' -f1)
    model_path=$(echo $model_info | cut -d' ' -f2)
    for split in "${POPQA_SPLITS[@]}"; do
        forget_split=$(echo "$split" | cut -d' ' -f1)
        retain_split=$(echo "$split" | cut -d' ' -f2)
        task_name="popqa_${base_model}_${forget_split}_base_eval"
        output_dir="saves/evals/popqa_base/${task_name}"

        if [ -f "${output_dir}/DUET_SUMMARY.json" ]; then
            echo "[PopQA] Skipping $base_model on $forget_split (already exists at $output_dir)"
            continue
        fi

        echo "[PopQA] Evaluating $base_model on $forget_split..."
        conda run --no-capture-output -n MU python src/eval.py \
            experiment=eval/popqa/default.yaml \
            model=${base_model} \
            forget_split=${forget_split} \
            holdout_split=${retain_split} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            ++model.model_args.device_map=auto \
            ++model.model_args.low_cpu_mem_usage=true \
            paths.output_dir=${output_dir} \
            retain_logs_path=null
    done
done
