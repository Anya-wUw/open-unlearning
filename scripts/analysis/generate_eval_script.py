import os
import json
import glob
from pathlib import Path

base_dir = "/mnt/extremessd10tb/borisiuk/new_MU_exps/open-unlearning/saves/unlearn"
datasets = ["duet", "rwku"]
target_lr = "1e-4"

models_to_eval = []
origin_models = set()

# Find unlearned models
for dataset in datasets:
    search_path = os.path.join(base_dir, dataset, "NEW_BUGFIX_*", f"*{target_lr}*")
    for d in glob.glob(search_path):
        if os.path.isdir(d):
            adapter_conf = os.path.join(d, "adapter_config.json")
            if os.path.exists(adapter_conf):
                with open(adapter_conf, "r") as f:
                    conf = json.load(f)
                    base_model = conf.get("base_model_name_or_path")
                    if base_model:
                        origin_models.add(base_model)
                        
                        # Extract algorithm
                        folder_name = os.path.basename(d)
                        parent_folder = os.path.basename(os.path.dirname(d))
                        algo = parent_folder.replace("NEW_BUGFIX_", "")
                        
                        # We also need model name to distinguish
                        if "gemma" in folder_name.lower():
                            model_name = "gemma-7b-it"
                        elif "llama" in folder_name.lower():
                            model_name = "Llama-3.1-8B-Instruct"
                        elif "qwen" in folder_name.lower():
                            model_name = "Qwen2.5-7B-Instruct"
                        else:
                            model_name = "Unknown"
                            
                        models_to_eval.append({
                            "type": "unlearned",
                            "path": d,
                            "base_model": base_model,
                            "dataset": dataset,
                            "algo": algo,
                            "lr": target_lr,
                            "model_name": model_name
                        })

# Generate bash script
output_dir = "/mnt/extremessd10tb/borisiuk/new_MU_exps/open-unlearning/Benchmark_Evaluation"
bash_lines = [
    "#!/bin/bash",
    "export CUDA_VISIBLE_DEVICES=0",
    f"mkdir -p {output_dir}",
    "echo 'Starting evaluations...'",
    ""
]

# Add origin models
for i, base in enumerate(origin_models):
    model_name = os.path.basename(base)
    output_path = os.path.join(output_dir, f"origin_{model_name}")
    cmd = f"lm_eval --model hf --model_args pretrained={base} --apply_chat_template --tasks mmlu,hellaswag --device cuda:0 --batch_size auto:4 --output_path {output_path}"
    bash_lines.append(f"echo 'Evaluating Origin: {model_name}'")
    bash_lines.append(cmd)
    bash_lines.append("")

# Add unlearned models
for m in models_to_eval:
    run_name = f"{m['dataset']}_{m['algo']}_{m['model_name']}_lr{m['lr']}"
    output_path = os.path.join(output_dir, run_name)
    cmd = f"lm_eval --model hf --model_args pretrained={m['base_model']},peft={m['path']} --apply_chat_template --tasks mmlu,hellaswag --device cuda:0 --batch_size auto:4 --output_path {output_path}"
    bash_lines.append(f"echo 'Evaluating Unlearned: {run_name}'")
    bash_lines.append(cmd)
    bash_lines.append("")

with open("run_all_evals.sh", "w") as f:
    for line in bash_lines:
        f.write(line + "\\n")

print(f"Generated bash script to evaluate {len(origin_models)} origin models and {len(models_to_eval)} unlearned models.")
