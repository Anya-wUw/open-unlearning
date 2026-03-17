#!/usr/bin/env python3
"""adapop_anchor_sensitivity.py

Sensitivity analysis of AdaPop's power-law coefficients a and b.

Each training sample has its own pop_sum_i. Beta is computed per sample as:
    beta_i = clip(a * pop_sum_i^(-b), 0.05, 2.0)

We perturb a and b independently by ±20% to produce 4 configs,
train AdaPop at lr=1e-4 for each, and report forget/retain ROUGE-L.
The baseline (a=58.7, b=0.796) reuses existing NEW_BUGFIX_ada_pop results.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n MU \\
        python scripts/analysis/adapop_anchor_sensitivity.py
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n MU \\
        python scripts/analysis/adapop_anchor_sensitivity.py --skip-train
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import subprocess
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME   = "Llama-3.1-8B-Instruct"
LORA_MODEL   = "Llama-3.1-8B-Instruct-lora"
SFT_PATH     = ("/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune"
                "/llama3.1-8b_full_3ep_ft_tripunlamb")

FORGET_SPLIT = "city_forget_rare_5+city_forget_popular_5"
RETAIN_SPLIT = "city_fast_retain_500"
FORGET_LABEL = "city_forget_5"

LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.0
LR           = "1e-4"
BS           = 1
GRAD_ACCUM   = 32
NUM_EPOCHS   = 5

# Baseline coefficients (used in all existing experiments)
A_BASE = 58.7
B_BASE = 0.796
PERTURB = 0.20  # ±20%

# ── Configs ───────────────────────────────────────────────────────────────────

CONFIGS = {}
for tag, a_mult, b_mult in [
    ("baseline", 1.0,          1.0         ),
    ("+a+b",     1 + PERTURB,  1 + PERTURB ),
    ("+a-b",     1 + PERTURB,  1 - PERTURB ),
    ("-a+b",     1 - PERTURB,  1 + PERTURB ),
    ("-a-b",     1 - PERTURB,  1 - PERTURB ),
]:
    CONFIGS[tag] = {
        "a": round(A_BASE * a_mult, 4),
        "b": round(B_BASE * b_mult, 4),
    }

# Path to existing baseline result (reused, no retraining needed)
BASELINE_EVAL_GLOB = (
    f"saves/unlearn/duet/NEW_BUGFIX_ada_pop"
    f"/duet_{MODEL_NAME}_{FORGET_LABEL}_ada_pop_lora_r{LORA_R}"
    f"_lalpha{LORA_ALPHA}_ldrop0p0_lr{LR}_adyn_bdyn_gamma1p0/evals/DUET_SUMMARY.json"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def run(cmd):
    print(f"\n[CMD] {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def task_name_for(tag: str) -> str:
    safe = tag.replace("+", "p").replace("-", "m")
    return (
        f"duet_{MODEL_NAME}_{FORGET_LABEL}_ada_pop"
        f"_lora_r{LORA_R}_lalpha{LORA_ALPHA}_ldrop0p0"
        f"_lr{LR}_coef_sens_{safe}"
    )


def run_dir_for(tag: str) -> Path:
    return (REPO_ROOT / "saves" / "unlearn" / "duet"
            / "adapop_coef_sensitivity" / task_name_for(tag))


def eval_dir_for(tag: str) -> Path:
    return run_dir_for(tag) / "evals"


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


# ── Train ──────────────────────────────────────────────────────────────────────

def train(tag: str, cfg: dict) -> None:
    rd = run_dir_for(tag)
    if (rd / "adapter_model.safetensors").exists():
        print(f"[SKIP train] {tag} — adapter already exists")
        return
    rd.mkdir(parents=True, exist_ok=True)
    run([
        "python", "src/train.py",
        "--config-name=unlearn.yaml",
        "experiment=unlearn/duet/wga_lora.yaml",
        "trainer=AdaPop",
        f"task_name={task_name_for(tag)}",
        f"model={LORA_MODEL}",
        f"forget_split={FORGET_SPLIT}",
        f"retain_split={RETAIN_SPLIT}",
        f"model.model_args.pretrained_model_name_or_path={SFT_PATH}",
        "model.model_args.device_map=auto",
        "++model.model_args.low_cpu_mem_usage=true",
        f"model.lora_config.r={LORA_R}",
        f"model.lora_config.lora_alpha={LORA_ALPHA}",
        f"model.lora_config.lora_dropout={LORA_DROPOUT}",
        f"trainer.args.per_device_train_batch_size={BS}",
        f"trainer.args.gradient_accumulation_steps={GRAD_ACCUM}",
        f"trainer.args.num_train_epochs={NUM_EPOCHS}",
        f"trainer.args.learning_rate={LR}",
        "trainer.method_args.gamma=1.0",
        "trainer.method_args.retain_loss_type=NLL",
        f"trainer.method_args.beta_a={cfg['a']}",
        f"trainer.method_args.beta_b={cfg['b']}",
        "retain_logs_path=null",
        f"paths.output_dir={run_dir_for(tag)}",
    ])


# ── Eval ───────────────────────────────────────────────────────────────────────

def evaluate(tag: str) -> dict:
    ed = eval_dir_for(tag)
    summary = ed / "DUET_SUMMARY.json"
    if summary.exists():
        print(f"[SKIP eval] {tag} — summary exists")
        return load_summary(summary)
    ed.mkdir(parents=True, exist_ok=True)
    run([
        "python", "src/eval.py",
        "experiment=eval/duet/default.yaml",
        f"model={LORA_MODEL}",
        f"forget_split={FORGET_SPLIT}",
        f"holdout_split={RETAIN_SPLIT}",
        f"task_name={task_name_for(tag)}_eval",
        f"model.model_args.pretrained_model_name_or_path={run_dir_for(tag)}",
        f"model.model_args.base_model_name_or_path={SFT_PATH}",
        "model.model_args.device_map=auto",
        "++model.model_args.low_cpu_mem_usage=true",
        f"model.lora_config.r={LORA_R}",
        f"model.lora_config.lora_alpha={LORA_ALPHA}",
        f"model.lora_config.lora_dropout={LORA_DROPOUT}",
        "eval.duet.overwrite=true",
        f"paths.output_dir={ed}",
        "retain_logs_path=null",
    ])
    return load_summary(summary)


# ── Table ──────────────────────────────────────────────────────────────────────

def print_and_save_table(results: dict) -> None:
    out_dir = REPO_ROOT / "saves" / "plots" / "adapop_coef_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag, cfg in CONFIGS.items():
        res = results.get(tag, {})
        rows.append({
            "tag":    tag,
            "a":      cfg["a"],
            "b":      cfg["b"],
            "forget": res.get("forget_qa_rouge", float("nan")),
            "retain": res.get("holdout_qa_rouge", float("nan")),
        })

    print(f"\n{'='*60}")
    print(f"AdaPop Coefficient Sensitivity — {MODEL_NAME} / DUET / lr={LR}")
    print(f"  beta_i = clip(a * pop_sum_i^(-b), 0.05, 2.0)")
    print(f"{'='*60}")
    print(f"{'Config':<10} {'a':>7} {'b':>6} {'Forget↓':>10} {'Retain↑':>10}")
    print("-" * 48)
    for r in rows:
        print(f"{r['tag']:<10} {r['a']:>7.3f} {r['b']:>6.4f} {r['forget']:>10.4f} {r['retain']:>10.4f}")

    csv_lines = ["config,a,b,forget_rouge_l,retain_rouge_l"]
    for r in rows:
        csv_lines.append(f"{r['tag']},{r['a']},{r['b']},{r['forget']:.4f},{r['retain']:.4f}")
    csv_path = out_dir / "coef_sensitivity.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n")

    md_lines = [
        "| Config | a | b | Forget ROUGE-L↓ | Retain ROUGE-L↑ |",
        "|:-------|--:|--:|----------------:|----------------:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['tag']} | {r['a']:.3f} | {r['b']:.4f} | {r['forget']:.4f} | {r['retain']:.4f} |"
        )
    md_path = out_dir / "coef_sensitivity.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"\nCSV → {csv_path}")
    print(f"MD  → {md_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; only eval existing checkpoints")
    args = parser.parse_args()

    print("\nConfigs (beta_i = clip(a * pop_sum_i^(-b), 0.05, 2.0)):")
    print(f"{'Config':<10} {'a':>7} {'b':>6}")
    print("-" * 28)
    for tag, cfg in CONFIGS.items():
        print(f"{tag:<10} {cfg['a']:>7.3f} {cfg['b']:>6.4f}")

    results = {}

    for tag, cfg in CONFIGS.items():
        print(f"\n{'='*55}\n[{tag}]  a={cfg['a']}  b={cfg['b']}\n{'='*55}")

        if tag == "baseline":
            existing = list(REPO_ROOT.glob(BASELINE_EVAL_GLOB))
            if existing:
                print(f"[baseline] Reusing existing result: {existing[0]}")
                results[tag] = load_summary(existing[0])
                continue
            print("[baseline] No existing result found — will train fresh.")

        if not args.skip_train:
            train(tag, cfg)

        try:
            results[tag] = evaluate(tag)
        except Exception as e:
            print(f"[WARN] eval failed for {tag}: {e}")
            results[tag] = {}

    print_and_save_table(results)


if __name__ == "__main__":
    main()
