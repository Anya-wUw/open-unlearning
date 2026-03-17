#!/usr/bin/env python3
"""adapop_component_abl.py

2×2 ablation isolating AdaPop's two independent contributions,
swept over a range of learning rates.

  Axis 1 (rows):    popularity-sensitive beta  vs  fixed beta=1
  Axis 2 (columns): dual-ascent controller ON  vs  dual-ascent controller OFF

  | Dual ON (alpha_const=None)       | Dual OFF (alpha_const=1.0)          |
  |----------------------------------|-------------------------------------|
  | AdaPop-Full   (beta dyn, a dyn)  | AdaPop-NoDual  (beta dyn, a=1)      |
  | AdaPop-BetaC  (beta=1,   a dyn)  | AdaPop-NoPop-NoDual (beta=1, a=1)   |

Produces:
  - CSV with columns: variant, lr, forget_rouge_l, retain_rouge_l
  - Line plot: x=LR (log scale), 4 lines per panel (forget / retain)
  - Markdown 2×2 table at best LR

Existing checkpoints (e.g. from a previous single-LR run) are reused automatically.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u scripts/adapop_component_abl.py
    CUDA_VISIBLE_DEVICES=0 python -u scripts/adapop_component_abl.py --skip-train
    CUDA_VISIBLE_DEVICES=0 python -u scripts/adapop_component_abl.py \\
        --lrs "1e-5 1e-4 5e-4"
    CUDA_VISIBLE_DEVICES=0 python -u scripts/adapop_component_abl.py \\
        --reuse-full-adapop
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Default LR sweep (matches main scripts) ───────────────────────────────────
DEFAULT_LRS = "1e-6 5e-6 1e-5 4e-5 5e-5 1e-4 5e-4"

# ── Shared training config ────────────────────────────────────────────────────
FORGET_SPLIT = "city_forget_rare_5+city_forget_popular_5"
RETAIN_SPLIT = "city_fast_retain_500"
FORGET_LABEL = "city_forget_5"
LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.0
BS           = 1
GRAD_ACCUM   = 32
NUM_EPOCHS   = 5

# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "Llama-3.1-8B-Instruct": {
        "lora_model":  "Llama-3.1-8B-Instruct-lora",
        "plain_model": "Llama-3.1-8B-Instruct",
        "sft_path": (
            "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune"
            "/llama3.1-8b_full_3ep_ft_tripunlamb"
        ),
    },
    "gemma-7b-it": {
        "lora_model":  "gemma-7b-it-lora",
        "plain_model": "gemma-7b-it",
        "sft_path": (
            "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune"
            "/gemma-7b-it_full_3ep_ft_tripunlamb"
        ),
    },
    "Qwen2.5-7B-Instruct": {
        "lora_model":  "Qwen2.5-7B-Instruct-lora",
        "plain_model": "Qwen2.5-7B-Instruct",
        "sft_path": (
            "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune"
            "/Qwen2.5-7B-Instruct_full_3ep_ft_tripunlamb"
        ),
    },
}

# ── 2×2 variant definitions ───────────────────────────────────────────────────
VARIANTS = {
    "AdaPop-Full": {
        "beta_const":  None,
        "alpha_const": None,
        "label": "AdaPop-Full (β dyn, dual ON)",
        "color": "#4C72B0",
        "row": 0, "col": 0,
    },
    "AdaPop-NoDual": {
        "beta_const":  None,
        "alpha_const": 1.0,
        "label": "AdaPop-NoDual (β dyn, dual OFF)",
        "color": "#DD8452",
        "row": 0, "col": 1,
    },
    "AdaPop-BetaConst": {
        "beta_const":  1.0,
        "alpha_const": None,
        "label": "AdaPop-BetaConst (β=1, dual ON)",
        "color": "#55A868",
        "row": 1, "col": 0,
    },
    "AdaPop-NoPop-NoDual": {
        "beta_const":  1.0,
        "alpha_const": 1.0,
        "label": "AdaPop-NoPop-NoDual (β=1, dual OFF)",
        "color": "#C44E52",
        "row": 1, "col": 1,
    },
}

VARIANT_ORDER = list(VARIANTS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd: list[str]) -> None:
    print(f"\n[CMD] {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def lr_tag(lr: str) -> str:
    return lr.replace(".", "p")


def task_name_for(variant: str, model_name: str, lr: str) -> str:
    v = VARIANTS[variant]
    btag = f"b{str(v['beta_const']).replace('.','p')}" if v["beta_const"] is not None else "bdyn"
    atag = f"a{str(v['alpha_const']).replace('.','p')}" if v["alpha_const"] is not None else "adyn"
    return (
        f"duet_{model_name}_{FORGET_LABEL}_adapop_abl"
        f"_lora_r{LORA_R}_la{LORA_ALPHA}_lr{lr}_{atag}_{btag}"
    )


def _main_adapop_summary(model_name: str, lr: str) -> Path | None:
    """Path to DUET_SUMMARY.json for the main AdaPop run at this lr, if it exists."""
    run_name = (
        f"duet_{model_name}_{FORGET_LABEL}_ada_pop"
        f"_lora_r{LORA_R}_lalpha{LORA_ALPHA}_ldrop0p0"
        f"_lr{lr}_adyn_bdyn_gamma1p0"
    )
    p = (
        REPO_ROOT / "saves" / "unlearn" / "duet" / "NEW_BUGFIX_ada_pop"
        / run_name / "evals" / "DUET_SUMMARY.json"
    )
    return p if p.exists() else None


# ── Train ─────────────────────────────────────────────────────────────────────
def train_variant(
    variant: str, run_dir: Path, lr: str,
    model_name: str, base_model_path: str, lora_model: str,
) -> None:
    v = VARIANTS[variant]
    run_dir.mkdir(parents=True, exist_ok=True)
    extra: list[str] = []
    if v["beta_const"] is not None:
        extra.append(f"trainer.method_args.beta_const={v['beta_const']}")
    if v["alpha_const"] is not None:
        extra.append(f"trainer.method_args.alpha_const={v['alpha_const']}")

    run([
        "python", "src/train.py",
        "--config-name=unlearn.yaml",
        "experiment=unlearn/duet/wga_lora.yaml",
        "trainer=AdaPop",
        f"task_name={task_name_for(variant, model_name, lr)}",
        f"model={lora_model}",
        f"forget_split={FORGET_SPLIT}",
        f"retain_split={RETAIN_SPLIT}",
        f"model.model_args.pretrained_model_name_or_path={base_model_path}",
        "model.model_args.device_map=auto",
        "++model.model_args.low_cpu_mem_usage=true",
        f"model.lora_config.r={LORA_R}",
        f"model.lora_config.lora_alpha={LORA_ALPHA}",
        f"model.lora_config.lora_dropout={LORA_DROPOUT}",
        f"trainer.args.per_device_train_batch_size={BS}",
        f"trainer.args.gradient_accumulation_steps={GRAD_ACCUM}",
        f"trainer.args.num_train_epochs={NUM_EPOCHS}",
        f"trainer.args.learning_rate={lr}",
        "trainer.method_args.gamma=1.0",
        "trainer.method_args.retain_loss_type=NLL",
        "retain_logs_path=null",
        f"paths.output_dir={run_dir}",
    ] + extra)


# ── Evaluate ──────────────────────────────────────────────────────────────────
def eval_variant(
    variant: str, run_dir: Path, eval_dir: Path, lr: str,
    model_name: str, base_model_path: str, lora_model: str,
) -> dict:
    summary_path = eval_dir / "DUET_SUMMARY.json"
    if summary_path.exists():
        print(f"[SKIP eval] {eval_dir} — summary exists", flush=True)
        with open(summary_path) as f:
            return json.load(f)

    eval_dir.mkdir(parents=True, exist_ok=True)
    run([
        "python", "src/eval.py",
        "experiment=eval/duet/default.yaml",
        f"model={lora_model}",
        f"forget_split={FORGET_SPLIT}",
        f"holdout_split={RETAIN_SPLIT}",
        f"task_name={task_name_for(variant, model_name, lr)}",
        f"model.model_args.pretrained_model_name_or_path={run_dir}",
        f"model.model_args.base_model_name_or_path={base_model_path}",
        "model.model_args.device_map=auto",
        "++model.model_args.low_cpu_mem_usage=true",
        f"model.lora_config.r={LORA_R}",
        f"model.lora_config.lora_alpha={LORA_ALPHA}",
        f"model.lora_config.lora_dropout={LORA_DROPOUT}",
        "eval.duet.overwrite=true",
        f"paths.output_dir={eval_dir}",
        "retain_logs_path=null",
    ])
    with open(summary_path) as f:
        return json.load(f)


# ── Save results ──────────────────────────────────────────────────────────────
# results: {variant: {lr: summary_dict}}

def save_csv(results: dict, out_dir: Path, model_name: str) -> Path:
    rows = []
    for variant in VARIANT_ORDER:
        if variant not in results:
            continue
        v = VARIANTS[variant]
        for lr, summary in sorted(results[variant].items(), key=lambda x: float(x[0])):
            rows.append({
                "variant":        variant,
                "lr":             lr,
                "beta_const":     str(v["beta_const"]),
                "alpha_const":    str(v["alpha_const"]),
                "forget_rouge_l": round(summary.get("forget_qa_rouge", float("nan")), 4),
                "retain_rouge_l": round(summary.get("holdout_qa_rouge", float("nan")), 4),
            })
    csv_path = out_dir / f"adapop_abl_{model_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "lr", "beta_const", "alpha_const",
                           "forget_rouge_l", "retain_rouge_l"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def save_plot(results: dict, out_dir: Path, model_name: str) -> None:
    fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

    for variant in VARIANT_ORDER:
        if variant not in results or not results[variant]:
            continue
        v = VARIANTS[variant]
        lr_vals = sorted(results[variant].keys(), key=float)
        xs = [float(lr) for lr in lr_vals]
        forget_ys = [results[variant][lr].get("forget_qa_rouge", float("nan")) for lr in lr_vals]
        retain_ys = [results[variant][lr].get("holdout_qa_rouge", float("nan")) for lr in lr_vals]

        ax_f.plot(xs, forget_ys, marker="o", linewidth=2,
                  color=v["color"], label=v["label"])
        ax_r.plot(xs, retain_ys, marker="s", linewidth=2,
                  color=v["color"], label=v["label"])

    for ax, title in [
        (ax_f, f"Forget ROUGE-L ↓ — {model_name} / DUET"),
        (ax_r, f"Retain ROUGE-L ↑ — {model_name} / DUET"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("ROUGE-L (recall)", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    fig.suptitle(
        f"AdaPop Component Ablation — LR sweep\n"
        f"LoRA r={LORA_R}/α={LORA_ALPHA}, {NUM_EPOCHS} epochs",
        fontsize=12,
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"adapop_abl_{model_name}.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"[PLOT] → {p}", flush=True)
    plt.close()


def print_table(results: dict, model_name: str) -> None:
    print(f"\n{'='*75}", flush=True)
    print(f"AdaPop Component Ablation — {model_name} / DUET", flush=True)
    print(f"{'='*75}", flush=True)
    print(f"{'Variant':<28}  {'LR':>8}  {'Forget↓':>10}  {'Retain↑':>10}", flush=True)
    print("-" * 65, flush=True)
    for variant in VARIANT_ORDER:
        if variant not in results:
            continue
        for lr in sorted(results[variant].keys(), key=float):
            s = results[variant][lr]
            f_val = s.get("forget_qa_rouge", float("nan"))
            r_val = s.get("holdout_qa_rouge", float("nan"))
            print(f"{variant:<28}  {lr:>8}  {f_val:>10.4f}  {r_val:>10.4f}", flush=True)


def save_all(results: dict, out_dir: Path, model_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_csv(results, out_dir, model_name)
    save_plot(results, out_dir, model_name)
    print_table(results, model_name)
    print(f"\nCSV → {csv_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--lrs", default=DEFAULT_LRS,
                        help="Space-separated list of learning rates")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; eval existing checkpoints only.")
    parser.add_argument("--reuse-full-adapop", action="store_true",
                        help="Load AdaPop-Full results from main NEW_BUGFIX_ada_pop runs.")
    args = parser.parse_args()

    lrs = args.lrs.replace(",", " ").split()
    cfg             = MODEL_CONFIGS[args.model]
    lora_model      = cfg["lora_model"]
    base_model_path = cfg["sft_path"]

    abl_root = REPO_ROOT / "saves" / "unlearn" / "duet" / "adapop_component_abl"
    out_dir  = REPO_ROOT / "saves" / "plots" / "adapop_component_abl"

    os.chdir(REPO_ROOT)

    # results[variant][lr] = summary_dict
    results: dict[str, dict[str, dict]] = {v: {} for v in VARIANTS}

    for lr in lrs:
        print(f"\n{'#'*70}", flush=True)
        print(f"LR = {lr}", flush=True)
        print(f"{'#'*70}", flush=True)

        for variant in VARIANTS:
            print(f"\n{'='*65}", flush=True)
            print(f"{variant}  lr={lr}  —  {args.model}", flush=True)
            print(f"{'='*65}", flush=True)

            # AdaPop-Full: optionally reuse main results
            if variant == "AdaPop-Full" and args.reuse_full_adapop:
                existing = _main_adapop_summary(args.model, lr)
                if existing is not None:
                    print(f"[REUSE] {existing}", flush=True)
                    with open(existing) as f:
                        results[variant][lr] = json.load(f)
                    save_all(results, out_dir, args.model)
                    continue
                else:
                    print(f"[WARN] No main AdaPop-Full result for lr={lr}; training.", flush=True)

            run_dir  = abl_root / task_name_for(variant, args.model, lr)
            eval_dir = run_dir / "evals"

            # ── Train ────────────────────────────────────────────────────────
            adapter = run_dir / "adapter_model.safetensors"
            if args.skip_train:
                if not (run_dir / "trainer_state.json").exists():
                    print(f"[SKIP {variant} lr={lr}] No checkpoint found.", flush=True)
                    continue
                print(f"[--skip-train] {run_dir}", flush=True)
            elif adapter.exists():
                print(f"[SKIP training] Adapter exists: {adapter}", flush=True)
            else:
                print(f"Training {variant} lr={lr} …", flush=True)
                train_variant(variant, run_dir, lr, args.model, base_model_path, lora_model)

            # ── Evaluate ─────────────────────────────────────────────────────
            print(f"Evaluating {variant} lr={lr} …", flush=True)
            results[variant][lr] = eval_variant(
                variant, run_dir, eval_dir, lr, args.model, base_model_path, lora_model,
            )

            # ── Incremental save ──────────────────────────────────────────────
            save_all(results, out_dir, args.model)

    print(f"\n{'='*65}", flush=True)
    print("All done.", flush=True)
    save_all(results, out_dir, args.model)
    print(f"Outputs → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
