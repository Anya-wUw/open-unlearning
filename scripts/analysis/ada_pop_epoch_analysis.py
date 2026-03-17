#!/usr/bin/env python3
"""ada_pop_epoch_analysis.py

Train AdaPop on DUET for MAX_EPOCHS epochs (saving a checkpoint each epoch),
evaluate ROUGE-L (forget + retain) at epoch 0 (base model) and epochs 1..MAX_EPOCHS,
then save a CSV table, Markdown table, and PDF/PNG plot.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n MU python scripts/analysis/ada_pop_epoch_analysis.py
    CUDA_VISIBLE_DEVICES=0 conda run -n MU python scripts/analysis/ada_pop_epoch_analysis.py --model gemma-7b-it
    CUDA_VISIBLE_DEVICES=0 conda run -n MU python scripts/analysis/ada_pop_epoch_analysis.py --skip-train
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths & constants ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

FORGET_SPLIT = "city_forget_rare_5+city_forget_popular_5"
RETAIN_SPLIT = "city_fast_retain_500"
FORGET_LABEL = "city_forget_5"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LR = "1e-4"
BS = 1
GRAD_ACCUM = 32
MAX_EPOCHS = 10

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str]) -> None:
    print(f"\n[CMD] {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def train_ada_pop(run_dir: Path, model_name: str, base_model_path: str,
                  lora_model: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    task_name = (
        f"duet_{model_name}_{FORGET_LABEL}_ada_pop"
        f"_lora_r{LORA_R}_lalpha{LORA_ALPHA}_ldrop0p0"
        f"_lr{LR}_adyn_bdyn_gamma1p0_epoch_analysis"
    )
    run([
        "python", "src/train.py",
        "--config-name=unlearn.yaml",
        "experiment=unlearn/duet/wga_lora.yaml",
        "trainer=AdaPop",
        f"task_name={task_name}",
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
        f"trainer.args.num_train_epochs={MAX_EPOCHS}",
        f"trainer.args.learning_rate={LR}",
        "trainer.args.save_strategy=epoch",
        f"+trainer.args.save_total_limit={MAX_EPOCHS + 1}",
        "trainer.method_args.gamma=1.0",
        "trainer.method_args.retain_loss_type=NLL",
        "retain_logs_path=null",
        f"paths.output_dir={run_dir}",
    ])


def get_epoch_checkpoints(run_dir: Path) -> dict[int, Path]:
    """Return {epoch: checkpoint_dir} for epochs 1..MAX_EPOCHS."""
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found in {run_dir}")
    with open(state_path) as f:
        state = json.load(f)

    max_steps = state["max_steps"]
    num_epochs = int(round(state["num_train_epochs"]))
    steps_per_epoch = max_steps / num_epochs

    checkpoints: dict[int, Path] = {}
    for epoch in range(1, num_epochs + 1):
        expected_step = round(epoch * steps_per_epoch)
        ckpt_dir = run_dir / f"checkpoint-{expected_step}"
        if ckpt_dir.exists():
            checkpoints[epoch] = ckpt_dir
        else:
            print(
                f"[WARN] checkpoint-{expected_step} not found for epoch {epoch}",
                file=sys.stderr,
            )
    return checkpoints


def _load_summary(summary_path: Path) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def eval_lora_checkpoint(
    ckpt_dir: Path, eval_dir: Path, base_model_path: str,
    lora_model: str, task_name: str,
) -> dict:
    """Eval a LoRA checkpoint. Returns DUET_SUMMARY dict."""
    summary_path = eval_dir / "DUET_SUMMARY.json"
    if summary_path.exists():
        print(f"[SKIP eval] {eval_dir} — summary exists")
        return _load_summary(summary_path)

    eval_dir.mkdir(parents=True, exist_ok=True)
    run([
        "python", "src/eval.py",
        "experiment=eval/duet/default.yaml",
        f"model={lora_model}",
        f"forget_split={FORGET_SPLIT}",
        f"holdout_split={RETAIN_SPLIT}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={ckpt_dir}",
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
    return _load_summary(summary_path)


def eval_base_model(
    eval_dir: Path, base_model_path: str,
    plain_model: str, task_name: str,
) -> dict:
    """Eval the un-unlearned SFT checkpoint (epoch 0, no LoRA adapter)."""
    summary_path = eval_dir / "DUET_SUMMARY.json"
    if summary_path.exists():
        print(f"[SKIP eval] {eval_dir} — summary exists")
        return _load_summary(summary_path)

    eval_dir.mkdir(parents=True, exist_ok=True)
    run([
        "python", "src/eval.py",
        "experiment=eval/duet/default.yaml",
        f"model={plain_model}",
        f"forget_split={FORGET_SPLIT}",
        f"holdout_split={RETAIN_SPLIT}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={base_model_path}",
        "model.model_args.device_map=auto",
        "++model.model_args.low_cpu_mem_usage=true",
        "eval.duet.overwrite=true",
        f"paths.output_dir={eval_dir}",
        "retain_logs_path=null",
    ])
    return _load_summary(summary_path)


# ── Output: table + plot ──────────────────────────────────────────────────────

def make_table_and_plot(
    results: dict[int, dict], out_dir: Path, model_name: str,
) -> None:
    rows = [
        {
            "epoch": ep,
            "forget_rouge_l": round(v.get("forget_qa_rouge", float("nan")), 4),
            "retain_rouge_l": round(v.get("holdout_qa_rouge", float("nan")), 4),
        }
        for ep, v in sorted(results.items())
    ]

    # ── table ──────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / f"ada_pop_epochs_{model_name}.csv"
    md_path   = out_dir / f"ada_pop_epochs_{model_name}.md"

    # Write CSV manually (avoid pandas dependency issues)
    header = "epoch,forget_rouge_l,retain_rouge_l"
    lines  = [header] + [f"{r['epoch']},{r['forget_rouge_l']},{r['retain_rouge_l']}" for r in rows]
    csv_path.write_text("\n".join(lines) + "\n")

    # Markdown table
    md_lines = [
        "| epoch | forget_rouge_l | retain_rouge_l |",
        "|------:|---------------:|---------------:|",
    ] + [f"| {r['epoch']} | {r['forget_rouge_l']} | {r['retain_rouge_l']} |" for r in rows]
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"\n{'='*55}")
    print(f"AdaPop ROUGE-L vs Epoch  —  {model_name} / DUET")
    print(f"{'='*55}")
    print(f"{'Epoch':>6}  {'Forget ROUGE-L':>14}  {'Retain ROUGE-L':>14}")
    print(f"{'------':>6}  {'------------------':>14}  {'------------------':>14}")
    for r in rows:
        print(f"{r['epoch']:>6}  {r['forget_rouge_l']:>14.4f}  {r['retain_rouge_l']:>14.4f}")
    print(f"\nCSV  → {csv_path}")
    print(f"MD   → {md_path}")

    # ── plot ───────────────────────────────────────────────────────────────
    epochs       = [r["epoch"]        for r in rows]
    forget_vals  = [r["forget_rouge_l"] for r in rows]
    retain_vals  = [r["retain_rouge_l"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, forget_vals, marker="o", linewidth=2,
            label="Forget ROUGE-L", color="tab:red")
    ax.plot(epochs, retain_vals, marker="s", linewidth=2,
            label="Retain ROUGE-L", color="tab:blue")

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("ROUGE-L (recall)", fontsize=13)
    ax.set_title(f"AdaPop ROUGE-L vs Epoch — {model_name} / DUET", fontsize=13)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"ada_pop_epochs_{model_name}.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Plot → {p}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="Llama-3.1-8B-Instruct",
        choices=list(MODEL_CONFIGS.keys()),
        help="Which model to analyse (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; evaluate existing per-epoch checkpoints only",
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    lora_model   = cfg["lora_model"]
    plain_model  = cfg["plain_model"]
    base_model_path = cfg["sft_path"]

    run_dir = (
        REPO_ROOT / "saves" / "unlearn" / "duet" / "epoch_analysis_ada_pop"
        / f"duet_{args.model}_{FORGET_LABEL}_ada_pop_lora_r{LORA_R}"
          f"_lalpha{LORA_ALPHA}_ldrop0p0_lr{LR}_adyn_bdyn_gamma1p0_epoch_analysis"
    )
    out_dir = REPO_ROOT / "saves" / "plots" / "ada_pop_epoch_analysis"

    os.chdir(REPO_ROOT)

    # ── Train ────────────────────────────────────────────────────────────────
    final_adapter = run_dir / "adapter_model.safetensors"
    if args.skip_train:
        print(f"[--skip-train] Using existing checkpoints in {run_dir}")
    elif final_adapter.exists():
        print(f"[SKIP training] Final adapter already exists: {final_adapter}")
    else:
        print(f"\n{'='*55}\nTraining AdaPop for {MAX_EPOCHS} epochs\n{'='*55}")
        train_ada_pop(run_dir, args.model, base_model_path, lora_model)

    # ── Map epoch → checkpoint dir ───────────────────────────────────────────
    epoch_to_ckpt = get_epoch_checkpoints(run_dir)
    print(f"\nCheckpoints found for epochs: {sorted(epoch_to_ckpt)}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    results: dict[int, dict] = {}

    # Epoch 0: base SFT model (no unlearning)
    print(f"\n{'='*55}\nEvaluating epoch 0 (base model, no unlearning)\n{'='*55}")
    results[0] = eval_base_model(
        eval_dir=run_dir / "epoch_evals" / "epoch_0",
        base_model_path=base_model_path,
        plain_model=plain_model,
        task_name=f"ada_pop_epoch_0_{args.model}",
    )

    # Epochs 1..MAX_EPOCHS: per-epoch LoRA checkpoints
    for epoch, ckpt_dir in sorted(epoch_to_ckpt.items()):
        print(f"\n{'='*55}\nEvaluating epoch {epoch}\n{'='*55}")
        results[epoch] = eval_lora_checkpoint(
            ckpt_dir=ckpt_dir,
            eval_dir=run_dir / "epoch_evals" / f"epoch_{epoch}",
            base_model_path=base_model_path,
            lora_model=lora_model,
            task_name=f"ada_pop_epoch_{epoch}_{args.model}",
        )

    # ── Table + Plot ─────────────────────────────────────────────────────────
    make_table_and_plot(results, out_dir, args.model)


if __name__ == "__main__":
    main()
