#!/usr/bin/env python3
"""baselines_epoch_analysis.py

Train WGA, GA, GD, and NPO on DUET for MAX_EPOCHS epochs (one checkpoint per
epoch), evaluate ROUGE-L (forget + retain) at epoch 0 and each subsequent
epoch, then save a combined CSV table, Markdown table, and PDF/PNG plot.

The plot and table are updated after every newly evaluated point so you can
monitor progress live.

Solid lines  = forget ROUGE-L
Dashed lines = retain ROUGE-L
One colour per algorithm.

Also loads AdaPop results from the existing epoch-analysis CSV (if present)
and overlays them on the combined plot.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/baselines_epoch_analysis.py
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/baselines_epoch_analysis.py --model gemma-7b-it
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/baselines_epoch_analysis.py --algo WGA NPO
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/baselines_epoch_analysis.py --skip-train
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

# ── Paths & training constants ────────────────────────────────────────────────
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

# ── Algorithm definitions ─────────────────────────────────────────────────────
#   trainer    : Hydra trainer= value
#   method_args: list of extra "trainer.method_args.*=*" overrides (empty for GA)
#   color      : matplotlib colour string
ALGO_CONFIGS = {
    "WGA": {
        "trainer": "WGA",
        "method_args": [
            "trainer.method_args.beta=1.0",
            "trainer.method_args.alpha=1.0",
            "trainer.method_args.gamma=1.0",
            "trainer.method_args.retain_loss_type=NLL",
        ],
        "color": "#1f77b4",  # blue
        "marker": "o",
    },
    "GA": {
        "trainer": "GradAscent",
        # wga_lora.yaml injects beta/alpha/gamma/retain_loss_type — delete them all
        # since GradAscent.__init__ accepts none of them.
        "method_args": [
            "~trainer.method_args.beta",
            "~trainer.method_args.alpha",
            "~trainer.method_args.gamma",
            "~trainer.method_args.retain_loss_type",
        ],
        "color": "#d62728",  # red
        "marker": "o",
    },
    "GD": {
        "trainer": "GradDiff",
        # GradDiff accepts alpha/gamma/retain_loss_type but not beta — delete beta.
        "method_args": [
            "~trainer.method_args.beta",
            "trainer.method_args.gamma=1.0",
            "trainer.method_args.alpha=1.0",
            "trainer.method_args.retain_loss_type=NLL",
        ],
        "color": "#ff7f0e",  # orange
        "marker": "o",
    },
    "NPO": {
        "trainer": "NPO",
        "method_args": [
            "trainer.method_args.beta=0.1",
            "trainer.method_args.alpha=1.0",
            "trainer.method_args.gamma=1.0",
            "trainer.method_args.retain_loss_type=NLL",
        ],
        "color": "#2ca02c",  # green
        "marker": "o",
    },
    "UNDIAL": {
        "trainer": "UNDIAL",
        # Accepts beta (distillation temp), alpha, gamma, retain_loss_type — all fine from wga_lora.yaml.
        "method_args": [
            "trainer.method_args.beta=1.0",
            "trainer.method_args.alpha=1.0",
            "trainer.method_args.gamma=1.0",
            "trainer.method_args.retain_loss_type=NLL",
        ],
        "color": "#17becf",  # cyan
        "marker": "o",
    },
    "RMU": {
        "trainer": "RMU_lora",
        # Use RMU_lora config: sets module_regex=".*\.layers\.7" (PEFT-prefix-agnostic)
        # and trainable_params_regex=[".*lora.*"] so only LoRA adapter weights are updated.
        # All RMU_lora.yaml defaults are correct; only override what we need.
        "method_args": [
            "trainer.method_args.steering_coeff=2",
            "trainer.method_args.alpha=1.0",
            "trainer.method_args.gamma=1.0",
        ],
        "color": "#8c564b",  # brown
        "marker": "o",
    },
    "PDU": {
        "trainer": "PDU",
        # beta is absorbed (ignored) by PDU; other params match main duet script defaults.
        "method_args": [
            "trainer.method_args.retain_loss_eps=0.3",
            "trainer.method_args.primal_dual=true",
            "trainer.method_args.dual_step_size=1.0",
            "trainer.method_args.dual_warmup_epochs=3",
            "trainer.method_args.dual_update_upon=step",
            "trainer.method_args.alpha=1.0",
            "trainer.method_args.gamma=1.0",
        ],
        "color": "#e377c2",  # pink
        "marker": "o",
    },
}

# AdaPop colour for overlay (loaded from existing CSV, not retrained here)
ADAPOP_COLOR = "#9467bd"  # purple — distinct from all ALGO_CONFIGS colors

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


# ── Shell helpers ─────────────────────────────────────────────────────────────

def run(cmd: list[str]) -> None:
    print(f"\n[CMD] {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


# ── Training ──────────────────────────────────────────────────────────────────

def train_algo(
    algo: str, run_dir: Path, model_name: str,
    base_model_path: str, lora_model: str,
) -> None:
    acfg = ALGO_CONFIGS[algo]
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "src/train.py",
        "--config-name=unlearn.yaml",
        "experiment=unlearn/duet/wga_lora.yaml",
        f"trainer={acfg['trainer']}",
        f"task_name=duet_{model_name}_{FORGET_LABEL}_{algo}_lora"
            f"_r{LORA_R}_la{LORA_ALPHA}_lr{LR}_epoch_analysis",
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
        "retain_logs_path=null",
        f"paths.output_dir={run_dir}",
    ] + acfg["method_args"]
    run(cmd)


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def get_epoch_checkpoints(run_dir: Path) -> dict[int, Path]:
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found in {run_dir}")
    with open(state_path) as f:
        state = json.load(f)

    num_epochs = int(round(state["num_train_epochs"]))
    max_steps  = state["max_steps"]
    steps_per_epoch = max_steps / num_epochs

    checkpoints: dict[int, Path] = {}
    for epoch in range(1, num_epochs + 1):
        expected_step = round(epoch * steps_per_epoch)
        ckpt_dir = run_dir / f"checkpoint-{expected_step}"
        if ckpt_dir.exists():
            checkpoints[epoch] = ckpt_dir
        else:
            print(f"[WARN] checkpoint-{expected_step} not found for epoch {epoch}",
                  file=sys.stderr)
    return checkpoints


# ── Evaluation ────────────────────────────────────────────────────────────────

def _load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def eval_lora_checkpoint(
    ckpt_dir: Path, eval_dir: Path,
    base_model_path: str, lora_model: str, task_name: str,
) -> dict:
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


# ── Output helpers ────────────────────────────────────────────────────────────

def load_adapop_csv(model_name: str, out_dir: Path) -> dict[int, dict] | None:
    """Load existing AdaPop epoch results from the ada_pop_epoch_analysis CSV."""
    csv_path = out_dir / f"ada_pop_epochs_{model_name}.csv"
    if not csv_path.exists():
        return None
    results: dict[int, dict] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["epoch"])
            results[ep] = {
                "forget_qa_rouge": float(row["forget_rouge_l"]),
                "holdout_qa_rouge": float(row["retain_rouge_l"]),
            }
    print(f"[INFO] Loaded AdaPop results from {csv_path} ({len(results)} epochs)")
    return results


def save_combined_table(
    all_results: dict[str, dict[int, dict]],
    out_dir: Path, model_name: str,
) -> None:
    """Write combined CSV and Markdown table for all algorithms."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all (algo, epoch) rows
    rows: list[dict] = []
    for algo, res in sorted(all_results.items()):
        for ep, v in sorted(res.items()):
            rows.append({
                "algo": algo,
                "epoch": ep,
                "forget_rouge_l": round(v.get("forget_qa_rouge", float("nan")), 4),
                "retain_rouge_l": round(v.get("holdout_qa_rouge", float("nan")), 4),
            })

    csv_path = out_dir / f"baselines_epochs_{model_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "epoch", "forget_rouge_l", "retain_rouge_l"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / f"baselines_epochs_{model_name}.md"
    lines = [
        "| algo | epoch | forget_rouge_l | retain_rouge_l |",
        "|:-----|------:|---------------:|---------------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['algo']} | {r['epoch']} "
            f"| {r['forget_rouge_l']} | {r['retain_rouge_l']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")


def save_combined_plot(
    all_results: dict[str, dict[int, dict]],
    out_dir: Path, model_name: str,
) -> None:
    """Save combined plot: left panel=forget ROUGE-L, right panel=retain ROUGE-L."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for algo, res in sorted(all_results.items()):
        if not res:
            continue
        epochs      = sorted(res)
        forget_vals = [res[ep].get("forget_qa_rouge", float("nan")) for ep in epochs]
        retain_vals = [res[ep].get("holdout_qa_rouge", float("nan")) for ep in epochs]

        if algo == "AdaPop":
            color  = ADAPOP_COLOR
            marker = "*"
            ms     = 12
            lw     = 2.5
            zorder = 5  # draw on top
        else:
            color  = ALGO_CONFIGS[algo]["color"]
            marker = ALGO_CONFIGS[algo]["marker"]
            ms     = 7
            lw     = 2
            zorder = 3

        kw = dict(marker=marker, linewidth=lw, color=color,
                  markersize=ms, zorder=zorder, label=algo)
        ax_f.plot(epochs, forget_vals, **kw)
        ax_r.plot(epochs, retain_vals, **kw)

    for ax, title in [(ax_f, "Forget ROUGE-L"), (ax_r, "Retain ROUGE-L")]:
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("ROUGE-L (recall)", fontsize=13)
        ax.set_title(f"{title} — {model_name} / DUET", fontsize=13)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"baselines_epochs_{model_name}.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"[PLOT] → {p}", flush=True)
    plt.close()


def print_table(all_results: dict[str, dict[int, dict]], model_name: str) -> None:
    print(f"\n{'='*65}")
    print(f"Baseline ROUGE-L vs Epoch  —  {model_name} / DUET")
    print(f"{'='*65}")
    print(f"{'Algo':>8}  {'Epoch':>5}  {'Forget':>10}  {'Retain':>10}")
    print(f"{'--------':>8}  {'-----':>5}  {'----------':>10}  {'----------':>10}")
    for algo, res in sorted(all_results.items()):
        for ep, v in sorted(res.items()):
            f_val = v.get("forget_qa_rouge", float("nan"))
            r_val = v.get("holdout_qa_rouge", float("nan"))
            print(f"{algo:>8}  {ep:>5}  {f_val:>10.4f}  {r_val:>10.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="Llama-3.1-8B-Instruct",
        choices=list(MODEL_CONFIGS.keys()),
    )
    parser.add_argument(
        "--algo", nargs="+", default=list(ALGO_CONFIGS.keys()),
        choices=list(ALGO_CONFIGS.keys()),
        help="Which algorithm(s) to run (default: all four)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; evaluate existing per-epoch checkpoints only",
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    lora_model      = cfg["lora_model"]
    plain_model     = cfg["plain_model"]
    base_model_path = cfg["sft_path"]

    base_run_dir = (
        REPO_ROOT / "saves" / "unlearn" / "duet" / "epoch_analysis_baselines"
    )
    # Output tables + plots go here (same folder as ada_pop analysis for easy overlay)
    ada_pop_out_dir = REPO_ROOT / "saves" / "plots" / "ada_pop_epoch_analysis"
    out_dir = REPO_ROOT / "saves" / "plots" / "baselines_epoch_analysis"

    os.chdir(REPO_ROOT)

    # ── Epoch-0 base model eval (shared across all algorithms) ───────────────
    base_eval_dir = base_run_dir / "base_model_epoch_0" / args.model
    print(f"\n{'='*60}\nEvaluating epoch 0 (base model)\n{'='*60}")
    base_result = eval_base_model(
        eval_dir=base_eval_dir,
        base_model_path=base_model_path,
        plain_model=plain_model,
        task_name=f"baselines_epoch_0_{args.model}",
    )

    # ── Load AdaPop overlay (if available) ───────────────────────────────────
    adapop_results = load_adapop_csv(args.model, ada_pop_out_dir)
    all_results: dict[str, dict[int, dict]] = {}
    if adapop_results is not None:
        all_results["AdaPop"] = adapop_results

    # ── Per-algorithm loop ────────────────────────────────────────────────────
    for algo in args.algo:
        print(f"\n{'='*60}\n{algo}  —  {args.model}\n{'='*60}")

        run_dir = (
            base_run_dir
            / f"duet_{args.model}_{FORGET_LABEL}_{algo}_lora"
              f"_r{LORA_R}_la{LORA_ALPHA}_lr{LR}_epoch_analysis"
        )

        # ── Train ─────────────────────────────────────────────────────────
        final_adapter = run_dir / "adapter_model.safetensors"
        if args.skip_train:
            print(f"[--skip-train] Using existing checkpoints in {run_dir}")
        elif final_adapter.exists():
            print(f"[SKIP training] Adapter already exists: {final_adapter}")
        else:
            print(f"\nTraining {algo} for {MAX_EPOCHS} epochs …")
            train_algo(algo, run_dir, args.model, base_model_path, lora_model)

        # ── Discover checkpoints ──────────────────────────────────────────
        try:
            epoch_to_ckpt = get_epoch_checkpoints(run_dir)
        except FileNotFoundError as e:
            print(f"[SKIP {algo}] {e}", flush=True)
            continue
        print(f"Checkpoints found for epochs: {sorted(epoch_to_ckpt)}", flush=True)

        # Seed with epoch-0 baseline
        algo_results: dict[int, dict] = {0: base_result}
        all_results[algo] = algo_results

        # ── Evaluate epoch 0 (update plot immediately) ────────────────────
        save_combined_table(all_results, out_dir, args.model)
        save_combined_plot(all_results, out_dir, args.model)
        print_table(all_results, args.model)

        # ── Evaluate each epoch ───────────────────────────────────────────
        for epoch, ckpt_dir in sorted(epoch_to_ckpt.items()):
            print(f"\n{algo} — evaluating epoch {epoch}")
            result = eval_lora_checkpoint(
                ckpt_dir=ckpt_dir,
                eval_dir=run_dir / "epoch_evals" / f"epoch_{epoch}",
                base_model_path=base_model_path,
                lora_model=lora_model,
                task_name=f"{algo}_epoch_{epoch}_{args.model}",
            )
            algo_results[epoch] = result

            # ── Incremental update after each new data point ──────────────
            save_combined_table(all_results, out_dir, args.model)
            save_combined_plot(all_results, out_dir, args.model)
            print_table(all_results, args.model)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nAll done.\n{'='*60}")
    print_table(all_results, args.model)
    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
