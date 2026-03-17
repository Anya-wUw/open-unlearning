#!/usr/bin/env python3
"""
Benchmark evaluation script for LLM unlearning checkpoints.

Scans saves/unlearn/{duet,rwku}/NEW_BUGFIX_* for lr1e-4 checkpoints,
adds origin (pre-unlearning) baselines, runs lm_eval (mmlu + hellaswag),
and generates a summary CSV + Markdown table.

Usage:
    # See what would run (no GPU needed):
    conda run -n MU python run_benchmarks.py --dry-run

    # Run all evaluations:
    conda run -n MU python run_benchmarks.py --run

    # Parse existing results and generate table only:
    conda run -n MU python run_benchmarks.py --summarize
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
UNLEARN_ROOT = REPO_ROOT / "saves" / "unlearn"
RESULTS_DIR = Path(__file__).parent / "results"
LM_EVAL_BIN = "lm_eval"   # must be on PATH inside conda env MU
TARGET_LR = "lr1e-4"

# ── Model entry dataclass ────────────────────────────────────────────────────
@dataclass
class ModelEntry:
    model_path: str          # path passed to --model_args pretrained=
    model_name: str          # short display name (e.g. "gemma-7b-it")
    benchmark: str           # "duet" or "rwku"
    mu: str                  # algorithm name or "origin"
    lr: str                  # "1e-4" or "N/A"
    result_tag: str = field(init=False)  # unique tag for output file

    def __post_init__(self):
        # Build a filesystem-safe tag
        safe_path = re.sub(r"[/\s]", "_", self.model_path.strip("/"))
        safe_path = re.sub(r"[^a-zA-Z0-9_\-.]", "", safe_path)
        self.result_tag = safe_path[-120:]  # cap length


# ── Discovery ────────────────────────────────────────────────────────────────

def _read_base_model(ckpt_dir: str) -> Optional[str]:
    cfg = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            return json.load(f)["base_model_name_or_path"]
    return None


def _model_name_from_path(path: str) -> str:
    """Extract a short human-readable model name from any path."""
    basename = os.path.basename(path.rstrip("/"))
    # Mapping for known finetune dir names
    mappings = {
        "gemma-7b-it_full_3ep_ft_tripunlamb": "gemma-7b-it",
        "Qwen2.5-7B-Instruct_full_3ep_ft_tripunlamb": "Qwen2.5-7B-Instruct",
        "llama3.1-8b_full_3ep_ft_tripunlamb": "Llama-3.1-8B-Instruct",
        "llama3.1-8b_full_3ep_full_ft_duet": "Llama-3.1-8B-Instruct",
    }
    if basename in mappings:
        return mappings[basename]
    # HuggingFace hub ids: "google/gemma-7b-it" → "gemma-7b-it"
    if "/" in path and not path.startswith("/"):
        return basename
    # LoRA checkpoint dirs: "{benchmark}_{model}_{dataset_...}"
    # e.g. "duet_gemma-7b-it_city_forget_5_ga_..." → "gemma-7b-it"
    # Model names never contain underscores (they use hyphens/dots), so
    # splitting on '_' and taking index 1 is reliable.
    parts = basename.split("_")
    if len(parts) > 1 and parts[0] in ("duet", "rwku"):
        return parts[1]
    return basename


def collect_models() -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    seen_origins: set[tuple[str, str]] = set()  # (benchmark, base_model_path)

    for benchmark in ("duet", "rwku"):
        algo_dirs = sorted(glob.glob(str(UNLEARN_ROOT / benchmark / "NEW_BUGFIX_*")))
        for algo_dir in algo_dirs:
            algo = os.path.basename(algo_dir).replace("NEW_BUGFIX_", "")
            ckpt_dirs = sorted(glob.glob(f"{algo_dir}/*{TARGET_LR}*"))
            for ckpt in ckpt_dirs:
                if not os.path.isdir(ckpt):
                    continue
                # Skip sub-dirs like evals/.hydra/logs inside a checkpoint
                if os.path.basename(ckpt) in ("evals", "logs", ".hydra"):
                    continue
                # Only include checkpoints that are direct children of algo_dir
                if os.path.dirname(ckpt) != algo_dir:
                    continue

                base = _read_base_model(ckpt)
                model_name = _model_name_from_path(ckpt)

                entries.append(ModelEntry(
                    model_path=ckpt,
                    model_name=model_name,
                    benchmark=benchmark,
                    mu=algo,
                    lr="1e-4",
                ))

                # Register the origin (base) model once per (benchmark, base)
                if base and (benchmark, base) not in seen_origins:
                    seen_origins.add((benchmark, base))
                    entries.append(ModelEntry(
                        model_path=base,
                        model_name=_model_name_from_path(base),
                        benchmark=benchmark,
                        mu="origin",
                        lr="N/A",
                    ))

    # Deduplicate (same path may appear across algo dirs)
    seen_paths: set[tuple[str, str]] = set()
    unique: list[ModelEntry] = []
    for e in entries:
        key = (e.benchmark, e.model_path)
        if key not in seen_paths:
            seen_paths.add(key)
            unique.append(e)

    return unique


# ── Result path helpers ───────────────────────────────────────────────────────

def result_dir_for(entry: ModelEntry) -> Path:
    return RESULTS_DIR / entry.result_tag


def result_json_for(entry: ModelEntry) -> Optional[Path]:
    """Return the first *.json result file in the entry's result dir, if any.

    lm_eval writes results into a sub-directory named after the model path, so
    we need rglob (not glob) to find them.
    """
    rdir = result_dir_for(entry)
    if rdir.exists():
        jsons = sorted(rdir.rglob("results_*.json"))
        if jsons:
            return jsons[0]
    return None


# ── Evaluation ───────────────────────────────────────────────────────────────

def run_eval(entry: ModelEntry, dry_run: bool = False) -> None:
    if result_json_for(entry) is not None:
        print(f"[SKIP] Already evaluated: {entry.model_name} / {entry.benchmark} / {entry.mu}")
        return

    out_dir = result_dir_for(entry)
    out_dir.mkdir(parents=True, exist_ok=True)

    # LoRA adapters must be loaded as pretrained=<base>,peft=<lora_dir>
    base = _read_base_model(entry.model_path)
    if base:
        model_args = f"pretrained={base},peft={entry.model_path}"
    else:
        model_args = f"pretrained={entry.model_path}"

    cmd = [
        LM_EVAL_BIN,
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", "mmlu,hellaswag",
        "--device", "cuda:0",
        "--batch_size", "auto:4",
        "--apply_chat_template",
        "--output_path", str(out_dir),
    ]

    label = f"{entry.benchmark}/{entry.mu}/{entry.model_name} (lr={entry.lr})"
    if dry_run:
        print(f"[DRY-RUN] {label}")
        print("  " + " ".join(cmd))
        print()
        return

    print(f"\n{'='*70}")
    print(f"[RUN] {label}")
    print("  " + " ".join(cmd))
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] lm_eval failed for {label} (exit {result.returncode})", file=sys.stderr)


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_result(json_path: Path) -> dict:
    """Extract mmlu acc and hellaswag acc_norm from a lm_eval result JSON."""
    with open(json_path) as f:
        data = json.load(f)

    results = data.get("results", {})

    def _get(task: str, metric: str) -> Optional[float]:
        # lm_eval stores metrics as "acc,none" or "acc_norm,none"
        task_data = results.get(task, {})
        # Try aggregate key first (e.g. "mmlu" aggregates sub-tasks)
        val = task_data.get(f"{metric},none")
        if val is not None:
            return round(float(val), 4)
        # Fallback: average across sub-task keys
        subtask_vals = []
        for k, v in results.items():
            if k.startswith(task + "_") or k.startswith(task + ":"):
                sv = v.get(f"{metric},none")
                if sv is not None:
                    subtask_vals.append(float(sv))
        if subtask_vals:
            return round(sum(subtask_vals) / len(subtask_vals), 4)
        return None

    return {
        "mmlu_acc": _get("mmlu", "acc"),
        "hellaswag_acc_norm": _get("hellaswag", "acc_norm"),
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def summarize(entries: list[ModelEntry]) -> None:
    rows = []
    for entry in entries:
        jfile = result_json_for(entry)
        if jfile is None:
            continue
        metrics = parse_result(jfile)
        rows.append({
            "Model Name": entry.model_name,
            "Benchmark": entry.benchmark,
            "LR": entry.lr,
            "MU": entry.mu,
            "MMLU acc": metrics["mmlu_acc"] if metrics["mmlu_acc"] is not None else "N/A",
            "HellaSwag acc_norm": metrics["hellaswag_acc_norm"] if metrics["hellaswag_acc_norm"] is not None else "N/A",
        })

    if not rows:
        print("No results found yet. Run evaluations first with --run.")
        return

    # Sort: benchmark → model_name → mu (origin first)
    rows.sort(key=lambda r: (r["Benchmark"], r["Model Name"], r["MU"] == "origin", r["MU"]))

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "summary.csv"
    headers = ["Model Name", "Benchmark", "LR", "MU", "MMLU acc", "HellaSwag acc_norm"]
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    print(f"CSV saved to: {csv_path}")

    # ── Markdown ─────────────────────────────────────────────────────────────
    md_path = RESULTS_DIR / "summary.md"
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    def md_row(row):
        return "| " + " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers) + " |"

    sep = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    header_row = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"

    with open(md_path, "w") as f:
        f.write("# Benchmark Evaluation Summary\n\n")
        f.write("Tasks: `mmlu` (acc), `hellaswag` (acc_norm)  \n")
        f.write(f"Filter: `NEW_BUGFIX_*` checkpoints with `lr=1e-4` + origin baselines\n\n")
        f.write(header_row + "\n")
        f.write(sep + "\n")
        for row in rows:
            f.write(md_row(row) + "\n")
    print(f"Markdown saved to: {md_path}")

    # ── Console preview ───────────────────────────────────────────────────────
    print("\n" + header_row)
    print(sep)
    for row in rows:
        print(md_row(row))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    group.add_argument("--run", action="store_true", help="Execute all lm_eval evaluations (GPU required)")
    group.add_argument("--summarize", action="store_true", help="Parse results and generate summary table")
    args = parser.parse_args()

    entries = collect_models()

    if args.dry_run or args.run:
        print(f"Found {len(entries)} model entries to evaluate:\n")
        for e in entries:
            tag = f"[origin]" if e.mu == "origin" else f"[{e.mu}  lr={e.lr}]"
            already = "✓ done" if result_json_for(e) else "○ pending"
            print(f"  {already}  {e.benchmark:<6}  {e.model_name:<30}  {tag}")
        print()

        for entry in entries:
            run_eval(entry, dry_run=args.dry_run)

    if args.summarize:
        summarize(entries)


if __name__ == "__main__":
    main()
