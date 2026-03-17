#!/usr/bin/env python3
"""
Evaluate existing NEW_BUGFIX_* unlearned models (lr=1e-4) on additional data subsets.

DUET (city_forget_5 / merged runs, each NEW_BUGFIX_* algo):
  - paraphrases_city_forget_popular_5  →  run_dir/evals/paraphrase_popular/
  - paraphrases_city_forget_rare_5     →  run_dir/evals/paraphrase_rare/

RWKU (forget_level2 runs, each NEW_BUGFIX_* algo):
  - forget_level3 (every 6th row)      →  run_dir/evals/level3/

Saves summary table:
  notebooks/saves/unified_tables/table4_additional_subsets.csv

Usage:
  python scripts/analysis/eval_additional_subsets.py [--dry-run] [--force] [--gpu 0]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import datasets as hf_datasets
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = Path("/mnt/extremessd10tb/borisiuk/new_MU_exps/open-unlearning")
TARGET_LR = "1e-4"

MODEL_CONFIG = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct-lora",
    "Llama-3.1-8B":          "Llama-3.1-8B-lora",
    "gemma-7b-it":            "gemma-7b-it-lora",
    "Qwen2.5-7B-Instruct":   "Qwen2.5-7B-Instruct-lora",
}

DUET_EXTRA_SUBSETS = [
    {"forget_split": "paraphrases_city_forget_popular_5", "out_subdir": "paraphrase_popular", "label": "paraphrase_popular"},
    {"forget_split": "paraphrases_city_forget_rare_5",    "out_subdir": "paraphrase_rare",    "label": "paraphrase_rare"},
]

RWKU_EXTRA_SUBSETS = [
    {
        "forget_split": "forget_level3",
        "out_subdir":   "level3",
        "label":        "level3",
        # Pre-subsample every 6th element (~7k → ~1k); cached as local JSONL
        "every_nth":    6,
        "hf_path":      "SwetieePawsss/exp_r",
        "hf_split":     "test",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_every_nth(base_dir: Path, hf_path: str, hf_name: str, hf_split: str, every_nth: int) -> Path:
    """Download and subsample dataset, cache as local JSONL. Returns path."""
    cache_dir  = base_dir / "saves" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_file = cache_dir / f"{hf_name}_{hf_split}_every{every_nth}.jsonl"
    if local_file.exists():
        print(f"  [cache] {local_file.name}")
        return local_file
    print(f"  [subsample] Downloading {hf_path}/{hf_name}/{hf_split} ...")
    ds     = hf_datasets.load_dataset(hf_path, name=hf_name, split=hf_split)
    ds_sub = ds.select(range(0, len(ds), every_nth))
    print(f"  [subsample] {len(ds)} → {len(ds_sub)} rows (every {every_nth}th)")
    ds_sub.to_json(str(local_file), lines=True)
    return local_file


def _get_model_tag(run_name: str) -> Optional[str]:
    # Check longer tags first to avoid "Llama-3.1-8B" matching before "Llama-3.1-8B-Instruct"
    for tag in sorted(MODEL_CONFIG, key=len, reverse=True):
        if tag in run_name:
            return tag
    return None


def _read_base_model_path(run_dir: Path) -> Optional[str]:
    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        return None
    try:
        cfg = yaml.safe_load(hydra_cfg.read_text())
        return cfg.get("model", {}).get("model_args", {}).get("pretrained_model_name_or_path")
    except Exception:
        return None


def _parse_lora_config(run_dir: Path) -> Dict[str, str]:
    defaults  = {"r": "32", "lora_alpha": "64", "lora_dropout": "0.0"}
    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        return defaults
    try:
        cfg = yaml.safe_load(hydra_cfg.read_text())
        lc  = cfg.get("model", {}).get("lora_config", {})
        return {k: str(lc.get(k, defaults[k])) for k in defaults}
    except Exception:
        return defaults


def _find_runs(bench_dir: Path, split_pattern: str, lr_tag: str) -> List[Path]:
    runs = []
    for bugfix_dir in sorted(bench_dir.glob("NEW_BUGFIX_*")):
        for run_dir in sorted(bugfix_dir.iterdir()):
            if run_dir.is_dir() and split_pattern in run_dir.name and f"_lr{lr_tag}" in run_dir.name:
                runs.append(run_dir)
    return runs


def _algo_from_run_dir(run_dir: Path) -> str:
    return run_dir.parent.name.replace("NEW_BUGFIX_", "").lower()


def _build_eval_cmd(
    bench: str,
    run_dir: Path,
    base_model_path: str,
    model_tag: str,
    lora: Dict[str, str],
    forget_split: str,
    output_dir: Path,
    task_name: str,
    local_data_file: Optional[Path] = None,
) -> List[str]:
    model_cfg = MODEL_CONFIG.get(model_tag, f"{model_tag}-lora")
    cmd = [
        sys.executable, "src/eval.py",
        f"experiment=eval/{bench}/forget_only.yaml",
        f"model={model_cfg}",
        f"forget_split={forget_split}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={run_dir}",
        f"model.model_args.base_model_name_or_path={base_model_path}",
        "model.model_args.device_map=auto",
        "model.model_args.low_cpu_mem_usage=true",
        f"model.lora_config.r={lora['r']}",
        f"model.lora_config.lora_alpha={lora['lora_alpha']}",
        f"model.lora_config.lora_dropout={lora['lora_dropout']}",
        "eval.duet.overwrite=false",
        f"paths.output_dir={output_dir}",
        "retain_logs_path=null",
    ]
    if local_data_file is not None:
        # Load from pre-saved local JSONL; data_files is not in the struct so use + to add it
        pfx = "eval.duet.metrics.forget_qa_rouge.datasets.RWKU_QA_forget.args.hf_args"
        cmd += [
            f"{pfx}.path=json",
            f"+{pfx}.data_files={local_data_file}",
            f"{pfx}.name=null",
            f"{pfx}.split=train",
        ]
    return cmd


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evals(base_dir: Path, dry_run: bool, force: bool, gpu: str) -> List[Dict]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    bench_subsets = {
        "duet": ("city_forget_5", DUET_EXTRA_SUBSETS),
        "rwku": ("forget_level2", RWKU_EXTRA_SUBSETS),
    }

    # ── Collect all jobs ──────────────────────────────────────────────────────
    jobs: List[Dict] = []
    for bench, (split_pattern, subsets) in bench_subsets.items():
        bench_dir = base_dir / "saves" / "unlearn" / bench
        runs      = _find_runs(bench_dir, split_pattern, TARGET_LR)
        print(f"\n[{bench}] Found {len(runs)} runs at lr={TARGET_LR} matching '{split_pattern}'")

        for run_dir in runs:
            model_tag = _get_model_tag(run_dir.name)
            if model_tag is None:
                print(f"  [skip] Cannot identify model: {run_dir.name}")
                continue
            base_model_path = _read_base_model_path(run_dir)
            if base_model_path is None:
                print(f"  [skip] No .hydra/config.yaml: {run_dir.name}")
                continue
            for subset in subsets:
                out_dir = run_dir / "evals" / subset["out_subdir"]
                jobs.append({
                    "bench":           bench,
                    "algo":            _algo_from_run_dir(run_dir),
                    "model_tag":       model_tag,
                    "run_dir":         run_dir,
                    "subset":          subset,
                    "out_dir":         out_dir,
                    "summary":         out_dir / "DUET_SUMMARY.json",
                    "base_model_path": base_model_path,
                    "lora":            _parse_lora_config(run_dir),
                })

    total = len(jobs)
    done  = sum(1 for j in jobs if j["summary"].exists())
    print(f"\n[progress] {done}/{total} already completed, {total - done} to run\n")

    # ── Run missing jobs ──────────────────────────────────────────────────────
    records: List[Dict] = []
    for i, job in enumerate(jobs):
        bench, algo, model_tag = job["bench"], job["algo"], job["model_tag"]
        run_dir, subset        = job["run_dir"], job["subset"]
        out_dir, summary       = job["out_dir"], job["summary"]

        records.append({
            "bench":   bench, "algo": algo, "model": model_tag,
            "label":   subset["label"],
            "run_dir": str(run_dir), "out_dir": str(out_dir), "summary": str(summary),
        })

        if summary.exists() and not force:
            print(f"  [skip {i}/{total}] {run_dir.name} / {subset['out_subdir']}")
            continue

        local_data_file: Optional[Path] = None
        if subset.get("every_nth"):
            local_data_file = _prepare_every_nth(
                base_dir, subset["hf_path"], subset["forget_split"],
                subset["hf_split"], subset["every_nth"],
            )

        task_name = f"{run_dir.name}_{subset['out_subdir']}"
        cmd = _build_eval_cmd(
            bench=bench, run_dir=run_dir, base_model_path=job["base_model_path"],
            model_tag=model_tag, lora=job["lora"], forget_split=subset["forget_split"],
            output_dir=out_dir, task_name=task_name, local_data_file=local_data_file,
        )

        print(f"\n  [{i}/{total}] [{bench}/{algo}/{model_tag}] {subset['out_subdir']}")
        print(f"  CMD: {' '.join(cmd)}")

        if dry_run:
            print("  [dry-run] skipping execution")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, cwd=str(base_dir), env=env)
        if result.returncode != 0:
            print(f"  [ERROR] eval failed for {run_dir.name} / {subset['out_subdir']}")

    return records


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        summary_path = Path(rec["summary"])
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text())
        except Exception:
            continue
        forget_val = data.get("forget_qa_rouge")
        if forget_val is None:
            continue
        rows.append({
            "benchmark":      rec["bench"],
            "model":          rec["model"],
            "algo":           rec["algo"],
            "eval_subset":    rec["label"],
            "rouge_l_forget": round(float(forget_val), 4),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["benchmark", "model", "algo", "eval_subset"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir",  type=Path, default=BASE_DIR)
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--force",     action="store_true")
    p.add_argument("--gpu",       default="0")
    p.add_argument("--table-dir", type=Path, default=BASE_DIR / "notebooks" / "saves" / "unified_tables")
    args = p.parse_args()

    records = run_evals(base_dir=args.base_dir, dry_run=args.dry_run, force=args.force, gpu=args.gpu)

    print("\n[summary] Building results table...")
    df = build_summary_table(records)
    if df.empty:
        print("[summary] No completed evaluations found yet.")
        return

    args.table_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.table_dir / "table4_additional_subsets.csv"
    df.to_csv(out_path, index=False)
    print(f"[summary] Saved {len(df)} rows → {out_path}")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
