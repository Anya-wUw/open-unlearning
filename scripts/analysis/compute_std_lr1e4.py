#!/usr/bin/env python3
"""
Compute mean ± std across random seeds (42, 1, 219) at lr=1e-4.

For each (benchmark, algo, model, metric) we gather one value per seed,
then report mean ± std (ddof=1) across seeds.

Metrics:
  - ROUGE-L forget / retain          (DUET_SUMMARY.json)
  - Cosine Similarity forget / retain (COS_SIM_EVAL.json)

Algorithms: NPO, WGA, AdaPop
Models:     Llama-3.1-8B-Instruct, gemma-7b-it, Qwen2.5-7B-Instruct
Benchmarks: duet, rwku
Seeds:      42 (baseline, no _seedN tag), 1, 219

Usage:
    conda run -n MU python scripts/compute_std_lr1e4.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR  = Path("/mnt/extremessd10tb/borisiuk/new_MU_exps/open-unlearning")
TARGET_LR = 1e-4
LR_RE     = re.compile(r"_lr([^_]+)")
SEED_RE   = re.compile(r"_seed(\d+)")

BENCHMARKS = ["duet", "rwku"]
MODEL_TAGS = [
    "Llama-3.1-8B-Instruct",
    "gemma-7b-it",
    "Qwen2.5-7B-Instruct",
]
ALGOS = ["npo", "wga", "ada_pop"]
ALGO_LABELS = {"npo": "NPO", "wga": "WGA", "ada_pop": "AdaPop"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_lr(run_name: str) -> Optional[float]:
    m = LR_RE.search(run_name)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def _parse_seed(run_name: str) -> int:
    """Return seed number; runs without _seedN tag are seed 42 (baseline)."""
    m = SEED_RE.search(run_name)
    return int(m.group(1)) if m else 42


def _get_model(run_name: str) -> Optional[str]:
    for tag in MODEL_TAGS:
        if tag in run_name:
            return tag
    return None


def _is_merged_split(run_name: str, benchmark: str) -> bool:
    rn = run_name.lower()
    if benchmark == "duet":
        if "city_forget_rare" in rn or "city_forget_popular" in rn:
            return False
        return "city_forget" in rn
    return True  # RWKU has a single split


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_rouge(benchmark: str) -> pd.DataFrame:
    root = BASE_DIR / "saves" / "unlearn" / benchmark
    rows: List[Dict] = []
    for bugfix_dir in sorted(root.glob("NEW_BUGFIX_*")):
        algo = bugfix_dir.name.replace("NEW_BUGFIX_", "").lower()
        if algo not in ALGOS:
            continue
        for run_dir in sorted(bugfix_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not _is_merged_split(run_dir.name, benchmark):
                continue
            lr = _parse_lr(run_dir.name)
            if lr is None or abs(lr - TARGET_LR) / TARGET_LR > 0.01:
                continue
            model = _get_model(run_dir.name)
            if model is None:
                continue
            sp = run_dir / "evals" / "DUET_SUMMARY.json"
            if not sp.exists():
                continue
            try:
                d = json.loads(sp.read_text())
            except Exception:
                continue
            rows.append({
                "benchmark":    benchmark,
                "algo":         algo,
                "model":        model,
                "seed":         _parse_seed(run_dir.name),
                "forget_rouge": d.get("forget_qa_rouge"),
                "retain_rouge": d.get("holdout_qa_rouge"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("forget_rouge", "retain_rouge"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_cossim(benchmark: str) -> pd.DataFrame:
    root = BASE_DIR / "saves" / "unlearn" / benchmark
    rows: List[Dict] = []
    for bugfix_dir in sorted(root.glob("NEW_BUGFIX_*")):
        algo = bugfix_dir.name.replace("NEW_BUGFIX_", "").lower()
        if algo not in ALGOS:
            continue
        for run_dir in sorted(bugfix_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not _is_merged_split(run_dir.name, benchmark):
                continue
            lr = _parse_lr(run_dir.name)
            if lr is None or abs(lr - TARGET_LR) / TARGET_LR > 0.01:
                continue
            model = _get_model(run_dir.name)
            if model is None:
                continue
            sp = run_dir / "evals" / "COS_SIM_EVAL.json"
            if not sp.exists():
                continue
            try:
                d = json.loads(sp.read_text())
            except Exception:
                continue
            rows.append({
                "benchmark":     benchmark,
                "algo":          algo,
                "model":         model,
                "seed":          _parse_seed(run_dir.name),
                "forget_cossim": d.get("forget_qa_cos_sim", {}).get("agg_value"),
                "retain_cossim": d.get("holdout_qa_cos_sim", {}).get("agg_value"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("forget_cossim", "retain_cossim"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def agg(vals: pd.Series) -> str:
    vals = vals.dropna()
    n = len(vals)
    if n == 0:
        return "—"
    mean = vals.mean()
    std  = vals.std(ddof=1) if n > 1 else float("nan")
    std_str = f"{std:.3f}" if not np.isnan(std) else "n/a"
    return f"{mean:.3f} ± {std_str}  (n={n})"


def print_table(title: str, df: pd.DataFrame, metrics: List[Tuple[str, str]]) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"  std computed across seeds [42, 1, 219] per (algo, model, benchmark)")
    print(f"{'='*80}")

    for bench in BENCHMARKS:
        db = df[df["benchmark"] == bench]
        print(f"\n  [{bench.upper()}]")
        for model in MODEL_TAGS:
            dm = db[db["model"] == model]
            print(f"\n    {model}")
            col_w = 28
            header = f"    {'Algo':<10}" + "".join(f"  {lbl:>{col_w}}" for _, lbl in metrics)
            print(header)
            print("    " + "-" * (10 + (col_w + 2) * len(metrics)))
            for algo in ALGOS:
                da = dm[dm["algo"] == algo]
                row = f"    {ALGO_LABELS[algo]:<10}"
                for key, _ in metrics:
                    row += f"  {agg(da[key]):>{col_w}}"
                print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rouge_df  = pd.concat([load_rouge(b)  for b in BENCHMARKS], ignore_index=True)
    cossim_df = pd.concat([load_cossim(b) for b in BENCHMARKS], ignore_index=True)

    print_table(
        "ROUGE-L  (forget ↓ / retain ↑)",
        rouge_df,
        [("forget_rouge", "Forget ROUGE-L"), ("retain_rouge", "Retain ROUGE-L")],
    )
    print_table(
        "Cosine Similarity  (forget ↓ / retain ↑)",
        cossim_df,
        [("forget_cossim", "Forget CosSim"), ("retain_cossim", "Retain CosSim")],
    )

    # Summary: mean std across models (one number per algo/benchmark)
    print(f"\n{'='*80}")
    print("  Average std across models  (how stable is the method across architectures?)")
    print(f"{'='*80}")
    for bench in BENCHMARKS:
        print(f"\n  [{bench.upper()}]")
        for df, metrics in [
            (rouge_df,  [("forget_rouge", "Forget ROUGE-L"), ("retain_rouge", "Retain ROUGE-L")]),
            (cossim_df, [("forget_cossim", "Forget CosSim"), ("retain_cossim", "Retain CosSim")]),
        ]:
            db = df[df["benchmark"] == bench]
            for key, lbl in metrics:
                print(f"\n    {lbl}")
                for algo in ALGOS:
                    da = db[db["algo"] == algo]
                    # std per model (across seeds), then average that std
                    stds = []
                    for model in MODEL_TAGS:
                        vals = da[da["model"] == model][key].dropna()
                        if len(vals) > 1:
                            stds.append(vals.std(ddof=1))
                    if stds:
                        print(f"      {ALGO_LABELS[algo]:<10}  avg_seed_std = {np.mean(stds):.4f}  "
                              f"(per-model stds: {[f'{s:.4f}' for s in stds]})")
                    else:
                        print(f"      {ALGO_LABELS[algo]:<10}  no multi-seed data yet")

    print()


if __name__ == "__main__":
    main()
