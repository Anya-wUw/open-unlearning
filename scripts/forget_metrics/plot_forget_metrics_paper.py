#!/usr/bin/env python3
"""
Paper-ready forget-metrics figure: 4 columns (metrics) × 2 rows (forget / retain).

Each panel combines both benchmarks and all 3 models:
  y-axis  = algorithm (sorted)
  x-axis  = metric value
  colour  = model
  marker  = benchmark  (circle = DUET, square = RWKU)

Only lr=1e-4 NEW_BUGFIX_* runs are included.
Algo names have the NEW_BUGFIX_ prefix stripped.

Output:
  scripts/forget_metrics/paper_forget_metrics.png
  scripts/forget_metrics/paper_forget_metrics_table.csv
  scripts/forget_metrics/paper_forget_metrics_table.md
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = str(Path("/tmp") / f"matplotlib-{os.getlogin()}")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.grid":         True,
    "grid.alpha":        0.18,
    "grid.linestyle":    ":",
    "axes.axisbelow":    True,
    "axes.linewidth":    0.7,
    "figure.autolayout": False,
})

# ── Constants ─────────────────────────────────────────────────────────────────
SUMMARY_NAME = "FORGET_METRICS_SUMMARY.json"
LR_RE        = re.compile(r"_lr([^_]+)")
TARGET_LR    = "1e-4"

MODEL_TAGS = [
    "Llama-3.1-8B-Instruct",
    "gemma-7b-it",
    "Qwen2.5-7B-Instruct",
]

BENCHMARKS = ["duet", "rwku"]

PAPER_METRICS = [
    ("delta_logprob_mean", "Δ Log-Prob"),
    ("delta_rank_mean",    "Δ Rank"),
    ("hidden_cos_mean",    "Hid. Cosine"),
    ("kl_mean",            "KL Div"),
]

SPLITS = ["forget", "retain"]

# Algorithms to show, in display order
ALGO_ORDER = ["ga", "gd", "wga", "npo", "ada_pop"]
ALGO_LABELS = {
    "ga":      "GA",
    "gd":      "GD",
    "wga":     "WGA",
    "npo":     "NPO",
    "ada_pop": "AdaPop",
}

MODEL_COLORS = {
    "Llama-3.1-8B-Instruct": "#e63946",
    "gemma-7b-it":            "#2a9d8f",
    "Qwen2.5-7B-Instruct":    "#457b9d",
}
MODEL_LABELS = {
    "Llama-3.1-8B-Instruct": "Llama-3.1 8B-Inst",
    "gemma-7b-it":            "Gemma 7B-It",
    "Qwen2.5-7B-Instruct":   "Qwen2.5 7B-Inst",
}

BENCH_MARKERS = {
    "duet": "o",
    "rwku": "s",
}


# ── Data collection ───────────────────────────────────────────────────────────

def _parse_lr(name: str) -> Optional[str]:
    m = LR_RE.search(name)
    return m.group(1) if m else None


def _parse_model_tag(name: str) -> Optional[str]:
    for tag in MODEL_TAGS:
        if tag in name:
            return tag
    return None


def _strip_bugfix(algo: str) -> str:
    return re.sub(r"^NEW_BUGFIX_", "", algo, flags=re.IGNORECASE).lower()


def collect(bench: str, root_dir: Path) -> pd.DataFrame:
    bench_dir = root_dir / "saves" / "unlearn" / bench
    rows: List[Dict] = []

    pattern = f"**/NEW_BUGFIX_*/*/evals/forget_metrics__*/{SUMMARY_NAME}"
    for sp in sorted(bench_dir.glob(pattern)):
        try:
            data = json.loads(sp.read_text())
        except Exception:
            continue

        # Navigate up to run_dir (parent of evals/)
        evals_dir = sp.parent.parent  # forget_metrics__* → evals/
        run_dir   = evals_dir.parent   # run_dir
        algo_dir  = run_dir.parent     # NEW_BUGFIX_*

        algo     = _strip_bugfix(algo_dir.name)
        run_name = run_dir.name

        lr = _parse_lr(run_name)
        if lr != TARGET_LR:
            continue

        model_tag = _parse_model_tag(run_name)
        if model_tag is None:
            continue

        for split_name, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            rows.append({
                "benchmark": bench,
                "algo":      algo,
                "model_tag": model_tag,
                "lr":        lr,
                "split":     split_name,
                **metrics,
            })

    return pd.DataFrame(rows)


# ── Styling ───────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.tick_params(left=False)


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    """
    4 rows (metrics) × 2 cols (forget | retain).
    Designed for single-column layout in a two-column paper (~84mm wide).
    Each panel: y = algorithm, x = metric value.
                colour = model family, marker = benchmark.
    """
    n_rows  = len(PAPER_METRICS)   # 4
    n_cols  = len(SPLITS)          # 2
    n_algos = len(ALGO_ORDER)      # 5
    n_dots  = len(BENCHMARKS) * len(MODEL_TAGS)   # 2 × 3 = 6 dots per algo

    # Vertical jitter: spread 6 dots within each algo band
    y_offsets = np.linspace(-0.28, 0.28, n_dots)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.4, 11.2),
        gridspec_kw={"wspace": 0.10, "hspace": 0.72},
    )
    fig.subplots_adjust(top=0.94, bottom=0.10)

    # ── AdaPop y-position for highlight band ──────────────────────────────────
    ada_idx = ALGO_ORDER.index("ada_pop")

    for ri, (metric, metric_label) in enumerate(PAPER_METRICS):
        for ci, split in enumerate(SPLITS):
            ax = axes[ri, ci]

            # Background bands: alternating light grey + pale violet for AdaPop
            for ai in range(n_algos):
                if ai == ada_idx:
                    ax.axhspan(ai - 0.45, ai + 0.45, color="#ede8fa", zorder=0)
                elif ai % 2 == 0:
                    ax.axhspan(ai - 0.45, ai + 0.45, color="#f6f6f6", zorder=0)

            # ── Dots ──────────────────────────────────────────────────────────
            dot_idx = 0
            for bench in BENCHMARKS:
                for model in MODEL_TAGS:
                    color  = MODEL_COLORS[model]
                    marker = BENCH_MARKERS[bench]
                    y_off  = y_offsets[dot_idx]

                    df_bm = df[
                        (df["split"]     == split) &
                        (df["benchmark"] == bench) &
                        (df["model_tag"] == model)
                    ]

                    for ai, algo in enumerate(ALGO_ORDER):
                        sub = df_bm[df_bm["algo"] == algo]
                        if sub.empty or metric not in sub.columns:
                            continue
                        val = float(sub[metric].mean())
                        ax.scatter(
                            val, ai + y_off,
                            color=color, marker=marker,
                            s=26, zorder=4,
                            linewidths=0.4, edgecolors="white",
                        )
                    dot_idx += 1

            # ── Y-axis ────────────────────────────────────────────────────────
            ax.set_yticks(range(n_algos))
            if ci == 0:
                labels = [ALGO_LABELS.get(a, a.upper()) for a in ALGO_ORDER]
                # Bold the AdaPop label
                ax.set_yticklabels(labels, fontsize=8)
                ax.get_yticklabels()[ada_idx].set_fontweight("bold")
            else:
                ax.set_yticklabels([])

            # ── X-axis: 3 ticks, scientific notation for large values ─────────
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
            fmt = mticker.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 3))
            ax.xaxis.set_major_formatter(fmt)
            ax.tick_params(axis="x", labelsize=6, rotation=40, pad=1, length=2)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")

            ax.set_ylim(-0.52, n_algos - 0.48)

            # ── Column header: top row only ───────────────────────────────────
            if ri == 0:
                direction = "↓ lower is better" if split == "forget" else "↑ higher is better"
                ax.set_title(
                    f"{split.capitalize()}\n{direction}",
                    fontsize=8.5, fontweight="bold", pad=8, linespacing=1.4,
                )

            # ── Metric label: below the panel as x-axis label ─────────────────
            ax.set_xlabel(metric_label, fontsize=7, fontweight="bold",
                          color="#444", labelpad=3)

            _style_ax(ax)

    # ── Legend ────────────────────────────────────────────────────────────────
    model_handles = [
        plt.scatter([], [], color=MODEL_COLORS[m], marker="o", s=28,
                    label=MODEL_LABELS[m], linewidths=0.4, edgecolors="white")
        for m in MODEL_TAGS
    ]
    bench_handles = [
        plt.scatter([], [], color="#666", marker=BENCH_MARKERS[b], s=28,
                    label=b.upper(), linewidths=0.4, edgecolors="white")
        for b in BENCHMARKS
    ]

    leg = fig.legend(
        handles=model_handles + bench_handles,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=0.7,
        handletextpad=0.35,
    )
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out_path}")


# ── Table ─────────────────────────────────────────────────────────────────────

def make_table(df: pd.DataFrame, out_stem: Path) -> None:
    """
    Per-benchmark tables: rows = algo, columns = split × metric (mean across models).
    Saves CSV and Markdown side by side with the plot.
    """
    metric_keys = [m for m, _ in PAPER_METRICS]

    all_tables: Dict[str, pd.DataFrame] = {}

    for bench in BENCHMARKS:
        df_b = df[df["benchmark"] == bench]
        if df_b.empty:
            continue

        records = []
        for algo in ALGO_ORDER:
            df_a = df_b[df_b["algo"] == algo]
            row: Dict = {"algo": ALGO_LABELS.get(algo, algo.upper())}
            for split in SPLITS:
                df_s = df_a[df_a["split"] == split]
                for mk in metric_keys:
                    col = f"{split[:3]}_{mk.replace('_mean','')}"
                    if not df_s.empty and mk in df_s.columns:
                        row[col] = round(float(df_s[mk].mean()), 4)
                    else:
                        row[col] = float("nan")
            records.append(row)

        tbl = pd.DataFrame(records).set_index("algo")
        all_tables[bench] = tbl

    # ── CSV (one sheet per benchmark, stacked with a blank row separator) ──
    csv_path = out_stem.with_suffix(".csv")
    with open(csv_path, "w") as f:
        for bench, tbl in all_tables.items():
            f.write(f"# {bench.upper()}\n")
            tbl.to_csv(f)
            f.write("\n")
    print(f"Saved CSV   → {csv_path}")

    # ── Markdown ──
    md_path = out_stem.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# Forget Metrics — lr=1e-4 (mean across models)\n\n")
        for bench, tbl in all_tables.items():
            f.write(f"## {bench.upper()}\n\n")
            # Pretty column headers
            col_renames = {}
            for split in SPLITS:
                for mk, ml in PAPER_METRICS:
                    col_renames[f"{split[:3]}_{mk.replace('_mean','')}"] = \
                        f"{split[:3].capitalize()} {ml}"
            tbl_display = tbl.rename(columns=col_renames)
            f.write(tbl_display.to_markdown(floatfmt=".4g"))
            f.write("\n\n")
    print(f"Saved MD    → {md_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Resolve repo root from script location
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent.parent

    frames: List[pd.DataFrame] = []
    for bench in BENCHMARKS:
        df_b = collect(bench, repo_root)
        if df_b.empty:
            print(f"  [warn] no lr={TARGET_LR} data for '{bench}'")
        else:
            print(f"  [{bench}] {len(df_b)} rows, "
                  f"algos={sorted(df_b['algo'].unique())}, "
                  f"models={sorted(df_b['model_tag'].unique())}")
            frames.append(df_b)

    if not frames:
        print("No data collected. Run the forget-metrics scripts first.")
        return

    df = pd.concat(frames, ignore_index=True)

    stem = script_dir / "paper_forget_metrics"
    make_figure(df, stem.with_suffix(".png"))
    make_table(df, stem.with_name("paper_forget_metrics_table"))
    print("Done.")


if __name__ == "__main__":
    main()
