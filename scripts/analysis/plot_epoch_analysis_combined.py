"""
Plot baseline + AdaPop epoch analysis for Llama-3.1-8B-Instruct.

Generates two JPEG files in saves/plots/baselines_epoch_analysis/:
  1) epoch_analysis_all_colored_Llama-3.1-8B-Instruct.jpeg
       — all baselines colored, AdaPop as black line with star markers
  2) epoch_analysis_adapop_highlight_Llama-3.1-8B-Instruct.jpeg
       — AdaPop in red, all baselines in shades of grey

Usage (from repo root):
    conda run -n MU python scripts/analysis/plot_epoch_analysis_combined.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
MODEL_NAME  = "Llama-3.1-8B-Instruct"
BASELINES_CSV = (
    REPO_ROOT / "saves/plots/baselines_epoch_analysis"
    / f"baselines_epochs_{MODEL_NAME}.csv"
)
ADAPOP_CSV = (
    REPO_ROOT
    / "saves/plots/baselines_epoch_analysis/ada_pop_epoch_analysis_llama"
    / f"ada_pop_epochs_{MODEL_NAME}.csv"
)
OUT_DIR = REPO_ROOT / "saves/plots/baselines_epoch_analysis"

# ── Load data ─────────────────────────────────────────────────────────────────
baselines = pd.read_csv(BASELINES_CSV)
adapop    = pd.read_csv(ADAPOP_CSV)

algos = baselines["algo"].unique().tolist()

# Colors for the "all colored" plot
ALGO_COLORS = {
    "GA":     "#4e79a7",
    "GD":     "#f28e2b",
    "NPO":    "#e15759",
    "PDU":    "#76b7b2",
    "RMU":    "#59a14f",
    "UNDIAL": "#edc948",
    "WGA":    "#b07aa1",
}

Y_COLS  = ["forget_rouge_l", "retain_rouge_l"]
TITLES  = ["Forget ROUGE-L vs Epoch", "Retain ROUGE-L vs Epoch"]
Y_LABEL = "ROUGE-L"


def _add_axes(axes, y_col, title, baseline_kwargs_fn, adapop_kwargs):
    """Draw one subplot column. baseline_kwargs_fn(algo) → dict of plot kwargs."""
    for ax, yc, ttl in zip(axes, [y_col] if not isinstance(y_col, list) else y_col,
                           [title] if not isinstance(title, list) else title):
        pass  # unused – kept for symmetry

def draw_subplot(ax, y_col, title, baseline_kw_fn, adapop_kw):
    for algo in algos:
        sub = baselines[baselines["algo"] == algo].sort_values("epoch")
        kw  = baseline_kw_fn(algo)
        ax.plot(sub["epoch"], sub[y_col], **kw)
    # AdaPop on top
    ax.plot(adapop["epoch"].values, adapop[y_col].values, **adapop_kw)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(Y_LABEL, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — all algos colored, AdaPop with star markers
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

def baseline_kw_colored(algo):
    return dict(
        marker="o", markersize=5, linewidth=1.8,
        color=ALGO_COLORS.get(algo, "grey"),
        label=algo,
    )

adapop_kw_colored = dict(
    marker="*", markersize=13, linestyle="-",
    linewidth=2.2, color="black", zorder=5,
    label="AdaPop",
)

for ax, y_col, title in zip(axes, Y_COLS, TITLES):
    draw_subplot(ax, y_col, title, baseline_kw_colored, adapop_kw_colored)

fig.suptitle(f"Epoch Analysis — {MODEL_NAME}", fontsize=14, fontweight="bold")
plt.tight_layout()
out1 = OUT_DIR / f"epoch_analysis_all_colored_{MODEL_NAME}.jpeg"
fig.savefig(out1, dpi=150, bbox_inches="tight", format="jpeg")
print(f"Saved: {out1}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — AdaPop in red, baselines in shades of grey
# ═══════════════════════════════════════════════════════════════════════════════
# distribute grey values from dark (0.20) to light (0.72) so all are visible
grey_values = np.linspace(0.20, 0.72, len(algos))

def baseline_kw_grey(algo):
    idx   = algos.index(algo)
    shade = str(grey_values[idx])
    return dict(
        marker="o", markersize=5, linewidth=1.8,
        color=shade, label=algo,
    )

adapop_kw_red = dict(
    marker="*", markersize=13, linestyle="-",
    linewidth=2.5, color="red", zorder=5,
    label="AdaPop",
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, y_col, title in zip(axes, Y_COLS, TITLES):
    draw_subplot(ax, y_col, title, baseline_kw_grey, adapop_kw_red)

fig.suptitle(f"Epoch Analysis — {MODEL_NAME}", fontsize=14, fontweight="bold")
plt.tight_layout()
out2 = OUT_DIR / f"epoch_analysis_adapop_highlight_{MODEL_NAME}.jpeg"
fig.savefig(out2, dpi=150, bbox_inches="tight", format="jpeg")
print(f"Saved: {out2}")
plt.close(fig)
