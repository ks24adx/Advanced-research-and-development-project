"""
analysis/ablation_plots.py
─────────────────────────────────────────────────────────────
Generates ablation-study charts:
  1. Top-k retrieval curve
  2. Error recovery bar chart
  3. Domain-level heatmap
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from config import OUTPUT_DIR, COLORS

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "figure.dpi"      : 150,
})

SAVE = OUTPUT_DIR / "charts"
SAVE.mkdir(parents=True, exist_ok=True)


def plot_topk_curve(topk_df: pd.DataFrame) -> Path:
    """Line plot: top-k vs EM for knowledge / reasoning / all."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(topk_df["top_k"], topk_df["EM_knowledge"], "o-",
            color=COLORS["blue"],   lw=2, ms=7, label="Knowledge Errors")
    ax.plot(topk_df["top_k"], topk_df["EM_reasoning"], "s-",
            color=COLORS["orange"], lw=2, ms=7, label="Reasoning Errors")
    ax.plot(topk_df["top_k"], topk_df["EM_all"],       "^-",
            color=COLORS["green"],  lw=2, ms=7, label="All Questions")
    ax.set_xlabel("Retrieval Top-K", fontsize=11)
    ax.set_ylabel("Exact Match (%)", fontsize=11)
    ax.set_title("Effect of Retrieval Top-K on Accuracy",
                 fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = SAVE / "abl1_topk_curve.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


def plot_recovery_rate(recovery_df: pd.DataFrame) -> Path:
    """Grouped bar chart: recovery rate per condition × error type."""
    if recovery_df.empty:
        return None

    from config import CONDITION_LABELS
    recovery_df = recovery_df.copy()
    recovery_df["cond_label"] = recovery_df["condition"].map(
        lambda c: CONDITION_LABELS.get(c, c)
    )

    conds  = recovery_df["cond_label"].unique()
    et_map = {"knowledge": COLORS["blue"], "reasoning": COLORS["orange"]}
    x, w   = np.arange(len(conds)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (et, color) in enumerate(et_map.items()):
        vals = []
        for cond in conds:
            sub = recovery_df[(recovery_df["cond_label"] == cond) &
                               (recovery_df["error_type"] == et)]
            vals.append(sub["recovery_%"].values[0] if len(sub) else 0)
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w, color=color,
                      label=f"{et.title()} Errors", zorder=3)
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}%",
                        xy=(b.get_x()+b.get_width()/2, b.get_height()+0.5),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Error Recovery Rate (%)", fontsize=11)
    ax.set_title("Error Recovery Rate per Condition",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = SAVE / "abl2_recovery.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


def plot_domain_heatmap(domain_df: pd.DataFrame) -> Path:
    """Heatmap: EM by (condition × domain)."""
    from config import CONDITION_LABELS
    if domain_df.empty:
        return None

    domain_df = domain_df.copy()
    domain_df["cond_label"] = domain_df["condition"].map(
        lambda c: CONDITION_LABELS.get(c, c)
    )
    pivot = domain_df.pivot_table(
        index="cond_label", columns="domain", values="EM (%)", aggfunc="mean"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=30, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Exact Match (%) by Condition and Domain",
                 fontsize=12, fontweight="bold", pad=12)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=9, color="white" if val > 70 else "black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="EM (%)")
    plt.tight_layout()
    path = SAVE / "abl3_domain_heatmap.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path
