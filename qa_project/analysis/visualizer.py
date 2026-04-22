"""
analysis/visualizer.py
─────────────────────────────────────────────────────────────
Generates every chart used in the research report.

Changes from original:
  - Removed matplotlib.use("Agg") — causes display issues in Jupyter
  - Fixed FutureWarning: added observed=False to groupby("bin")
  - Added plot_error_distribution_real() — uses original CSV annotations
    instead of auto-classifier output (fixes 100% single-colour bug)
  - generate_all() now calls plot_error_distribution_real()
  - All plt.show() replaced with plt.close() — prevents duplicate images
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

from config import OUTPUT_DIR, COLORS

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.dpi"        : 150,
})

SAVE = OUTPUT_DIR / "charts"
SAVE.mkdir(parents=True, exist_ok=True)


# ── 1. Closed-Book vs Open-Book accuracy ─────────────────────────────────────

def plot_closed_vs_open(results: dict[str, pd.DataFrame],
                         datasets: list[str] | None = None) -> Path:
    """Bar chart: closed_book vs rag_only accuracy per dataset."""
    if datasets is None:
        datasets = ["Natural Questions", "TriviaQA", "HotpotQA"]

    def em_pct(cond, ds):
        df = results[cond]
        sub = df[df["source_dataset"] == ds]
        return round(sub["em"].mean() * 100, 1) if len(sub) else 0

    closed = [em_pct("closed_book", d) for d in datasets]
    open_  = [em_pct("rag_only",    d) for d in datasets]

    x, w = np.arange(len(datasets)), 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, closed, w, color=COLORS["blue"],   label="Closed-Book",     zorder=3)
    b2 = ax.bar(x + w/2, open_,  w, color=COLORS["orange"], label="Open-Book (RAG)", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
    ax.set_title("Closed-Book vs. Open-Book Accuracy Across Datasets",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 115); ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=10)
    for b in [*b1, *b2]:
        ax.annotate(f"{b.get_height()}%",
                    xy=(b.get_x() + b.get_width()/2, b.get_height() + 0.8),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = SAVE / "fig1_closed_vs_open.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


# ── 2a. Error distribution (auto-classifier — kept for reference) ─────────────

def plot_error_distribution(df: pd.DataFrame) -> Path:
    """
    Pie charts using auto-classifier output.
    NOTE: In mock mode this shows 100% single colour. Use
    plot_error_distribution_real() for correct results.
    """
    datasets = df["source_dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5))
    colors = [COLORS["blue"], COLORS["orange"], COLORS["grey"]]
    labels = ["Knowledge Error", "Reasoning Error", "Other / Ambiguous"]

    for ax, ds in zip(axes, datasets):
        sub   = df[df["source_dataset"] == ds]
        total = len(sub[sub["error_type"] != "none"])
        k_n   = len(sub[sub["error_type"] == "knowledge"])
        r_n   = len(sub[sub["error_type"] == "reasoning"])
        o_n   = total - k_n - r_n
        wedges, _, autos = ax.pie(
            [k_n, r_n, max(o_n, 0)],
            colors=colors, autopct="%1.0f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        for at in autos: at.set_fontsize(10); at.set_fontweight("bold")
        ax.set_title(ds, fontsize=10, fontweight="bold")

    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Distribution of Error Types per Dataset",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = SAVE / "fig2_error_distribution.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


# ── 2b. Error distribution (REAL — uses original CSV annotations) ─────────────

def plot_error_distribution_real(df: pd.DataFrame) -> Path:
    """
    Pie charts using the original manually annotated error_type column
    from the CSV — NOT the auto-classifier output.

    Fixes the 100% single-colour bug caused by mock mode RAG, where
    every wrong answer is labelled a knowledge error because RAG
    fixes everything perfectly in mock mode.

    Real values (from pipeline output):
      Natural Questions : knowledge 30.6%, reasoning  0.0%, none 69.4%
      TriviaQA          : knowledge 15.2%, reasoning  0.0%, none 84.8%
      HotpotQA          : knowledge  6.5%, reasoning 83.9%, none  9.7%
    """
    datasets = df["source_dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5))
    fig.patch.set_facecolor("white")

    colors = [COLORS["blue"], COLORS["orange"], COLORS["grey"]]
    labels = ["Knowledge Error", "Reasoning Error", "Correct / None"]

    for ax, ds in zip(axes, datasets):
        sub   = df[df["source_dataset"] == ds]
        total = len(sub)

        k_n = len(sub[sub["error_type"] == "knowledge"])
        r_n = len(sub[sub["error_type"] == "reasoning"])
        o_n = total - k_n - r_n

        k_pct = round(k_n / total * 100, 1)
        r_pct = round(r_n / total * 100, 1)
        o_pct = round(100 - k_pct - r_pct, 1)

        wedges, texts, autotexts = ax.pie(
            [k_pct, r_pct, o_pct],
            colors      = colors,
            autopct     = "%1.1f%%",
            startangle  = 90,
            pctdistance = 0.72,
            wedgeprops  = dict(edgecolor="white", linewidth=2.5),
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
            at.set_color("white")

        ax.set_title(ds, fontsize=13, fontweight="bold", pad=14)

    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(colors, labels)]
    fig.legend(
        handles        = handles,
        loc            = "lower center",
        ncol           = 3,
        fontsize       = 11,
        frameon        = False,
        bbox_to_anchor = (0.5, -0.02),
    )
    fig.suptitle(
        "Distribution of Error Types per Dataset",
        fontsize   = 14,
        fontweight = "bold",
        y          = 1.02,
    )
    plt.tight_layout()
    path = SAVE / "fig2_error_distribution_real.png"
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# ── 3. CoT impact by error category ──────────────────────────────────────────

def plot_cot_impact(results: dict[str, pd.DataFrame]) -> Path:
    """Grouped bars: standard vs CoT for each error category."""
    def em(cond, filt_col, filt_val):
        df = results[cond]
        sub = df[df[filt_col] == filt_val] if filt_col in df.columns else df
        return round(sub["em"].mean() * 100, 1) if len(sub) else 0

    categories = ["Knowledge\nErrors", "Reasoning\nErrors",
                  "Multi-hop\nReasoning", "Single-hop\nFactual"]
    std = [
        em("closed_book", "error_type",        "knowledge"),
        em("closed_book", "error_type",        "reasoning"),
        em("closed_book", "num_hops_required", 2),
        em("closed_book", "num_hops_required", 1),
    ]
    cot = [
        em("cot_only", "error_type",        "knowledge"),
        em("cot_only", "error_type",        "reasoning"),
        em("cot_only", "num_hops_required", 2),
        em("cot_only", "num_hops_required", 1),
    ]

    x, w = np.arange(len(categories)), 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, std, w, color=COLORS["grey"],  label="Standard Prompting", zorder=3)
    b2 = ax.bar(x + w/2, cot, w, color=COLORS["green"], label="CoT Prompting",      zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Chain-of-Thought Prompting Impact by Error Category",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 115); ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=10)
    for b in [*b1, *b2]:
        ax.annotate(f"{b.get_height()}",
                    xy=(b.get_x() + b.get_width()/2, b.get_height() + 0.6),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = SAVE / "fig3_cot_impact.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


# ── 4. Retrieval quality vs accuracy ─────────────────────────────────────────

def plot_retrieval_curve(df: pd.DataFrame) -> Path:
    """Line plot: confidence bins vs accuracy for knowledge vs reasoning errors."""
    bins = np.linspace(0, 1, 7)
    labels_bin = [f"{int(b*100)}" for b in bins[:-1]]

    def binned_em(error_type, score_col="confidence_score_open",
                  correct_col="open_book_correct"):
        sub = df[df["error_type"] == error_type].copy()
        if sub.empty: return [0]*6
        sub["bin"] = pd.cut(sub[score_col], bins=bins, labels=labels_bin,
                            include_lowest=True)
        # observed=False keeps all bins even if empty — fixes FutureWarning
        grouped = sub.groupby("bin", observed=False)[correct_col].mean().mul(100).reindex(labels_bin)
        return grouped.fillna(0).tolist()

    k_acc = binned_em("knowledge")
    r_acc = binned_em("reasoning")
    x     = [int(l) for l in labels_bin]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, k_acc, "o-", color=COLORS["blue"],   lw=2, ms=7, label="Knowledge-Error Questions")
    ax.plot(x, r_acc, "s-", color=COLORS["orange"], lw=2, ms=7, label="Reasoning-Error Questions")
    ax.fill_between(x, k_acc, alpha=0.12, color=COLORS["blue"])
    ax.fill_between(x, r_acc, alpha=0.12, color=COLORS["orange"])
    ax.set_xlabel("Retrieval Confidence Bin (%)", fontsize=11)
    ax.set_ylabel("Correct Answer Rate (%)",      fontsize=11)
    ax.set_title("Effect of Retrieval Quality on Answer Accuracy by Error Type",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(-2, 102); ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = SAVE / "fig4_retrieval_curve.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


# ── 5. All conditions comparison ─────────────────────────────────────────────

def plot_all_conditions(results: dict[str, pd.DataFrame],
                         datasets: list[str] | None = None) -> Path:
    """Grouped bar chart across all 5 conditions × 3 datasets."""
    if datasets is None:
        datasets = ["Natural Questions", "TriviaQA", "HotpotQA"]

    conditions  = ["closed_book", "rag_only", "cot_only",
                   "rag_cot", "rag_cot_selfverify"]
    cond_labels = ["Baseline\n(Closed-Book)", "RAG Only", "CoT Only",
                   "RAG + CoT", "RAG + CoT\n+ Self-Verify"]
    ds_colors   = [COLORS["dark"], COLORS["orange"], COLORS["green"]]

    x, w = np.arange(len(conditions)), 0.22
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (ds, color) in enumerate(zip(datasets, ds_colors)):
        vals = []
        for cond in conditions:
            df  = results[cond]
            sub = df[df["source_dataset"] == ds]
            vals.append(round(sub["em"].mean() * 100, 1) if len(sub) else 0)
        offset = (i - 1) * w
        ax.bar(x + offset, vals, w, color=color, label=ds, zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(cond_labels, fontsize=9.5)
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy Across All Experimental Conditions",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 115); ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=9.5)
    plt.tight_layout()
    path = SAVE / "fig5_all_conditions.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


# ── 6. Confidence calibration ─────────────────────────────────────────────────

def plot_calibration(df: pd.DataFrame) -> Path:
    """Reliability diagram: model confidence vs observed accuracy."""
    bins  = np.linspace(0, 1, 11)
    mids  = (bins[:-1] + bins[1:]) / 2 * 100
    ideal = mids

    def binned(conf_col, correct_col):
        vals = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sub = df[(df[conf_col] >= lo) & (df[conf_col] < hi)]
            vals.append(sub[correct_col].mean() * 100 if len(sub) > 0 else np.nan)
        return vals

    cb = binned("confidence_score_closed", "closed_book_correct")
    ob = binned("confidence_score_open",   "open_book_correct")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mids, ideal, "--", color="grey",         lw=1.5, label="Perfect Calibration")
    ax.plot(mids, cb, "o-", color=COLORS["blue"],   lw=2, ms=6, label="Closed-Book")
    ax.plot(mids, ob, "s-", color=COLORS["orange"], lw=2, ms=6, label="Open-Book (RAG)")
    ax.set_xlabel("Confidence Score (%)",  fontsize=11)
    ax.set_ylabel("Observed Accuracy (%)", fontsize=11)
    ax.set_title("Confidence Calibration Diagram",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = SAVE / "fig6_calibration.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


# ── Convenience: generate all charts ─────────────────────────────────────────

def generate_all(df: pd.DataFrame,
                 results: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """
    Generate all 6 report charts. Returns dict of {name: Path}.
    Uses plot_error_distribution_real() so original CSV annotations
    are used — not the auto-classifier which breaks in mock mode.
    """
    paths = {}
    paths["closed_vs_open"]     = plot_closed_vs_open(results)
    paths["error_distribution"] = plot_error_distribution_real(df)   # ← real annotations
    paths["cot_impact"]         = plot_cot_impact(results)
    paths["retrieval_curve"]    = plot_retrieval_curve(df)
    paths["all_conditions"]     = plot_all_conditions(results)
    paths["calibration"]        = plot_calibration(df)
    return paths
