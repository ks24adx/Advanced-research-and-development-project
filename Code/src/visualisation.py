"""
visualisation.py
----------------
Produces all plots required for the research paper:

  Fig 1 — Error type breakdown per (domain × condition) — stacked bar
  Fig 2 — Accuracy comparison across conditions (grouped bar + error bars)
  Fig 3 — Knowledge vs Reasoning error % heatmap
  Fig 4 — Partial-credit and logical-consistency distributions (violin)
  Fig 5 — Cohen's Kappa per slice (reliability)
  Fig 6 — Hallucination rate across conditions
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import CONDITIONS, DOMAINS, RESULTS_DIR

logger = logging.getLogger(__name__)

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

CONDITION_COLORS = {
    "reasoning_only": "#4C72B0",
    "rag":            "#DD8452",
    "hybrid":         "#55A868",
}
ERROR_COLORS = {
    "pure_knowledge_failure": "#E05C5C",
    "pure_reasoning_failure": "#5C7EE0",
    "mixed_failure":          "#E0A85C",
    "ambiguous":              "#AAAAAA",
    "correct":                "#5CB85C",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _save(fig, name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


# ── Figure 1: Error type stacked bar ────────────────────────────────────────

def plot_error_breakdown(eval_df: pd.DataFrame,
                         output_dir: str = RESULTS_DIR) -> None:
    """Stacked bar: error label proportions per (domain, condition)."""
    labels = ["correct", "pure_knowledge_failure",
              "pure_reasoning_failure", "mixed_failure", "ambiguous"]
    col_map = {
        "correct":               "n_correct",
        "pure_knowledge_failure":"n_knowledge_errors",
        "pure_reasoning_failure":"n_reasoning_errors",
        "mixed_failure":         "n_mixed_errors",
        "ambiguous":             "n_ambiguous",
    }

    n_domains = len(eval_df["domain"].unique())
    fig, axes = plt.subplots(1, n_domains, figsize=(7 * n_domains, 5),
                             sharey=False)
    if n_domains == 1:
        axes = [axes]

    for ax, domain in zip(axes, eval_df["domain"].unique()):
        sub = eval_df[eval_df["domain"] == domain].copy()
        sub = sub.set_index("condition")
        bottom = np.zeros(len(sub))

        x = np.arange(len(sub))
        for lbl in labels:
            col = col_map.get(lbl)
            if col not in sub.columns:
                continue
            vals = sub[col].values.astype(float)
            ax.bar(x, vals, bottom=bottom, color=ERROR_COLORS[lbl],
                   label=lbl.replace("_", " ").title(), width=0.55)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=15, ha="right")
        ax.set_title(f"Domain: {domain.capitalize()}", fontweight="bold")
        ax.set_ylabel("Number of responses")
        ax.set_xlabel("Condition")

    handles = [mpatches.Patch(color=ERROR_COLORS[l], label=l.replace("_", " ").title())
               for l in labels]
    axes[-1].legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                    fontsize=9, title="Error Type")

    fig.suptitle("Error Type Breakdown by Domain and Condition", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig1_error_breakdown.png", output_dir)


# ── Figure 2: Accuracy grouped bar ──────────────────────────────────────────

def plot_accuracy_comparison(eval_df: pd.DataFrame,
                              output_dir: str = RESULTS_DIR) -> None:
    """Grouped bar chart: accuracy per (condition, domain)."""
    fig, ax = plt.subplots(figsize=(9, 5))

    domains    = eval_df["domain"].unique()
    conditions = eval_df["condition"].unique()
    n = len(domains)
    x = np.arange(n)
    width = 0.8 / len(conditions)

    for i, cond in enumerate(conditions):
        sub   = eval_df[eval_df["condition"] == cond].set_index("domain")
        accs  = [sub.loc[d, "accuracy"] if d in sub.index else 0.0 for d in domains]
        offset = (i - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width * 0.9,
                      color=CONDITION_COLORS.get(cond, "#888888"),
                      label=cond.replace("_", " ").title())
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_ylabel("Accuracy (semantic similarity ≥ 0.65)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Accuracy by Condition and Domain", fontweight="bold")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, "fig2_accuracy_comparison.png", output_dir)


# ── Figure 3: Knowledge vs Reasoning heatmap ────────────────────────────────

def plot_error_heatmap(eval_df: pd.DataFrame,
                       output_dir: str = RESULTS_DIR) -> None:
    """Heatmap of (knowledge %, reasoning %) across (domain, condition)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, title in zip(
        axes,
        ["knowledge_error_pct", "reasoning_error_pct"],
        ["Knowledge Error %", "Reasoning Error %"],
    ):
        pivot = eval_df.pivot(index="condition", columns="domain", values=col)
        pivot = pivot.fillna(0)
        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".1%",
            cmap="YlOrRd", vmin=0, vmax=1,
            linewidths=0.5, cbar_kws={"label": "Proportion of errors"},
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Domain")
        ax.set_ylabel("Condition")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.suptitle("Error Attribution Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig3_error_heatmap.png", output_dir)


# ── Figure 4: Distribution plots ────────────────────────────────────────────

def plot_metric_distributions(
    metrics_records: list,
    output_dir: str = RESULTS_DIR,
) -> None:
    """
    Violin plots for partial-credit and logical-consistency distributions.

    Parameters
    ----------
    metrics_records : list of dicts with keys
        {domain, condition, partial_credit, logical_consistency}
    """
    df = pd.DataFrame(metrics_records)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes,
        ["partial_credit", "logical_consistency"],
        ["Partial Credit Score", "Logical Consistency Score"],
    ):
        if metric not in df.columns:
            continue
        sns.violinplot(
            data=df, x="condition", y=metric, hue="domain",
            ax=ax, inner="quartile", split=False,
            palette={"law": "#4C72B0", "medicine": "#DD8452"},
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Score [0, 1]")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)

    fig.suptitle("Score Distributions by Condition and Domain",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig4_score_distributions.png", output_dir)


# ── Figure 5: Cohen's Kappa ──────────────────────────────────────────────────

def plot_kappa(eval_df: pd.DataFrame,
               output_dir: str = RESULTS_DIR) -> None:
    """Bar chart of Cohen's Kappa per (domain, condition) slice."""
    if "cohen_kappa" not in eval_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    domains    = eval_df["domain"].unique()
    conditions = eval_df["condition"].unique()
    n = len(domains)
    x = np.arange(n)
    width = 0.7 / len(conditions)

    for i, cond in enumerate(conditions):
        sub    = eval_df[eval_df.condition == cond].set_index("domain")
        kappas = [sub.loc[d, "cohen_kappa"] if d in sub.index else 0.0
                  for d in domains]
        offset = (i - len(conditions) / 2 + 0.5) * width
        ax.bar(x + offset, kappas, width * 0.9,
               color=CONDITION_COLORS.get(cond, "#888888"),
               label=cond.replace("_", " ").title())

    ax.axhline(0.6, color="red", linestyle="--", linewidth=1.2,
               label="Acceptable threshold (κ=0.60)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_ylabel("Cohen's Kappa")
    ax.set_ylim(0, 1.1)
    ax.set_title("Inter-Annotator Agreement (Cohen's κ)", fontweight="bold")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, "fig5_cohens_kappa.png", output_dir)


# ── Figure 6: Hallucination rate ────────────────────────────────────────────

def plot_hallucination_rate(eval_df: pd.DataFrame,
                             output_dir: str = RESULTS_DIR) -> None:
    """Line plot of hallucination rate across conditions, per domain."""
    if "hallucination_rate" not in eval_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for domain in eval_df["domain"].unique():
        sub   = eval_df[eval_df.domain == domain]
        conds = sub["condition"].tolist()
        rates = sub["hallucination_rate"].tolist()
        ax.plot(conds, rates, marker="o", linewidth=2,
                label=domain.capitalize())

    ax.set_title("Hallucination Rate by Condition", fontweight="bold")
    ax.set_ylabel("Hallucination Rate")
    ax.set_ylim(0, 1)
    ax.legend(title="Domain")
    ax.set_xlabel("Condition")
    fig.tight_layout()
    _save(fig, "fig6_hallucination_rate.png", output_dir)


# ── Master function ──────────────────────────────────────────────────────────

def generate_all_plots(
    eval_df: pd.DataFrame,
    metrics_records: Optional[list] = None,
    output_dir: str = RESULTS_DIR,
) -> None:
    """Generate and save all figures."""
    plot_error_breakdown(eval_df, output_dir)
    plot_accuracy_comparison(eval_df, output_dir)
    plot_error_heatmap(eval_df, output_dir)
    if metrics_records:
        plot_metric_distributions(metrics_records, output_dir)
    plot_kappa(eval_df, output_dir)
    plot_hallucination_rate(eval_df, output_dir)
    logger.info("All figures saved to %s", output_dir)
