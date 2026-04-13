"""
evaluation/error_classifier.py
─────────────────────────────────────────────────────────────
Automatically classifies the source of each error using the
two-stage closed-book / open-book evaluation protocol described
in the paper.

Classification rules
────────────────────
  KNOWLEDGE error  : closed-book = wrong AND
                     supporting facts retrieved = YES AND
                     open-book = correct

  REASONING error  : closed-book = wrong AND
                     supporting facts retrieved = YES AND
                     open-book = STILL wrong

  AMBIGUOUS        : closed-book = wrong AND
                     supporting facts retrieved = NO
                     (cannot distinguish error source)

  CORRECT          : closed-book = correct (no error)
"""

from __future__ import annotations
import pandas as pd
from utils import exact_match, log_info


LABELS = {
    "correct"   : "none",
    "knowledge" : "knowledge",
    "reasoning" : "reasoning",
    "ambiguous" : "ambiguous",
}


def classify_row(row: pd.Series) -> str:
    """
    Apply the classification protocol to a single DataFrame row.

    Expected columns:
        closed_book_correct, open_book_correct,
        supporting_facts_retrieved
    """
    cb_ok  = int(row.get("closed_book_correct", 0))
    ob_ok  = int(row.get("open_book_correct",   0))
    facts  = int(row.get("supporting_facts_retrieved", 0))

    if cb_ok:
        return LABELS["correct"]

    # Closed-book is wrong —— check retrieval quality
    if not facts:
        return LABELS["ambiguous"]

    # Supporting passage was retrieved
    if ob_ok:
        return LABELS["knowledge"]   # retrieval fixed it → was a knowledge gap
    else:
        return LABELS["reasoning"]   # retrieval did NOT fix it → reasoning failure


def classify_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply classify_row to every row and add a 'predicted_error_type' column.
    Also computes agreement with the manual 'error_type' annotation.
    """
    df = df.copy()
    df["predicted_error_type"] = df.apply(classify_row, axis=1)

    if "error_type" in df.columns:
        # Map 'none' to correct for comparison
        agree = (df["predicted_error_type"] == df["error_type"])
        pct   = round(agree.mean() * 100, 1)
        log_info(f"Auto-classifier agrees with manual annotations: {pct}%")

    return df


def error_distribution(df: pd.DataFrame,
                        label_col: str = "error_type") -> pd.DataFrame:
    """
    Return count and percentage of each error type per dataset.
    """
    grouped = (
        df.groupby(["source_dataset", label_col])
          .size()
          .reset_index(name="count")
    )
    totals = grouped.groupby("source_dataset")["count"].transform("sum")
    grouped["percentage"] = (grouped["count"] / totals * 100).round(1)
    return grouped


def subtype_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts of each (error_type, error_subtype) combination.
    """
    return (
        df.groupby(["error_type", "error_subtype"])
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
    )


def hop_vs_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulate num_hops_required against error_type.
    """
    return pd.crosstab(
        df["num_hops_required"],
        df["error_type"],
        margins=True,
        margins_name="Total"
    )
