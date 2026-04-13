"""
evaluation/metrics.py
─────────────────────────────────────────────────────────────
All evaluation metrics used in the study:
  - Exact Match (EM)
  - Token-level F1
  - Per-condition aggregation tables
  - Per-error-type breakdowns
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from utils import exact_match, token_f1


# ── Single-example scoring ────────────────────────────────────────────────────

def score_answer(prediction: str, ground_truth: str) -> dict:
    """Return a dict with em and f1 for one prediction."""
    return {
        "em" : int(exact_match(prediction, ground_truth)),
        "f1" : token_f1(prediction, ground_truth),
    }


# ── Batch scoring ─────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """
    Aggregate a list of per-question result dicts.

    Each result dict should contain at minimum:
        prediction, ground_truth, error_type, source_dataset

    Returns overall EM, F1 and breakdowns by dataset and error_type.
    """
    if not results:
        return {}

    df = pd.DataFrame(results)
    df["em"] = df.apply(
        lambda r: int(exact_match(str(r["prediction"]),
                                   str(r["ground_truth"]))), axis=1)
    df["f1"] = df.apply(
        lambda r: token_f1(str(r["prediction"]),
                            str(r["ground_truth"])), axis=1)

    overall = {
        "em_mean" : round(df["em"].mean() * 100, 1),
        "f1_mean" : round(df["f1"].mean() * 100, 1),
        "n"       : len(df),
    }

    # Breakdown by source_dataset
    by_dataset = (
        df.groupby("source_dataset")[["em", "f1"]]
          .mean()
          .mul(100)
          .round(1)
          .rename(columns={"em": "EM (%)", "f1": "F1 (%)"})
          .to_dict(orient="index")
    )

    # Breakdown by error_type
    by_error = (
        df.groupby("error_type")[["em", "f1"]]
          .mean()
          .mul(100)
          .round(1)
          .rename(columns={"em": "EM (%)", "f1": "F1 (%)"})
          .to_dict(orient="index")
    )

    return {
        "overall"    : overall,
        "by_dataset" : by_dataset,
        "by_error"   : by_error,
        "_df"        : df,         # raw dataframe for further analysis
    }


# ── Condition comparison table ─────────────────────────────────────────────────

def condition_table(condition_results: dict[str, pd.DataFrame],
                    dataset_col: str = "source_dataset") -> pd.DataFrame:
    """
    Build a summary table: rows = conditions, columns = datasets + average.

    condition_results: mapping of condition_name → DataFrame with columns
                       [prediction, ground_truth, source_dataset]
    """
    rows = []
    for cond, df in condition_results.items():
        df = df.copy()
        df["em"] = df.apply(
            lambda r: int(exact_match(str(r["prediction"]),
                                       str(r["ground_truth"]))), axis=1)
        row = {"condition": cond}
        for ds in df[dataset_col].unique():
            sub = df[df[dataset_col] == ds]
            row[ds] = round(sub["em"].mean() * 100, 1)
        row["Average"] = round(df["em"].mean() * 100, 1)
        rows.append(row)

    return pd.DataFrame(rows).set_index("condition")


# ── Error-type accuracy breakdown ─────────────────────────────────────────────

def error_breakdown(df: pd.DataFrame,
                    pred_col: str,
                    gt_col: str = "ground_truth_answer",
                    error_col: str = "error_type") -> pd.DataFrame:
    """
    For each error_type compute mean EM and F1.
    """
    df = df.copy()
    df["em"] = df.apply(
        lambda r: int(exact_match(str(r[pred_col]), str(r[gt_col]))), axis=1)
    df["f1"] = df.apply(
        lambda r: token_f1(str(r[pred_col]), str(r[gt_col])), axis=1)

    return (
        df.groupby(error_col)[["em", "f1"]]
          .agg(["mean", "count"])
          .round(3)
    )


# ── Cohen's Kappa (inter-annotator agreement) ─────────────────────────────────

def cohen_kappa(annotator_a: list, annotator_b: list) -> float:
    """
    Compute Cohen's kappa between two lists of categorical labels.
    """
    from sklearn.metrics import cohen_kappa_score
    return round(cohen_kappa_score(annotator_a, annotator_b), 3)


# ── Confidence calibration ────────────────────────────────────────────────────

def calibration_stats(df: pd.DataFrame,
                       conf_col: str,
                       correct_col: str) -> dict:
    """
    Bin predictions by confidence and compute mean accuracy per bin.
    Returns a dict suitable for a reliability diagram.
    """
    bins = np.linspace(0, 1, 11)
    labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    df = df.copy()
    df["bin"] = pd.cut(df[conf_col], bins=bins, labels=labels, include_lowest=True)
    stats = (
        df.groupby("bin")[correct_col]
          .agg(["mean", "count"])
          .rename(columns={"mean": "accuracy", "count": "n"})
          .reset_index()
    )
    return stats.to_dict(orient="records")
