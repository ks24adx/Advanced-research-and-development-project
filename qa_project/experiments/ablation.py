"""
experiments/ablation.py
─────────────────────────────────────────────────────────────
Ablation study utilities:
  1. Effect of retrieval top-k on accuracy
  2. Error recovery rate per condition
  3. Self-verify gain decomposition
  4. Domain-level accuracy breakdown
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from utils import exact_match, token_f1, log_info


# ── 1. Retrieval top-k ablation (mock) ───────────────────────────────────────

def topk_ablation(df: pd.DataFrame,
                  k_values: list[int] | None = None) -> pd.DataFrame:
    """
    Simulate how accuracy changes as top-k increases.
    In mock mode we use supporting_facts_retrieved as the oracle.
    For k >= 1, assume supporting fact is retrieved if flag = 1.
    Returns a DataFrame with columns: k, em_knowledge, em_reasoning, em_all
    """
    if k_values is None:
        k_values = [1, 2, 3, 5, 8, 10]

    rows = []
    for k in k_values:
        # Probability of retrieving the fact increases with k (simulated)
        retrieval_prob = min(1.0, k * 0.18)  # heuristic curve
        rng = np.random.default_rng(42 + k)

        em_k, em_r, em_n = [], [], []
        for _, row in df.iterrows():
            fact_retrieved = (
                int(row["supporting_facts_retrieved"]) == 1 or
                rng.random() < retrieval_prob
            )
            if fact_retrieved and int(row["open_book_correct"]) == 1:
                pred = str(row["ground_truth_answer"])
            elif fact_retrieved and int(row["open_book_correct"]) == 0:
                pred = str(row["open_book_answer"])
            else:
                pred = str(row["closed_book_answer"])

            em = int(exact_match(pred, str(row["ground_truth_answer"])))
            et = str(row["error_type"])
            if et == "knowledge":  em_k.append(em)
            elif et == "reasoning": em_r.append(em)
            else:                   em_n.append(em)

        rows.append({
            "top_k"        : k,
            "EM_knowledge" : round(np.mean(em_k)*100, 1) if em_k else 0,
            "EM_reasoning" : round(np.mean(em_r)*100, 1) if em_r else 0,
            "EM_all"       : round(np.mean(em_k + em_r + em_n)*100, 1),
        })

    return pd.DataFrame(rows)


# ── 2. Error recovery rate ────────────────────────────────────────────────────

def recovery_rate(df: pd.DataFrame,
                  condition_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each condition, compute the proportion of baseline errors that
    were recovered (i.e., became correct) — separately for each error type.
    """
    baseline = condition_results.get("closed_book")
    if baseline is None:
        log_info("No closed_book condition found — skipping recovery analysis")
        return pd.DataFrame()

    base_errors = baseline[baseline["em"] == 0][["question_id"]].copy()
    base_errors = base_errors.merge(
        df[["question_id", "error_type"]], on="question_id", how="left"
    )

    rows = []
    for cond, res in condition_results.items():
        if cond == "closed_book":
            continue
        merged = base_errors.merge(
            res[["question_id", "em"]].rename(columns={"em": "recovered"}),
            on="question_id", how="left"
        )
        merged["recovered"] = merged["recovered"].fillna(0).astype(int)
        for et in ["knowledge", "reasoning"]:
            sub = merged[merged["error_type"] == et]
            if len(sub) == 0:
                continue
            rate = round(sub["recovered"].mean() * 100, 1)
            rows.append({
                "condition"  : cond,
                "error_type" : et,
                "n_errors"   : len(sub),
                "recovered"  : sub["recovered"].sum(),
                "recovery_%" : rate,
            })

    return pd.DataFrame(rows)


# ── 3. Self-verify decomposition ─────────────────────────────────────────────

def selfverify_decomposition(df: pd.DataFrame,
                              rag_cot_results: pd.DataFrame,
                              sv_results: pd.DataFrame) -> pd.DataFrame:
    """
    Break down the self-verify gain: how many answers changed, and
    how many of those changes were improvements vs regressions.
    """
    merged = rag_cot_results[["question_id", "em", "prediction"]].merge(
        sv_results[["question_id", "em", "prediction",
                    "initial_prediction"]].rename(
            columns={"em": "em_sv", "prediction": "pred_sv"}
        ),
        on="question_id", how="inner"
    )
    merged = merged.merge(
        df[["question_id", "ground_truth_answer", "error_type"]],
        on="question_id", how="left"
    )

    merged["changed"]    = merged["pred_sv"] != merged["initial_prediction"]
    merged["improved"]   = (merged["em"] == 0) & (merged["em_sv"] == 1)
    merged["regressed"]  = (merged["em"] == 1) & (merged["em_sv"] == 0)

    summary = {
        "total"            : len(merged),
        "changed"          : merged["changed"].sum(),
        "improved"         : merged["improved"].sum(),
        "regressed"        : merged["regressed"].sum(),
        "net_gain"         : merged["improved"].sum() - merged["regressed"].sum(),
    }
    return pd.DataFrame([summary])


# ── 4. Domain-level breakdown ────────────────────────────────────────────────

def domain_accuracy(condition_results: dict[str, pd.DataFrame],
                    df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    For each condition, compute EM per domain.
    """
    rows = []
    for cond, res in condition_results.items():
        meta   = df_meta[["question_id", "domain"]].rename(columns={"domain": "_domain"})
        merged = res.merge(meta, on="question_id", how="left")
        for domain, grp in merged.groupby("_domain"):
            rows.append({
                "condition" : cond,
                "domain"    : domain,
                "EM (%)"    : round(grp["em"].mean() * 100, 1),
                "n"         : len(grp),
            })
    return pd.DataFrame(rows)
