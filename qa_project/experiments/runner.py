"""
experiments/runner.py
─────────────────────────────────────────────────────────────
Orchestrates all five experimental conditions for a given
DataFrame (train or test).

Conditions
──────────
  1. closed_book          – parametric memory only
  2. rag_only             – retrieved context, standard prompt
  3. cot_only             – CoT prompt, no retrieval
  4. rag_cot              – retrieved context + CoT prompt
  5. rag_cot_selfverify   – retrieved context + CoT + self-verify
"""

from __future__ import annotations
import pandas as pd
from tqdm import tqdm

from models import LLMInterface, Retriever
from utils import exact_match, token_f1, log_info, log_header


def run_all_conditions(df: pd.DataFrame,
                       llm: LLMInterface | None = None,
                       retriever: Retriever | None = None) -> dict[str, pd.DataFrame]:
    """
    Run all five conditions on the supplied DataFrame.

    Returns a dict mapping condition name → results DataFrame.
    Each results DataFrame has columns:
        question_id, source_dataset, question, ground_truth,
        prediction, em, f1, error_type, num_hops_required, domain
    """
    if llm is None:
        llm = LLMInterface(mock_df=df)
    if retriever is None:
        retriever = Retriever(mock_df=df)

    results = {}

    log_header("Running Experimental Conditions")

    results["closed_book"]       = _run_closed_book(df, llm)
    results["rag_only"]          = _run_rag_only(df, llm, retriever)
    results["cot_only"]          = _run_cot_only(df, llm)
    results["rag_cot"]           = _run_rag_cot(df, llm, retriever)
    results["rag_cot_selfverify"]= _run_rag_cot_selfverify(df, llm, retriever)

    return results


# ── Individual condition runners ──────────────────────────────────────────────

def _base_record(row: pd.Series, prediction: str) -> dict:
    """Build a result record from a df row and a prediction."""
    gt = str(row["ground_truth_answer"])
    return {
        "question_id"      : row["question_id"],
        "source_dataset"   : row["source_dataset"],
        "question"         : row["question"],
        "ground_truth"     : gt,
        "prediction"       : prediction,
        "em"               : int(exact_match(prediction, gt)),
        "f1"               : round(token_f1(prediction, gt), 3),
        "error_type"       : row.get("error_type", "unknown"),
        "num_hops_required": int(row.get("num_hops_required", 1)),
        "domain"           : row.get("domain", ""),
    }


def _run_closed_book(df: pd.DataFrame, llm: LLMInterface) -> pd.DataFrame:
    log_info("Condition 1/5 → Closed-Book")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Closed-Book"):
        pred = llm.answer_closed_book(row["question_id"], row["question"])
        records.append(_base_record(row, pred))
    result = pd.DataFrame(records)
    _log_accuracy(result, "Closed-Book")
    return result


def _run_rag_only(df: pd.DataFrame, llm: LLMInterface,
                  retriever: Retriever) -> pd.DataFrame:
    log_info("Condition 2/5 → RAG Only")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="RAG-Only"):
        ctx, recall = retriever.retrieve(row["question_id"], row["question"])
        pred = llm.answer_open_book(row["question_id"], row["question"], ctx)
        rec  = _base_record(row, pred)
        rec["recall_at5"] = recall
        records.append(rec)
    result = pd.DataFrame(records)
    _log_accuracy(result, "RAG Only")
    return result


def _run_cot_only(df: pd.DataFrame, llm: LLMInterface) -> pd.DataFrame:
    log_info("Condition 3/5 → CoT Only")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="CoT-Only"):
        pred = llm.answer_closed_book_cot(row["question_id"], row["question"])
        records.append(_base_record(row, pred))
    result = pd.DataFrame(records)
    _log_accuracy(result, "CoT Only")
    return result


def _run_rag_cot(df: pd.DataFrame, llm: LLMInterface,
                 retriever: Retriever) -> pd.DataFrame:
    log_info("Condition 4/5 → RAG + CoT")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="RAG+CoT"):
        ctx, recall = retriever.retrieve(row["question_id"], row["question"])
        pred = llm.answer_open_book_cot(row["question_id"], row["question"], ctx)
        rec  = _base_record(row, pred)
        rec["recall_at5"] = recall
        records.append(rec)
    result = pd.DataFrame(records)
    _log_accuracy(result, "RAG + CoT")
    return result


def _run_rag_cot_selfverify(df: pd.DataFrame, llm: LLMInterface,
                             retriever: Retriever) -> pd.DataFrame:
    log_info("Condition 5/5 → RAG + CoT + Self-Verify")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Self-Verify"):
        ctx, recall   = retriever.retrieve(row["question_id"], row["question"])
        init_pred     = llm.answer_open_book_cot(row["question_id"],
                                                  row["question"], ctx)
        final_pred    = llm.self_verify(row["question_id"], row["question"],
                                         ctx, init_pred)
        rec = _base_record(row, final_pred)
        rec.update({"recall_at5": recall, "initial_prediction": init_pred})
        records.append(rec)
    result = pd.DataFrame(records)
    _log_accuracy(result, "RAG + CoT + Self-Verify")
    return result


# ── Logging helper ────────────────────────────────────────────────────────────

def _log_accuracy(result: pd.DataFrame, label: str) -> None:
    em = result["em"].mean() * 100
    f1 = result["f1"].mean() * 100
    log_info(f"  [{label}]  EM = {em:.1f}%   F1 = {f1:.1f}%")
