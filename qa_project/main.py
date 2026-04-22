"""
main.py
─────────────────────────────────────────────────────────────
Master entry point for the QA Error Analysis project.

Usage
─────
  python main.py                    # run on train set (mock mode)
  python main.py --split test       # run on test set
  python main.py --split both       # run on both sets
  python main.py --mode live        # call OpenAI API (needs key)
  python main.py --no-charts        # skip chart generation
  python main.py --report           # also print a text report

All outputs are saved to ./outputs/
"""

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Import config first ───────────────────────────────────────────────────────
import config
from config import OUTPUT_DIR, CONDITIONS, CONDITION_LABELS


# ─────────────────────────────────────────────────────────────────────────────
#  CLI PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="QA Error Analysis: Knowledge vs Reasoning Failures in LLMs"
    )
    p.add_argument("--split",     default="train",
                   choices=["train", "test", "both"],
                   help="Which dataset split to evaluate")
    p.add_argument("--mode",      default=None,
                   choices=["mock", "live"],
                   help="Override config.py MODE setting")
    p.add_argument("--no-charts", action="store_true",
                   help="Skip chart generation")
    p.add_argument("--report",    action="store_true",
                   help="Print a full text report to stdout")
    p.add_argument("--conditions", nargs="+", default=None,
                   choices=CONDITIONS,
                   help="Run only specific conditions")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  APPLY MODE OVERRIDE BEFORE ANY MODEL IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# Quick check: if --mode is in sys.argv, apply it NOW before any imports
# This ensures models read the correct MODE value when they're imported
if '--mode' in sys.argv:
    try:
        mode_idx = sys.argv.index('--mode')
        if mode_idx + 1 < len(sys.argv):
            requested_mode = sys.argv[mode_idx + 1]
            if requested_mode in ['mock', 'live']:
                config.MODE = requested_mode
                print(f"[OVERRIDE] Mode set to: {config.MODE}")
    except (ValueError, IndexError):
        pass

# ── NOW import everything else (models will see correct MODE) ─────────────────
from data    import load_train, load_test, dataset_summary
from models  import LLMInterface, Retriever
from experiments import run_all_conditions
from evaluation  import (compute_metrics, condition_table,
                          classify_dataset, error_distribution,
                          subtype_breakdown, hop_vs_error)
from analysis    import generate_all
from utils       import log_info, log_ok, log_warn, log_header


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, split_name: str,
                 skip_charts: bool = False,
                 print_report: bool = False,
                 conditions_to_run: list[str] | None = None) -> dict:
    """
    Full evaluation pipeline on a single DataFrame.
    Returns a results summary dict.
    """
    log_header(f"Pipeline: {split_name.upper()} split  ({len(df)} questions)")

    # ── Dataset summary ───────────────────────────────────────────────────────
    log_info("Dataset composition:")
    print(dataset_summary(df).to_string())
    print()

    # ── Instantiate models ────────────────────────────────────────────────────
    llm       = LLMInterface(mock_df=df)
    retriever = Retriever(mock_df=df)

    # ── Run all conditions ────────────────────────────────────────────────────
    condition_results = run_all_conditions(df, llm=llm, retriever=retriever, 
                                          conditions_to_run=conditions_to_run)

    # ── Error classification ──────────────────────────────────────────────────
    log_header("Error Classification")
    df_classified = classify_dataset(df)
    dist  = error_distribution(df_classified)
    log_info("Error distribution by dataset:")
    print(dist.to_string(index=False))
    print()
    log_info("Sub-type breakdown:")
    print(subtype_breakdown(df_classified).to_string(index=False))
    print()
    log_info("Hops vs error type:")
    print(hop_vs_error(df_classified).to_string())
    print()

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    log_header("Accuracy Summary")
    summary_rows = []
    for cond, res_df in condition_results.items():
        for ds in ["Natural Questions", "TriviaQA", "HotpotQA"]:
            sub = res_df[res_df["source_dataset"] == ds]
            if len(sub) == 0:
                continue
            em = round(sub["em"].mean() * 100, 1)
            f1 = round(sub["f1"].mean() * 100, 1)
            summary_rows.append({
                "condition": CONDITION_LABELS.get(cond, cond),
                "dataset":   ds,
                "EM (%)":    em,
                "F1 (%)":    f1,
                "n":         len(sub),
            })

    summary_df = pd.DataFrame(summary_rows)
    pivot = summary_df.pivot_table(
        index="condition", columns="dataset", values="EM (%)"
    )
    pivot["Average"] = pivot.mean(axis=1).round(1)
    print(pivot.to_string())
    print()

    # ── Save CSV outputs ──────────────────────────────────────────────────────
    out_prefix = OUTPUT_DIR / split_name
    out_prefix.mkdir(parents=True, exist_ok=True)

    df_classified.to_csv(out_prefix / "classified_dataset.csv", index=False)
    summary_df.to_csv(out_prefix / "condition_accuracy.csv", index=False)
    pivot.to_csv(out_prefix / "pivot_table.csv")

    for cond, res_df in condition_results.items():
        res_df.to_csv(out_prefix / f"results_{cond}.csv", index=False)

    log_ok(f"Results saved to {out_prefix}/")

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_paths = {}
    if not skip_charts:
        log_header("Generating Charts")
        chart_paths = generate_all(df_classified, condition_results)
        for name, path in chart_paths.items():
            log_ok(f"  Chart saved: {path.name}")

    # ── Optional text report ──────────────────────────────────────────────────
    if print_report:
        _print_text_report(df_classified, condition_results, pivot, split_name)

    return {
        "split"            : split_name,
        "n_questions"      : len(df),
        "condition_results": condition_results,
        "summary_pivot"    : pivot,
        "error_dist"       : dist,
        "chart_paths"      : chart_paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _print_text_report(df: pd.DataFrame,
                       results: dict,
                       pivot: pd.DataFrame,
                       split_name: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep  = "=" * 70
    sep2 = "-" * 70

    print(f"\n{sep}")
    print(f"  QA ERROR ANALYSIS — FULL REPORT  |  {split_name.upper()}  |  {ts}")
    print(sep)

    # 1. Overview
    print("\n1. DATASET OVERVIEW")
    print(sep2)
    total  = len(df)
    k_err  = (df["error_type"] == "knowledge").sum()
    r_err  = (df["error_type"] == "reasoning").sum()
    correct= (df["error_type"] == "none").sum()
    ambig  = total - k_err - r_err - correct
    print(f"   Total questions   : {total}")
    print(f"   Correct (no error): {correct} ({correct/total*100:.1f}%)")
    print(f"   Knowledge errors  : {k_err}  ({k_err/total*100:.1f}%)")
    print(f"   Reasoning errors  : {r_err}  ({r_err/total*100:.1f}%)")
    print(f"   Ambiguous         : {ambig}  ({ambig/total*100:.1f}%)")

    # 2. Per-dataset
    print("\n2. ACCURACY BY CONDITION AND DATASET")
    print(sep2)
    print(pivot.to_string())

    # 3. Key findings
    print("\n3. KEY FINDINGS")
    print(sep2)
    cb_avg = results["closed_book"]["em"].mean() * 100
    rag_avg= results["rag_only"]["em"].mean() * 100
    cot_avg= results["cot_only"]["em"].mean() * 100
    best_avg=results["rag_cot_selfverify"]["em"].mean() * 100

    print(f"   Baseline (closed-book) EM : {cb_avg:.1f}%")
    print(f"   RAG Only EM              : {rag_avg:.1f}%  (Δ = +{rag_avg-cb_avg:.1f}pp)")
    print(f"   CoT Only EM              : {cot_avg:.1f}%  (Δ = +{cot_avg-cb_avg:.1f}pp)")
    print(f"   Best condition EM        : {best_avg:.1f}%  (Δ = +{best_avg-cb_avg:.1f}pp)")

    # 4. Error type vs condition
    print("\n4. ERROR-TYPE BREAKDOWN PER CONDITION")
    print(sep2)
    for cond, res in results.items():
        meta   = df[["question_id","error_type"]].rename(columns={"error_type":"_et"})
        merged = res.merge(meta, on="question_id", how="left")
        label  = CONDITION_LABELS.get(cond, cond)
        for et in ["knowledge", "reasoning", "none"]:
            sub = merged[merged["_et"] == et]
            if len(sub) == 0: continue
            em = sub["em"].mean() * 100
            print(f"   {label:<38} | {et:<12} | EM={em:.1f}%  n={len(sub)}")
        print()

    print(sep)
    print("  END OF REPORT")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Mode was already overridden at module level above
    # Just log it if it was set
    if args.mode:
        log_info(f"Running in {config.MODE} mode")

    splits_to_run = []
    if args.split in ("train", "both"):
        splits_to_run.append(("train", load_train))
    if args.split in ("test", "both"):
        splits_to_run.append(("test", load_test))

    all_summaries = {}
    for split_name, loader in splits_to_run:
        df = loader()
        summary = run_pipeline(
            df,
            split_name   = split_name,
            skip_charts  = args.no_charts,
            print_report = args.report,
            conditions_to_run = args.conditions  
        )
        all_summaries[split_name] = summary

    log_header("Run Complete")
    for split_name, summary in all_summaries.items():
        avg_em = summary["summary_pivot"]["Average"].mean()
        log_ok(f"  [{split_name}]  Mean EM across conditions = {avg_em:.1f}%")

    log_ok(f"All outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()