"""
run_ablation.py
─────────────────────────────────────────────────────────────
Runs the ablation study experiments and saves all ablation charts.

Usage
─────
  python run_ablation.py              # run on train set
  python run_ablation.py --split test
"""

import argparse
import sys
from data  import load_train, load_test
from models import LLMInterface, Retriever
from experiments import run_all_conditions
from experiments.ablation import (
    topk_ablation, recovery_rate, selfverify_decomposition, domain_accuracy
)
from analysis.ablation_plots import (
    plot_topk_curve, plot_recovery_rate, plot_domain_heatmap
)
from config import OUTPUT_DIR
from utils  import log_header, log_info, log_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "test"])
    args = p.parse_args()

    df = load_train() if args.split == "train" else load_test()

    log_header("Ablation Study")

    # Run all conditions first
    llm       = LLMInterface(mock_df=df)
    retriever = Retriever(mock_df=df)
    results   = run_all_conditions(df, llm=llm, retriever=retriever)

    # ── 1. Top-K ablation ─────────────────────────────────────────────────────
    log_info("Running top-k retrieval ablation …")
    topk_df = topk_ablation(df)
    print("\nTop-K Ablation Results:")
    print(topk_df.to_string(index=False))
    path = plot_topk_curve(topk_df)
    log_ok(f"Saved: {path.name}")

    # ── 2. Recovery rate ──────────────────────────────────────────────────────
    log_info("Computing error recovery rates …")
    rec_df = recovery_rate(df, results)
    if not rec_df.empty:
        print("\nRecovery Rates:")
        print(rec_df.to_string(index=False))
        path = plot_recovery_rate(rec_df)
        if path:
            log_ok(f"Saved: {path.name}")

    # ── 3. Self-verify decomposition ──────────────────────────────────────────
    if "rag_cot" in results and "rag_cot_selfverify" in results:
        log_info("Decomposing self-verify gains …")
        sv_decomp = selfverify_decomposition(
            df,
            results["rag_cot"],
            results["rag_cot_selfverify"]
        )
        print("\nSelf-Verify Decomposition:")
        print(sv_decomp.to_string(index=False))

    # ── 4. Domain accuracy ─────────────────────────────────────────────────────
    log_info("Computing domain-level accuracy …")
    dom_df = domain_accuracy(results, df)
    print("\nDomain Accuracy (first 20 rows):")
    print(dom_df.head(20).to_string(index=False))
    path = plot_domain_heatmap(dom_df)
    if path:
        log_ok(f"Saved: {path.name}")

    # Save CSVs
    out = OUTPUT_DIR / args.split
    out.mkdir(parents=True, exist_ok=True)
    topk_df.to_csv(out / "ablation_topk.csv", index=False)
    if not rec_df.empty:
        rec_df.to_csv(out / "ablation_recovery.csv", index=False)
    dom_df.to_csv(out / "ablation_domain.csv", index=False)

    log_ok(f"Ablation results saved to {out}/")


if __name__ == "__main__":
    main()
