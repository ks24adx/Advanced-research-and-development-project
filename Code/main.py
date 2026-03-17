"""
main.py
-------
Master orchestrator for the Knowledge vs Reasoning research project.

Usage:
  python main.py --mock                   # full pipeline, no API calls needed
  python main.py --model gpt-4o-mini      # use OpenAI (requires OPENAI_API_KEY)
  python main.py --model claude-3-haiku   # use Anthropic (requires ANTHROPIC_API_KEY)
  python main.py --domain law             # run only law domain
  python main.py --domain medicine        # run only medicine domain
  python main.py --n 20 --mock            # quick test with 20 items per domain
  python main.py --no-plots               # skip matplotlib generation

Pipeline:
  1. Build / load domain datasets
  2. Build FAISS vector stores
  3. Run experiments (all items × all conditions)
  4. Annotate errors (heuristic + LLM judge)
  5. Compute evaluation metrics
  6. Run statistical tests
  7. Answer research questions
  8. Generate plots and save all results
"""

import argparse
import json
import logging
import os
import sys
import time

import pandas as pd

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    DOMAINS, CONDITIONS, N_SAMPLES_PER_DOMAIN, RESULTS_DIR, DATA_DIR,
    DEFAULT_MODEL
)
from dataset_builder import build_dataset, dataset_summary
from rag_module       import get_vector_store
from llm_runner       import run_all_conditions
from error_attribution import annotate_all
from evaluation       import (
    evaluate_all, StatisticalAnalyser,
    answer_research_questions, save_report,
    score_partial_credit, score_logical_consistency, score_hallucination,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Knowledge vs Reasoning: LLM QA error attribution pipeline"
    )
    p.add_argument("--model",   default=DEFAULT_MODEL,
                   help="LLM model identifier (e.g. gpt-4o-mini, claude-3-haiku-20240307)")
    p.add_argument("--n",       type=int, default=N_SAMPLES_PER_DOMAIN,
                   help="Number of QA items per domain")
    p.add_argument("--domain",  choices=DOMAINS + ["all"], default="all",
                   help="Which domain(s) to run")
    p.add_argument("--mock",    action="store_true",
                   help="Skip LLM API calls — use mock responses for testing")
    p.add_argument("--offline", action="store_true",
                   help="Do not attempt to download datasets from HuggingFace")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib figure generation")
    p.add_argument("--results-dir", default=RESULTS_DIR,
                   help="Directory to save results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    domains = DOMAINS if args.domain == "all" else [args.domain]
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("Knowledge vs Reasoning — Research Pipeline")
    logger.info("  model   : %s%s", args.model, " (MOCK)" if args.mock else "")
    logger.info("  domains : %s", domains)
    logger.info("  n/domain: %d", args.n)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Build datasets
    # ------------------------------------------------------------------
    logger.info("[1/7] Building datasets …")
    all_items  = []
    items_by_domain = {}
    for domain in domains:
        items = build_dataset(domain, n=args.n, csv_dir=DATA_DIR)
        items_by_domain[domain] = items
        all_items.extend(items)
        summary = dataset_summary(items)
        logger.info("  %s: %d items  |  has_context: %d  |  has_chain: %d",
                    domain, len(items),
                    summary["has_context"].sum(),
                    summary["has_reasoning_chain"].sum())

    qa_by_id = {it.item_id: it for it in all_items}

    # ------------------------------------------------------------------
    # Step 2 — Build vector stores
    # ------------------------------------------------------------------
    logger.info("[2/7] Building vector stores …")
    stores = {}
    for domain in domains:
        try:
            store = get_vector_store(domain, items=items_by_domain[domain])
            stores[domain] = store
            logger.info("  %s: index built with %d chunks.", domain, store.ntotal)
        except Exception as exc:
            logger.warning("  Could not build store for %s: %s", domain, exc)
            stores[domain] = None

    # ------------------------------------------------------------------
    # Step 3 — Run experiments
    # ------------------------------------------------------------------
    logger.info("[3/7] Running experiments (%d conditions × %d items) …",
                len(CONDITIONS), len(all_items))
    all_results = []
    for domain in domains:
        store = stores.get(domain)
        if store is not None and store.ntotal == 0:
            store = None
        results = run_all_conditions(
            items_by_domain[domain],
            model=args.model,
            vector_store=store,
            mock=args.mock,
            progress=True,
        )
        all_results.extend(results)

    # Persist raw results
    raw_path = os.path.join(args.results_dir, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump([vars(r) for r in all_results], f, indent=2)
    logger.info("  Raw results saved to %s", raw_path)

    # ------------------------------------------------------------------
    # Step 4 — Error annotation
    # ------------------------------------------------------------------
    logger.info("[4/7] Annotating errors …")
    heuristic_annots, judge_annots = annotate_all(
        all_results,
        qa_items_by_id=qa_by_id,
        use_llm_judge=not args.mock,
        judge_model=args.model,
        mock=args.mock,
    )
    label_counts = {}
    for a in heuristic_annots:
        label_counts[a.error_label] = label_counts.get(a.error_label, 0) + 1
    logger.info("  Heuristic label distribution: %s", label_counts)

    # ------------------------------------------------------------------
    # Step 5 — Evaluation metrics
    # ------------------------------------------------------------------
    logger.info("[5/7] Computing evaluation metrics …")
    eval_df = evaluate_all(
        all_results, heuristic_annots, judge_annots, qa_by_id
    )
    logger.info("\n%s", eval_df.to_string(index=False))

    # Per-item metric records (for distribution plots)
    metrics_records = []
    h_map = {a.item_id + "|" + a.condition: a for a in heuristic_annots}
    for res in all_results:
        qa = qa_by_id.get(res.item_id)
        exp_steps = qa.reasoning_chain if qa else []
        metrics_records.append({
            "item_id":            res.item_id,
            "domain":             res.domain,
            "condition":          res.condition,
            "partial_credit":     score_partial_credit(res),
            "logical_consistency": score_logical_consistency(res, exp_steps),
            "hallucination":      score_hallucination(res),
        })

    # ------------------------------------------------------------------
    # Step 6 — Statistical tests
    # ------------------------------------------------------------------
    logger.info("[6/7] Running statistical tests …")
    analyser   = StatisticalAnalyser(eval_df)
    stat_reports = analyser.full_report(all_results)

    for name, df in stat_reports.items():
        if not df.empty:
            logger.info("  [%s]\n%s\n", name, df.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 7 — Research question answers + save
    # ------------------------------------------------------------------
    logger.info("[7/7] Answering research questions & saving …")
    rq_text = answer_research_questions(eval_df, stat_reports)
    print("\n" + rq_text)

    rq_path = os.path.join(args.results_dir, "research_questions.txt")
    with open(rq_path, "w") as f:
        f.write(rq_text)

    save_report(eval_df, stat_reports, args.results_dir)

    # Per-item metrics
    pd.DataFrame(metrics_records).to_csv(
        os.path.join(args.results_dir, "per_item_metrics.csv"), index=False
    )

    # Annotations
    pd.DataFrame([vars(a) for a in heuristic_annots]).to_csv(
        os.path.join(args.results_dir, "annotations_heuristic.csv"), index=False
    )
    pd.DataFrame([vars(a) for a in judge_annots]).to_csv(
        os.path.join(args.results_dir, "annotations_judge.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Optional: plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        try:
            from visualisation import generate_all_plots
            generate_all_plots(eval_df, metrics_records, args.results_dir)
        except ImportError as exc:
            logger.warning("Skipping plots (missing library): %s", exc)

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1f s. Results: %s", elapsed, args.results_dir)


if __name__ == "__main__":
    main()
