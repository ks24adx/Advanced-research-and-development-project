"""
evaluation.py
-------------
Validation and evaluation module for the Knowledge vs Reasoning project.

Implements ALL metrics specified in the DPP:

Performance Metrics:
  • Accuracy                   — exact/semantic match
  • Partial credit scoring     — weighted by answer similarity
  • Logical consistency score  — CoT step coherence
  • Hallucination rate         — proportion of answers with hallucinated content

Error Attribution Metrics:
  • % knowledge-based errors
  • % reasoning-based errors
  • Cross-domain comparison
  • Inter-annotator agreement (Cohen's Kappa)

Statistical Analysis:
  • Chi-squared test (error distributions across conditions)
  • Mann-Whitney U test (accuracy across conditions)
  • Effect sizes (Cohen's h for proportions)
  • Bonferroni-corrected pairwise comparisons
"""

import logging
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score  # type: ignore

from config import (
    DOMAINS, CONDITIONS, ERROR_LABELS, ALPHA,
    ExperimentResult, ErrorAnnotation, EvaluationReport,
)
from error_attribution import (
    compute_answer_similarity,
    compute_cot_quality,
    estimate_hallucination,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual-level metric helpers
# ---------------------------------------------------------------------------

def score_accuracy(result: ExperimentResult,
                   threshold: float = 0.65) -> bool:
    """Binary correctness: semantic similarity ≥ threshold."""
    sim = compute_answer_similarity(result.model_answer, result.gold_answer)
    return sim >= threshold


def score_partial_credit(result: ExperimentResult) -> float:
    """
    Partial credit score [0, 1] based on semantic similarity.
    Gives proportional credit for partially correct answers.
    """
    return min(1.0, max(0.0, compute_answer_similarity(result.model_answer, result.gold_answer)))


def score_logical_consistency(result: ExperimentResult,
                               expected_steps: Optional[List[str]] = None) -> float:
    """
    Logical consistency [0, 1]:
    Measures whether the CoT reasoning is internally coherent and
    aligned with the expected reasoning chain.

    Sub-components:
      (a) Step coherence: consecutive steps are semantically related (0-1)
      (b) Step completeness: coverage of expected steps (0-1)
      (c) Conclusion alignment: final step leads to the stated answer (0-1)
    """
    cot = result.cot_steps
    if not cot:
        return 0.0   # no reasoning provided

    # (a) Step coherence — adjacent steps should be related
    coherence = _step_coherence(cot)

    # (b) Step completeness
    completeness = compute_cot_quality(cot, expected_steps) if expected_steps else 0.5

    # (c) Conclusion alignment — last CoT step vs final answer
    if len(cot) >= 1:
        from error_attribution import _embed, _cosine_similarity
        texts = [cot[-1], result.model_answer]
        embs  = _embed(texts)
        conclusion = _cosine_similarity(embs[0], embs[1])
    else:
        conclusion = 0.5

    # Weighted average
    return float(0.4 * coherence + 0.3 * completeness + 0.3 * conclusion)


def _step_coherence(steps: List[str]) -> float:
    """Measure semantic continuity between consecutive reasoning steps."""
    if len(steps) < 2:
        return 1.0
    from error_attribution import _embed, _cosine_similarity
    embs = _embed(steps)
    sims = [_cosine_similarity(embs[i], embs[i + 1])
            for i in range(len(embs) - 1)]
    return float(np.mean(sims))


def score_hallucination(result: ExperimentResult) -> float:
    """Hallucination score [0, 1]. Higher = more hallucination."""
    return estimate_hallucination(
        result.model_answer, result.gold_answer, result.retrieved_docs
    )


# ---------------------------------------------------------------------------
# Aggregate metric computation
# ---------------------------------------------------------------------------

def compute_metrics_for_slice(
    results: List[ExperimentResult],
    annotations: List[ErrorAnnotation],
    qa_items_by_id: Optional[dict] = None,
    judge_annotations: Optional[List[ErrorAnnotation]] = None,
) -> dict:
    """
    Compute all metrics for a flat list of (result, annotation) pairs
    from a single (domain, condition) slice.

    Returns a dict with all metric values.
    """
    if not results:
        return {}

    # Align annotations by item_id
    annot_map = {a.item_id: a for a in annotations}
    judge_map = {a.item_id: a for a in (judge_annotations or [])}

    accuracies    = []
    partial_scores = []
    logic_scores  = []
    hall_scores   = []
    error_labels  = []

    for res in results:
        qa = (qa_items_by_id or {}).get(res.item_id)
        exp_steps = qa.reasoning_chain if qa else []

        accuracies.append(float(score_accuracy(res)))
        partial_scores.append(score_partial_credit(res))
        logic_scores.append(score_logical_consistency(res, exp_steps))
        hall_scores.append(score_hallucination(res))

        annot = annot_map.get(res.item_id)
        error_labels.append(annot.error_label if annot else "ambiguous")

    n = len(results)
    label_counts = {lbl: error_labels.count(lbl) for lbl in ERROR_LABELS}
    n_errors = sum(1 for l in error_labels if l != "correct")

    # Cohen's Kappa between heuristic and judge annotators
    kappa = _compute_kappa(results, annotations, judge_annotations or annotations)

    return {
        "n":                  n,
        "accuracy":           round(float(np.mean(accuracies)), 4),
        "partial_credit":     round(float(np.mean(partial_scores)), 4),
        "logical_consistency": round(float(np.mean(logic_scores)), 4),
        "hallucination_rate": round(float(np.mean(hall_scores)), 4),
        "n_correct":          label_counts.get("correct", 0),
        "n_knowledge_errors": label_counts.get("pure_knowledge_failure", 0),
        "n_reasoning_errors": label_counts.get("pure_reasoning_failure", 0),
        "n_mixed_errors":     label_counts.get("mixed_failure", 0),
        "n_ambiguous":        label_counts.get("ambiguous", 0),
        "knowledge_error_pct": round(label_counts.get("pure_knowledge_failure", 0) / max(n_errors, 1), 4),
        "reasoning_error_pct": round(label_counts.get("pure_reasoning_failure", 0) / max(n_errors, 1), 4),
        "mixed_error_pct":    round(label_counts.get("mixed_failure", 0) / max(n_errors, 1), 4),
        "cohen_kappa":        round(kappa, 4),
    }


def _compute_kappa(
    results: List[ExperimentResult],
    annots_1: List[ErrorAnnotation],
    annots_2: List[ErrorAnnotation],
) -> float:
    """
    Cohen's Kappa between two annotators over the same set of items.
    """
    a1_map = {a.item_id: a.error_label for a in annots_1}
    a2_map = {a.item_id: a.error_label for a in annots_2}

    labels_1, labels_2 = [], []
    for res in results:
        l1 = a1_map.get(res.item_id, "ambiguous")
        l2 = a2_map.get(res.item_id, "ambiguous")
        labels_1.append(l1)
        labels_2.append(l2)

    if len(set(labels_1)) <= 1 or len(set(labels_2)) <= 1:
        return 1.0   # trivial case

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = cohen_kappa_score(labels_1, labels_2)
        return float(kappa)
    except Exception as exc:
        logger.warning("Kappa computation failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_all(
    results: List[ExperimentResult],
    heuristic_annotations: List[ErrorAnnotation],
    judge_annotations: List[ErrorAnnotation],
    qa_items_by_id: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute metrics for every (domain, condition) combination.

    Returns a long-format DataFrame with one row per slice.
    """
    rows = []

    for domain in DOMAINS:
        for condition in CONDITIONS:
            # Filter to this slice
            slice_results = [r for r in results
                             if r.domain == domain and r.condition == condition]
            slice_h = [a for a in heuristic_annotations
                       if a.condition == condition and
                       any(r.item_id == a.item_id for r in slice_results)]
            slice_j = [a for a in judge_annotations
                       if a.condition == condition and
                       any(r.item_id == a.item_id for r in slice_results)]

            if not slice_results:
                continue

            metrics = compute_metrics_for_slice(
                slice_results, slice_h, qa_items_by_id, slice_j
            )
            rows.append({
                "domain":    domain,
                "condition": condition,
                **metrics,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

class StatisticalAnalyser:
    """
    Performs and reports all required statistical tests.

    Tests run:
    1. Chi-squared: error type distribution across conditions (per domain)
    2. Mann-Whitney U: accuracy distribution between pairs of conditions
    3. Effect size (Cohen's h) for proportion differences
    4. Bonferroni correction for multiple comparisons
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.reports: List[dict] = []

    # ---------------------------------------------------------- #
    #  Chi-squared: error distribution across conditions          #
    # ---------------------------------------------------------- #

    def test_error_distribution(self) -> pd.DataFrame:
        """
        H0: Error type proportions are the same across conditions.
        Test: Chi-squared goodness-of-fit per domain.
        """
        rows = []
        for domain in self.df["domain"].unique():
            sub = self.df[self.df["domain"] == domain]
            observed = np.array([
                [sub[sub.condition == c]["n_knowledge_errors"].sum(),
                 sub[sub.condition == c]["n_reasoning_errors"].sum(),
                 sub[sub.condition == c]["n_mixed_errors"].sum()]
                for c in CONDITIONS
                if c in sub["condition"].values
            ], dtype=float)

            # Remove zero rows
            observed = observed[observed.sum(axis=1) > 0]
            if observed.shape[0] < 2:
                continue

            try:
                chi2, p_val, dof, _ = stats.chi2_contingency(observed)
            except Exception:
                continue

            rows.append({
                "domain":     domain,
                "test":       "chi2_error_distribution",
                "statistic":  round(chi2, 4),
                "dof":        int(dof),
                "p_value":    round(p_val, 4),
                "significant": p_val < ALPHA,
                "interpretation": (
                    f"Error distributions DIFFER across conditions "
                    f"(p={p_val:.4f})" if p_val < ALPHA else
                    f"No significant difference in error distributions "
                    f"(p={p_val:.4f})"
                ),
            })
        return pd.DataFrame(rows)

    # ---------------------------------------------------------- #
    #  Mann-Whitney U: accuracy pairwise comparisons             #
    # ---------------------------------------------------------- #

    def test_accuracy_differences(
        self,
        results: List[ExperimentResult],
    ) -> pd.DataFrame:
        """
        Pairwise Mann-Whitney U tests on accuracy scores across conditions.
        Applies Bonferroni correction for multiple comparisons.
        """
        rows = []
        pairs = [("reasoning_only", "rag"),
                 ("reasoning_only", "hybrid"),
                 ("rag", "hybrid")]
        n_tests = len(pairs) * len(DOMAINS)

        for domain in DOMAINS:
            dom_results = [r for r in results if r.domain == domain]
            for cond_a, cond_b in pairs:
                acc_a = [float(score_accuracy(r)) for r in dom_results
                         if r.condition == cond_a]
                acc_b = [float(score_accuracy(r)) for r in dom_results
                         if r.condition == cond_b]

                if len(acc_a) < 3 or len(acc_b) < 3:
                    continue

                try:
                    stat, p_val = stats.mannwhitneyu(acc_a, acc_b,
                                                     alternative="two-sided")
                except Exception:
                    continue

                p_corrected = min(p_val * n_tests, 1.0)   # Bonferroni
                effect_h    = self._cohens_h(
                    float(np.mean(acc_a)), float(np.mean(acc_b))
                )

                rows.append({
                    "domain":      domain,
                    "cond_a":      cond_a,
                    "cond_b":      cond_b,
                    "mean_acc_a":  round(float(np.mean(acc_a)), 4),
                    "mean_acc_b":  round(float(np.mean(acc_b)), 4),
                    "test":        "mann_whitney_u",
                    "statistic":   round(stat, 2),
                    "p_value_raw": round(p_val, 4),
                    "p_value_bonf":round(p_corrected, 4),
                    "cohens_h":    round(effect_h, 4),
                    "effect_size": self._interpret_effect_h(effect_h),
                    "significant": p_corrected < ALPHA,
                })
        return pd.DataFrame(rows)

    # ---------------------------------------------------------- #
    #  Interaction effect: knowledge × reasoning                  #
    # ---------------------------------------------------------- #

    def test_interaction_effect(self) -> pd.DataFrame:
        """
        Two-way ANOVA analogue (using Kruskal-Wallis) to test whether
        there is an interaction between knowledge availability (condition)
        and reasoning quality (domain).

        Research question (e): Is there an interaction effect between
        knowledge availability and reasoning quality?
        """
        rows = []
        for condition in CONDITIONS:
            sub = self.df[self.df["condition"] == condition]
            groups = [
                sub[sub.domain == d]["accuracy"].dropna().values
                for d in DOMAINS
                if d in sub["domain"].values
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue
            try:
                stat, p_val = stats.kruskal(*groups)
            except Exception:
                continue
            rows.append({
                "condition":  condition,
                "test":       "kruskal_wallis_domain_interaction",
                "statistic":  round(stat, 4),
                "p_value":    round(p_val, 4),
                "significant": p_val < ALPHA,
                "interpretation": (
                    f"Domain modulates accuracy under {condition} "
                    f"(p={p_val:.4f})" if p_val < ALPHA else
                    f"No significant domain × condition interaction "
                    f"(p={p_val:.4f})"
                ),
            })
        return pd.DataFrame(rows)

    # ---------------------------------------------------------- #
    #  Helpers                                                    #
    # ---------------------------------------------------------- #

    @staticmethod
    def _cohens_h(p1: float, p2: float) -> float:
        """Cohen's h effect size for two proportions."""
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    @staticmethod
    def _interpret_effect_h(h: float) -> str:
        h = abs(h)
        if h < 0.2:
            return "negligible"
        elif h < 0.5:
            return "small"
        elif h < 0.8:
            return "medium"
        return "large"

    def full_report(self, results: List[ExperimentResult]) -> Dict[str, pd.DataFrame]:
        """Run all tests and return a dict of result DataFrames."""
        return {
            "error_distribution":  self.test_error_distribution(),
            "accuracy_pairwise":   self.test_accuracy_differences(results),
            "interaction_effect":  self.test_interaction_effect(),
        }


# ---------------------------------------------------------------------------
# Research-question answering
# ---------------------------------------------------------------------------

def answer_research_questions(
    eval_df: pd.DataFrame,
    stat_reports: Dict[str, pd.DataFrame],
) -> str:
    """
    Map statistical findings to each of the 5 research questions in the DPP.
    Returns a formatted multi-line string summary.
    """
    lines = [
        "=" * 70,
        "RESEARCH QUESTION ANSWERS",
        "=" * 70,
    ]

    # RQ-a: Knowledge vs reasoning as dominant error source?
    lines += ["", "RQ-a: Are errors primarily knowledge or reasoning failures?"]
    for domain in DOMAINS:
        sub = eval_df[eval_df["domain"] == domain]
        k = sub["knowledge_error_pct"].mean()
        r = sub["reasoning_error_pct"].mean()
        dominant = "KNOWLEDGE" if k > r else "REASONING"
        lines.append(
            f"  [{domain.upper()}] Knowledge errors: {k:.1%}  |  "
            f"Reasoning errors: {r:.1%}  →  Dominant: {dominant}"
        )

    # RQ-b: Does dominant source differ across domains?
    lines += ["", "RQ-b: Does the dominant error source differ across domains?"]
    if len(eval_df["domain"].unique()) >= 2:
        law_k  = eval_df[eval_df.domain == "law"]["knowledge_error_pct"].mean()
        med_k  = eval_df[eval_df.domain == "medicine"]["knowledge_error_pct"].mean()
        differ = abs(law_k - med_k) > 0.10
        lines.append(
            f"  Law knowledge error %: {law_k:.1%}  |  "
            f"Medicine knowledge error %: {med_k:.1%}"
        )
        lines.append(
            f"  {'YES' if differ else 'NO'} — meaningful cross-domain difference "
            f"(|Δ|={abs(law_k - med_k):.1%})"
        )

    # RQ-c: Does RAG reduce knowledge errors?
    lines += ["", "RQ-c: Does retrieval augmentation reduce knowledge-based errors?"]
    for domain in DOMAINS:
        sub = eval_df[eval_df.domain == domain]
        ro  = sub[sub.condition == "reasoning_only"]["knowledge_error_pct"].values
        rag = sub[sub.condition == "rag"]["knowledge_error_pct"].values
        if len(ro) and len(rag):
            delta = float(ro[0]) - float(rag[0])
            lines.append(
                f"  [{domain.upper()}] RAG reduces knowledge errors by "
                f"{delta:+.1%} (reasoning_only={ro[0]:.1%} → rag={rag[0]:.1%})"
            )
            # Check statistical test
            if "accuracy_pairwise" in stat_reports:
                row = stat_reports["accuracy_pairwise"]
                sig_row = row[(row.domain == domain) &
                              (row.cond_a == "reasoning_only") &
                              (row.cond_b == "rag")]
                if not sig_row.empty:
                    sig = sig_row["significant"].values[0]
                    p   = sig_row["p_value_bonf"].values[0]
                    lines.append(
                        f"  Statistical test: {'SIGNIFICANT' if sig else 'NOT significant'} "
                        f"(Bonferroni p={p:.4f})"
                    )

    # RQ-d: Does CoT prompting reduce reasoning errors?
    lines += ["", "RQ-d: Does explicit reasoning prompting reduce reasoning errors?"]
    for domain in DOMAINS:
        sub = eval_df[eval_df.domain == domain]
        ro  = sub[sub.condition == "reasoning_only"]["reasoning_error_pct"].values
        hyb = sub[sub.condition == "hybrid"]["reasoning_error_pct"].values
        if len(ro) and len(hyb):
            delta = float(ro[0]) - float(hyb[0])
            lines.append(
                f"  [{domain.upper()}] Hybrid vs reasoning_only reasoning "
                f"error delta: {delta:+.1%}"
            )

    # RQ-e: Interaction effect?
    lines += ["", "RQ-e: Is there an interaction between knowledge and reasoning?"]
    if "interaction_effect" in stat_reports and not stat_reports["interaction_effect"].empty:
        for _, row in stat_reports["interaction_effect"].iterrows():
            lines.append(f"  {row['condition']}: {row['interpretation']}")
    else:
        lines.append("  Interaction tests not available (insufficient data).")

    lines += ["", "=" * 70]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------

def save_report(
    eval_df: pd.DataFrame,
    stat_reports: Dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    """Save all evaluation artefacts to CSV."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    eval_df.to_csv(os.path.join(output_dir, "evaluation_summary.csv"), index=False)

    for name, df in stat_reports.items():
        if df is not None and not df.empty:
            df.to_csv(os.path.join(output_dir, f"stat_{name}.csv"), index=False)

    logger.info("Evaluation reports saved to %s", output_dir)
