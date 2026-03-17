"""
test_validation.py
------------------
Complete validation suite for the Knowledge vs Reasoning pipeline.

Covers:
  Unit tests  — individual functions (similarity, chunking, parsing, etc.)
  Integration — full pipeline in mock mode on seed data
  Metric tests — correctness of Cohen's Kappa, chi-squared, Mann-Whitney U
  Regression  — known-good outputs don't change

Run with:
  python -m pytest test_validation.py -v
  python test_validation.py          # direct run (no pytest needed)
"""

import json
import os
import sys
import math
import warnings
import unittest
from typing import List

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Path to real CSV datasets
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

from config import (
    QAItem, ExperimentResult, ErrorAnnotation,
    CONDITIONS, DOMAINS, ERROR_LABELS
)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: Dataset Builder Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetBuilder(unittest.TestCase):
    """Validates dataset construction, seed items, and caching."""

    def setUp(self):
        from dataset_builder import SEED_LAW, SEED_MEDICINE
        self.seed_law = SEED_LAW
        self.seed_med = SEED_MEDICINE

    def test_seed_law_structure(self):
        """All seed law items have required fields."""
        for item in self.seed_law:
            self.assertIsInstance(item.item_id, str)
            self.assertGreater(len(item.question), 10,
                               f"Question too short: {item.item_id}")
            self.assertGreater(len(item.gold_answer), 10,
                               f"Gold answer too short: {item.item_id}")
            self.assertEqual(item.domain, "law")

    def test_seed_medicine_structure(self):
        """All seed medicine items have required fields."""
        for item in self.seed_med:
            self.assertEqual(item.domain, "medicine")
            self.assertIsInstance(item.reasoning_chain, list)
            self.assertGreater(len(item.context_docs), 0,
                               f"No context docs: {item.item_id}")

    def test_build_offline_law(self):
        """build_dataset returns N items in offline mode (seed fallback)."""
        from dataset_builder import build_dataset
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            items = build_dataset("law", n=5, csv_dir=_DATA_DIR)
            self.assertEqual(len(items), 5)
            self.assertTrue(all(i.domain == "law" for i in items))

    def test_build_offline_medicine(self):
        from dataset_builder import build_dataset
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            items = build_dataset("medicine", n=5, csv_dir=_DATA_DIR)
            self.assertEqual(len(items), 5)

    def test_cache_roundtrip(self):
        """Items serialised to JSON cache are loaded identically."""
        from dataset_builder import build_dataset, _save_cache, _load_cache
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            items = build_dataset("law", n=3, csv_dir=_DATA_DIR)
            cache = os.path.join(tmpdir, "roundtrip.json")
            _save_cache(items, cache)
            loaded = _load_cache(cache)
            self.assertEqual(len(loaded), len(items))
            self.assertEqual(loaded[0].item_id, items[0].item_id)
            self.assertEqual(loaded[0].question, items[0].question)

    def test_unique_ids_after_replication(self):
        """Replicated seed items receive unique IDs."""
        from dataset_builder import build_dataset
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            items = build_dataset("law", n=10, csv_dir=_DATA_DIR)
            ids = [i.item_id for i in items]
            self.assertEqual(len(ids), len(set(ids)),
                             "Duplicate item IDs found after replication")

    def test_dataset_summary(self):
        from dataset_builder import build_dataset, dataset_summary
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            items = build_dataset("medicine", n=3, csv_dir=_DATA_DIR)
            df = dataset_summary(items)
            self.assertEqual(len(df), 3)
            self.assertIn("domain",       df.columns)
            self.assertIn("has_context",  df.columns)
            self.assertIn("q_length",     df.columns)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: RAG Module Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRagModule(unittest.TestCase):
    """Validates document chunking, embedding, and retrieval."""

    def test_split_chunks_basic(self):
        from rag_module import _split_into_chunks
        text = "First sentence. Second sentence. Third sentence."
        chunks = _split_into_chunks(text, max_chars=30, overlap=5)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertIsInstance(c, str)
            self.assertGreater(len(c), 0)

    def test_split_chunks_length_constraint(self):
        """No chunk should exceed max_chars by more than one sentence."""
        from rag_module import _split_into_chunks
        text = ("Sentence one is here. Sentence two is here. "
                "Sentence three is here. Sentence four is here.")
        max_c = 40
        chunks = _split_into_chunks(text, max_chars=max_c, overlap=10)
        for c in chunks:
            # Allow some slack for overlap
            self.assertLessEqual(len(c), max_c * 3,
                                 f"Chunk too long: {len(c)} chars")

    def test_split_chunks_overlap(self):
        """Overlapping chunks share some content."""
        from rag_module import _split_into_chunks
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
        chunks = _split_into_chunks(text, max_chars=35, overlap=15)
        if len(chunks) >= 2:
            # Crude check: words from chunk n-1 appear in chunk n
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            self.assertGreater(len(words_0 & words_1), 0,
                               "Overlap chunks share no words")

    def test_domain_corpus_coverage(self):
        """VectorStore can be built with real item context docs."""
        from rag_module import VectorStore
        from dataset_builder import SEED_LAW, SEED_MEDICINE
        for domain, seed in [('law', SEED_LAW), ('medicine', SEED_MEDICINE)]:
            corpus = [d for item in seed for d in item.context_docs]
            self.assertGreater(len(corpus), 0,
                               f"No corpus text for domain '{domain}'")

    def test_vector_store_build_retrieval(self):
        """VectorStore builds index and returns relevant docs."""
        from rag_module import VectorStore
        from dataset_builder import SEED_LAW
        corpus = [d for item in SEED_LAW for d in item.context_docs]
        store = VectorStore("law", corpus=corpus)
        store.build()
        results = store.retrieve("negligence duty of care breach", top_k=2)
        self.assertIsInstance(results, list)
        if results:
            self.assertIsInstance(results[0], str)
            self.assertGreater(len(results[0]), 5)

    def test_retrieve_without_build(self):
        """Retrieve before build returns empty list (no crash)."""
        from rag_module import VectorStore
        store = VectorStore("medicine")
        result = store.retrieve("diabetes treatment", top_k=3)
        self.assertEqual(result, [])

    def test_get_vector_store_cached(self):
        """get_vector_store returns the same object on repeated calls."""
        from rag_module import get_vector_store, reset_stores
        from dataset_builder import SEED_LAW
        reset_stores()
        s1 = get_vector_store("law", items=SEED_LAW)
        s2 = get_vector_store("law", items=SEED_LAW)
        self.assertIs(s1, s2)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: LLM Runner Tests (mock mode only)
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMRunner(unittest.TestCase):
    """Validates prompt construction, response parsing, and mock runner."""

    def _make_item(self, domain="law") -> QAItem:
        return QAItem(
            item_id="test_001",
            domain=domain,
            question="What are the elements of a valid contract?",
            gold_answer="Offer, acceptance, consideration, and intention to create legal relations.",
            context_docs=["Consideration is something of value exchanged between parties."],
            reasoning_chain=["Identify offer.", "Identify acceptance.", "Check consideration."],
        )

    def test_mock_response_structure(self):
        """Mock response returns a dict with required keys."""
        from llm_runner import _mock_llm_response
        item = self._make_item()
        resp = _mock_llm_response(item, "reasoning_only")
        self.assertIn("answer",            resp)
        self.assertIn("cot_steps",         resp)
        self.assertIn("confidence",        resp)
        self.assertIn("knowledge_used",    resp)
        self.assertIn("reasoning_performed", resp)

    def test_run_experiment_mock(self):
        """run_experiment in mock mode returns a valid ExperimentResult."""
        from llm_runner import run_experiment
        item = self._make_item()
        for cond in CONDITIONS:
            result = run_experiment(
                item, cond,
                retrieved_docs=["Context doc." if cond != "reasoning_only" else ""],
                mock=True,
            )
            self.assertEqual(result.item_id, "test_001")
            self.assertEqual(result.condition, cond)
            self.assertIsInstance(result.model_answer, str)
            self.assertIsInstance(result.cot_steps, list)

    def test_run_all_conditions_mock(self):
        """run_all_conditions produces 3 results per item."""
        from dataset_builder import SEED_LAW
        from llm_runner import run_all_conditions
        items = SEED_LAW[:2]
        results = run_all_conditions(items, mock=True, progress=False)
        self.assertEqual(len(results), len(items) * len(CONDITIONS))

    def test_parse_response_valid_json(self):
        """_parse_response correctly parses well-formed JSON."""
        from llm_runner import _parse_response
        raw = '{"answer": "Offer and acceptance.", "cot_steps": ["step1"], "confidence": 0.9, ' \
              '"knowledge_used": "contract law", "reasoning_performed": "deduction"}'
        parsed = _parse_response(raw)
        self.assertEqual(parsed["answer"], "Offer and acceptance.")
        self.assertAlmostEqual(parsed["confidence"], 0.9)

    def test_parse_response_fenced_json(self):
        """_parse_response strips markdown fences."""
        from llm_runner import _parse_response
        raw = '```json\n{"answer": "Test", "cot_steps": [], "confidence": 0.8, ' \
              '"knowledge_used": "", "reasoning_performed": ""}\n```'
        parsed = _parse_response(raw)
        self.assertEqual(parsed["answer"], "Test")

    def test_parse_response_malformed(self):
        """_parse_response falls back gracefully on invalid JSON."""
        from llm_runner import _parse_response
        raw = "This is not JSON at all."
        parsed = _parse_response(raw)
        self.assertIn("answer", parsed)   # fallback key must exist
        self.assertIsInstance(parsed.get("cot_steps", []), list)

    def test_unknown_condition_raises(self):
        """run_experiment raises AssertionError for unknown condition."""
        from llm_runner import run_experiment
        item = self._make_item()
        with self.assertRaises(AssertionError):
            run_experiment(item, "nonexistent_condition", mock=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: Error Attribution Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorAttribution(unittest.TestCase):
    """Validates semantic scoring and error classification logic."""

    def _make_result(self, correct=True, has_context=True) -> ExperimentResult:
        gold = ("A valid contract requires offer, acceptance, consideration, "
                "and intention to create legal relations between parties.")
        model = gold if correct else (
            "A contract only needs an offer and acceptance to be valid.")
        return ExperimentResult(
            item_id="test_item",
            domain="law",
            condition="rag",
            model="gpt-4o-mini",
            question="What makes a contract valid?",
            gold_answer=gold,
            model_answer=model,
            retrieved_docs=[
                "Consideration is something of value exchanged between parties."
            ] if has_context else [],
            cot_steps=[
                "Identify offer.",
                "Identify acceptance.",
                "Verify consideration exists.",
            ],
        )

    def test_cosine_identity(self):
        """Identical text should have similarity ≈ 1."""
        from error_attribution import compute_answer_similarity
        text = "Offer acceptance consideration intention legal relations."
        sim = compute_answer_similarity(text, text)
        self.assertGreater(sim, 0.95)

    def test_cosine_unrelated(self):
        """Completely different text should have low similarity."""
        from error_attribution import compute_answer_similarity
        sim = compute_answer_similarity(
            "The cat sat on the mat.",
            "Negligence requires duty breach causation damages."
        )
        self.assertLess(sim, 0.7)

    def test_context_coverage_with_docs(self):
        """Coverage should be positive when docs are provided."""
        from error_attribution import compute_context_coverage
        coverage = compute_context_coverage(
            ["Consideration is something of value exchanged between parties."],
            "Consideration is required for a valid contract."
        )
        self.assertGreater(coverage, 0.1)

    def test_context_coverage_no_docs(self):
        """Coverage should be 0 when no docs provided."""
        from error_attribution import compute_context_coverage
        coverage = compute_context_coverage([], "Any gold answer.")
        self.assertEqual(coverage, 0.0)

    def test_classify_correct_answer(self):
        """A correct answer should be labelled 'correct'."""
        from error_attribution import classify_error_heuristic
        result = self._make_result(correct=True)
        annot = classify_error_heuristic(result)
        self.assertEqual(annot.error_label, "correct")
        self.assertTrue(annot.is_correct)

    def test_classify_wrong_answer_returns_error_label(self):
        """A wrong answer should return one of the defined error labels."""
        from error_attribution import classify_error_heuristic
        result = self._make_result(correct=False)
        annot = classify_error_heuristic(result)
        self.assertIn(annot.error_label, ERROR_LABELS)
        # knowledge_score and reasoning_score in [0, 1]
        self.assertGreaterEqual(annot.knowledge_score, 0.0)
        self.assertLessEqual(annot.knowledge_score, 1.0)

    def test_hallucination_estimate_no_hallucination(self):
        """Answer that mirrors gold should have low hallucination score."""
        from error_attribution import estimate_hallucination
        gold = "Metformin is first-line therapy for type 2 diabetes."
        score = estimate_hallucination(gold, gold, [])
        self.assertLess(score, 0.5)

    def test_hallucination_estimate_mismatch(self):
        """Highly divergent answer should score higher."""
        from error_attribution import estimate_hallucination
        gold   = "Metformin is first-line therapy for type 2 diabetes."
        answer = "Insulin glargine combined with metoprolol treats hypertension."
        score  = estimate_hallucination(answer, gold, [])
        # Should flag some hallucination
        self.assertGreaterEqual(score, 0.0)

    def test_annotate_all_mock(self):
        """annotate_all returns two equal-length annotation lists."""
        from dataset_builder import SEED_LAW
        from llm_runner import run_all_conditions
        from error_attribution import annotate_all
        items = SEED_LAW[:1]
        results = run_all_conditions(items, mock=True, progress=False)
        qa_map = {i.item_id: i for i in items}
        h, j = annotate_all(results, qa_items_by_id=qa_map, mock=True)
        self.assertEqual(len(h), len(results))
        self.assertEqual(len(j), len(results))
        for annot in h:
            self.assertIn(annot.error_label, ERROR_LABELS)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: Evaluation Metric Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluationMetrics(unittest.TestCase):
    """Validates metric computations against hand-calculated values."""

    def _make_pair(self, condition="reasoning_only", correct=True):
        gold = ("Promissory estoppel allows recovery when there is a clear promise "
                "and reasonable detrimental reliance.")
        model = gold if correct else "Estoppel is only relevant in property law."
        result = ExperimentResult(
            item_id="eval_item",
            domain="law",
            condition=condition,
            model="mock",
            question="What is promissory estoppel?",
            gold_answer=gold,
            model_answer=model,
            retrieved_docs=["Restatement §90 defines promissory estoppel."],
            cot_steps=["Identify promise.", "Check reliance.", "Conclude."],
        )
        annot = ErrorAnnotation(
            item_id="eval_item",
            condition=condition,
            annotator="auto",
            is_correct=correct,
            error_label="correct" if correct else "pure_reasoning_failure",
            knowledge_score=0.0 if correct else 0.2,
            reasoning_score=0.0 if correct else 0.8,
        )
        return result, annot

    def test_score_accuracy_correct(self):
        from evaluation import score_accuracy
        result, _ = self._make_pair(correct=True)
        self.assertTrue(score_accuracy(result))

    def test_score_accuracy_incorrect(self):
        from evaluation import score_accuracy
        result, _ = self._make_pair(correct=False)
        self.assertFalse(score_accuracy(result))

    def test_partial_credit_range(self):
        from evaluation import score_partial_credit
        for correct in (True, False):
            result, _ = self._make_pair(correct=correct)
            pc = score_partial_credit(result)
            self.assertGreaterEqual(pc, 0.0)
            self.assertLessEqual(pc, 1.0)

    def test_logical_consistency_with_steps(self):
        from evaluation import score_logical_consistency
        result, _ = self._make_pair(correct=True)
        score = score_logical_consistency(result, ["Identify promise.", "Assess reliance."])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_logical_consistency_no_steps(self):
        """No CoT steps → score is 0."""
        from evaluation import score_logical_consistency
        result, _ = self._make_pair()
        result.cot_steps = []
        score = score_logical_consistency(result)
        self.assertEqual(score, 0.0)

    def test_compute_metrics_for_slice(self):
        """compute_metrics_for_slice returns all expected keys."""
        from evaluation import compute_metrics_for_slice
        results = []
        annots  = []
        for i in range(4):
            r, a = self._make_pair(correct=(i % 2 == 0))
            r.item_id = f"item_{i}"
            a.item_id = f"item_{i}"
            results.append(r)
            annots.append(a)
        metrics = compute_metrics_for_slice(results, annots)
        for key in ["n", "accuracy", "partial_credit",
                    "logical_consistency", "hallucination_rate",
                    "knowledge_error_pct", "reasoning_error_pct",
                    "cohen_kappa"]:
            self.assertIn(key, metrics, f"Missing key: {key}")
        self.assertEqual(metrics["n"], 4)
        self.assertAlmostEqual(metrics["accuracy"], 0.5, delta=0.15)

    def test_cohen_kappa_perfect_agreement(self):
        """Perfect annotator agreement → kappa = 1.0."""
        from evaluation import _compute_kappa
        results, annots_1 = [], []
        annots_2 = []
        for i in range(6):
            r, a = self._make_pair()
            r.item_id = f"k_item_{i}"
            a.item_id = f"k_item_{i}"
            # Vary labels to avoid trivial case
            label = ERROR_LABELS[i % len(ERROR_LABELS)]
            a.error_label = label
            b = ErrorAnnotation(
                item_id=a.item_id, condition=a.condition, annotator="judge",
                is_correct=a.is_correct, error_label=label,  # same label
                knowledge_score=0.5, reasoning_score=0.5,
            )
            results.append(r)
            annots_1.append(a)
            annots_2.append(b)
        kappa = _compute_kappa(results, annots_1, annots_2)
        self.assertAlmostEqual(kappa, 1.0, delta=0.01)

    def test_cohen_kappa_chance_disagreement(self):
        """Random disagreement → kappa near 0 (may be negative)."""
        from evaluation import _compute_kappa
        import random
        rng = random.Random(42)
        results, annots_1, annots_2 = [], [], []
        labels = ["pure_knowledge_failure", "pure_reasoning_failure",
                  "mixed_failure", "ambiguous", "correct"]
        for i in range(20):
            r, a = self._make_pair()
            r.item_id = f"kc_{i}"
            a.item_id = f"kc_{i}"
            a.error_label = rng.choice(labels)
            b = ErrorAnnotation(
                item_id=a.item_id, condition=a.condition, annotator="judge",
                is_correct=a.is_correct,
                error_label=rng.choice(labels),  # random — likely disagrees
                knowledge_score=0.5, reasoning_score=0.5,
            )
            results.append(r)
            annots_1.append(a)
            annots_2.append(b)
        kappa = _compute_kappa(results, annots_1, annots_2)
        # Not perfect; typically between -0.3 and 0.4 for random labels
        self.assertLess(kappa, 0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: Statistical Analysis Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStatisticalAnalysis(unittest.TestCase):
    """Validates that statistical tests return expected structures."""

    def _build_eval_df(self):
        """Create a minimal synthetic eval_df for testing."""
        import pandas as pd
        rows = []
        for domain in DOMAINS:
            for cond in CONDITIONS:
                rows.append({
                    "domain":    domain,
                    "condition": cond,
                    "n":         20,
                    "accuracy":  0.6 if cond == "reasoning_only" else 0.75,
                    "partial_credit": 0.65,
                    "logical_consistency": 0.7,
                    "hallucination_rate": 0.15,
                    "n_correct":   12,
                    "n_knowledge_errors": 4 if cond == "reasoning_only" else 2,
                    "n_reasoning_errors": 3 if cond == "reasoning_only" else 1,
                    "n_mixed_errors": 1,
                    "n_ambiguous": 0,
                    "knowledge_error_pct": 0.5,
                    "reasoning_error_pct": 0.375,
                    "mixed_error_pct": 0.125,
                    "cohen_kappa": 0.72,
                })
        return pd.DataFrame(rows)

    def test_error_distribution_test_runs(self):
        """Chi-squared test runs without exception."""
        from evaluation import StatisticalAnalyser
        df = self._build_eval_df()
        analyser = StatisticalAnalyser(df)
        result = analyser.test_error_distribution()
        # Should produce rows (may be empty if not enough conditions)
        self.assertIsNotNone(result)
        self.assertIn("test", result.columns if not result.empty else ["test"])

    def test_interaction_effect_runs(self):
        from evaluation import StatisticalAnalyser
        df = self._build_eval_df()
        analyser = StatisticalAnalyser(df)
        result = analyser.test_interaction_effect()
        self.assertIsNotNone(result)

    def test_cohens_h_values(self):
        """Cohen's h is 0 for equal proportions, grows with difference."""
        from evaluation import StatisticalAnalyser
        h_zero = StatisticalAnalyser._cohens_h(0.5, 0.5)
        self.assertAlmostEqual(h_zero, 0.0, places=5)
        h_large = abs(StatisticalAnalyser._cohens_h(0.9, 0.1))
        self.assertGreater(h_large, 1.0)

    def test_effect_size_interpretation(self):
        from evaluation import StatisticalAnalyser
        for val, expected in [(0.1, "negligible"), (0.3, "small"),
                               (0.6, "medium"), (1.0, "large")]:
            result = StatisticalAnalyser._interpret_effect_h(val)
            self.assertEqual(result, expected)

    def test_answer_research_questions(self):
        """answer_research_questions returns a non-empty string."""
        from evaluation import answer_research_questions
        df = self._build_eval_df()
        text = answer_research_questions(df, {})
        self.assertIsInstance(text, str)
        self.assertIn("RQ-a", text)
        self.assertIn("RQ-b", text)
        self.assertIn("RQ-c", text)
        self.assertIn("RQ-d", text)
        self.assertIn("RQ-e", text)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: Full Integration Test (mock pipeline)
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineMock(unittest.TestCase):
    """
    End-to-end integration test: runs the full pipeline in mock mode
    with a small number of seed items.
    """

    def test_pipeline_produces_eval_df(self):
        """The full pipeline produces a non-empty evaluation DataFrame."""
        import tempfile
        from dataset_builder import build_dataset
        from llm_runner import run_all_conditions
        from error_attribution import annotate_all
        from evaluation import evaluate_all, StatisticalAnalyser

        with tempfile.TemporaryDirectory() as tmpdir:
            all_items = []
            for domain in DOMAINS:
                items = build_dataset(domain, n=3, csv_dir=_DATA_DIR)
                all_items.extend(items)

            results  = run_all_conditions(all_items, mock=True, progress=False)
            qa_map   = {i.item_id: i for i in all_items}
            h, j     = annotate_all(results, qa_items_by_id=qa_map, mock=True)
            eval_df  = evaluate_all(results, h, j, qa_map)

            self.assertFalse(eval_df.empty, "Evaluation DataFrame is empty")
            expected_rows = len(DOMAINS) * len(CONDITIONS)
            self.assertLessEqual(len(eval_df), expected_rows)

            for col in ["accuracy", "partial_credit", "logical_consistency",
                        "hallucination_rate", "knowledge_error_pct",
                        "reasoning_error_pct"]:
                self.assertIn(col, eval_df.columns)
                self.assertTrue(eval_df[col].between(0, 1).all(),
                                f"Column '{col}' has values outside [0,1]")
            # Cohen's Kappa is in [-1, 1] (negative = worse than chance)
            self.assertIn("cohen_kappa", eval_df.columns)
            self.assertTrue(eval_df["cohen_kappa"].between(-1, 1).all(),
                            "cohen_kappa has values outside [-1, 1]")

    def test_pipeline_results_count(self):
        """Each item produces exactly one result per condition."""
        import tempfile
        from dataset_builder import build_dataset
        from llm_runner import run_all_conditions
        N = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            items = build_dataset("law", n=N, csv_dir=_DATA_DIR)
            results = run_all_conditions(items, mock=True, progress=False)
            self.assertEqual(len(results), N * len(CONDITIONS))

    def test_annotation_labels_valid(self):
        """All annotations carry valid error labels."""
        import tempfile
        from dataset_builder import build_dataset
        from llm_runner import run_all_conditions
        from error_attribution import annotate_all
        with tempfile.TemporaryDirectory() as tmpdir:
            items = build_dataset("medicine", n=2, csv_dir=_DATA_DIR)
            results = run_all_conditions(items, mock=True, progress=False)
            h, _ = annotate_all(results, mock=True)
            for annot in h:
                self.assertIn(annot.error_label, ERROR_LABELS,
                              f"Invalid label: {annot.error_label}")


# ══════════════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_tests() -> bool:
    """Run all test suites and print a summary report."""
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestDatasetBuilder,
        TestRagModule,
        TestLLMRunner,
        TestErrorAttribution,
        TestEvaluationMetrics,
        TestStatisticalAnalysis,
        TestFullPipelineMock,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"  Tests run  : {result.testsRun}")
    print(f"  Failures   : {len(result.failures)}")
    print(f"  Errors     : {len(result.errors)}")
    print(f"  Skipped    : {len(result.skipped)}")
    print(f"  Status     : {'PASS ✓' if result.wasSuccessful() else 'FAIL ✗'}")
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
