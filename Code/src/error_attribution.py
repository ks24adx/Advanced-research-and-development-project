"""
error_attribution.py
--------------------
The core diagnostic module of this research project.

For each LLM response, this module determines:
  • Is the answer correct?
  • If not, is the failure due to:
      - pure_knowledge_failure  (model lacked required facts)
      - pure_reasoning_failure  (facts were available but inference failed)
      - mixed_failure           (both gaps contributed)
      - ambiguous               (cannot reliably classify)

Attribution strategy (two-stage):
  Stage 1 — Automatic heuristic scoring using:
      (a) Semantic similarity (answer vs gold) for correctness
      (b) RAG context coverage score (was needed info retrievable?)
      (c) CoT step analysis (did the model articulate the right reasoning?)
      (d) Presence of hallucinated entities
  
  Stage 2 — LLM-as-judge (optional, requires API key):
      A separate LLM call prompts the model to classify the error,
      acting as the automated "annotator 2" for Cohen's Kappa calculation.
"""

import re
import logging
from typing import List, Optional, Tuple

import numpy as np

from config import (
    ERROR_LABELS, ExperimentResult, ErrorAnnotation
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — Heuristic attribution
# ---------------------------------------------------------------------------

# Minimum cosine similarity to count an answer as correct
CORRECTNESS_THRESHOLD = 0.65
# Below this similarity, knowledge was clearly missing
KNOWLEDGE_COVERAGE_LOW = 0.35
# Above this, reasoning was probably at fault (model had info but got it wrong)
KNOWLEDGE_COVERAGE_HIGH = 0.65


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _embed(texts: List[str]) -> np.ndarray:
    """Embed texts using the same model as the RAG module."""
    try:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        model = SentenceTransformer(EMBEDDING_MODEL)
        return model.encode(texts, convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False)
    except ImportError:
        # Fallback: TF-IDF-style bag-of-words cosine similarity
        return _tfidf_embed(texts)


def _tfidf_embed(texts: List[str]) -> np.ndarray:
    """Simple token-overlap fallback when sentence-transformers unavailable."""
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(stop_words="english")
    try:
        matrix = vec.fit_transform(texts).toarray().astype(np.float32)
    except ValueError:
        # All texts empty
        return np.zeros((len(texts), 1), dtype=np.float32)
    return matrix


def compute_answer_similarity(model_answer: str, gold_answer: str) -> float:
    """
    Semantic similarity between model answer and gold answer.
    Returns a float in [0, 1].
    """
    if not model_answer or not gold_answer:
        return 0.0
    embs = _embed([model_answer, gold_answer])
    return _cosine_similarity(embs[0], embs[1])


def compute_context_coverage(
    retrieved_docs: List[str],
    gold_answer: str,
) -> float:
    """
    How well do the retrieved documents cover the knowledge needed
    to answer correctly?

    Returns 0 if no documents were retrieved (knowledge not accessible),
    1 if the retrieved context closely matches the gold answer content.
    """
    if not retrieved_docs:
        return 0.0
    # Use max similarity between any retrieved doc and the gold answer
    texts = retrieved_docs + [gold_answer]
    embs  = _embed(texts)
    gold_emb = embs[-1]
    doc_embs = embs[:-1]
    sims = [_cosine_similarity(d, gold_emb) for d in doc_embs]
    return float(max(sims))


def compute_cot_quality(cot_steps: List[str], expected_steps: List[str]) -> float:
    """
    Measure how well the model's chain-of-thought aligns with the
    expected reasoning steps (from the dataset).
    Returns 0 if no expected steps defined.
    """
    if not cot_steps or not expected_steps:
        return 0.5   # neutral — cannot determine
    # Pairwise best-match similarity
    if len(cot_steps) == 0 or len(expected_steps) == 0:
        return 0.5
    texts = cot_steps + expected_steps
    embs  = _embed(texts)
    n_pred = len(cot_steps)
    pred_embs = embs[:n_pred]
    exp_embs  = embs[n_pred:]
    # For each expected step, find best matching predicted step
    scores = []
    for exp in exp_embs:
        sims = [_cosine_similarity(p, exp) for p in pred_embs]
        scores.append(max(sims))
    return float(np.mean(scores))


HALLUCINATION_PATTERNS = [
    r"\b(invented|fabricated|made.up|non-existent)\b",
    r"\b(case number \d{4,})\b",      # suspiciously specific case numbers
    r"\b[A-Z][a-z]+ v\.? [A-Z][a-z]+, \d{4} \(\d{4}\)\b",  # invented citations
]


def estimate_hallucination(model_answer: str, gold_answer: str,
                            retrieved_docs: List[str]) -> float:
    """
    Heuristic hallucination score [0, 1].
    High when the answer contains assertions unsupported by gold or context.
    """
    if not model_answer:
        return 0.0

    score = 0.0

    # Hard hallucination patterns
    for pat in HALLUCINATION_PATTERNS:
        if re.search(pat, model_answer, re.IGNORECASE):
            score += 0.3

    # Soft signal: very low overlap with gold + retrieved docs
    reference = gold_answer + " " + " ".join(retrieved_docs)
    if reference.strip():
        embs = _embed([model_answer, reference])
        sim  = _cosine_similarity(embs[0], embs[1])
        if sim < 0.2:
            score += 0.4
        elif sim < 0.35:
            score += 0.15

    return min(score, 1.0)


def classify_error_heuristic(
    result: ExperimentResult,
    expected_steps: Optional[List[str]] = None,
    annotator: str = "auto_heuristic",
) -> ErrorAnnotation:
    """
    Stage 1: heuristic error attribution.

    Decision logic:
    ┌─────────────────────────────────────────────────────────┐
    │  Is answer correct?  →  YES  →  label = "correct"       │
    │         ↓ NO                                            │
    │  Context coverage low AND condition has retrieval?       │
    │    YES  →  knowledge_score high → pure_knowledge_failure │
    │    NO   →  Did CoT match expected steps?                 │
    │              LOW  →  pure_reasoning_failure              │
    │              HIGH →  mixed_failure (had info, bad cot)   │
    │              MEDIUM or unavailable → ambiguous           │
    └─────────────────────────────────────────────────────────┘
    """
    # --- Correctness ---
    ans_sim = compute_answer_similarity(result.model_answer, result.gold_answer)
    is_correct = ans_sim >= CORRECTNESS_THRESHOLD

    if is_correct:
        return ErrorAnnotation(
            item_id=result.item_id,
            condition=result.condition,
            annotator=annotator,
            is_correct=True,
            error_label="correct",
            knowledge_score=0.0,
            reasoning_score=0.0,
            notes=f"answer_sim={ans_sim:.3f}",
        )

    # --- Knowledge coverage ---
    ctx_cov = compute_context_coverage(
        result.retrieved_docs, result.gold_answer
    )
    had_retrieval = result.condition in ("rag", "hybrid")

    # --- CoT quality ---
    cot_q = compute_cot_quality(result.cot_steps, expected_steps or [])

    # --- Hallucination ---
    hall = estimate_hallucination(
        result.model_answer, result.gold_answer, result.retrieved_docs
    )

    # --- Attribution logic ---
    # Knowledge score: how much of the failure is due to missing knowledge
    if had_retrieval:
        # If RAG was active, poor context coverage → knowledge gap
        knowledge_score = 1.0 - ctx_cov
    else:
        # In reasoning-only, knowledge is entirely from model training.
        # We cannot distinguish training knowledge gaps, so we flag as 0.5
        # unless hallucination is high (strong signal of knowledge gap).
        knowledge_score = 0.5 + 0.5 * hall

    # Reasoning score: derived inversely from CoT quality
    reasoning_score = 1.0 - cot_q if cot_q != 0.5 else 0.5

    # Normalise so they sum ≤ 1 (some failures are partially both)
    total = knowledge_score + reasoning_score + 1e-9
    k_norm = knowledge_score / total
    r_norm = reasoning_score / total

    DELTA = 0.25
    if k_norm - r_norm > DELTA:
        label = "pure_knowledge_failure"
    elif r_norm - k_norm > DELTA:
        label = "pure_reasoning_failure"
    elif k_norm > 0.35 and r_norm > 0.35:
        label = "mixed_failure"
    else:
        label = "ambiguous"

    return ErrorAnnotation(
        item_id=result.item_id,
        condition=result.condition,
        annotator=annotator,
        is_correct=False,
        error_label=label,
        knowledge_score=round(knowledge_score, 3),
        reasoning_score=round(reasoning_score, 3),
        notes=(
            f"ans_sim={ans_sim:.3f} | ctx_cov={ctx_cov:.3f} | "
            f"cot_q={cot_q:.3f} | hall={hall:.3f}"
        ),
    )


# ---------------------------------------------------------------------------
# Stage 2 — LLM-as-judge (automated second annotator)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert AI evaluator conducting error analysis.

Given:
- QUESTION: {question}
- GOLD ANSWER: {gold}
- MODEL ANSWER: {model_answer}
- RETRIEVED CONTEXT (if any): {context}
- MODEL'S REASONING STEPS: {cot_steps}

Task: Classify the error in the model's answer.

Error categories:
1. pure_knowledge_failure — model lacked key factual knowledge
2. pure_reasoning_failure — model had enough information but reasoned incorrectly
3. mixed_failure — both knowledge and reasoning deficiencies
4. ambiguous — cannot clearly distinguish the source
5. correct — the answer is acceptable

Rate the knowledge gap [0-1] and reasoning gap [0-1].

Respond ONLY with valid JSON:
{{
  "is_correct": <bool>,
  "error_label": "<category>",
  "knowledge_score": <0-1>,
  "reasoning_score": <0-1>,
  "explanation": "<one sentence>"
}}"""


def classify_error_llm_judge(
    result: ExperimentResult,
    model: str = "gpt-4o-mini",
    mock: bool = False,
) -> ErrorAnnotation:
    """
    Stage 2: LLM-as-judge annotation.
    Uses a separate LLM call to act as the second human annotator.
    """
    if mock:
        # Return a slightly randomised version of the heuristic for Kappa calc
        import random
        rng = random.Random(hash(result.item_id + "judge") & 0xFFFFFFFF)
        heuristic = classify_error_heuristic(result, annotator="auto_heuristic")
        # Occasionally disagree (realistic inter-annotator agreement ~75%)
        if rng.random() < 0.25:
            options = [l for l in ERROR_LABELS if l != heuristic.error_label]
            label = rng.choice(options)
        else:
            label = heuristic.error_label
        return ErrorAnnotation(
            item_id=result.item_id,
            condition=result.condition,
            annotator="llm_judge_mock",
            is_correct=heuristic.is_correct,
            error_label=label,
            knowledge_score=heuristic.knowledge_score + rng.uniform(-0.1, 0.1),
            reasoning_score=heuristic.reasoning_score + rng.uniform(-0.1, 0.1),
            notes="mock judge annotation",
        )

    from llm_runner import _call_llm, _parse_response
    prompt = JUDGE_PROMPT.format(
        question=result.question,
        gold=result.gold_answer[:300],
        model_answer=result.model_answer[:300],
        context=" | ".join(result.retrieved_docs)[:400] or "(none)",
        cot_steps=str(result.cot_steps[:3]),
    )
    try:
        raw    = _call_llm("You are an expert error analyser.", prompt, model=model)
        parsed = _parse_response(raw)
        return ErrorAnnotation(
            item_id=result.item_id,
            condition=result.condition,
            annotator="llm_judge",
            is_correct=bool(parsed.get("is_correct", False)),
            error_label=parsed.get("error_label", "ambiguous"),
            knowledge_score=float(parsed.get("knowledge_score", 0.5)),
            reasoning_score=float(parsed.get("reasoning_score", 0.5)),
            notes=parsed.get("explanation", ""),
        )
    except Exception as exc:
        logger.error("LLM judge failed for item=%s: %s", result.item_id, exc)
        return classify_error_heuristic(result, annotator="llm_judge_fallback")


# ---------------------------------------------------------------------------
# Batch annotation
# ---------------------------------------------------------------------------

def annotate_all(
    results: List[ExperimentResult],
    qa_items_by_id: Optional[dict] = None,
    use_llm_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
    mock: bool = False,
) -> Tuple[List[ErrorAnnotation], List[ErrorAnnotation]]:
    """
    Annotate all results with both the heuristic and the LLM-judge annotator.

    Returns
    -------
    (heuristic_annotations, judge_annotations) — parallel lists, one per result
    """
    heuristic_annots = []
    judge_annots     = []

    for res in results:
        expected = (qa_items_by_id or {}).get(res.item_id, None)
        exp_steps = expected.reasoning_chain if expected else []

        h = classify_error_heuristic(res, expected_steps=exp_steps)
        heuristic_annots.append(h)

        if use_llm_judge or mock:
            j = classify_error_llm_judge(res, model=judge_model, mock=mock)
        else:
            j = h   # use heuristic as both annotators if judge disabled
        judge_annots.append(j)

    return heuristic_annots, judge_annots
