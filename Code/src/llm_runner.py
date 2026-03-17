"""
llm_runner.py
-------------
Runs LLM experiments under the three conditions defined in the DPP:

  1. reasoning_only  — zero-shot + Chain-of-Thought (CoT) prompt, no retrieval
  2. rag             — retrieved context injected, no explicit CoT
  3. hybrid          — retrieved context + CoT prompt

Supports OpenAI (GPT-4o-mini default) and Anthropic Claude via a
simple adapter pattern. Extend _call_llm() for other providers.

Key design decisions:
  • temperature=0 for reproducibility
  • Structured JSON output enforced via prompt — allows automatic parsing
    of (answer, cot_steps, confidence) in one pass.
  • All API calls wrapped in retry logic with exponential back-off.
"""

import os
import re
import json
import time
import logging
from typing import List, Optional, Tuple

from config import (
    DEFAULT_MODEL, MAX_TOKENS, TEMPERATURE, CONDITIONS,
    QAItem, ExperimentResult
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert researcher conducting a controlled experiment.
Your task is to answer domain-specific questions accurately and then explain
your reasoning step-by-step.

Always respond in valid JSON with this exact schema:
{
  "answer": "<your final answer>",
  "cot_steps": ["<step 1>", "<step 2>", ...],
  "confidence": <float 0-1>,
  "knowledge_used": "<what factual knowledge was needed>",
  "reasoning_performed": "<what logical steps were needed>"
}
Do not include markdown fences. Output only the JSON object."""

RAG_SYSTEM_PROMPT = (
    "You are an expert researcher. The following retrieved documents\n"
    "may contain information relevant to the question. Read them carefully.\n\n"
    "Retrieved Documents:\n{context}\n\n"
    "Always respond in valid JSON:\n"
    '{{"answer":"<answer>","cot_steps":["<step1>"],"confidence":0.9,'
    '"knowledge_used":"<facts>","reasoning_performed":"<logic>",'
    '"context_relevance":0.8}}\n'
    "Do not include markdown fences. Output only the JSON object."
)

HYBRID_SYSTEM_PROMPT = (
    "You are an expert researcher. Retrieved context is provided.\n"
    "Use it together with step-by-step reasoning to answer the question.\n\n"
    "Retrieved Documents:\n{context}\n\n"
    "Instructions:\n"
    "1. Read the retrieved documents carefully.\n"
    "2. Think step-by-step, explicitly listing your reasoning.\n"
    "3. Identify what knowledge comes from the documents vs prior knowledge.\n"
    "4. Provide a confident, well-justified answer.\n\n"
    "Respond in valid JSON:\n"
    '{{"answer":"<answer>","cot_steps":["<step1>"],"confidence":0.9,'
    '"knowledge_used":"<facts>","reasoning_performed":"<logic>",'
    '"context_relevance":0.8}}'
)

REASONING_ONLY_USER_TEMPLATE = """Domain: {domain}
Question: {question}

Think step-by-step before giving your final answer."""

RAG_USER_TEMPLATE = """Domain: {domain}
Question: {question}"""

HYBRID_USER_TEMPLATE = """Domain: {domain}
Question: {question}

Use the retrieved documents and reason step-by-step."""


# ---------------------------------------------------------------------------
# LLM adapters
# ---------------------------------------------------------------------------

def _call_openai(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call the OpenAI chat completions API."""
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error("OpenAI API error: %s", exc)
        raise


def _call_anthropic(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call the Anthropic Messages API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text if resp.content else ""
    except Exception as exc:
        logger.error("Anthropic API error: %s", exc)
        raise


def _call_llm(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    max_retries: int = 3,
) -> str:
    """
    Route to the correct provider based on model name prefix.
    Retries with exponential back-off.
    """
    for attempt in range(max_retries):
        try:
            if model.startswith(("gpt-", "o1-", "o3-")):
                return _call_openai(system, user, model, max_tokens, temperature)
            elif model.startswith(("claude-",)):
                return _call_anthropic(system, user, model, max_tokens, temperature)
            else:
                # Default: try OpenAI-compatible endpoint
                return _call_openai(system, user, model, max_tokens, temperature)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning("LLM call failed (attempt %d): %s. Retry in %ds.",
                           attempt + 1, exc, wait)
            time.sleep(wait)
    return ""   # unreachable


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> dict:
    """
    Parse the structured JSON response from the LLM.
    Falls back gracefully if JSON is malformed.
    """
    # Strip any accidental markdown fences
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract a JSON object with regex
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    # Ultimate fallback: treat entire response as the answer
    logger.warning("Could not parse structured JSON from LLM response. "
                   "Using raw text as answer.")
    return {
        "answer":  raw[:500],
        "cot_steps": [],
        "confidence": 0.5,
        "knowledge_used": "",
        "reasoning_performed": "",
    }


# ---------------------------------------------------------------------------
# Mock LLM for offline / unit-testing
# ---------------------------------------------------------------------------

def _mock_llm_response(item: QAItem, condition: str) -> dict:
    """
    Returns a plausible-looking structured response without an API key.
    Used in --mock mode or unit tests.
    """
    import random
    rng = random.Random(hash(item.item_id + condition) & 0xFFFFFFFF)
    is_correct = rng.random() > 0.35   # ~65% correct on average
    if is_correct:
        answer = item.gold_answer[:200]
    else:
        # Simulate a hallucinated / partially wrong answer
        answer = item.gold_answer[:80] + " [incorrect extension due to hallucination]"
    steps = item.reasoning_chain if item.reasoning_chain else [
        "Identified key concepts in the question.",
        "Retrieved relevant domain knowledge.",
        "Applied logical inference to derive conclusion.",
    ]
    return {
        "answer":              answer,
        "cot_steps":           steps,
        "confidence":          rng.uniform(0.5, 0.95),
        "knowledge_used":      "Domain-specific facts from training data.",
        "reasoning_performed": "Deductive reasoning from premises to conclusion.",
        "context_relevance":   rng.uniform(0.6, 0.99) if condition != "reasoning_only" else 0.0,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    item: QAItem,
    condition: str,
    model: str = DEFAULT_MODEL,
    retrieved_docs: Optional[List[str]] = None,
    mock: bool = False,
) -> ExperimentResult:
    """
    Run a single QAItem under a given experimental condition.

    Parameters
    ----------
    item          : QAItem to evaluate
    condition     : "reasoning_only" | "rag" | "hybrid"
    model         : LLM model identifier
    retrieved_docs: Documents returned by the RAG pipeline (if any)
    mock          : If True, skip API call and return synthetic response

    Returns
    -------
    ExperimentResult with model answer and metadata
    """
    assert condition in CONDITIONS, f"Unknown condition: {condition}"

    context_str = "\n\n---\n\n".join(retrieved_docs or [])

    # Build prompts
    if condition == "reasoning_only":
        system = SYSTEM_PROMPT
        user   = REASONING_ONLY_USER_TEMPLATE.format(
            domain=item.domain, question=item.question
        )
    elif condition == "rag":
        system = RAG_SYSTEM_PROMPT.format(context=context_str or "(no documents retrieved)")
        user   = RAG_USER_TEMPLATE.format(
            domain=item.domain, question=item.question
        )
    else:  # hybrid
        system = HYBRID_SYSTEM_PROMPT.format(context=context_str or "(no documents retrieved)")
        user   = HYBRID_USER_TEMPLATE.format(
            domain=item.domain, question=item.question
        )

    # Call LLM (or mock)
    t0 = time.time()
    if mock:
        parsed = _mock_llm_response(item, condition)
    else:
        raw    = _call_llm(system, user, model=model)
        parsed = _parse_response(raw)
    latency = time.time() - t0

    return ExperimentResult(
        item_id      = item.item_id,
        domain       = item.domain,
        condition    = condition,
        model        = model,
        question     = item.question,
        gold_answer  = item.gold_answer,
        model_answer = parsed.get("answer", ""),
        retrieved_docs = retrieved_docs or [],
        cot_steps    = parsed.get("cot_steps", []),
        latency_s    = latency,
    )


def run_all_conditions(
    items: List[QAItem],
    model: str = DEFAULT_MODEL,
    vector_store=None,      # VectorStore instance (optional)
    top_k: int = 3,
    mock: bool = False,
    progress: bool = True,
) -> List[ExperimentResult]:
    """
    Run every item under all three conditions and collect results.

    Parameters
    ----------
    items        : list of QAItem
    model        : LLM to use
    vector_store : a rag_module.VectorStore (built and ready); if None,
                   RAG and hybrid conditions use no retrieved context.
    top_k        : documents to retrieve per query
    mock         : bypass API calls
    progress     : show tqdm progress bar

    Returns
    -------
    Flat list of ExperimentResult (len = len(items) * 3)
    """
    try:
        from tqdm import tqdm  # type: ignore
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    results: List[ExperimentResult] = []
    iterator = tqdm(items, desc="Running experiments") if (progress and _tqdm_available) else items

    for item in iterator:
        # Retrieve documents for RAG / hybrid conditions
        retrieved: List[str] = []
        if vector_store is not None:
            retrieved = vector_store.retrieve(item.question, top_k=top_k)

        for cond in CONDITIONS:
            docs = retrieved if cond in ("rag", "hybrid") else []
            try:
                result = run_experiment(item, cond, model, docs, mock=mock)
            except Exception as exc:
                logger.error("Failed item=%s cond=%s: %s", item.item_id, cond, exc)
                # Record an empty result so downstream code doesn't lose the item
                result = ExperimentResult(
                    item_id=item.item_id, domain=item.domain, condition=cond,
                    model=model, question=item.question,
                    gold_answer=item.gold_answer, model_answer="ERROR",
                )
            results.append(result)

    logger.info("Collected %d results (%d items × %d conditions).",
                len(results), len(items), len(CONDITIONS))
    return results
