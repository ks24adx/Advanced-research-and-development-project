"""
models/llm_interface.py
─────────────────────────────────────────────────────────────
Unified interface for calling the LLM.

  - MOCK mode  : returns answers pre-loaded from the CSV dataset,
                 simulating what the model would say. This is the
                 default and requires no API key.
  - LIVE mode  : sends requests to the OpenAI ChatCompletion API.

Both modes expose the same public API so downstream code is
unaffected by the mode switch.
"""

from __future__ import annotations
import time
import pandas as pd
import config  
from config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, MAX_TOKENS
from utils import log_info, log_warn

# ── Prompt templates ─────────────────────────────────────────────────────────

STANDARD_PROMPT = """Answer the following question as concisely as possible.
Give only the answer — no explanation.

Question: {question}
Answer:"""

COT_PROMPT = """Answer the following question step by step.
First write your reasoning, then give a final answer on a new line starting with "Final Answer:".

Question: {question}
Answer:"""

RAG_STANDARD_PROMPT = """Use the following passages to answer the question.
Give only the answer — no explanation.

Passages:
{context}

Question: {question}
Answer:"""

RAG_COT_PROMPT = """Use the following passages to answer the question step by step.
First write your reasoning using the passages, then give a final answer on a new line starting with "Final Answer:".

Passages:
{context}

Question: {question}
Answer:"""

SELF_VERIFY_PROMPT = """You previously answered a question. Check whether your answer is consistent
with the provided passages. If not, correct it.

Passages:
{context}

Question: {question}
Your previous answer: {previous_answer}

Verified answer (give ONLY the answer, no explanation):"""


class LLMInterface:
    """Thin wrapper around the LLM, supporting mock and live modes."""

    def __init__(self, mock_df: pd.DataFrame | None = None):
        self.mode    = config.MODE   
        self.mock_df = mock_df    # pre-loaded dataframe for mock mode
        self._client = None

        if self.mode == "live":  
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=OPENAI_API_KEY)
                log_info(f"LLM initialised in LIVE mode → {LLM_MODEL}")
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        else:
            log_info("LLM initialised in MOCK mode (no API calls)")

    # ── Public methods ────────────────────────────────────────────────────────

    def answer_closed_book(self, question_id: str, question: str) -> str:
        if self.mode == "mock":
            return self._mock_get(question_id, "closed_book_answer")
        prompt = STANDARD_PROMPT.format(question=question)
        return self._call(prompt)

    def answer_closed_book_cot(self, question_id: str, question: str) -> str:
        if self.mode == "mock":
            # Simulate: CoT does not change the answer, but may improve it
            row = self._mock_row(question_id)
            if row is not None and int(row.get("cot_closed_correct", 0)):
                return str(row["ground_truth_answer"])
            return str(row["closed_book_answer"]) if row is not None else ""
        prompt = COT_PROMPT.format(question=question)
        raw    = self._call(prompt)
        return _extract_final_answer(raw)

    def answer_open_book(self, question_id: str, question: str,
                         context: str) -> str:
        if self.mode == "mock":
            return self._mock_get(question_id, "open_book_answer")
        prompt = RAG_STANDARD_PROMPT.format(context=context, question=question)
        return self._call(prompt)

    def answer_open_book_cot(self, question_id: str, question: str,
                              context: str) -> str:
        if self.mode == "mock":
            row = self._mock_row(question_id)
            if row is not None and int(row.get("cot_open_correct", 0)):
                return str(row["ground_truth_answer"])
            return str(row["open_book_answer"]) if row is not None else ""
        prompt = RAG_COT_PROMPT.format(context=context, question=question)
        raw    = self._call(prompt)
        return _extract_final_answer(raw)

    def self_verify(self, question_id: str, question: str,
                    context: str, previous_answer: str) -> str:
        if self.mode == "mock":
            # Self-verify gives a small extra boost in mock mode
            row = self._mock_row(question_id)
            if row is not None:
                truth = str(row["ground_truth_answer"])
                # ~85 % chance of being correct after self-verify if open book was right
                if int(row.get("open_book_correct", 0)):
                    return truth
                # Small chance of self-correcting
                import random
                rng = random.Random(config.RANDOM_SEED)  
                return truth if rng.random() < 0.15 else previous_answer
            return previous_answer
        prompt = SELF_VERIFY_PROMPT.format(
            context=context, question=question, previous_answer=previous_answer
        )
        return self._call(prompt)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call(self, prompt: str) -> str:
        """Send a prompt to the OpenAI API and return the response text."""
        try:
            response = self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            time.sleep(0.3)   # gentle rate-limiting
            return response.choices[0].message.content.strip()
        except Exception as exc:
            log_warn(f"API call failed: {exc}")
            return ""

    def _mock_row(self, question_id: str) -> dict | None:
        if self.mock_df is None:
            return None
        rows = self.mock_df[self.mock_df["question_id"] == question_id]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def _mock_get(self, question_id: str, column: str) -> str:
        row = self._mock_row(question_id)
        if row is None:
            return ""
        return str(row.get(column, ""))


# ── Utility ───────────────────────────────────────────────────────────────────

def _extract_final_answer(cot_output: str) -> str:
    """Parse 'Final Answer: ...' from a CoT response."""
    for line in reversed(cot_output.splitlines()):
        if "final answer" in line.lower():
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return cot_output.strip()