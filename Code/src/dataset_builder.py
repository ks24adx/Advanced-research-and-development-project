"""
dataset_builder.py
------------------
Loads, pre-processes, and curates domain-specific QA datasets.

Primary sources (uploaded CSV files):
  • Law:      all_criminal_and_family_qa.csv
              Columns: id, text (context), question, answers (JSON list)
  • Medicine: all_diabetes_and_breast_cancer_questions.csv
              Columns: id, question, answer, source, focus_area

Falls back to a curated seed set if CSVs are not found.
"""

import os
import ast
import json
import copy
import logging
from typing import List, Optional

import pandas as pd

from config import (
    DATA_DIR, DOMAINS, N_SAMPLES_PER_DOMAIN, QAItem
)

logger = logging.getLogger(__name__)

# ── CSV filenames (placed in DATA_DIR) ─────────────────────────────────────
LAW_CSV_NAME = "all_criminal_and_family_qa.csv"
MED_CSV_NAME = "top_80_frequent_med_questions.csv"


# ── Answer parser for law CSV ───────────────────────────────────────────────

def _parse_law_answers(ans_str: str) -> str:
    """Extract the primary gold answer text from the JSON-like answers field."""
    try:
        ans = ast.literal_eval(str(ans_str))
        if isinstance(ans, list) and len(ans) > 0:
            return str(ans[0].get("text", "")).strip()
    except Exception:
        pass
    return str(ans_str).strip()


# ── CSV loaders ─────────────────────────────────────────────────────────────

def _load_law_csv(path: str, n: int) -> List[QAItem]:
    """Load law QA items from the uploaded criminal/family CSV."""
    df = pd.read_csv(path)
    items = []
    for i, row in df.iterrows():
        if len(items) >= n:
            break
        gold = _parse_law_answers(row.get("answers", ""))
        if not gold:
            continue
        context = str(row.get("text", "")).strip()
        items.append(QAItem(
            item_id      = f"law_{i:04d}",
            domain       = "law",
            question     = str(row.get("question", "")).strip(),
            gold_answer  = gold,
            context_docs = [context] if context else [],
            reasoning_chain = [],
            difficulty   = "medium",
            source       = LAW_CSV_NAME,
        ))
    logger.info("Loaded %d law items from %s", len(items), path)
    return items


def _load_med_csv(path: str, n: int) -> List[QAItem]:
    """Load medicine QA items from the uploaded diabetes/breast-cancer CSV."""
    df = pd.read_csv(path)
    items = []
    for i, row in df.iterrows():
        if len(items) >= n:
            break
        question = str(row.get("question", "")).strip()
        answer   = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        focus = str(row.get("focus_area", "")).strip()
        items.append(QAItem(
            item_id      = f"med_{i:04d}",
            domain       = "medicine",
            question     = question,
            gold_answer  = answer,
            context_docs = [],           # no pre-bundled context — RAG retrieves
            reasoning_chain = [],
            difficulty   = "medium",
            source       = f"{MED_CSV_NAME} | {focus}",
        ))
    logger.info("Loaded %d medicine items from %s", len(items), path)
    return items


# ── Minimal seed items (fallback only) ─────────────────────────────────────

SEED_LAW = [
    QAItem(
        item_id="law_seed_001", domain="law",
        question="What are the essential elements a plaintiff must prove to succeed in a negligence claim?",
        gold_answer="A plaintiff must prove four elements: (1) Duty, (2) Breach, (3) Causation, (4) Damages.",
        context_docs=["Donoghue v Stevenson [1932] AC 562 established the neighbour principle for duty of care."],
        reasoning_chain=["Identify duty of care.", "Assess breach.", "Apply causation test.", "Verify damages."],
        difficulty="easy", source="seed",
    ),
    QAItem(
        item_id="law_seed_002", domain="law",
        question="Can a plaintiff recover damages under promissory estoppel without consideration?",
        gold_answer="Yes. Promissory estoppel allows recovery when a clear promise induced reasonable detrimental reliance.",
        context_docs=["Restatement (Second) Contracts §90: a promise binding if injustice avoided only by enforcement."],
        reasoning_chain=["Identify promise.", "Assess reliance.", "Determine detriment.", "Conclude."],
        difficulty="medium", source="seed",
    ),
    QAItem(
        item_id="law_seed_003", domain="law",
        question="Is a warrantless search of a backpack carried at time of arrest constitutional?",
        gold_answer="Generally yes as a search incident to lawful arrest under Chimel v. California (1969).",
        context_docs=["Chimel v. California (1969): search incident to arrest covers the person and area within immediate control."],
        reasoning_chain=["Confirm lawful arrest.", "Apply Chimel standard.", "Determine scope.", "Conclude."],
        difficulty="hard", source="seed",
    ),
]

SEED_MEDICINE = [
    QAItem(
        item_id="med_seed_001", domain="medicine",
        question="What is first-line treatment for type 2 diabetes with no contraindications?",
        gold_answer="Metformin is first-line per ADA 2023 and NICE guidelines.",
        context_docs=["ADA 2023: metformin remains the preferred initial pharmacological agent for type 2 diabetes."],
        reasoning_chain=["Identify disease.", "Recall guideline.", "Justify with mechanism."],
        difficulty="easy", source="seed",
    ),
    QAItem(
        item_id="med_seed_002", domain="medicine",
        question="What are the symptoms of breast cancer?",
        gold_answer="A lump in the breast, changes in breast shape, skin dimpling, nipple discharge, or redness.",
        context_docs=["Breast cancer symptoms include lumps, skin changes, nipple discharge and axillary lymphadenopathy."],
        reasoning_chain=["Recall breast cancer hallmarks.", "List key symptoms."],
        difficulty="easy", source="seed",
    ),
    QAItem(
        item_id="med_seed_003", domain="medicine",
        question="What causes type 1 diabetes?",
        gold_answer="Autoimmune destruction of pancreatic beta cells, resulting in insulin deficiency.",
        context_docs=["Type 1 diabetes is an autoimmune condition in which immune cells destroy insulin-producing beta cells."],
        reasoning_chain=["Identify autoimmune mechanism.", "Describe beta cell destruction.", "State insulin deficiency."],
        difficulty="medium", source="seed",
    ),
]


# ── Public API ──────────────────────────────────────────────────────────────

def build_dataset(
    domain: str,
    n: int = N_SAMPLES_PER_DOMAIN,
    offline: bool = False,
    cache_dir: Optional[str] = None,
    csv_dir: Optional[str] = None,
) -> List[QAItem]:
    """
    Return a list of QAItem objects for the given domain.

    Priority:
    1. Load from uploaded CSV (csv_dir or DATA_DIR).
    2. Fall back to seed items if CSV not found.
    """
    csv_dir = csv_dir or cache_dir or DATA_DIR

    if domain == "law":
        csv_path = os.path.join(csv_dir, LAW_CSV_NAME)
        if os.path.exists(csv_path):
            items = _load_law_csv(csv_path, n)
        else:
            logger.warning("Law CSV not found at %s — using seed items.", csv_path)
            items = _replicate_seeds(SEED_LAW, n)

    elif domain == "medicine":
        csv_path = os.path.join(csv_dir, MED_CSV_NAME)
        if os.path.exists(csv_path):
            items = _load_med_csv(csv_path, n)
        else:
            logger.warning("Medicine CSV not found at %s — using seed items.", csv_path)
            items = _replicate_seeds(SEED_MEDICINE, n)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return items[:n]


def _replicate_seeds(seed: List[QAItem], n: int) -> List[QAItem]:
    """Replicate seed items to fill n slots, giving each a unique ID."""
    items = []
    for idx in range(n):
        base = seed[idx % len(seed)]
        new  = copy.deepcopy(base)
        copy_num = idx // len(seed)
        if copy_num > 0:
            new.item_id = f"{base.item_id}_dup{copy_num}"
        items.append(new)
    return items


def dataset_summary(items: List[QAItem]) -> pd.DataFrame:
    """Return a summary DataFrame for a list of QAItems."""
    rows = []
    for it in items:
        rows.append({
            "item_id":   it.item_id,
            "domain":    it.domain,
            "difficulty": it.difficulty,
            "has_context": len(it.context_docs) > 0,
            "has_reasoning_chain": len(it.reasoning_chain) > 0,
            "q_length":  len(it.question.split()),
            "a_length":  len(it.gold_answer.split()),
            "source":    it.source,
        })
    return pd.DataFrame(rows)


def _save_cache(items: List[QAItem], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([vars(i) for i in items], f, indent=2)


def _load_cache(path: str) -> List[QAItem]:
    with open(path) as f:
        raw = json.load(f)
    return [QAItem(**r) for r in raw]
