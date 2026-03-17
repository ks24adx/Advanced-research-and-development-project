"""
config.py
---------
Central configuration for the Knowledge vs Reasoning research project.
All hyper-parameters, paths, and prompt templates live here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
DOMAINS = ["law", "medicine"]

# Number of QA items per domain (50-80 as specified in the DPP)
N_SAMPLES_PER_DOMAIN = 60

# Experimental conditions
CONDITIONS = ["reasoning_only", "rag", "hybrid"]

# LLM settings
DEFAULT_MODEL   = "gpt-4o-mini"          # swap for any OpenAI / Anthropic model
MAX_TOKENS      = 1024
TEMPERATURE     = 0.0                    # deterministic for reproducibility

# RAG settings
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"   # fast, good quality
RETRIEVAL_TOP_K      = 3                     # documents to retrieve per query
CHUNK_SIZE           = 512                   # characters per document chunk
CHUNK_OVERLAP        = 64

# Error attribution labels
ERROR_LABELS = [
    "pure_knowledge_failure",
    "pure_reasoning_failure",
    "mixed_failure",
    "ambiguous",
    "correct",
]

# Annotation thresholds
COHEN_KAPPA_THRESHOLD = 0.6   # minimum acceptable inter-annotator agreement

# Statistical tests
ALPHA = 0.05                  # significance level

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QAItem:
    """A single question-answer task."""
    item_id:        str
    domain:         str          # "law" | "medicine"
    question:       str
    gold_answer:    str
    context_docs:   List[str] = field(default_factory=list)   # for RAG
    reasoning_chain: List[str] = field(default_factory=list)  # expected steps
    difficulty:     str  = "medium"   # easy | medium | hard
    source:         str  = ""


@dataclass
class ExperimentResult:
    """LLM output for one (item, condition) pair."""
    item_id:    str
    domain:     str
    condition:  str     # reasoning_only | rag | hybrid
    model:      str
    question:   str
    gold_answer: str
    model_answer: str
    retrieved_docs: List[str] = field(default_factory=list)
    cot_steps:      List[str] = field(default_factory=list)
    latency_s:      float = 0.0


@dataclass
class ErrorAnnotation:
    """Human / automated error annotation for one result."""
    item_id:    str
    condition:  str
    annotator:  str          # "auto" | "human_1" | "human_2"
    is_correct: bool
    error_label: str         # one of ERROR_LABELS
    knowledge_score: float   # 0-1: how much is a knowledge gap?
    reasoning_score: float   # 0-1: how much is a reasoning gap?
    notes:      str = ""


@dataclass
class EvaluationReport:
    """Aggregated metrics for one (domain, condition) slice."""
    domain:    str
    condition: str
    n:         int
    accuracy:  float
    partial_credit: float
    logical_consistency: float
    hallucination_rate:  float
    knowledge_error_pct: float
    reasoning_error_pct: float
    mixed_error_pct:     float
    kappa:               float   # Cohen's Kappa (auto vs human)
