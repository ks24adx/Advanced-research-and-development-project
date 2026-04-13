"""
config.py
─────────────────────────────────────────────────────────────
Central configuration for the QA Error Analysis project.
Edit the values here to switch between MOCK mode (uses the
pre-annotated CSV datasets) and LIVE mode (calls OpenAI API).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env if present

# ── Project paths ────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent
DATASET_DIR  = ROOT_DIR / "datasets"
OUTPUT_DIR   = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV    = DATASET_DIR / "qa_train_dataset.csv"
TEST_CSV     = DATASET_DIR / "qa_test_dataset.csv"

# ── Experiment mode ───────────────────────────────────────────────────────────
# "mock"  → uses pre-annotated answers from the CSV (no API key needed)
# "live"  → calls the OpenAI API in real time
#MODE = "mock"   # change to "live" for real API calls
MODE = "live"

# ── OpenAI settings (only used in LIVE mode) ─────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-YOUR-KEY-HERE")
LLM_MODEL      = "gpt-4-turbo"
TEMPERATURE    = 0
MAX_TOKENS     = 512

# ── Retrieval settings ────────────────────────────────────────────────────────
TOP_K_PASSAGES      = 5          # passages retrieved per question
RETRIEVAL_MODEL     = "facebook/dpr-ctx_encoder-single-nq-base"
EMBEDDING_CACHE_DIR = ROOT_DIR / ".cache" / "embeddings"

# ── Evaluation ────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
NUM_RUNS      = 3                # average results over N runs

# ── Conditions (order matters for tables/charts) ──────────────────────────────
CONDITIONS = [
    "closed_book",
    "rag_only",
    "cot_only",
    "rag_cot",
    "rag_cot_selfverify",
]

CONDITION_LABELS = {
    "closed_book"        : "Baseline (Closed-Book)",
    "rag_only"           : "RAG Only",
    "cot_only"           : "CoT Only",
    "rag_cot"            : "RAG + CoT",
    "rag_cot_selfverify" : "RAG + CoT + Self-Verify",
}

# ── Colour palette (consistent across all charts) ─────────────────────────────
COLORS = {
    "nq"      : "#1565C0",
    "hotpot"  : "#E07B39",
    "trivia"  : "#4CAF50",
    "blue"    : "#2E75B6",
    "orange"  : "#E07B39",
    "green"   : "#4CAF50",
    "red"     : "#E53935",
    "grey"    : "#90A4AE",
    "dark"    : "#1F4E79",
}
