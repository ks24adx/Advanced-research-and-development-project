"""
models/retriever.py
─────────────────────────────────────────────────────────────
Retrieval module for the open-book experimental conditions.

  - MOCK mode  : simulates retrieval using the
                 supporting_facts_retrieved flag already in the
                 CSV.  If the flag is 1 the relevant passage is
                 "retrieved"; otherwise a distractor passage is
                 returned. No models are loaded.
  - LIVE mode  : uses a DPR bi-encoder to encode the query and
                 retrieve the closest passages from a small
                 Wikipedia-subset FAISS index.
"""

from __future__ import annotations
import random
import pandas as pd
from config import MODE, TOP_K_PASSAGES, RANDOM_SEED, ROOT_DIR
from utils import log_info, log_warn

rng = random.Random(RANDOM_SEED)

# ── Small distractor passages for mock mode ──────────────────────────────────
_DISTRACTOR = (
    "This passage does not contain information directly relevant to the question. "
    "It discusses general background context that may be loosely related but does "
    "not provide the specific answer required."
)

_MOCK_TEMPLATES = {
    "knowledge": (
        "According to available records, the answer to this question is {answer}. "
        "This fact is well-documented in encyclopaedic sources and historical records."
    ),
    "reasoning": (
        "The following facts are relevant: {answer}. "
        "Note that deriving the final answer requires combining information "
        "from more than one passage."
    ),
    "none": (
        "The answer is {answer}. This is a widely known fact confirmed by "
        "multiple independent sources."
    ),
}


class Retriever:
    """Simulates or performs passage retrieval for open-book QA."""

    def __init__(self, mock_df: pd.DataFrame | None = None):
        self.mode    = MODE
        self.mock_df = mock_df
        self._index  = None
        self._model  = None

        if self.mode == "live":
            self._load_live_index()
        else:
            log_info("Retriever initialised in MOCK mode")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, question_id: str, question: str) -> tuple[str, float]:
        """
        Returns:
            context   : concatenated passages as a single string
            recall_at5: 1.0 if supporting fact retrieved, else 0.0
        """
        if self.mode == "mock":
            return self._mock_retrieve(question_id)
        return self._live_retrieve(question)

    # ── Mock retrieval ────────────────────────────────────────────────────────

    def _mock_retrieve(self, question_id: str) -> tuple[str, float]:
        row = self._get_row(question_id)
        if row is None:
            return _DISTRACTOR, 0.0

        retrieved   = int(row.get("supporting_facts_retrieved", 0))
        error_type  = str(row.get("error_type", "none"))
        answer      = str(row.get("ground_truth_answer", ""))
        recall_at5  = float(retrieved)

        if retrieved:
            template = _MOCK_TEMPLATES.get(error_type, _MOCK_TEMPLATES["none"])
            main_passage = template.format(answer=answer)
        else:
            main_passage = _DISTRACTOR

        # Pad to TOP_K_PASSAGES with distractors
        passages = [main_passage] + [_DISTRACTOR] * (TOP_K_PASSAGES - 1)
        rng.shuffle(passages)
        context = "\n\n".join(f"[Passage {i+1}] {p}"
                               for i, p in enumerate(passages))
        return context, recall_at5

    def _get_row(self, question_id: str) -> dict | None:
        if self.mock_df is None:
            return None
        rows = self.mock_df[self.mock_df["question_id"] == question_id]
        return rows.iloc[0].to_dict() if not rows.empty else None

    # ── Live retrieval (DPR + FAISS) ─────────────────────────────────────────

    def _load_live_index(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss, numpy as np, pickle
    
            index_path    = ROOT_DIR / ".cache" / "faiss" / "wiki.index"
            passages_path = ROOT_DIR / ".cache" / "faiss" / "passages.pkl"
    
            if not index_path.exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {index_path}. "
                    "Run build_faiss_index.py first."
                )
    
            log_info("Loading DPR question encoder...")
            self._model = SentenceTransformer("facebook-dpr-question_encoder-single-nq-base")
    
            log_info("Loading FAISS index...")
            self._index = faiss.read_index(str(index_path))

            with open(passages_path, "rb") as f:
                self._passages = pickle.load(f)
    
            log_info(f"Retriever ready — {self._index.ntotal:,} passages indexed.")

        except (ImportError, FileNotFoundError) as exc:
            log_warn(f"Live retrieval unavailable ({exc}). Falling back to mock.")
            self.mode = "mock"  


    def _live_retrieve(self, question: str) -> tuple[str, float]:
        import numpy as np
    
        query_vec = self._model.encode([question], convert_to_numpy=True)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec.astype("float32")
    
        _scores, indices = self._index.search(query_vec, TOP_K_PASSAGES)
    
        passages = [self._passages[i] for i in indices[0] if i >= 0]
        context  = "\n\n".join(f"[Passage {i+1}] {p}"
                                for i, p in enumerate(passages))
        return context, 1.0   # recall is unknown without gold labels
