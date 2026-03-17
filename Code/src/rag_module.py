"""
rag_module.py
-------------
Retrieval-Augmented Generation (RAG) pipeline.

For Law:      context passages come directly from the CSV 'text' column.
For Medicine: the full answer paragraphs from the CSV are indexed as the corpus.

Embedding: sentence-transformers all-MiniLM-L6-v2
Retrieval: FAISS IndexFlatIP (cosine similarity after L2 normalisation)
Fallback:  TF-IDF cosine similarity when sentence-transformers / FAISS unavailable
"""

import os
import json
import logging
import re
from typing import List, Optional, Tuple

import numpy as np

from config import (
    DATA_DIR, EMBEDDING_MODEL, RETRIEVAL_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
)

logger = logging.getLogger(__name__)


# ── Chunk utilities ─────────────────────────────────────────────────────────

def _split_into_chunks(text: str,
                       max_chars: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-level chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = (overlap_text + " " + sent).strip()
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


# ── FAISS vector store ──────────────────────────────────────────────────────

class VectorStore:
    """
    Lightweight FAISS-based vector store.

    Usage:
        store = VectorStore(domain="law")
        store.build(corpus_texts)
        docs = store.retrieve("Is a verbal contract binding?", top_k=3)
    """

    def __init__(self, domain: str, corpus: Optional[List[str]] = None):
        self.domain = domain
        self.corpus: List[str] = corpus or []
        self.chunks: List[str] = []
        self.index = None
        self._model = None
        self._tfidf_vec = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)
            except ImportError:
                logger.warning("sentence-transformers not installed; using TF-IDF fallback.")
                self._model = "tfidf"
        return self._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        if model == "tfidf":
            return self._tfidf_embed(texts)
        embs = model.encode(texts, convert_to_numpy=True,
                            show_progress_bar=False, normalize_embeddings=True)
        return embs.astype(np.float32)

    def _tfidf_embed(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Fit on corpus the first time; reuse for queries
        if not hasattr(self, "_tfidf_vec") or self._tfidf_vec is None:
            self._tfidf_vec = TfidfVectorizer(stop_words="english", max_features=2048)
            self._tfidf_vec.fit(self.chunks or texts)
        try:
            mat = self._tfidf_vec.transform(texts).toarray().astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            return mat / norms
        except ValueError:
            return np.zeros((len(texts), 1), dtype=np.float32)

    def build(self, additional_docs: Optional[List[str]] = None) -> None:
        """Chunk all corpus documents and build the FAISS (or numpy) index."""
        all_docs = self.corpus + (additional_docs or [])
        if not all_docs:
            logger.warning("No documents to index for domain '%s'.", self.domain)
            return

        self.chunks = []
        for doc in all_docs:
            if doc and doc.strip():
                self.chunks.extend(_split_into_chunks(doc))

        if not self.chunks:
            return

        logger.info("Indexing %d chunks for domain='%s'.", len(self.chunks), self.domain)
        embeddings = self._embed(self.chunks)
        d = embeddings.shape[1]

        try:
            import faiss
            self.index = faiss.IndexFlatIP(d)
            self.index.add(embeddings)
            self._index_type = "faiss"
            logger.info("FAISS index built: %d vectors (dim=%d).", len(self.chunks), d)
        except ImportError:
            # Fallback: store embeddings as numpy array
            self.index = embeddings          # shape (n, d)
            self._index_type = "numpy"
            logger.info("Numpy index built: %d vectors (dim=%d).", len(self.chunks), d)

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[str]:
        """Return top-k most similar chunks to the query."""
        if self.index is None:
            logger.warning(
                "VectorStore for domain='%s' not built. Call build() first.", self.domain)
            return []

        q_emb = self._embed([query])   # (1, d)
        k = min(top_k, len(self.chunks))

        if self._index_type == "faiss":
            distances, indices = self.index.search(q_emb, k)
            results = [(float(distances[0][i]), self.chunks[indices[0][i]])
                       for i in range(k) if indices[0][i] != -1]
        else:
            # Numpy dot product
            sims = (self.index @ q_emb.T).flatten()
            top_idx = np.argsort(sims)[::-1][:k]
            results = [(float(sims[i]), self.chunks[i]) for i in top_idx]

        results.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in results]

    @property
    def ntotal(self) -> int:
        if self.index is None:
            return 0
        try:
            return self.index.ntotal   # FAISS
        except AttributeError:
            return self.index.shape[0]  # numpy


# ── Factory ─────────────────────────────────────────────────────────────────

_stores: dict = {}


def get_vector_store(domain: str, items=None) -> VectorStore:
    """
    Return a cached, pre-built VectorStore for the given domain.

    Corpus is built from:
      - Law:      the 'context_docs' (passage text) of every QAItem
      - Medicine: the 'gold_answer' paragraphs of every QAItem
                  (the long NIH answer texts serve as the knowledge base)
    """
    if domain not in _stores:
        corpus = []
        if items:
            for item in items:
                if item.context_docs:
                    corpus.extend(item.context_docs)
                elif domain == "medicine" and item.gold_answer:
                    # For medicine, index the answer text as the knowledge base
                    corpus.append(item.gold_answer)

        store = VectorStore(domain, corpus=corpus)
        store.build()
        _stores[domain] = store
        logger.info("VectorStore ready for domain='%s': %d total chunks.", domain, store.ntotal)
    return _stores[domain]


def reset_stores():
    """Clear cached stores (useful between notebook runs)."""
    global _stores
    _stores = {}
