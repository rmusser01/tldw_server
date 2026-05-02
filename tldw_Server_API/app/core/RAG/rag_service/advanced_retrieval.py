"""
advanced_retrieval.py

Lightweight advanced retrieval helpers that operate before rerankers.

Includes:
- Multi-vector passage selection (ColBERT-style approximation):
  Splits candidate documents into overlapping spans, embeds spans and the
  query with the async embeddings service, and uses max-similarity over spans
  to re-order candidates or optionally replace documents with their best span.

Notes:
- This is a non-invasive approximation that does not require re-indexing
  documents with multi-vector backends. It runs on the shortlist produced by
  the primary retrievers and adds span-level focus for factoid-style queries.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from loguru import logger

try:
    from .types import DataSource, Document
except ImportError:  # pragma: no cover - import guard for isolated tests
    # Minimal fallback for type hints
    class Document:  # type: ignore
        def __init__(self, id: str, content: str, source=None, metadata: dict[str, Any] | None = None, score: float = 0.0):
            self.id = id
            self.content = content
            self.source = source
            self.metadata = metadata or {}
            self.score = score
    class DataSource:  # type: ignore
        MEDIA_DB = "media_db"

try:
    from tldw_Server_API.app.core.Embeddings.async_embeddings import get_async_embedding_service
except ImportError:  # pragma: no cover - import guard for environments without embeddings
    get_async_embedding_service = None  # type: ignore


@dataclass
class MultiVectorConfig:
    span_chars: int = 300
    stride: int = 150
    max_spans_per_doc: int = 8
    flatten_to_spans: bool = False
    batch_size: int = 32


def _sliding_spans(text: str, span_chars: int, stride: int, max_spans: int) -> list[tuple[int, int, str]]:
    """
    Produce overlapping spans (start_idx, end_idx, span_text).
    """
    if not text:
        return []
    n = len(text)
    if span_chars <= 0 or stride <= 0:
        return [(0, n, text)]
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < n and len(spans) < max_spans:
        start = i
        end = min(i + span_chars, n)
        spans.append((start, end, text[start:end]))
        if end >= n:
            break
        i += stride
    # Ensure at least one span exists
    if not spans:
        spans = [(0, n, text)]
    return spans


def _cosine(a: list[float], b: list[float]) -> float:
    try:
        import numpy as np  # local import to avoid hard dep at import time
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(va.dot(vb) / (na * nb))
    except (ImportError, AttributeError, TypeError, ValueError, ZeroDivisionError):
        # Fallback: Jaccard over rounded floats as tokens (very rough)
        sa = {round(x, 3) for x in a}
        sb = {round(x, 3) for x in b}
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0


async def apply_multi_vector_passages(
    query: str,
    documents: list[Document],
    config: MultiVectorConfig | None = None,
    user_id: str | None = None,
) -> list[Document]:
    """
    Reorder documents using a ColBERT-style max-sim over span embeddings.

    If flatten_to_spans is True, returns the top span per document as a
    pseudo-document (keeping a link to the parent via metadata).
    """
    if not documents:
        return []
    cfg = config or MultiVectorConfig()

    # Acquire embedding service
    if get_async_embedding_service is None:
        logger.warning("Async embedding service unavailable; skipping multi-vector passages")
        return documents

    svc = get_async_embedding_service()

    # Embed query
    try:
        q_vec = await svc.create_embedding(text=query, user_id=user_id)
    except (AttributeError, ConnectionError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
        logger.warning("Query embedding failed; skipping multi-vector passages")
        return documents

    # Build spans for all docs
    all_spans: list[str] = []
    span_meta: list[tuple[int, int, int]] = []  # (doc_idx, start, end)
    for idx, doc in enumerate(documents):
        spans = _sliding_spans(doc.content or "", cfg.span_chars, cfg.stride, cfg.max_spans_per_doc)
        for (s, e, txt) in spans:
            all_spans.append(txt)
            span_meta.append((idx, s, e))

    if not all_spans:
        return documents

    # Embed spans in batches
    span_vectors: list[list[float]] = []
    try:
        for i in range(0, len(all_spans), cfg.batch_size):
            batch = all_spans[i : i + cfg.batch_size]
            vecs = await svc.create_embeddings_batch(batch, user_id=user_id)
            span_vectors.extend(vecs)
    except (AttributeError, ConnectionError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
        logger.warning("Span embeddings failed; skipping multi-vector passages")
        return documents

    # For each document, compute best span similarity
    best_by_doc: dict[int, tuple[float, int]] = {}  # doc_idx -> (best_sim, span_global_idx)
    for g_idx, (doc_idx, _s, _e) in enumerate(span_meta):
        sim = _cosine(q_vec, span_vectors[g_idx])
        prev = best_by_doc.get(doc_idx)
        if prev is None or sim > prev[0]:
            best_by_doc[doc_idx] = (sim, g_idx)

    # Attach metadata and produce ordering
    ranked: list[tuple[float, int]] = []
    for doc_idx, (sim, g_idx) in best_by_doc.items():
        d = documents[doc_idx]
        s_doc_idx, start, end = span_meta[g_idx]
        if s_doc_idx != doc_idx:
            # Defensive: should not happen
            start, end = 0, min(cfg.span_chars, len(d.content or ""))
        span_text = (d.content or "")[start:end]
        meta = d.metadata or {}
        meta = {
            **meta,
            "mv_best_span": {
                "start": int(start),
                "end": int(end),
                "text": span_text,
                "score": float(sim),
            },
            "mv_span_chars": int(cfg.span_chars),
            "mv_stride": int(cfg.stride),
        }
        d.metadata = meta
        # Keep original score but expose mv_score for downstream consumers
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            d.metadata["mv_score"] = float(sim)
        ranked.append((sim, doc_idx))

    ranked.sort(key=lambda x: x[0], reverse=True)

    if not cfg.flatten_to_spans:
        # Reorder original documents by best span similarity
        return [documents[idx] for (_, idx) in ranked]

    # Replace each doc with its best span as a pseudo-document
    out_docs: list[Document] = []
    for sim, doc_idx in ranked:
        d = documents[doc_idx]
        mv = (d.metadata or {}).get("mv_best_span", {})
        start = int(mv.get("start", 0))
        end = int(mv.get("end", len(d.content or "")))
        span_text = (d.content or "")[start:end]
        new_id = f"{getattr(d, 'id', str(id(d)))}#span:{start}:{end}"
        new_meta = {
            **(d.metadata or {}),
            "parent_id": getattr(d, 'id', None),
            "parent_score": float(getattr(d, 'score', 0.0) or 0.0),
            "mv_adapted": True,
        }
        source = getattr(d, 'source', None)
        if source is None:
            source = DataSource.MEDIA_DB
        out_docs.append(Document(id=new_id, content=span_text, source=source, metadata=new_meta, score=float(sim)))

    return out_docs
