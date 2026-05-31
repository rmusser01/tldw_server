"""Agentic execution helpers for RAG evidence assembly.

This module owns the agentic-only execution helpers: query decomposition,
deterministic tool-loop assembly, ephemeral chunk construction, and the
derived-evidence boundary.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputOptions,
    parse_structured_output,
)
from .agentic_tools import make_default_registry
from .evidence_models import DerivedEvidence, RetrievedEvidence
from .types import Document

# Expose AnswerGenerator at module level for tests/patching parity with the chunker.
AnswerGenerator: Any
try:
    from .generation import AnswerGenerator as _AnswerGenerator

    AnswerGenerator = _AnswerGenerator
except ImportError:
    AnswerGenerator = None


_INTRA_DOC_VEC_CACHE: dict[str, Any] = {}


@dataclass
class AgenticConfig:
    """Configuration for agentic execution and evidence assembly."""

    top_k_docs: int = 3
    window_chars: int = 1200
    max_tokens_read: int = 6000
    max_tool_calls: int = 8
    extractive_only: bool = True
    quote_spans: bool = True
    enable_tools: bool = False
    use_llm_planner: bool = False
    time_budget_sec: float | None = None
    cache_ttl_sec: int = 600
    debug_trace: bool = False
    enable_query_decomposition: bool = False
    subgoal_max: int = 3
    enable_semantic_within: bool = True
    semantic_dim: int = 2048
    enable_section_index: bool = True
    prefer_structural_anchors: bool = True
    enable_table_support: bool = True
    table_trigger_keywords: tuple[str, ...] = ("table", "figure", "tabular", "dataset")
    table_min_bar_count: int = 3
    agentic_enable_vlm_late_chunking: bool = False
    agentic_vlm_backend: str | None = None
    agentic_vlm_detect_tables_only: bool = True
    agentic_vlm_max_pages: int | None = None
    agentic_vlm_late_chunk_top_k_docs: int = 2
    agentic_use_provider_embeddings_within: bool = False
    agentic_provider_embedding_model_id: str | None = None
    adaptive_budgets: bool = True
    coverage_target: float = 0.8
    min_corroborating_docs: int = 2
    max_redundancy: float = 0.9
    enable_metrics: bool = True


_STRUCT_DB: Any = None
_STRUCT_DB_LOCK = threading.Lock()


def build_agentic_execution_context(
    *,
    resolved_request: Any,
    retrieval_plan: Any,
    payload_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Build effective agentic payload/config from canonical request contracts."""
    effective_payload = dict(payload_override or getattr(resolved_request, "payload", {}) or {})
    effective_payload["sources"] = list(getattr(retrieval_plan, "sources", ()) or ())
    effective_payload["search_mode"] = getattr(retrieval_plan, "search_mode", "hybrid")
    effective_payload["top_k"] = getattr(retrieval_plan, "top_k", 10)
    effective_payload["min_score"] = getattr(retrieval_plan, "min_score", 0.0)
    effective_payload["index_namespace"] = getattr(retrieval_plan, "index_namespace", None)

    def _payload_bool(name: str, fallback: bool = False) -> bool:
        return bool(effective_payload.get(name, fallback))

    def _payload_int(name: str, fallback: int) -> int:
        raw = effective_payload.get(name, fallback)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return fallback

    def _payload_float(name: str, fallback: float) -> float:
        raw = effective_payload.get(name, fallback)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback

    agentic_cfg = AgenticConfig(
        top_k_docs=max(1, _payload_int("agentic_top_k_docs", 3)),
        window_chars=max(200, _payload_int("agentic_window_chars", 1200)),
        max_tokens_read=max(500, _payload_int("agentic_max_tokens_read", 6000)),
        max_tool_calls=max(1, _payload_int("agentic_max_tool_calls", 8)),
        extractive_only=_payload_bool("agentic_extractive_only", True),
        quote_spans=_payload_bool("agentic_quote_spans", True),
        enable_tools=_payload_bool("agentic_enable_tools", False),
        use_llm_planner=_payload_bool("agentic_use_llm_planner", False),
        time_budget_sec=effective_payload.get("agentic_time_budget_sec", None),
        cache_ttl_sec=max(1, _payload_int("agentic_cache_ttl_sec", 600)),
        debug_trace=_payload_bool("agentic_debug_trace", False)
        or _payload_bool("debug_mode", False),
        enable_query_decomposition=_payload_bool("agentic_enable_query_decomposition", False),
        subgoal_max=max(1, _payload_int("agentic_subgoal_max", 3)),
        enable_semantic_within=_payload_bool("agentic_enable_semantic_within", True),
        enable_section_index=_payload_bool("agentic_enable_section_index", True),
        prefer_structural_anchors=_payload_bool("agentic_prefer_structural_anchors", True),
        enable_table_support=_payload_bool("agentic_enable_table_support", True),
        agentic_enable_vlm_late_chunking=_payload_bool("agentic_enable_vlm_late_chunking", False),
        agentic_vlm_backend=effective_payload.get("agentic_vlm_backend", None),
        agentic_vlm_detect_tables_only=_payload_bool("agentic_vlm_detect_tables_only", True),
        agentic_vlm_max_pages=effective_payload.get("agentic_vlm_max_pages", None),
        agentic_vlm_late_chunk_top_k_docs=max(1, _payload_int("agentic_vlm_late_chunk_top_k_docs", 2)),
        agentic_use_provider_embeddings_within=_payload_bool(
            "agentic_use_provider_embeddings_within", False
        ),
        agentic_provider_embedding_model_id=effective_payload.get(
            "agentic_provider_embedding_model_id",
            None,
        ),
        adaptive_budgets=_payload_bool("agentic_adaptive_budgets", True),
        coverage_target=_payload_float("agentic_coverage_target", 0.8),
        min_corroborating_docs=max(1, _payload_int("agentic_min_corroborating_docs", 2)),
        max_redundancy=_payload_float("agentic_max_redundancy", 0.9),
        enable_metrics=_payload_bool("agentic_enable_metrics", True),
    )

    return effective_payload, agentic_cfg


def _now() -> float:
    return time.time()


def _token_estimate(text: str) -> int:
    return max(1, int(len(text) / 4))


def _keyword_terms(query: str) -> list[str]:
    terms = [term.lower() for term in re.findall(r"[A-Za-z0-9_-]{3,}", query or "")]
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        if term not in seen:
            out.append(term)
            seen.add(term)
    return out[:12]


def _split_headings_and_paragraphs(text: str) -> tuple[list[tuple[str, int, int]], list[tuple[int, int]]]:
    if not text:
        return [], []

    lines = text.splitlines()
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    section_indices: list[int] = []
    section_titles: list[str] = []
    for idx, line in enumerate(lines):
        if re.match(r"^\s*#{1,6}\s+", line):
            section_indices.append(idx)
            section_titles.append(re.sub(r"^\s*#+\s+", "", line).strip())
        elif idx + 1 < len(lines) and (set(lines[idx + 1].strip()) <= set("=-") and len(lines[idx + 1].strip()) >= min(3, len(line))):
            section_indices.append(idx)
            section_titles.append(line.strip())
        elif len(line) <= 80 and len(line) >= 3 and line.strip().isupper():
            section_indices.append(idx)
            section_titles.append(line.strip())

    sections: list[tuple[str, int, int]] = []
    for idx, title in zip(section_indices, section_titles):
        start = offsets[idx]
        next_idx = None
        for candidate in section_indices:
            if candidate > idx:
                next_idx = candidate
                break
        end = len(text) if next_idx is None else offsets[next_idx]
        sections.append((title, start, end))

    paragraphs: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n", text):
        end = match.start()
        if end > start:
            paragraphs.append((start, end))
        start = match.end()
    if start < len(text):
        paragraphs.append((start, len(text)))
    return sections, paragraphs


def _hash_embed(text: str, dim: int = 2048) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    if not text:
        return vector
    for token in re.findall(r"[A-Za-z0-9_-]{2,}", text.lower()):
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
        vector[int(digest, 16) % dim] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def _find_spans(text: str, terms: list[str], max_spans: int = 6, window: int = 300) -> list[tuple[int, int]]:
    if not text:
        return []

    lowered = text.lower()
    hits: list[tuple[int, int]] = []
    for term in terms:
        start = 0
        while True:
            idx = lowered.find(term, start)
            if idx == -1:
                break
            left = max(0, idx - window)
            right = min(len(text), idx + len(term) + window)
            hits.append((left, right))
            start = idx + len(term)
            if len(hits) >= max_spans * 3:
                break
        if len(hits) >= max_spans * 3:
            break

    if not hits:
        return [(0, min(len(text), window * 2))]

    hits.sort(key=lambda span: span[0])
    merged: list[tuple[int, int]] = []
    for start, end in hits:
        if not merged or start > merged[-1][1] + 20:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))

    merged.sort(key=lambda span: (span[1] - span[0]), reverse=True)
    return sorted(merged[:max_spans], key=lambda span: span[0])


def _should_use_structure_index(default: bool = True) -> bool:
    try:
        from tldw_Server_API.app.core.config import rag_enable_structure_index

        return bool(rag_enable_structure_index(default=default))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return default


def _get_media_db_for_structure() -> Any:
    """Return a MediaDatabase instance bound to the configured content backend."""
    global _STRUCT_DB
    with _STRUCT_DB_LOCK:
        if _STRUCT_DB is not None:
            return _STRUCT_DB

        try:
            from tldw_Server_API.app.core.config import load_comprehensive_config as _load_cfg
            from tldw_Server_API.app.core.DB_Management.content_backend import get_content_backend as _get_cb
            from tldw_Server_API.app.core.DB_Management.media_db.api import (
                create_media_database,
            )

            cfg = _load_cfg()
            backend = _get_cb(cfg) if cfg else None
            if backend is None:
                return None

            _STRUCT_DB = create_media_database(
                "agentic_toolbox",
                db_path=":memory:",
                backend=backend,
            )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        return _STRUCT_DB


def _lookup_section_from_structure_index(doc: Document, heading: str) -> tuple[int, int] | None:
    if not heading or not _should_use_structure_index():
        return None

    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
    media_id = metadata.get("media_id")
    if media_id is None:
        return None

    try:
        db = _get_media_db_for_structure()
        if db is None:
            return None
        result = db.lookup_section_by_heading(int(str(media_id)), heading)
    except DatabaseError as exc:
        logger.warning(
            "Structure index lookup failed for document_id={} section_title={}: {}",
            doc.id,
            heading,
            exc,
        )
        return None
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None

    if not isinstance(result, tuple) or len(result) < 2:
        return None

    start = int(result[0])
    end = int(result[1])
    if 0 <= start < end:
        return (start, end)
    return None


def build_agentic_derived_evidence(
    *,
    retrieved_evidence: RetrievedEvidence,
    synthetic_chunk: Any,
    derived_from_document_ids: Sequence[str],
    coarse_docs_window: Sequence[Any],
) -> DerivedEvidence:
    metadata = dict(retrieved_evidence.metadata)
    metadata["coarse_docs"] = list(coarse_docs_window)
    return DerivedEvidence(
        retrieved=retrieved_evidence,
        documents=[*retrieved_evidence.documents, synthetic_chunk],
        metadata=metadata,
        citations=list(metadata.get("chunk_citations", []) or []),
        verification_report=metadata.get("verification_report"),
        derived_from_document_ids=tuple(str(document_id) for document_id in derived_from_document_ids),
    )


def assemble_ephemeral_chunk(
    docs: list[Document],
    query: str,
    cfg: Any,
) -> tuple[str, list[dict[str, Any]]]:
    terms = _keyword_terms(query)
    remaining_tokens = int(getattr(cfg, "max_tokens_read", 0) or 0)
    parts: list[str] = []
    provenance: list[dict[str, Any]] = []

    for doc in docs[: max(1, int(getattr(cfg, "top_k_docs", 1) or 1))]:
        if remaining_tokens <= 0:
            break
        text = doc.content or ""
        spans = _find_spans(text, terms, max_spans=4, window=int(getattr(cfg, "window_chars", 1200) / 4))
        for start, end in spans:
            snippet = text[start:end]
            tokens = _token_estimate(snippet)
            if tokens > remaining_tokens:
                allowed_chars = max(50, remaining_tokens * 4)
                snippet = snippet[:allowed_chars]
                tokens = _token_estimate(snippet)
            if tokens <= 0:
                continue
            parts.append(snippet.strip())
            provenance.append(
                {
                    "document_id": doc.id,
                    "title": (doc.metadata or {}).get("title"),
                    "start": int(start),
                    "end": int(start + len(snippet)),
                    "snippet_preview": snippet[:120],
                }
            )
            if getattr(cfg, "enable_metrics", False):
                with contextlib.suppress(ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter, observe_histogram

                    observe_histogram("agentic_span_length_chars", float(len(snippet)), labels={"phase": "assemble"})
                    increment_counter("span_bytes_read_total", float(len(snippet.encode("utf-8"))), labels={"tool": "heuristic"})
            remaining_tokens -= tokens
            if remaining_tokens <= 0:
                break

    glue = "\n\n---\n\n"
    chunk_text = glue.join(parts) if parts else (docs[0].content[: getattr(cfg, "window_chars", 1200)] if docs else "")
    return chunk_text, provenance


class AgenticToolbox:
    """Deterministic tool primitives used by the tool loop."""

    def __init__(self, docs: list[Document], cfg: Any):
        self.docs = docs
        self.cfg = cfg
        self._sections: dict[str, list[tuple[str, int, int]]] = {}
        self._paragraphs: dict[str, list[tuple[int, int]]] = {}
        self._para_vecs: dict[str, list[Any]] = {}
        if getattr(cfg, "enable_section_index", True) or getattr(cfg, "enable_semantic_within", True):
            self._build_indexes()

    def _build_indexes(self) -> None:
        for doc in self.docs:
            text = doc.content or ""
            sections, paragraphs = _split_headings_and_paragraphs(text)
            self._sections[doc.id] = sections
            self._paragraphs[doc.id] = paragraphs
            if getattr(self.cfg, "enable_semantic_within", True):
                if getattr(self.cfg, "agentic_use_provider_embeddings_within", False):
                    try:
                        key = f"{doc.id}|{len(text)}|{hash(text)}|{getattr(self.cfg, 'agentic_provider_embedding_model_id', '') or ''}|prov"
                        cached = _INTRA_DOC_VEC_CACHE.get(key)
                        if cached is not None:
                            self._para_vecs[doc.id] = cached
                            if getattr(self.cfg, "enable_metrics", False):
                                with contextlib.suppress(ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

                                    increment_counter("agentic_cache_hits_total", 1, labels={"cache_type": "intra_doc"})
                        else:
                            from tldw_Server_API.app.core.config import load_comprehensive_config
                            from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
                                create_embeddings_batch,
                            )

                            app_cfg = load_comprehensive_config() or {}
                            embedding_settings = app_cfg.get("EMBEDDING_CONFIG", {})
                            app_config = {"embedding_config": embedding_settings}
                            texts = [text[start:end] for (start, end) in paragraphs]
                            vecs_list = create_embeddings_batch(
                                texts,
                                app_config,
                                getattr(self.cfg, "agentic_provider_embedding_model_id", None),
                            )
                            vecs_np = [np.array(v, dtype=np.float32) for v in vecs_list]
                            for idx, vector in enumerate(vecs_np):
                                norm = float((vector ** 2).sum()) ** 0.5
                                if norm > 0:
                                    vecs_np[idx] = vector / norm
                            self._para_vecs[doc.id] = vecs_np
                            _INTRA_DOC_VEC_CACHE[key] = vecs_np
                            continue
                    except (ImportError, AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError, TimeoutError):
                        pass
                self._para_vecs[doc.id] = [_hash_embed(text[start:end], getattr(self.cfg, "semantic_dim", 2048)) for (start, end) in paragraphs]

    def search_within(self, doc: Document, query: str, max_hits: int = 8, window: int = 300) -> list[tuple[int, int]]:
        if getattr(self.cfg, "enable_semantic_within", True) and doc.id in self._para_vecs:
            try:
                qv = _hash_embed(query, getattr(self.cfg, "semantic_dim", 2048))
                vecs = self._para_vecs.get(doc.id) or []
                if not vecs:
                    return []
                sims = [float(np.dot(qv, vector)) for vector in vecs]
                idxs = sorted(range(len(sims)), key=lambda idx: sims[idx], reverse=True)[:max_hits]
                paras = self._paragraphs.get(doc.id) or []
                return [paras[idx] for idx in idxs]
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                pass
        return _find_spans(doc.content or "", _keyword_terms(query), max_spans=max_hits, window=window)

    def open_section(self, doc: Document, heading: str) -> tuple[int, int] | None:
        structure_span = _lookup_section_from_structure_index(doc, heading)
        if structure_span is not None:
            return structure_span

        if getattr(self.cfg, "enable_section_index", True) and doc.id in self._sections:
            for title, start, end in self._sections.get(doc.id) or []:
                if heading.lower() in (title or "").lower():
                    return (start, end)

        text = doc.content or ""
        if not text:
            return None

        lines = text.splitlines()
        offsets: list[int] = []
        pos = 0
        for line in lines:
            offsets.append(pos)
            pos += len(line) + 1

        for idx, line in enumerate(lines):
            if re.match(r"^\s*(#+|\d+[\)\.]\s+)\s+", line) and heading.lower() in line.lower():
                start = offsets[idx]
                next_idx = idx + 1
                while next_idx < len(lines) and not re.match(r"^\s*(#+|\d+[\)\.]\s+)\s+", lines[next_idx]):
                    next_idx += 1
                end = len(text) if next_idx >= len(lines) else offsets[next_idx]
                return (start, end)
        return None

    def expand_window(self, doc: Document, start: int, end: int, delta: int = 200) -> tuple[int, int]:
        text = doc.content or ""
        return (max(0, start - delta), min(len(text), end + delta))

    def quote_spans(self, doc: Document, spans: list[tuple[int, int]]) -> list[str]:
        text = doc.content or ""
        return [text[start:end] for start, end in spans]

    def section_title_for(self, doc: Document, start: int) -> str | None:
        for title, section_start, section_end in self._sections.get(doc.id) or []:
            if section_start <= start < section_end:
                return title
        return None

    def looks_table(self, text: str) -> bool:
        if not text:
            return False
        bars = text.count("|")
        tabs = text.count("\t")
        nums = len(re.findall(r"\d", text))
        return bars >= int(getattr(self.cfg, "table_min_bar_count", 3) or 3) or tabs >= 2 or (nums >= 10 and ("|" in text or "\t" in text))


def decompose_query(query: str, cfg: Any) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []
    parts = re.split(r"\b(?:and then|then|and|,|;|\?)\b", q, flags=re.IGNORECASE)
    subgoals = [part.strip() for part in parts if part and len(part.strip()) >= 3]
    max_subgoals = int(getattr(cfg, "subgoal_max", 0) or 0)
    if max_subgoals and len(subgoals) > max_subgoals:
        subgoals = subgoals[:max_subgoals]
    return subgoals or [q]


async def tool_loop(docs: list[Document], query: str, cfg: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    tb = AgenticToolbox(docs, cfg)
    registry = make_default_registry(tb)
    remaining_tokens = int(getattr(cfg, "max_tokens_read", 0) or 0)
    max_steps = min(100, max(1, int(getattr(cfg, "max_tool_calls", 1) or 1)))
    deadline = (_now() + float(cfg.time_budget_sec)) if getattr(cfg, "time_budget_sec", None) is not None else None

    assembled: list[tuple[Document, int, int]] = []
    steps = 0
    tool_trace: list[dict[str, Any]] = []

    def time_left() -> bool:
        return (deadline is None) or (_now() < deadline)

    planned_headings: list[str] = []
    planned_terms: list[str] = []
    if getattr(cfg, "use_llm_planner", False):
        try:
            planner_cls = AnswerGenerator
            if planner_cls is None:
                raise RuntimeError("AnswerGenerator is unavailable")
            planner = planner_cls(model=None)
            gen = await planner.generate(query=query, context="", prompt_template="default", max_tokens=200)
            text = gen.get("answer", "") if isinstance(gen, dict) else str(gen)
            payload = parse_structured_output(
                text,
                options=StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
            )
            obj: dict[str, Any] | None = None
            if isinstance(payload, dict):
                obj = payload
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        obj = item
                        break
            if obj is not None:
                if isinstance(obj.get("headings"), list):
                    planned_headings = [str(item)[:80] for item in obj["headings"]][:5]
                if isinstance(obj.get("keywords"), list):
                    planned_terms = [str(item)[:40] for item in obj["keywords"]][:8]
        except (ImportError, AttributeError, ConnectionError, RuntimeError, TypeError, ValueError, TimeoutError):
            planned_headings = []
            planned_terms = []

    subgoals = decompose_query(query, cfg) if getattr(cfg, "enable_query_decomposition", False) else [query]

    def _compute_progress_metrics() -> dict[str, Any]:
        coverage = 0.0
        unique_docs = 0
        redundancy = 0.0
        try:
            terms = _keyword_terms(query)
            assembled_text = "\n".join([(doc.content or "")[start:end] for doc, start, end in assembled])
            term_hits = sum(1 for term in terms if term.lower() in (assembled_text or "").lower())
            coverage = term_hits / max(1, len(terms))
            unique_docs = len({getattr(doc, "id", "") for doc, _, _ in assembled})
            raw = 0
            merged = 0
            per_doc: dict[str, list[tuple[int, int]]] = {}
            for doc, start, end in assembled:
                per_doc.setdefault(getattr(doc, "id", ""), []).append((int(start), int(end)))
            for ranges in per_doc.values():
                ranges = sorted(ranges, key=lambda span: span[0])
                raw += sum(end - start for start, end in ranges)
                merged_ranges: list[tuple[int, int]] = []
                for start, end in ranges:
                    if not merged_ranges or start > merged_ranges[-1][1]:
                        merged_ranges.append((start, end))
                    else:
                        prev_start, prev_end = merged_ranges[-1]
                        merged_ranges[-1] = (prev_start, max(prev_end, end))
                merged += sum(end - start for start, end in merged_ranges)
            redundancy = 1.0 - (merged / max(1, raw))
        except (TypeError, ValueError, AttributeError):
            pass
        return {"coverage": coverage, "unique_docs": unique_docs, "redundancy": redundancy}

    for goal in subgoals:
        for doc in docs[: max(1, int(getattr(cfg, "top_k_docs", 1) or 1))]:
            if steps >= max_steps or not time_left():
                break
            local_query = " ".join([goal] + planned_terms) if planned_terms else goal

            t0 = time.time()
            search = registry.get("search_within")
            hits = search(doc, local_query, max_hits=4, window=int(getattr(cfg, "window_chars", 1200) / 4)) if search else tb.search_within(doc, local_query, max_hits=4, window=int(getattr(cfg, "window_chars", 1200) / 4))
            t1 = time.time()
            if getattr(cfg, "enable_metrics", False):
                with contextlib.suppress(ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter, observe_histogram

                    increment_counter("agentic_tool_calls_total", 1, labels={"tool": "search_within"})
                    observe_histogram("agentic_tool_duration_seconds", (t1 - t0), labels={"tool": "search_within"})

            if getattr(cfg, "enable_table_support", True) and any(keyword in local_query.lower() for keyword in getattr(cfg, "table_trigger_keywords", ())):
                hits = sorted(hits, key=lambda rng: int(not tb.looks_table((doc.content or "")[rng[0]:rng[1]])))

            if not hits and planned_headings:
                for heading in planned_headings[:3]:
                    t0 = time.time()
                    open_section = registry.get("open_section")
                    section = open_section(doc, heading) if open_section else tb.open_section(doc, heading)
                    t1 = time.time()
                    if getattr(cfg, "enable_metrics", False):
                        with contextlib.suppress(ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                            from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter, observe_histogram

                            increment_counter("agentic_tool_calls_total", 1, labels={"tool": "open_section"})
                            observe_histogram("agentic_tool_duration_seconds", (t1 - t0), labels={"tool": "open_section"})
                    if section:
                        hits = [section]
                        break

            for start, end in hits:
                if steps >= max_steps or not time_left():
                    break
                t0 = time.time()
                expand_window = registry.get("expand_window")
                expanded_start, expanded_end = (
                    expand_window(doc, start, end, delta=100)
                    if expand_window
                    else tb.expand_window(doc, start, end, delta=100)
                )
                t1 = time.time()
                if getattr(cfg, "enable_metrics", False):
                    with contextlib.suppress(ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter, observe_histogram

                        increment_counter("agentic_tool_calls_total", 1, labels={"tool": "expand_window"})
                        observe_histogram("agentic_tool_duration_seconds", (t1 - t0), labels={"tool": "expand_window"})
                assembled.append((doc, expanded_start, expanded_end))
                steps += 1
                snippet = (doc.content or "")[expanded_start:expanded_end]
                remaining_tokens -= _token_estimate(snippet)
                if getattr(cfg, "enable_metrics", False):
                    with contextlib.suppress(ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter, observe_histogram

                        observe_histogram("agentic_span_length_chars", float(len(snippet)), labels={"phase": "tool"})
                        increment_counter("span_bytes_read_total", float(len(snippet.encode("utf-8"))), labels={"tool": "expand_window"})
                if getattr(cfg, "debug_trace", False):
                    tool_trace.append(
                        {
                            "tool": "expand_window",
                            "doc_id": getattr(doc, "id", ""),
                            "start": int(expanded_start),
                            "end": int(expanded_end),
                            "duration_ms": int((t1 - t0) * 1000.0),
                            "bytes": int(len(snippet.encode("utf-8"))),
                            "reason": "around-hit",
                        }
                    )
                if remaining_tokens <= 0:
                    break

                if getattr(cfg, "adaptive_budgets", True):
                    progress = _compute_progress_metrics()
                    if (
                        progress.get("coverage", 0.0) >= float(getattr(cfg, "coverage_target", 1.0) or 1.0)
                        and progress.get("unique_docs", 0) >= int(getattr(cfg, "min_corroborating_docs", 1) or 1)
                    ):
                        steps = max_steps
                        break

    if not assembled and docs:
        first_doc = docs[0]
        assembled = [(first_doc, 0, min(len(first_doc.content or ""), int(getattr(cfg, "window_chars", 1200) or 1200)))]

    parts: list[str] = []
    provenance: list[dict[str, Any]] = []
    for doc, start, end in assembled:
        snippet = (doc.content or "")[start:end]
        parts.append(snippet.strip())
        provenance.append(
            {
                "document_id": doc.id,
                "title": (doc.metadata or {}).get("title"),
                "start": int(start),
                "end": int(end),
                "section_title": tb.section_title_for(doc, start),
                "snippet_preview": snippet[:120],
            }
        )

    glue = "\n\n---\n\n"
    return glue.join(parts), provenance, tool_trace


__all__ = [
    "AgenticConfig",
    "AgenticToolbox",
    "AnswerGenerator",
    # Exported for legacy tests and monkeypatch hooks; not part of the public API.
    "_get_media_db_for_structure",
    "assemble_ephemeral_chunk",
    "build_agentic_derived_evidence",
    "build_agentic_execution_context",
    "decompose_query",
    "tool_loop",
]
