"""
Agentic chunking orchestrator for query-time, LLM-guided evidence assembly.

Design goals:
- Perform coarse retrieval to get top-K candidate documents/sections.
- Use lightweight, deterministic heuristics to assemble a query-specific
  synthetic context ("ephemeral chunk") with provenance spans.
- Avoid introducing latency or external dependencies by default; upstream
  callers can pass generation knobs to produce an answer from the assembled
  chunk using existing generation utilities.

This module deliberately ships a conservative baseline that is test-friendly:
- It does not call external LLMs to plan tool use (that can be added later).
- It extracts spans by keyword proximity and returns a compact synthetic chunk
  with simple provenance metadata. This keeps behavior deterministic in CI.

Integration entrypoint: `agentic_rag_pipeline(...)` which mirrors a subset of
`unified_rag_pipeline` parameters and returns a `UnifiedSearchResult`.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import time
from typing import Any, Literal

from loguru import logger

from . import agentic_execution as _agentic_execution
from .advanced_cache import AGENTIC_CACHE
from .agentic_execution import (
    AgenticConfig,
    AgenticToolbox,
    _get_media_db_for_structure,
    assemble_ephemeral_chunk as _assemble_ephemeral_chunk,
    build_agentic_derived_evidence,
    decompose_query as _decompose_query,
    tool_loop as _tool_loop,
)
from .database_retrievers import MultiDatabaseRetriever, RetrievalConfig
from .evidence_models import RetrievedEvidence
from .request_resolution import ResolvedRAGRequest
from .retrieval_executor import execute_retrieval_phase
from .retrieval_plan import RetrievalPlan, build_retrieval_plan
from .types import DataSource, Document
from .unified_pipeline import UnifiedSearchResult

# Expose AnswerGenerator at module level for tests/patching parity with unified pipeline
AnswerGenerator: Any
try:
    from .generation import AnswerGenerator as _AnswerGenerator
    AnswerGenerator = _AnswerGenerator
except ImportError:
    AnswerGenerator = None


# Simple in-process caches (namespaced via adapter)
_EPHEMERAL_CACHE: dict[str, Any] = {}
_INTRA_DOC_VEC_CACHE = _agentic_execution._INTRA_DOC_VEC_CACHE


def _cache_get(key: str) -> dict[str, Any] | None:
    v = AGENTIC_CACHE.get("ephemeral_chunk", key)
    if isinstance(v, dict):
        return v
    # Fallback to legacy dict
    v2 = _EPHEMERAL_CACHE.get(key)
    return v2 if isinstance(v2, dict) else None


def _cache_set(key: str, value: dict[str, Any], ttl: int) -> None:
    AGENTIC_CACHE.set("ephemeral_chunk", key, value, ttl)
    # Mirror into legacy dict for test visibility
    _EPHEMERAL_CACHE[key] = value


def invalidate_intra_doc_vectors(media_id: str) -> int:
    """Invalidate cached intra-doc paragraph vectors for a given media/document id.

    Returns the number of entries removed.
    """
    if not media_id:
        return 0
    cache = _agentic_execution._INTRA_DOC_VEC_CACHE
    to_delete = [k for k in list(cache.keys()) if str(k).startswith(f"{media_id}|")]
    removed = 0
    for k in to_delete:
        try:
            cache.pop(k, None)
            removed += 1
        except (KeyError, TypeError):
            pass
    return removed


def clear_agentic_caches() -> None:
    """Clear ephemeral chunk cache and intra-doc vector cache."""
    with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
        AGENTIC_CACHE.invalidate_prefix("ephemeral_chunk", "")
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        _agentic_execution._INTRA_DOC_VEC_CACHE.clear()
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        _EPHEMERAL_CACHE.clear()


def _keyword_terms(query: str) -> list[str]:
    """Extract lightweight keyword set from query (alphanum >= 3 chars)."""
    terms = [t.lower() for t in re.findall(r"[A-Za-z0-9_-]{3,}", query or "")]
    # Deduplicate while preserving order
    seen = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out[:12]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _serialize_retrieval_plan(retrieval_plan: RetrievalPlan | None) -> dict[str, Any] | None:
    if retrieval_plan is None:
        return None
    return {
        "query": retrieval_plan.query,
        "sources": list(retrieval_plan.sources),
        "search_mode": retrieval_plan.search_mode,
        "top_k": retrieval_plan.top_k,
        "min_score": retrieval_plan.min_score,
        "index_namespace": retrieval_plan.index_namespace,
        "collection_names": dict(retrieval_plan.collection_names),
    }


def _resolve_agentic_request_contract(
    *,
    query: str,
    sources: list[str] | None,
    search_mode: str,
    top_k: int,
    min_score: float,
    index_namespace: str | None,
    resolved_request: ResolvedRAGRequest | None,
    retrieval_plan: RetrievalPlan | None,
) -> tuple[ResolvedRAGRequest, RetrievalPlan]:
    if resolved_request is None:
        resolved_request = ResolvedRAGRequest(
            query=query,
            strategy="agentic",
            payload={
                "query": query,
                "sources": list(sources or ["media_db"]),
                "search_mode": search_mode,
                "top_k": top_k,
                "min_score": min_score,
                "index_namespace": index_namespace,
            },
            index_namespace=index_namespace,
            rag_profile=None,
            user_id=None,
            feedback_user_id=None,
        )
    if retrieval_plan is None:
        retrieval_plan = build_retrieval_plan(resolved_request)
    return resolved_request, retrieval_plan


def _document_ids_from_provenance(
    provenance: list[dict[str, Any]] | None,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in provenance or []:
        if not isinstance(item, dict):
            continue
        raw_document_id = item.get("document_id")
        document_id = str(raw_document_id).strip() if raw_document_id is not None else ""
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        ordered.append(document_id)
    return ordered


def _normalize_fts_level(level: str | None) -> Literal["media", "chunk"]:
    return "chunk" if str(level).lower() == "chunk" else "media"


# Compatibility re-exports for callers and tests that still import the old names
# from agentic_chunker.py while execution now lives in agentic_execution.py.


async def agentic_rag_pipeline(
    *,
    query: str,
    # data sources
    sources: list[str] | None = None,
    media_db: Any = None,
    chacha_db: Any = None,
    media_db_path: str | None = None,
    notes_db_path: str | None = None,
    character_db_path: str | None = None,
    kanban_db_path: str | None = None,
    # retrieval config
    search_mode: str = "hybrid",
    fts_level: str = "media",
    hybrid_alpha: float = 0.7,
    top_k: int = 10,
    min_score: float = 0.0,
    index_namespace: str | None = None,
    # agentic config
    agentic: AgenticConfig | None = None,
    # generation config passthrough
    enable_generation: bool = True,
    generation_model: str | None = None,
    generation_provider: str | None = None,
    generation_prompt: str | None = None,
    max_generation_tokens: int = 500,
    # misc
    enable_citations: bool = False,
    include_chunk_citations: bool = True,
    debug_mode: bool = False,
    explain_only: bool = False,
    # verification/guardrails (optional)
    require_hard_citations: bool = False,
    enable_numeric_fidelity: bool = False,
    numeric_fidelity_behavior: str = "continue",  # continue|ask|decline|retry
    enable_claims: bool = False,
    claim_verifier: str = "hybrid",
    claims_top_k: int = 5,
    claims_conf_threshold: float = 0.7,
    claims_max: int = 25,
    nli_model: str | None = None,
    claims_concurrency: int = 8,
    # NLI/low-confidence gate
    adaptive_unsupported_threshold: float = 0.15,
    low_confidence_behavior: str = "continue",
    resolved_request: ResolvedRAGRequest | None = None,
    retrieval_plan: RetrievalPlan | None = None,
) -> UnifiedSearchResult:
    """Agentic RAG: coarse retrieve, assemble ephemeral chunk, optional answer.

    This function is intentionally lightweight and safe to call in tests.
    """
    t0 = time.time()
    cfg = agentic or AgenticConfig()
    resolved_request, effective_retrieval_plan = _resolve_agentic_request_contract(
        query=query,
        sources=sources,
        search_mode=search_mode,
        top_k=top_k,
        min_score=min_score,
        index_namespace=index_namespace,
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
    )
    effective_query = str(resolved_request.query or query)
    effective_sources = list(effective_retrieval_plan.sources or ("media_db",))
    effective_search_mode = effective_retrieval_plan.search_mode
    effective_top_k = max(1, int(effective_retrieval_plan.top_k or top_k or 10))
    effective_min_score = float(effective_retrieval_plan.min_score if effective_retrieval_plan.min_score is not None else min_score or 0.0)
    effective_index_namespace = effective_retrieval_plan.index_namespace
    allowed_media_ids = (resolved_request.payload or {}).get("include_media_ids")
    effective_hybrid_alpha = _coerce_float(
        (resolved_request.payload or {}).get("hybrid_alpha", hybrid_alpha),
        default=_coerce_float(hybrid_alpha, 0.7),
    )

    # Config-driven default: require_hard_citations toggle
    try:
        from tldw_Server_API.app.core.config import rag_require_hard_citations as _rag_req_hc
        if not bool(require_hard_citations) and bool(_rag_req_hc(default=False)):
            require_hard_citations = True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass

    # 1) Build retriever
    db_paths: dict[str, str] = {}
    if media_db_path:
        db_paths["media_db"] = media_db_path
    if notes_db_path:
        db_paths["notes_db"] = notes_db_path
    if character_db_path:
        db_paths["character_cards_db"] = character_db_path
    if kanban_db_path:
        db_paths["kanban_db"] = kanban_db_path

    retriever = MultiDatabaseRetriever(
        db_paths,
        user_id=str(resolved_request.user_id or "rag_agentic"),
        media_db=media_db,
        chacha_db=chacha_db,
    )

    # 2) Coarse retrieval (prefer media-level)
    config = RetrievalConfig(
        max_results=effective_top_k,
        min_score=effective_min_score,
        use_fts=(effective_search_mode in ("fts", "hybrid")),
        use_vector=(effective_search_mode in ("vector", "hybrid")),
        include_metadata=True,
        fts_level=_normalize_fts_level(fts_level),
    )

    try:
        retrieved_evidence = await execute_retrieval_phase(
            resolved_request=resolved_request,
            retrieval_plan=effective_retrieval_plan,
            retriever=retriever,
            retrieval_config=config,
            allowed_media_ids=allowed_media_ids,
        )
        docs = list(retrieved_evidence.documents)
    except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError, TimeoutError):
        logger.opt(exception=True).warning("Agentic coarse retrieval failed")
        docs = []

    # Fallback: if no documents were retrieved via MultiDatabaseRetriever but we
    # have a media DB path, run a direct Media DB FTS-only search to seed the
    # agentic ephemeral chunk. This mirrors the standard pipeline fallback and
    # ensures quality-gate tests have at least one document when media exists.
    if (not docs) and media_db_path and effective_search_mode in ("fts", "hybrid"):
        try:
            from .database_retrievers import MediaDBRetriever as _MDBR
            from .database_retrievers import RetrievalConfig as _RCfg
            fb_cfg = _RCfg(
                max_results=effective_top_k,
                min_score=effective_min_score,
                use_fts=True,
                use_vector=False,
                include_metadata=True,
                fts_level=_normalize_fts_level(fts_level),
            )
            fb_retriever = _MDBR(
                db_path=media_db_path,
                config=fb_cfg,
                user_id=str(resolved_request.user_id or "rag_agentic"),
                media_db=media_db,
            )
            fallback_docs = await fb_retriever.retrieve(
                query=effective_query,
                media_type=None,
                allowed_media_ids=allowed_media_ids,
            )
            if fallback_docs:
                docs = fallback_docs
        except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError, TimeoutError):
            logger.warning("Agentic Media DB fallback retrieval failed")

    # Optional: VLM late chunking to add table/figure hints for PDFs
    if cfg.agentic_enable_vlm_late_chunking and docs:
        try:
            try:
                from tldw_Server_API.app.core.Ingestion_Media_Processing.VLM.registry import (
                    get_backend as _get_vlm_backend,
                )
            except ImportError:
                def _get_vlm_backend(name=None):
                    return None
            backend = _get_vlm_backend(cfg.agentic_vlm_backend if cfg.agentic_vlm_backend not in (None, "auto") else None)
            if backend is not None:
                # Select top-k docs with local PDF path
                sel = []
                for d in docs:
                    md = d.metadata or {}
                    url = md.get("url") or md.get("pdf_path") or md.get("file_path")
                    if not url:
                        continue
                    try:
                        from pathlib import Path
                        p = Path(str(url))
                        if p.exists() and p.suffix.lower() == ".pdf":
                            sel.append((d, str(p)))
                    except (OSError, RuntimeError, TypeError, ValueError):
                        continue
                sel = sel[: max(1, int(cfg.agentic_vlm_late_chunk_top_k_docs or 1))]
                added: list[Document] = []
                for (doc0, pdf_path) in sel:
                    detections = []
                    # Prefer doc-level processing
                    if hasattr(backend, "process_pdf"):
                        res = backend.process_pdf(pdf_path, max_pages=cfg.agentic_vlm_max_pages)
                        by_page: list[dict[str, Any]] = []
                        if isinstance(getattr(res, "extra", None), dict):
                            by_page = res.extra.get("by_page") or []
                        for entry in by_page:
                            page_no = entry.get("page")
                            for d in (entry.get("detections") or []):
                                label = str(d.get("label"))
                                if cfg.agentic_vlm_detect_tables_only and label.lower() != "table":
                                    continue
                                detections.append({
                                    "label": label,
                                    "score": float(d.get("score", 0.0)),
                                    "bbox": d.get("bbox") or [0.0, 0.0, 0.0, 0.0],
                                    "page": page_no,
                                })
                    else:
                        # Per-page fallback via pymupdf
                        try:
                            import pymupdf
                            with pymupdf.open(pdf_path) as _doc:
                                total = len(_doc)
                                maxp = min(cfg.agentic_vlm_max_pages or total, total)
                                for i, page in enumerate(_doc, start=1):
                                    if i > maxp:
                                        break
                                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), alpha=False)
                                    img_bytes = pix.tobytes("png")
                                    res = backend.process_image(img_bytes, context={"page": i, "pdf_path": pdf_path})
                                    for det in (getattr(res, "detections", []) or []):
                                        label = str(getattr(det, "label", ""))
                                        if cfg.agentic_vlm_detect_tables_only and label.lower() != "table":
                                            continue
                                        detections.append({
                                            "label": label,
                                            "score": float(getattr(det, "score", 0.0)),
                                            "bbox": list(getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0])),
                                            "page": i,
                                        })
                        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                            pass
                    for idx, dct in enumerate(detections[:100]):
                        label_val = dct.get("label")
                        label = str(label_val) if label_val is not None else "vlm"
                        score = _coerce_float(dct.get("score", 0.0))
                        bbox = dct.get("bbox") or [0.0, 0.0, 0.0, 0.0]
                        page_no = dct.get("page")
                        chunk_text = f"Detected {label} ({score:.2f}) on page {page_no} at {bbox}"
                        added.append(
                            Document(
                                id=f"vlm:{doc0.id}:{idx}",
                                content=chunk_text,
                                source=doc0.source,
                                metadata={
                                    **(doc0.metadata or {}),
                                    "chunk_type": ("table" if str(label).lower() == "table" else "vlm"),
                                    "page": page_no,
                                    "bbox": bbox,
                                    "derived_from": doc0.id,
                                },
                                score=float(getattr(doc0, "score", 0.0)),
                            )
                        )
                if added:
                    docs.extend(added)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError, TimeoutError):
            logger.debug("Agentic VLM late chunking skipped")

    # 3) Cache key
    def _hashable_doc(d: Document) -> str:
        md = d.metadata or {}
        created = str(md.get("created_at") or md.get("ingestion_date") or "")
        length = str(len(d.content or ""))
        return f"{d.id}|{created}|{length}"

    key_raw = "|".join([effective_query.strip().lower()] + sorted(_hashable_doc(d) for d in docs[: cfg.top_k_docs]))
    cache_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        chunk_text = cached.get("chunk_text", "")
        prov = cached.get("provenance", [])
        cached_hit = True
        if cfg.enable_metrics:
            try:
                from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                increment_counter("agentic_cache_hits_total", 1, labels={"cache_type": "ephemeral"})
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                pass
    else:
        cached_hit = False
        # 4) Assemble ephemeral chunk (either tools or heuristics)
        tool_trace: list[dict[str, Any]] = []
        if cfg.enable_tools:
            chunk_text, prov, tool_trace = await _tool_loop(docs, effective_query, cfg)
        else:
            chunk_text, prov = _assemble_ephemeral_chunk(docs, effective_query, cfg)
            tool_trace = []
        _cache_set(cache_key, {"chunk_text": chunk_text, "provenance": prov}, cfg.cache_ttl_sec)

    # Represent the ephemeral chunk as a Document so the existing
    # generation and response formatting utilities can handle it.
    synthetic = Document(
        id=f"agentic:{hash((effective_query, len(chunk_text))) & 0xFFFFFFFF:x}",
        content=chunk_text,
        metadata={
            "title": "Agentic Ephemeral Chunk",
            "source": "agentic",
            "provenance": prov,
            "strategy": "agentic",
        },
        score=1.0,
        source=DataSource.MEDIA_DB,
    )

    retrieved_docs = list(docs[:effective_top_k])
    coarse_docs = list(docs[: max(1, cfg.top_k_docs)])
    derived_from_document_ids = _document_ids_from_provenance(prov)
    if not derived_from_document_ids:
        derived_from_document_ids = [
            str(d.id) for d in coarse_docs if getattr(d, "id", None)
        ]
    if not derived_from_document_ids:
        logger.warning(
            "Agentic derived evidence has no source lineage: retrieved_docs={}, coarse_docs={}, top_k_docs={}",
            len(retrieved_docs),
            len(coarse_docs),
            cfg.top_k_docs,
        )
    retrieved_evidence = RetrievedEvidence(
        documents=retrieved_docs,
        metadata={
            "strategy": "agentic",
            "coarse_docs": [
                {
                    "id": d.id,
                    "title": (d.metadata or {}).get("title"),
                    "score": float(getattr(d, "score", 0.0) or 0.0),
                }
                for d in coarse_docs
            ],
            "provenance": prov,
            "retrieval_plan": _serialize_retrieval_plan(effective_retrieval_plan),
        },
    )
    coordinated = build_agentic_derived_evidence(
        retrieved_evidence=retrieved_evidence,
        synthetic_chunk=synthetic,
        derived_from_document_ids=derived_from_document_ids,
        coarse_docs_window=[
            {
                "id": d.id,
                "title": (d.metadata or {}).get("title"),
                "score": float(getattr(d, "score", 0.0) or 0.0),
            }
            for d in coarse_docs
        ],
    )

    result = UnifiedSearchResult(
        documents=[synthetic],
        query=effective_query,
        expanded_queries=[],
        metadata=dict(coordinated.metadata),
        timings={},
        citations=[],
        cache_hit=bool(cached_hit),
        errors=[],
        security_report=None,
        total_time=0.0,
    )

    # Attach lightweight coverage/precision metrics
    try:
        terms = _keyword_terms(query)
        term_hits = sum(1 for t in terms if t in (chunk_text or "").lower())
        coverage = (term_hits / max(1, len(terms)))
        uniq_docs = len({str(p.get("document_id")) for p in (prov or []) if isinstance(p, dict)})
        per_doc: dict[str, list[tuple[int, int]]] = {}
        for p in (prov or []):
            try:
                per_doc.setdefault(str(p.get("document_id")), []).append((int(p.get("start", 0)), int(p.get("end", 0))))
            except (AttributeError, TypeError, ValueError):
                continue
        raw = 0
        merged = 0
        for _doc_id, ranges in per_doc.items():
            ranges = sorted(ranges, key=lambda x: x[0])
            raw += sum(end_pos - start_pos for start_pos, end_pos in ranges)
            merged_ranges: list[tuple[int, int]] = []
            for start_pos, end_pos in ranges:
                if not merged_ranges or start_pos > merged_ranges[-1][1]:
                    merged_ranges.append((start_pos, end_pos))
                else:
                    ps, pe = merged_ranges[-1]
                    merged_ranges[-1] = (ps, max(pe, end_pos))
            merged += sum(end_pos - start_pos for start_pos, end_pos in merged_ranges)
        redundancy = 1.0 - (merged / max(1, raw))
        result.metadata.setdefault("agentic_metrics", {})
        result.metadata["agentic_metrics"].update({
            "term_coverage": float(coverage),
            "unique_docs": int(uniq_docs),
            "redundancy": float(redundancy),
        })
    except (AttributeError, KeyError, TypeError, ValueError):
        pass

    if coordinated.derived_from_document_ids:
        result.metadata["derived_from_document_ids"] = list(coordinated.derived_from_document_ids)
    if coordinated.citations:
        result.metadata.setdefault("chunk_citations", list(coordinated.citations))
    if coordinated.verification_report is not None:
        result.metadata.setdefault("verification_report", coordinated.verification_report)

    # Explain-only dry run: return plan/provenance without answer or chunk body
    if explain_only and not enable_generation:
        try:
            # Remove documents to avoid heavy payloads; keep provenance and metrics
            result.documents = []
            result.metadata.setdefault("explain", {})
            # Include a minimal plan derived from tool trace and coverage
            result.metadata["explain"].update({
                "provenance": prov,
            })
        except (AttributeError, TypeError, ValueError):
            pass
        # Timings and return
        result.total_time = time.time() - t0
        result.timings["total"] = result.total_time
        result.timings["agentic_chunking"] = result.total_time
        return result

    # 4) Optional generation grounded in the synthetic chunk
    if enable_generation:
        try:
            from .generation import AnswerGenerator
            gen = AnswerGenerator(
                model=generation_model,
                provider=generation_provider,
            )
            ctx = chunk_text
            gen_out = await gen.generate(
                query=effective_query,
                context=ctx,
                prompt_template=generation_prompt or "default",
                max_tokens=max_generation_tokens,
            )
            ans = gen_out["answer"] if isinstance(gen_out, dict) else str(gen_out)
            result.generated_answer = ans
        except (ImportError, AttributeError, ConnectionError, RuntimeError, TypeError, ValueError, TimeoutError) as e:
            logger.warning("Agentic generation failed")
            result.errors.append(str(e))

    # Guardrails and verification: hard citations + numeric fidelity + optional claims/NLI
    if result.generated_answer:
        claims_payload = None
        # Optional claims verification (NLI/LLM) constrained to assembled spans
        if enable_claims:
            try:
                import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

                from .claims import ClaimsEngine
                def _analyze(api_name: str, input_data: Any, custom_prompt_arg: str | None = None,
                             api_key: str | None = None, system_message: str | None = None,
                             temp: float | None = None, **kwargs):
                    return sgl.analyze(api_name, input_data, custom_prompt_arg, api_key, system_message, temp, **kwargs)
                engine = ClaimsEngine(_analyze)
                async def _retrieve_for_claim(_c_text: str, top_k: int = 3):
                    return [synthetic]
                claims_run = await engine.run(
                    answer=result.generated_answer,
                    query=effective_query,
                    documents=[synthetic],
                    claim_extractor="auto",
                    claim_verifier=claim_verifier,
                    claims_top_k=claims_top_k,
                    claims_conf_threshold=claims_conf_threshold,
                    claims_max=claims_max,
                    retrieve_fn=_retrieve_for_claim,
                    nli_model=nli_model,
                    claims_concurrency=claims_concurrency,
                )
                claims_payload = claims_run.get("claims")
                result.metadata["claims"] = claims_payload
                result.metadata["factuality"] = claims_run.get("summary")
            except (ImportError, AttributeError, ConnectionError, RuntimeError, TypeError, ValueError, TimeoutError):
                logger.debug("Agentic claims verification skipped")

        # Hard citations using assembled spans
        try:
            from .guardrails import build_hard_citations
            hc = build_hard_citations(result.generated_answer, [synthetic], claims_payload=claims_payload)
            if isinstance(hc, dict):
                result.metadata["hard_citations"] = hc
                cov = float(hc.get("coverage") or 0.0)
                if require_hard_citations and cov < 1.0:
                    result.metadata.setdefault("generation_gate", {})
                    result.metadata["generation_gate"].update({
                        "reason": "missing_hard_citations",
                        "coverage": cov,
                        "at": time.time(),
                    })
                    # Abstain if strict
                    result.generated_answer = "Insufficient evidence: missing citations for some statements."
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _ec:
            result.errors.append(f"Hard citations failed: {str(_ec)}")

    # Numeric fidelity check and optional mitigation
        try:
            from .guardrails import check_numeric_fidelity
            if enable_numeric_fidelity:
                nf = check_numeric_fidelity(result.generated_answer, [synthetic])
                if nf:
                    result.metadata.setdefault("numeric_fidelity", {})
                    result.metadata["numeric_fidelity"].update({
                        "present": sorted(nf.present),
                        "missing": sorted(nf.missing),
                        "source_numbers": sorted(nf.union_source_numbers)[:100],
                    })
                    if nf.missing and numeric_fidelity_behavior in {"retry", "ask", "decline"}:
                        if numeric_fidelity_behavior == "ask":
                            note = "\n\n[Note] Some numeric values could not be verified against sources. Please clarify or provide references."
                            result.generated_answer = (result.generated_answer or "") + note
                        elif numeric_fidelity_behavior == "decline":
                            result.generated_answer = "Insufficient evidence to verify numeric claims in the current context."
                        elif numeric_fidelity_behavior == "retry":
                            try:
                                if media_db_path:
                                    mdr = MultiDatabaseRetriever(
                                        {"media_db": media_db_path},
                                        user_id=str(resolved_request.user_id or "rag_agentic"),
                                        media_db=media_db,
                                        chacha_db=chacha_db,
                                    )
                                    conf = RetrievalConfig(
                                        max_results=min(10, effective_top_k),
                                        min_score=effective_min_score,
                                        use_fts=True,
                                        use_vector=True,
                                        include_metadata=True,
                                        fts_level=_normalize_fts_level(fts_level),
                                    )
                                    added = []
                                    for tok in list(nf.missing)[:3]:
                                        try:
                                            added.extend(
                                                await mdr.retrieve(
                                                    query=f"{effective_query} {tok}",
                                                    sources=[DataSource.MEDIA_DB],
                                                    config=conf,
                                                    index_namespace=effective_index_namespace,
                                                )
                                            )
                                        except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError, TimeoutError):
                                            continue
                                    if added:
                                        by_id: dict[str, Document] = {getattr(d, 'id', ''): d for d in (result.documents or [])}
                                        for d in added:
                                            cur = by_id.get(getattr(d, 'id', ''))
                                            if cur is None or float(getattr(d, 'score', 0.0)) > float(getattr(cur, 'score', 0.0)):
                                                by_id[getattr(d, 'id', '')] = d
                                        result.documents = list(by_id.values())
                            except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError, TimeoutError):
                                pass
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _enf:
            result.errors.append(f"Numeric fidelity check failed: {str(_enf)}")

        # NLI low-confidence gate (lightweight, optional)
        try:
            if enable_claims and result.generated_answer:
                from .post_generation_verifier import PostGenerationVerifier as _PGV
                verifier = _PGV(max_retries=0, unsupported_threshold=float(adaptive_unsupported_threshold or 0.15), max_claims=min(10, int(claims_max or 25)))
                vres = await verifier.verify_and_maybe_fix(
                    query=effective_query,
                    answer=result.generated_answer,
                    base_documents=result.documents or [],
                    media_db_path=media_db_path,
                    notes_db_path=notes_db_path,
                    character_db_path=character_db_path,
                    user_id=str(resolved_request.user_id or "rag_agentic"),
                    generation_model=generation_model,
                    generation_provider=generation_provider,
                    existing_claims=None,
                    existing_summary=None,
                    search_mode=effective_search_mode,
                    hybrid_alpha=effective_hybrid_alpha,
                    top_k=effective_top_k,
                )
                result.metadata.setdefault("post_verification", {})
                result.metadata["post_verification"].update({
                    "unsupported_ratio": vres.unsupported_ratio,
                    "total_claims": vres.total_claims,
                    "unsupported_count": vres.unsupported_count,
                    "fixed": vres.fixed,
                    "reason": vres.reason,
                })
                # Gauge and gate behavior
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter, set_gauge
                    set_gauge("rag_nli_unsupported_ratio", float(vres.unsupported_ratio or 0.0), labels={"strategy": "agentic"})
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                low_conf = (vres.unsupported_ratio > float(adaptive_unsupported_threshold or 0.15)) and (not vres.fixed)
                if low_conf:
                    result.metadata.setdefault("generation_gate", {})
                    result.metadata["generation_gate"].update({
                        "reason": "nli_low_confidence",
                        "unsupported_ratio": float(vres.unsupported_ratio or 0.0),
                        "threshold": float(adaptive_unsupported_threshold or 0.15),
                        "at": time.time(),
                    })
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                        increment_counter("rag_nli_low_confidence_total", 1)
                    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                        pass
                    if low_confidence_behavior == "ask":
                        note = "\n\n[Note] Evidence is insufficient; please clarify or provide more context."
                        result.generated_answer = (result.generated_answer or "") + note
                    elif low_confidence_behavior == "decline":
                        result.generated_answer = "Insufficient evidence found to answer confidently."
        except (ImportError, AttributeError, ConnectionError, RuntimeError, TypeError, ValueError, TimeoutError) as _enlv:
            result.errors.append(f"NLI verification failed: {str(_enlv)}")

    # Include tool trace on debug
    if (debug_mode or cfg.debug_trace) and (not cached_hit) and cfg.enable_tools:
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            result.metadata["tool_trace"] = tool_trace

    # Timings
    result.total_time = time.time() - t0
    result.timings["total"] = result.total_time
    result.timings["agentic_chunking"] = result.total_time
    if debug_mode or cfg.debug_trace:
        logger.info(
            f"Agentic RAG built synthetic chunk of {len(chunk_text)} chars from {len(docs)} docs in {result.total_time:.3f}s"
        )

    # Sentence-level chunk citations (align sentences to chunk spans)
    try:
        if result.generated_answer:
            import re as _re
            sents = [s.strip() for s in _re.split(r"(?<=[\.!?])\s+", result.generated_answer.strip()) if s.strip()]
            chunk = synthetic.content or ""
            def _find_off(full: str, t: str) -> tuple[int, int]:
                i = full.find(t)
                return (i, i + len(t)) if i >= 0 else (0, 0)
            entries: list[dict[str, Any]] = []
            for sent in sents:
                st, en = _find_off(chunk, sent)
                entry = {"text": sent, "citations": []}
                if en > st:
                    entry["citations"].append({
                        "doc_id": synthetic.id,
                        "start": int(st),
                        "end": int(en),
                    })
                entries.append(entry)
            result.metadata["chunk_citations"] = {"sentences": entries}
    except (AttributeError, TypeError, ValueError):
        pass

    return result


__all__ = [
    "AgenticConfig",
    "AgenticToolbox",
    "AnswerGenerator",
    # Legacy re-exports retained for tests and old callers that patched these internals.
    "_decompose_query",
    "_get_media_db_for_structure",
    "agentic_rag_pipeline",
    "clear_agentic_caches",
    "invalidate_intra_doc_vectors",
]
