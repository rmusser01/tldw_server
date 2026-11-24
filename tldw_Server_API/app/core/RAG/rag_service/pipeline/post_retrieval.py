"""post_retrieval code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


async def run_web_fallback(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Self-correcting RAG web search fallback."""
    if state.enable_web_fallback and up.fallback_to_web_search:
        web_fallback_start = time.time()
        try:
            # Compute relevance signal for web fallback decision
            # Use reranking calibration if available, otherwise use avg document score
            relevance_signal = 0.5  # Default
            if isinstance(state.result.metadata, dict):
                cal = state.result.metadata.get("reranking_calibration", {})
                if isinstance(cal, dict) and "fused_score" in cal:
                    relevance_signal = float(cal.get("fused_score", 0.5))
                elif state.result.documents:
                    # Fall back to average document score
                    scores = [float(getattr(d, "score", 0.0) or 0.0) for d in state.result.documents]
                    relevance_signal = sum(scores) / len(scores) if scores else 0.0

            # Only trigger if below threshold
            if relevance_signal < rcfg.self_correct.web_fallback_threshold:
                up.logger.debug(
                    f"Web fallback triggered: relevance {relevance_signal:.2f} < threshold {rcfg.self_correct.web_fallback_threshold}"
                )

                merged_docs, web_metadata = await up.fallback_to_web_search(
                    query=state.query,
                    local_docs=state.result.documents or [],
                    relevance_signal=relevance_signal,
                    threshold=rcfg.self_correct.web_fallback_threshold,
                    engine=rcfg.self_correct.web_search_engine,
                    result_count=rcfg.self_correct.web_fallback_result_count,
                    merge_strategy=rcfg.self_correct.web_fallback_merge_strategy,
                    max_total=state.top_k * 2,  # Allow some extra docs from web
                )

                if web_metadata.get("triggered"):
                    state.result.documents = merged_docs
                    state.result.metadata["web_fallback"] = {
                        "enabled": True,
                        "triggered": True,
                        "relevance_signal": relevance_signal,
                        "threshold": rcfg.self_correct.web_fallback_threshold,
                        "web_results_count": web_metadata.get("web_results_count", 0),
                        "merged_count": web_metadata.get("merged_count", 0),
                        "engine_used": web_metadata.get("engine_used"),
                        "merge_strategy": rcfg.self_correct.web_fallback_merge_strategy,
                        "search_time_ms": web_metadata.get("search_time_ms", 0),
                    }

                    # --- OTEL span for web fallback ---
                    if rcfg.observability.enable_observability and up.get_telemetry_manager:
                        try:
                            _tm = up.get_telemetry_manager()
                            _tr = _tm.get_tracer("tldw.rag")
                            with _tr.start_as_current_span("rag.web_fallback") as _span:
                                _span.set_attribute("rag.phase", "web_fallback")
                                _span.set_attribute("rag.web_results", web_metadata.get("web_results_count", 0))
                                _span.set_attribute("rag.engine", rcfg.self_correct.web_search_engine)
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            pass
            else:
                state.result.metadata["web_fallback"] = {
                    "enabled": True,
                    "triggered": False,
                    "relevance_signal": relevance_signal,
                    "threshold": rcfg.self_correct.web_fallback_threshold,
                    "reason": "relevance_above_threshold",
                }

            state.result.timings["web_fallback"] = time.time() - web_fallback_start

        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Web fallback failed: {e}")
            up.logger.warning(f"Web fallback error: {e}")
            state.result.metadata["web_fallback"] = {"enabled": True, "error": str(e)}


def explain_and_expand_sources(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Why-these-sources metadata, sibling inclusion and the document cap."""
    try:
        docs = (state.result.documents or []) if rcfg.observability.include_retrieval_diagnostics else []
        if docs:
            import urllib.parse

            def _host(u: Optional[str]) -> Optional[str]:
                try:
                    if not u:
                        return None
                    return urllib.parse.urlparse(str(u)).hostname
                except (AttributeError, TypeError, ValueError):
                    return None

            hosts = []
            sources_ = []
            ages = []
            scores = []
            now_ts = time.time()
            for d in docs:
                md = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {}) or {}
                url = md.get("url")
                h = _host(url)
                if h:
                    hosts.append(h)
                src = md.get("source") or str(getattr(d, "source", "") or "")
                if src:
                    sources_.append(str(src))
                created = md.get("last_modified") or md.get("created_at")
                ts = None
                try:
                    if isinstance(created, (int, float)):
                        ts = float(created)
                    elif isinstance(created, str) and created:
                        ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                except (TypeError, ValueError):
                    ts = None
                if ts is not None:
                    ages.append(max(0.0, (now_ts - ts) / 86400.0))
                try:
                    scores.append(float(getattr(d, "score", d.get("score", 0.0) if isinstance(d, dict) else 0.0)))
                except (TypeError, ValueError):
                    scores.append(0.0)
            n = max(1, len(docs))
            uniq_hosts = len(set(hosts)) if hosts else 0
            uniq_sources = len(set(sources_)) if sources_ else 0
            diversity = min(1.0, max(uniq_hosts, uniq_sources) / float(n))
            fresh_portion = 0.5
            if ages:
                fresh = sum(1 for a in ages if a <= 90.0)
                fresh_portion = fresh / float(len(ages))
            if scores:
                smin, smax = min(scores), max(scores)
                if smax > smin:
                    topicality = sum((s - smin) / (smax - smin) for s in scores) / float(len(scores))
                else:
                    topicality = 1.0
            else:
                topicality = 0.0

            def _title(md):
                try:
                    return (md.get("title") or "") if isinstance(md, dict) else ""
                except (AttributeError, TypeError, ValueError):
                    return ""

            top_contexts = []
            for d in docs[: min(10, n)]:
                md = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {}) or {}
                top_contexts.append(
                    {
                        "id": getattr(d, "id", d.get("id") if isinstance(d, dict) else None),
                        "title": _title(md),
                        "score": float(
                            getattr(d, "score", md.get("score", 0.0) if isinstance(md, dict) else 0.0) or 0.0
                        ),
                        "url": md.get("url"),
                        "source": md.get("source") or str(getattr(d, "source", "") or ""),
                    }
                )
            state.result.metadata["why_these_sources"] = {
                "diversity": round(float(diversity), 4),
                "freshness": round(float(fresh_portion), 4),
                "topicality": round(float(topicality), 4),
                "top_contexts": top_contexts,
            }
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass

    # ========== SIBLING INCLUSION ==========
    if rcfg.processing.include_sibling_chunks and state.result.documents and rcfg.processing.sibling_window and rcfg.processing.sibling_window > 0:
        siblings_start = time.time()
        try:
            # Index docs by parent and index
            parents: dict[str, dict[int, up.Document]] = {}
            for d in state.result.documents:
                pid = str(d.metadata.get("parent_id", ""))
                cidx_md = d.metadata.get("chunk_index", -1)
                cidx = (
                    int(cidx_md)
                    if isinstance(cidx_md, int) or (isinstance(cidx_md, str) and cidx_md.isdigit())
                    else -1
                )
                if pid and cidx >= 0:
                    parents.setdefault(pid, {})[cidx] = d

            sibling_added: list[up.Document] = []
            seen_ids = {getattr(d, "id", None) for d in state.result.documents}

            for d in list(state.result.documents):
                pid = str(d.metadata.get("parent_id", ""))
                cidx_md = d.metadata.get("chunk_index", -1)
                cidx = (
                    int(cidx_md)
                    if isinstance(cidx_md, int) or (isinstance(cidx_md, str) and cidx_md.isdigit())
                    else -1
                )
                if not pid or cidx < 0:
                    continue
                siblings = parents.get(pid, {})
                # expand symmetrically up to window size
                for w in range(1, int(rcfg.processing.sibling_window) + 1):
                    for adj in (cidx - w, cidx + w):
                        sdoc = siblings.get(adj)
                        if sdoc is not None and getattr(sdoc, "id", None) not in seen_ids:
                            sibling_added.append(sdoc)
                            seen_ids.add(getattr(sdoc, "id", None))

            if sibling_added:
                state.result.documents.extend(sibling_added)
            state.result.metadata["siblings_added_count"] = len(sibling_added)
            state.result.timings["sibling_inclusion"] = time.time() - siblings_start
        except (AttributeError, TypeError, ValueError) as e:
            state.result.errors.append(f"Sibling inclusion failed: {str(e)}")

    # Cap documents to top_k when reranking is disabled
    if (not state.enable_reranking or state.reranking_strategy == "none") and state.result.documents:
        try:
            max_docs = int(state.top_k or 0)
        except (TypeError, ValueError):
            max_docs = 0
        if max_docs > 0 and len(state.result.documents) > max_docs:
            try:
                state.result.documents = sorted(
                    state.result.documents,
                    key=lambda d: getattr(d, "score", 0.0),
                    reverse=True,
                )[:max_docs]
            except (AttributeError, TypeError, ValueError):
                state.result.documents = list(state.result.documents)[:max_docs]


async def build_citations_and_chains(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Citations, knowledge strips and evidence chains."""
    evidence_chain_result = None

    # ========== CITATION GENERATION ==========
    if rcfg.citations.enable_citations and state.result.documents:
        citation_start = time.time()
        try:
            if up.CitationGenerator:
                state.generator = up.CitationGenerator()
                # Map style string to enum if available
                style_map = {
                    "apa": getattr(up.CitationStyle, "APA", None),
                    "mla": getattr(up.CitationStyle, "MLA", None),
                    "chicago": getattr(up.CitationStyle, "CHICAGO", None),
                    "harvard": getattr(up.CitationStyle, "HARVARD", None),
                    "ieee": getattr(up.CitationStyle, "IEEE", None),
                }
                style_enum = style_map.get(rcfg.citations.citation_style) or next(
                    iter([v for v in style_map.values() if v is not None]), None
                )

                if rcfg.citations.enable_evidence_chains and hasattr(state.generator, "generate_citations_with_chains"):
                    dual, chain_result = await state.generator.generate_citations_with_chains(
                        documents=state.result.documents,
                        query=state.query,
                        generated_answer=state.result.generated_answer,
                        style=(
                            style_enum if style_enum is not None else up.CitationStyle.MLA if up.CitationStyle else None
                        ),
                        include_chunks=bool(rcfg.citations.enable_chunk_citations),
                        max_citations=min(len(state.result.documents), (rcfg.rerank.rerank_top_k or state.top_k or 10)),
                        **(
                            {
                                "credential_runtime": rcfg.runtime.credential_runtime,
                                "llm_provider": rcfg.generation.generation_provider,
                                "llm_model": rcfg.generation.generation_model,
                            }
                            if rcfg.runtime.credential_runtime is not None
                            else {}
                        ),
                    )
                    if chain_result:
                        evidence_chain_result = chain_result
                else:
                    dual = await state.generator.generate_citations(
                        documents=state.result.documents,
                        query=state.query,
                        style=(
                            style_enum if style_enum is not None else up.CitationStyle.MLA if up.CitationStyle else None
                        ),
                        include_chunks=bool(rcfg.citations.enable_chunk_citations),
                        max_citations=min(len(state.result.documents), (rcfg.rerank.rerank_top_k or state.top_k or 10)),
                    )

                # Combined citations list for backward compatibility
                state.result.citations = [
                    {"type": "academic", "formatted": s} for s in (dual.academic_citations or [])
                ] + ([{"type": "chunk", **c.to_dict()} for c in (dual.chunk_citations or [])])
                # Expose detailed structures via metadata
                state.result.metadata["academic_citations"] = dual.academic_citations or []
                state.result.metadata["chunk_citations"] = [c.to_dict() for c in (dual.chunk_citations or [])]
                state.result.metadata["inline_citations"] = dual.inline_markers or {}
                state.result.metadata["citation_map"] = dual.citation_map or {}

                state.result.timings["citation_generation"] = time.time() - citation_start
            else:
                state.result.errors.append("Citation module not available")
                up.logger.warning("Citations requested but module not available")
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Citation generation failed: {str(e)}")
            up.logger.error(f"Citation error: {e}")

    # ========== SELF-CORRECTING RAG: KNOWLEDGE STRIPS (Stage 4) ==========
    # Partition documents into semantic strips and filter by relevance before generation
    if rcfg.self_correct.enable_knowledge_strips and state.result.documents and up.process_knowledge_strips:
        strips_start = time.time()
        try:
            filtered_docs, strips_metadata = await up.process_knowledge_strips(
                query=state.query,
                documents=state.result.documents,
                strip_size_tokens=rcfg.self_correct.strip_size_tokens,
                min_relevance=rcfg.self_correct.strip_min_relevance,
                max_strips=rcfg.self_correct.max_strips,
                use_llm_grading=False,  # Use heuristic for speed by default
                **(
                    {
                        "llm_provider": rcfg.research.classifier_provider
                        or rcfg.generation.generation_provider
                        or "openai",
                        "llm_model": rcfg.research.classifier_model or rcfg.generation.generation_model,
                        "credential_runtime": rcfg.runtime.credential_runtime,
                    }
                    if rcfg.runtime.credential_runtime is not None
                    else {}
                ),
            )

            if filtered_docs:
                state.result.documents = filtered_docs

            state.result.metadata["knowledge_strips"] = {
                "enabled": True,
                "total_strips": strips_metadata.get("total_strips", 0),
                "relevant_strips": strips_metadata.get("relevant_strips", 0),
                "filtered_strips": strips_metadata.get("filtered_strips", 0),
                "avg_relevance": strips_metadata.get("avg_relevance", 0.0),
                "strip_size_tokens": rcfg.self_correct.strip_size_tokens,
                "min_relevance": rcfg.self_correct.strip_min_relevance,
                "resulting_docs": len(filtered_docs) if filtered_docs else 0,
                **(
                    {
                        key: strips_metadata[key]
                        for key in ("verification_available", "failure_code")
                        if key in strips_metadata
                    }
                ),
            }

            state.result.timings["knowledge_strips"] = time.time() - strips_start

            # --- OTEL span for knowledge strips ---
            if rcfg.observability.enable_observability and up.get_telemetry_manager:
                try:
                    _tm = up.get_telemetry_manager()
                    _tr = _tm.get_tracer("tldw.rag")
                    with _tr.start_as_current_span("rag.knowledge_strips") as _span:
                        _span.set_attribute("rag.phase", "knowledge_strips")
                        _span.set_attribute("rag.total_strips", strips_metadata.get("total_strips", 0))
                        _span.set_attribute("rag.filtered_strips", strips_metadata.get("filtered_strips", 0))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass

        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Knowledge strips processing failed: {e}")
            up.logger.warning(f"Knowledge strips error: {e}")
            state.result.metadata["knowledge_strips"] = {"enabled": True, "error": str(e)}

    # ========== EVIDENCE CHAINS (before generation for chain-aware citations) ==========
    if rcfg.citations.enable_evidence_chains and state.result.documents and up.EvidenceChainBuilder:
        chain_start = time.time()
        try:
            if evidence_chain_result is None:
                chain_builder = up.EvidenceChainBuilder(
                    enable_llm_extraction=True,
                    llm_provider=rcfg.generation.generation_provider,
                    llm_model=rcfg.generation.generation_model,
                    **(
                        {"credential_runtime": rcfg.runtime.credential_runtime}
                        if rcfg.runtime.credential_runtime is not None
                        else {}
                    ),
                )

                # Build chains - note: we don't have the answer yet, so chains are built from docs
                # This will be re-run after generation if needed for claim extraction
                evidence_chain_result = await chain_builder.build_chains(
                    query=state.query,
                    documents=state.result.documents,
                    generated_answer=None,  # Will be updated post-generation
                )

            if evidence_chain_result:
                state.result.metadata["evidence_chains"] = {
                    "total_chains": len(evidence_chain_result.chains),
                    "overall_confidence": evidence_chain_result.overall_confidence,
                    "multi_hop_detected": evidence_chain_result.multi_hop_detected,
                    "total_nodes": evidence_chain_result.metadata.get("total_nodes", 0),
                    **(
                        {
                            key: evidence_chain_result.metadata[key]
                            for key in ("verification_available", "failure_code")
                            if key in evidence_chain_result.metadata
                        }
                    ),
                }

            state.result.timings["evidence_chains"] = time.time() - chain_start

        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Evidence chain building failed: {e}")
            up.logger.warning(f"Evidence chains error: {e}")
