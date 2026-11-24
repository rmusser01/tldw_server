"""query_prep code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import calendar
import contextlib
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from ..pipeline import run_state as _run_state
from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


async def clarify_and_spellcheck(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Pre-retrieval clarification (may raise EarlyReturn) and spell check."""
    effective_pre_clarify = (
        rcfg.query_prep.enable_pre_retrieval_clarification
        if rcfg.query_prep.enable_pre_retrieval_clarification is not None
        else state.effective_enable_generation
    )
    if effective_pre_clarify and up.assess_query_for_clarification is not None:
        clarification_start = time.time()
        decision = await up.assess_query_for_clarification(
            query=state.query,
            chat_history=rcfg.query_prep.chat_history,
            timeout_sec=float(rcfg.query_prep.clarification_timeout_sec or 1.5),
        )
        state.result.timings["pre_retrieval_clarification"] = time.time() - clarification_start
        if decision.required:
            state.result.generated_answer = decision.question or "Could you clarify your request?"
            state.result.documents = []
            state.result.timings["retrieval"] = 0.0
            state.result.metadata.setdefault("clarification", {})
            state.result.metadata["clarification"].update(
                {
                    "required": True,
                    "stage": "pre_retrieval",
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "detector": decision.detector,
                }
            )
            state.result.metadata.setdefault("retrieval_bypassed", {})
            state.result.metadata["retrieval_bypassed"]["reason"] = "pre_retrieval_clarification"
            try:
                from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

                increment_counter(
                    "rag_clarification_triggered_total",
                    1,
                    labels={"stage": "pre_retrieval", "detector": decision.detector},
                )
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                pass
            raise _run_state.EarlyReturn()

    # ========== SPELL CHECK ==========
    if rcfg.query_prep.spell_check:
        if up.spell_check_query:
            spell_start = time.time()
            corrected = await up.spell_check_query(state.query)
            if corrected != state.query:
                state.result.metadata["original_query"] = state.query
                state.result.metadata["corrected_query"] = corrected
                state.sync_effective_query(corrected)
            state.result.timings["spell_check"] = time.time() - spell_start
        else:
            state.result.errors.append("Spell check module not available")
            up.logger.warning("Spell check requested but module not available")


async def run_query_classification(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Query classification and reformulation, with the routed external-docs prefetch."""
    state._classification = None
    state._skip_retrieval_stack = False
    state._skip_retrieval_reason: Optional[str] = None
    state._skip_local_retrieval = False

    async def _prefetch_routed_external_docs(
        *,
        query_text: str,
        use_web: bool,
        use_academic: bool,
        use_discussions: bool,
    ) -> list["Document"]:  # noqa: F821 - string annotation never evaluated; Document is up.Document
        """Best-effort external prefetch when classification disables local DB search."""
        prefetched_docs: list[up.Document] = []
        max_results = max(1, min(int(state.retrieval_top_k or 10), 25))
        seen_ids: set[str] = set()

        def _append_doc(doc: up.Document) -> None:
            if doc.id in seen_ids:
                return
            seen_ids.add(doc.id)
            prefetched_docs.append(doc)

        # Discussion-prefetch first for opinion/experience-focused questions.
        if use_discussions and state.enable_discussion_search:
            try:
                from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import search_discussions

                discussion_results = await search_discussions(
                    query=query_text,
                    platforms=rcfg.research.discussion_platforms,
                    max_results=max_results,
                )
                for item in discussion_results or []:
                    doc_id = str(item.get("url") or item.get("id") or f"discussion:{len(prefetched_docs)}")
                    _append_doc(
                        up.Document(
                            id=doc_id,
                            content=str(item.get("content", "")),
                            metadata={
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "source_type": "discussion",
                                "platform": item.get("platform", ""),
                            },
                            source=up.DataSource.WEB_CONTENT,
                            score=float(item.get("score", 0.5) or 0.5),
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - best effort
                state.result.errors.append(f"Discussion prefetch failed: {exc}")

        # Academic/web prefetch via existing web search processor.
        if use_academic or use_web:
            try:
                from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import (
                    perform_websearch,
                    process_web_search_results,
                )

                web_query = query_text
                if use_academic:
                    web_query = (
                        f"{query_text} " "site:arxiv.org OR site:scholar.google.com OR site:semanticscholar.org"
                    )

                raw_results = await asyncio.to_thread(
                    perform_websearch,
                    search_engine="duckduckgo",
                    search_query=web_query,
                    content_country="US",
                    search_lang="en",
                    output_lang="en",
                    result_count=max_results,
                )
                processed = process_web_search_results(raw_results, "duckduckgo")
                for item in (processed.get("results", []) or [])[:max_results]:
                    url = str(item.get("url", ""))
                    if not url:
                        continue
                    source_type = "academic" if use_academic else "web"
                    _append_doc(
                        up.Document(
                            id=url,
                            content=str(item.get("content", item.get("snippet", ""))),
                            metadata={
                                "title": item.get("title", ""),
                                "url": url,
                                "source_type": source_type,
                            },
                            source=up.DataSource.WEB_CONTENT,
                            score=float(item.get("score", 0.5) or 0.5),
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - best effort
                state.result.errors.append(f"Web/academic prefetch failed: {exc}")

        return prefetched_docs[:max_results]

    if rcfg.query_prep.enable_query_classification and up.classify_and_reformulate is not None:
        try:
            _cls_start = time.time()
            _cls_provider = rcfg.research.classifier_provider or rcfg.generation.generation_provider or "openai"
            _cls_model = rcfg.research.classifier_model or rcfg.generation.generation_model
            _classification_stage_metadata: dict[str, Any] = {}
            state._classification = await up.classify_and_reformulate(
                query=state.query,
                chat_history=rcfg.query_prep.chat_history,
                llm_provider=_cls_provider,
                llm_model=_cls_model,
                **(
                    {
                        "credential_runtime": rcfg.runtime.credential_runtime,
                        "stage_metadata": _classification_stage_metadata,
                    }
                    if rcfg.runtime.credential_runtime is not None
                    else {}
                ),
            )
            state.result.timings["query_classification"] = time.time() - _cls_start
            state.result.metadata["query_classification"] = {
                "skip_search": state._classification.skip_search,
                "search_local_db": state._classification.search_local_db,
                "search_web": state._classification.search_web,
                "search_academic": state._classification.search_academic,
                "search_discussions": state._classification.search_discussions,
                "detected_intent": state._classification.detected_intent,
                "confidence": state._classification.confidence,
                "reasoning": state._classification.reasoning,
                **_classification_stage_metadata,
            }

            # Apply classification decisions
            if state._classification.skip_search:
                # Skip retrieval entirely, return early with generation only
                up.logger.info(f"Query classification: skip_search=True for '{state.query[:80]}'")
                state.result.metadata["classification_skip_search"] = True
                state._skip_retrieval_stack = True
                state._skip_retrieval_reason = "classification_skip_search"
                # If generation is enabled, we still want to generate a response
                # but without search context — this is handled downstream

            # Apply reformulated query
            if state._classification.standalone_query and state._classification.standalone_query != state.query:
                state.result.metadata["reformulated_query"] = state._classification.standalone_query
                state.sync_effective_query(state._classification.standalone_query)
                up.logger.debug(f"Query reformulated to: {state.query[:100]}")

            # Route web fallback based on classification
            if state._classification.search_web and not state.enable_web_fallback:
                state.enable_web_fallback = True
            if state._classification.search_discussions and not state.enable_discussion_search:
                state.enable_discussion_search = True
            if not state._classification.search_local_db:
                state._skip_local_retrieval = True
                state.result.metadata["classification_local_retrieval"] = "disabled"

                # If local retrieval is disabled but external routes are requested,
                # prefetch external docs directly in non-research-loop mode.
                if not rcfg.research.enable_research_loop and (
                    state._classification.search_web
                    or state._classification.search_academic
                    or state._classification.search_discussions
                ):
                    external_docs = await _prefetch_routed_external_docs(
                        query_text=state.query,
                        use_web=bool(state._classification.search_web),
                        use_academic=bool(state._classification.search_academic),
                        use_discussions=bool(state._classification.search_discussions),
                    )
                    if external_docs:
                        state.result.documents = external_docs
                        state._skip_retrieval_stack = True
                        state._skip_retrieval_reason = "classification_external_prefetch"
                        state.result.metadata["classification_external_prefetch"] = {
                            "enabled": True,
                            "document_count": len(external_docs),
                            "search_web": bool(state._classification.search_web),
                            "search_academic": bool(state._classification.search_academic),
                            "search_discussions": bool(state._classification.search_discussions),
                        }
                    else:
                        state._skip_retrieval_stack = True
                        state._skip_retrieval_reason = "classification_local_disabled"
                        state.result.metadata["classification_external_prefetch"] = {
                            "enabled": True,
                            "document_count": 0,
                            "search_web": bool(state._classification.search_web),
                            "search_academic": bool(state._classification.search_academic),
                            "search_discussions": bool(state._classification.search_discussions),
                        }

        except Exception as _cls_exc:  # noqa: BLE001 - moved from unified_pipeline.py (TASK-13950.24)
            up.logger.warning(f"Query classification failed: {_cls_exc!r}")
            state.result.errors.append(f"Query classification failed: {_cls_exc}")
    elif rcfg.query_prep.chat_history and rcfg.query_prep.enable_query_reformulation and up.classify_and_reformulate is not None:
        # Even without full classification, do reformulation if chat history present
        try:
            from ..query_classifier import reformulate_query as _reformulate_q

            _ref_start = time.time()
            _ref_provider = rcfg.research.classifier_provider or rcfg.generation.generation_provider or "openai"
            _ref_model = rcfg.research.classifier_model or rcfg.generation.generation_model
            _reformulation_stage_metadata: dict[str, Any] = {}
            reformulated = await _reformulate_q(
                query=state.query,
                chat_history=rcfg.query_prep.chat_history,
                llm_provider=_ref_provider,
                llm_model=_ref_model,
                **(
                    {
                        "credential_runtime": rcfg.runtime.credential_runtime,
                        "stage_metadata": _reformulation_stage_metadata,
                    }
                    if rcfg.runtime.credential_runtime is not None
                    else {}
                ),
            )
            if reformulated and reformulated != state.query:
                state.result.metadata["reformulated_query"] = reformulated
                state.sync_effective_query(reformulated)
            if _reformulation_stage_metadata:
                state.result.metadata["query_reformulation"] = dict(
                    _reformulation_stage_metadata
                )
            state.result.timings["query_reformulation"] = time.time() - _ref_start
        except Exception as _ref_exc:  # noqa: BLE001 - moved from unified_pipeline.py (TASK-13950.24)
            up.logger.warning(f"Query reformulation failed: {_ref_exc!r}")


def apply_guardrail_defaults(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Env-based production defaults and generation guardrails."""
    try:
        import os as _os

        _prod_env = up._shared_is_production_like_env()
        _strict_env = up._shared_is_truthy(_os.getenv("RAG_GUARDRAILS_STRICT", "false"))
        if _prod_env or _strict_env:
            if not state.enable_numeric_fidelity:
                state.enable_numeric_fidelity = True
            if not state.require_hard_citations:
                state.require_hard_citations = True
            # Behavior default can be tuned via env when it's left as "continue"
            if state.numeric_fidelity_behavior == "continue":
                _beh = _os.getenv("RAG_NUMERIC_FIDELITY_BEHAVIOR", "ask").strip().lower()
                if _beh in {"continue", "ask", "decline", "retry"}:
                    state.numeric_fidelity_behavior = _beh  # type: ignore
    except (TypeError, ValueError):
        pass

    # Apply config-driven defaults for confidence/citation gates when not explicitly set
    try:
        if up._rag_low_conf:
            cfg_lcb = up._rag_low_conf()
            if (state.low_confidence_behavior or "continue") == "continue" and cfg_lcb != "continue":
                state.low_confidence_behavior = cfg_lcb
        if up._rag_req_hc and not bool(state.require_hard_citations) and bool(up._rag_req_hc(default=False)):
            state.require_hard_citations = True
    except (TypeError, ValueError):
        pass

    # Apply config-driven default for strict extractive generation
    _rag_strict: Any = None
    with contextlib.suppress(ImportError):
        from tldw_Server_API.app.core.config import rag_strict_extractive as _rag_strict
    try:
        if _rag_strict is not None and not bool(state.strict_extractive) and bool(_rag_strict(default=False)):
            state.strict_extractive = True
    except (TypeError, ValueError):
        pass

    # Precompute query analysis once for reuse across phases
    state.analysis = None
    state.analysis_intent_val = None
    state.analysis_complexity_val = None
    state.analysis_domain = None
    if up.QueryAnalyzer and state.retrieval_query:
        try:
            qa = up.QueryAnalyzer()
            state.analysis = qa.analyze_query(state.retrieval_query)
            analysis_intent = getattr(state.analysis, "intent", None)
            analysis_complexity = getattr(state.analysis, "complexity", None)
            state.analysis_intent_val = (
                getattr(analysis_intent, "value", str(analysis_intent)) if analysis_intent is not None else None
            )
            state.analysis_complexity_val = (
                getattr(analysis_complexity, "value", str(analysis_complexity))
                if analysis_complexity is not None
                else None
            )
            state.analysis_domain = getattr(state.analysis, "domain", None)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            state.analysis = None
            state.analysis_intent_val = None
            state.analysis_complexity_val = None
            state.analysis_domain = None


async def run_query_expansion(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Query expansion, scoped to the user's cache namespace."""
    state.expanded_queries = [state.retrieval_query]
    if state.expand_query:
        expansion_start = time.time()
        try:
            # Try rewrite cache first
            cached_rewrites: list[str] = []
            intent_label = state.analysis_intent_val
            if up.RewriteCache and rcfg.scope.user_id:
                try:
                    rc = up.RewriteCache(user_id=rcfg.scope.user_id)
                    cached = rc.get(
                        state.retrieval_query,
                        intent=intent_label,
                        corpus=state.query_expansion_corpus,
                    )
                    if cached:
                        cached_rewrites = [c for c in cached if isinstance(c, str) and c.strip()]
                except ValueError as exc:
                    up.logger.debug(f"Rewrite cache disabled for user_id={rcfg.scope.user_id}: {exc}")
                except (AttributeError, OSError, RuntimeError, TypeError, sqlite3.Error):
                    pass
            strategies = (rcfg.query_prep.expansion_strategies or ["acronym", "synonym"]).copy()
            expanded_variants: list[str] = []
            if up.multi_strategy_expansion:
                if state.query_expansion_corpus:
                    expanded = await up.multi_strategy_expansion(
                        state.retrieval_query,
                        strategies=strategies,
                        corpus=state.query_expansion_corpus,
                    )
                else:
                    # Avoid passing None to preserve expected call signature in tests
                    expanded = await up.multi_strategy_expansion(
                        state.retrieval_query,
                        strategies=strategies,
                    )
                if isinstance(expanded, list):
                    expanded_variants.extend([q for q in expanded if isinstance(q, str)])
                elif isinstance(expanded, str) and expanded.strip():
                    expanded_variants.append(expanded)
            rewriter_variants: list[str] = []
            if up.QueryRewriter and state.query:
                try:
                    rewriter = up.QueryRewriter()
                    rw = rewriter.rewrite_query(
                        state.query,
                        strategies=["decompose", "generalize", "specify", "clarify"],
                    )
                    rewriter_variants = [r.rewritten_query for r in rw if getattr(r, "rewritten_query", None)]
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    rewriter_variants = []

            # Merge and dedupe all candidates while preserving order
            candidate_queries: list[str] = [state.query]
            candidate_queries.extend(cached_rewrites)
            candidate_queries.extend(expanded_variants)
            candidate_queries.extend(rewriter_variants)
            state.expanded_queries = []
            seen = set()
            for q in candidate_queries:
                if not isinstance(q, str):
                    continue
                s = q.strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                state.expanded_queries.append(s)

            # Bound the number of variations to avoid excessive retrieval fan-out
            try:
                max_variations = max(0, int(rcfg.query_prep.max_query_variations))
            except (TypeError, ValueError):
                max_variations = 3
            limit = 1 + max_variations
            if len(state.expanded_queries) > limit:
                state.expanded_queries = state.expanded_queries[:limit]
            # Persist effective rewrites for future reuse (best-effort)
            try:
                if up.RewriteCache and rcfg.scope.user_id and len(state.expanded_queries) > 1:
                    rew = [q for q in state.expanded_queries if q != state.query][:5]
                    if rew:
                        rc = up.RewriteCache(user_id=rcfg.scope.user_id)
                    rc.put(
                        state.retrieval_query,
                        rewrites=rew,
                        intent=intent_label,
                        corpus=state.retrieval_index_namespace,
                    )
            except ValueError as exc:
                up.logger.debug(f"Rewrite cache write disabled for user_id={rcfg.scope.user_id}: {exc}")
            except (AttributeError, TypeError):
                pass
            state.result.expanded_queries = [q for q in state.expanded_queries if q != state.query]
            if state.result.expanded_queries:
                state.result.metadata.setdefault("query_expansion", {})
                state.result.metadata["query_expansion"]["variations"] = len(state.result.expanded_queries)
            state.result.timings["query_expansion"] = time.time() - expansion_start
            if state.metrics:
                state.metrics.expansion_time = state.result.timings["query_expansion"]
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Query expansion failed: {str(e)}")
            up.logger.warning(f"Query expansion error: {e}")


def route_and_weight(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Intent routing, dynamic granularity and intent-based weighting."""
    if rcfg.query_prep.enable_intent_routing and up.QueryRouter:
        try:
            router = up.QueryRouter()
            routing = router.route_query(state.query)
            # Map routing decisions to current pipeline knobs conservatively
            # Keep search_mode hybrid by default; adjust hybrid_alpha and top_k
            strat = str(routing.get("retrieval_strategy", "")).lower()
            if strat == "precise":
                # Favor lexical; shift hybrid_alpha toward FTS
                with contextlib.suppress(TypeError, ValueError):
                    state.hybrid_alpha = min(max(0.0, float(state.hybrid_alpha)), 1.0)
                state.hybrid_alpha = min(state.hybrid_alpha, 0.5)
            elif strat == "broad":
                # Favor semantic
                with contextlib.suppress(TypeError, ValueError):
                    state.hybrid_alpha = min(max(0.0, float(state.hybrid_alpha)), 1.0)
                state.hybrid_alpha = max(state.hybrid_alpha, 0.7)
            # Respect suggested top_k when present
            try:
                tk = int(routing.get("top_k", state.top_k))
                if 1 <= tk <= 100:
                    if tk != state.top_k:
                        state.retrieval_scalar_overrides_applied = True
                    state.top_k = tk
            except (TypeError, ValueError):
                pass
            state.result.metadata["intent_routing"] = {
                "strategy": strat,
                "hybrid_alpha": state.hybrid_alpha,
                "top_k": state.top_k,
            }
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            state.result.errors.append(f"Intent routing failed: {e}")

    # ========== DYNAMIC GRANULARITY ROUTING ==========
    granularity_decision = None
    if rcfg.query_prep.enable_dynamic_granularity and up.GranularityRouter and up.route_query_granularity:
        try:
            granularity_start = time.time()
            granularity_decision = up.route_query_granularity(state.query)

            # Apply granularity-specific retrieval parameters
            if granularity_decision:
                params = granularity_decision.retrieval_params
                # Override top_k if not explicitly set by user
                if "top_k" in params:
                    if params["top_k"] != state.top_k:
                        state.retrieval_scalar_overrides_applied = True
                    state.top_k = params["top_k"]
                # Override fts_level
                if "fts_level" in params:
                    state.fts_level = params["fts_level"]
                # Override parent expansion settings
                if params.get("enable_parent_expansion"):
                    state.enable_parent_expansion = True
                    if "parent_context_size" in params:
                        state.parent_context_size = params["parent_context_size"]
                if params.get("include_parent_document"):
                    state.include_parent_document = True
                    if "parent_max_tokens" in params:
                        state.parent_max_tokens = params["parent_max_tokens"]
                # Override multi-vector passages
                if params.get("enable_multi_vector_passages"):
                    state.enable_multi_vector_passages = True
                    if "mv_span_chars" in params:
                        state.mv_span_chars = params["mv_span_chars"]
                    if "mv_stride" in params:
                        state.mv_stride = params["mv_stride"]
                    if "mv_max_spans" in params:
                        state.mv_max_spans = params["mv_max_spans"]

                state.result.metadata["granularity_routing"] = {
                    "query_type": granularity_decision.query_type.value,
                    "granularity": granularity_decision.granularity.value,
                    "confidence": granularity_decision.confidence,
                    "reasoning": granularity_decision.reasoning,
                }

            state.result.timings["granularity_routing"] = time.time() - granularity_start
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            state.result.errors.append(f"Granularity routing failed: {e}")
            up.logger.warning(f"Granularity routing error: {e}")

    state.refresh_retrieval_snapshot_from_current_knobs(
        prefer_current_scalars=state.retrieval_scalar_overrides_applied,
    )

    # ========== INTENT-BASED WEIGHTING (optional) ==========
    if rcfg.retrieval.adaptive_hybrid_weights and rcfg.retrieval.search_mode == "hybrid":
        try:
            local_analysis = state.analysis
            if local_analysis is None and up.QueryAnalyzer and state.query:
                try:
                    qa = up.QueryAnalyzer()
                    local_analysis = qa.analyze_query(state.query)
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    local_analysis = None
            # Conceptual queries favor semantic; specific factual favor keyword
            if local_analysis and getattr(local_analysis, "intent", None) is not None:
                if local_analysis.intent in {
                    getattr(up.QueryIntent, "EXPLORATORY", None),
                    getattr(up.QueryIntent, "DEFINITIONAL", None),
                    getattr(up.QueryIntent, "ANALYTICAL", None),
                    getattr(up.QueryIntent, "PROCEDURAL", None),
                }:
                    state.hybrid_alpha = 0.7
                elif local_analysis.intent in {
                    getattr(up.QueryIntent, "FACTUAL", None),
                    getattr(up.QueryIntent, "COMPARATIVE", None),
                    getattr(up.QueryIntent, "TEMPORAL", None),
                }:
                    state.hybrid_alpha = 0.4
                state.result.metadata["query_intent"] = getattr(
                    local_analysis.intent, "value", str(local_analysis.intent)
                )
            state.result.metadata["adaptive_hybrid_alpha"] = state.hybrid_alpha
        except (AttributeError, TypeError, ValueError):
            pass


async def prepare_hyde_and_temporal_filters(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """HyDE preparation and automatic temporal filters."""
    state.hyde_vector = None
    if state.enable_hyde and up.generate_hypothetical_answer and up.hyde_embed_text:
        try:
            hyde_start = time.time()
            # Read defaults if present
            try:
                from tldw_Server_API.app.core.config import load_and_log_configs

                hyde_config = load_and_log_configs()
                if not isinstance(hyde_config, dict):
                    hyde_config = {}
                raw_provider = hyde_config.get("RAG_HYDE_PROVIDER")
                raw_model = hyde_config.get("RAG_HYDE_MODEL")
                state.hyde_provider = state.hyde_provider or (str(raw_provider).strip() if raw_provider else None)
                state.hyde_model = state.hyde_model or (str(raw_model).strip() if raw_model else None)
            except (ImportError, AttributeError, OSError, TypeError, ValueError):
                pass
            hyde_embedding_metadata: dict[str, Any] = {}
            if rcfg.runtime.credential_runtime is None:
                # Off the loop: a sync LLM call on the loop's thread blocks it and cannot be quota-checked.
                hypo = await asyncio.to_thread(up.generate_hypothetical_answer, state.query, state.hyde_provider, state.hyde_model)
                vec = await up.hyde_embed_text(hypo)
            else:
                hypo = await up.generate_hypothetical_answer_async(
                    state.query,
                    state.hyde_provider,
                    state.hyde_model,
                    credential_runtime=rcfg.runtime.credential_runtime,
                    stage_metadata=hyde_embedding_metadata,
                )
                vec = await up.hyde_embed_text(
                    hypo,
                    credential_runtime=rcfg.runtime.credential_runtime,
                    stage_metadata=hyde_embedding_metadata,
                )
            if hyde_embedding_metadata:
                state.result.metadata["hyde_embedding"] = hyde_embedding_metadata
            if vec:
                state.hyde_vector = vec
                state.result.metadata["hyde_applied"] = True
            state.result.timings["hyde_prep"] = time.time() - hyde_start
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"HyDE prep failed: {e}")

    # ========== AUTO TEMPORAL FILTERS (optional) ==========
    if rcfg.query_prep.auto_temporal_filters:
        try:
            qlower = state.query.lower()
            start_dt = None
            end_dt = None

            now = datetime.utcnow()
            # Relative expressions
            if "yesterday" in qlower:
                start_dt = now - timedelta(days=1)
                end_dt = now
            elif "last week" in qlower or "past week" in qlower:
                start_dt = now - timedelta(days=7)
                end_dt = now
            elif "last month" in qlower:
                # Compute previous calendar month
                y = now.year
                m = now.month - 1 if now.month > 1 else 12
                y = y if now.month > 1 else y - 1
                start_dt = datetime(y, m, 1)
                _, last_day = calendar.monthrange(y, m)
                end_dt = datetime(y, m, last_day, 23, 59, 59)
            elif "past month" in qlower:
                start_dt = now - timedelta(days=30)
                end_dt = now

            # Quarters like Q1 2024
            m_quarter = re.search(r"\bq([1-4])\s*(20\d{2}|19\d{2})\b", qlower)
            if m_quarter:
                qn = int(m_quarter.group(1))
                y = int(m_quarter.group(2))
                qm = {1: 1, 2: 4, 3: 7, 4: 10}[qn]
                start_dt = datetime(y, qm, 1)
                end_month = qm + 2
                _, last_day = calendar.monthrange(y, end_month)
                end_dt = datetime(y, end_month, last_day, 23, 59, 59)

            # Month name + year, e.g., January 2023
            month_names = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
            m_month_year = re.search(
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2}|19\d{2})\b",
                qlower,
            )
            if m_month_year:
                mon = month_names.get(m_month_year.group(1))
                y = int(m_month_year.group(2))
                if mon:
                    start_dt = datetime(y, mon, 1)
                    _, last_day = calendar.monthrange(y, mon)
                    end_dt = datetime(y, mon, last_day, 23, 59, 59)

            # Year-only reference (prefer exact year range)
            m_year = re.search(r"\b(20\d{2}|19\d{2})\b", qlower)
            if m_year and start_dt is None and end_dt is None:
                y = int(m_year.group(1))
                start_dt = datetime(y, 1, 1)
                end_dt = datetime(y, 12, 31, 23, 59, 59)

            if start_dt is None and end_dt is None:
                # Conservative default: last 7 days window when auto filtering is enabled
                start_dt = now - timedelta(days=7)
                end_dt = now

            if start_dt and end_dt:
                state.enable_date_filter = True
                state.date_range = {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
                state.result.metadata["temporal_filter"] = {
                    "start": state.date_range["start"],
                    "end": state.date_range["end"],
                    "source": "auto",
                }
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
