"""retrieval code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import contextlib
import re
import sqlite3
import time
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


async def lookup_retrieval_cache(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Exact then semantic retrieval cache lookup, scoped by the per-owner cache namespace."""
    if state.retrieval_cache_eligible and not state._skip_retrieval_stack and not state._skip_local_retrieval:
        cache_start = time.time()
        cache = state.get_cache_instance()

        if cache:
            get_fn = getattr(cache, "get", None)
            try:
                direct = (
                    await up._invoke_cache_callable(get_fn, state.query)
                    if callable(get_fn)
                    else None
                )
            except (AttributeError, OSError, RuntimeError, TypeError):
                direct = None
            cached_documents = up._extract_cached_documents(direct)
            if cached_documents:
                state.result.cache_hit = True
            else:
                for q in state.expanded_queries:
                    try:
                        find_fn = getattr(cache, "find_similar", None)
                        if not callable(find_fn):
                            break
                        cached_result = await up._invoke_cache_callable(find_fn, q)
                    except (AttributeError, OSError, RuntimeError, TypeError):
                        cached_result = None
                    validated_match = up._validate_semantic_cache_match(cached_result)
                    if validated_match is None:
                        continue
                    cached_query, similarity = validated_match
                    try:
                        similar_payload = (
                            await up._invoke_cache_callable(get_fn, cached_query)
                            if callable(get_fn)
                            else None
                        )
                    except (AttributeError, OSError, RuntimeError, TypeError):
                        similar_payload = None
                    cached_documents = up._extract_cached_documents(similar_payload)
                    if cached_documents:
                        state.result.cache_hit = True
                        state.result.metadata["cache_similarity"] = similarity
                        break

            if state.result.cache_hit:
                state.result.documents = cached_documents
                state.result.generated_answer = None
                state.result.metadata["retrieval_cache_hit"] = True
                state.result.metadata["cached_flag"] = True

        state.result.timings["cache_check"] = time.time() - cache_start
        if state.metrics:
            state.metrics.cache_lookup_time = state.result.timings["cache_check"]
    elif state._skip_retrieval_stack or state._skip_local_retrieval:
        state.result.metadata["cache_bypassed"] = {
            "reason": (
                state._skip_retrieval_reason
                if state._skip_retrieval_stack
                else "classification_local_disabled"
            )
        }


async def prepare_retriever(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Build the multi-database retriever and config, then run the base retrieval."""
    db_paths = state.build_pipeline_db_paths()

    # Initialize retriever (tests may patch this constructor).
    state.retriever = state.build_multi_retriever(db_paths)
    state.source_status_retriever = state.retriever

    # Configure retrieval
    state.config = up.RetrievalConfig(
        max_results=state.retrieval_top_k,
        min_score=state.retrieval_min_score,
        use_fts=(state.retrieval_search_mode in ["fts", "hybrid"]),
        use_vector=(state.retrieval_search_mode in ["vector", "hybrid"]),
        include_metadata=True,
        fts_level=state.fts_level,
        enable_text_late_chunking=rcfg.processing.enable_text_late_chunking,
        chunk_method=rcfg.processing.chunk_method,
        chunk_size=rcfg.processing.chunk_size,
        chunk_overlap=rcfg.processing.chunk_overlap,
        chunk_language=rcfg.processing.chunk_language,
    )
    # Optional date filter
    if state.enable_date_filter and state.date_range and isinstance(state.date_range, dict):
        try:
            start = (
                datetime.fromisoformat(state.date_range.get("start", "")) if state.date_range.get("start") else None
            )
            end = datetime.fromisoformat(state.date_range.get("end", "")) if state.date_range.get("end") else None
            if start and end:
                state.config.date_filter = (start, end)
        except (TypeError, ValueError):
            pass
    # Fallback: use metadata-written temporal filter (auto)
    if getattr(state.config, "date_filter", None) is None:
        tf = state.result.metadata.get("temporal_filter") if isinstance(state.result.metadata, dict) else None
        if isinstance(tf, dict):
            try:
                start_val = tf.get("start")
                end_val = tf.get("end")
                if start_val and end_val:
                    state.config.date_filter = (
                        datetime.fromisoformat(start_val),
                        datetime.fromisoformat(end_val),
                    )
            except (TypeError, ValueError):
                pass

    def _execution_collection_names() -> dict[str, str]:
        user_key = str(rcfg.scope.user_id or rcfg.scope.feedback_user_id or "0").strip() or "0"
        source_names = {
            str(getattr(source, "value", source)).strip().lower()
            for source in state.retrieval_sources
            if str(getattr(source, "value", source)).strip()
        }
        collection_names: dict[str, str] = {
            "media_db": f"user_{user_key}_media_embeddings",
            "notes": f"user_{user_key}_notes_embeddings",
        }
        if {"character_cards", "characters"} & source_names:
            collection_names["character_cards"] = f"user_{user_key}_character_embeddings"
        if {"kanban", "kanban_db"} & source_names:
            collection_names["kanban"] = f"user_{user_key}_kanban_embeddings"
        return collection_names

    def _effective_retrieval_plan_for_query(
        query_text: str,
        *,
        top_k_override: Optional[int] = None,
    ) -> up.RetrievalPlan:
        plan = state.retrieval_plan_for(query_text)
        if plan is None:
            plan = up.RetrievalPlan(
                query=query_text,
                sources=tuple(str(source) for source in state.retrieval_sources),
                search_mode=str(state.retrieval_search_mode),
                top_k=int(state.retrieval_top_k),
                min_score=float(state.retrieval_min_score),
                index_namespace=state.retrieval_index_namespace,
                collection_names=_execution_collection_names(),
            )
        if top_k_override is not None:
            try:
                return replace(plan, top_k=max(1, int(top_k_override)))
            except (TypeError, ValueError):
                return plan
        return plan

    async def _execute_retrieval_variant(
        component: str,
        query_text: str,
        *,
        retrieval_cfg: Optional[Any] = None,
        top_k_override: Optional[int] = None,
    ) -> list[up.Document]:
        if state.latched_optional_provider_failure(component) is not None:
            return []
        try:
            evidence = await state.resilient_call(
                component,
                up.execute_retrieval_phase,
                resolved_request=state.resolved_request,
                retrieval_plan=_effective_retrieval_plan_for_query(
                    query_text,
                    top_k_override=top_k_override,
                ),
                retriever=state.retriever,
                retrieval_config=retrieval_cfg or state.config,
                source_failures=state.source_failures,
                allowed_media_ids=rcfg.scope.include_media_ids,
                allowed_note_ids=rcfg.scope.include_note_ids,
                bypass_provider_failures=rcfg.runtime.credential_runtime is not None,
            )
        except asyncio.TimeoutError:
            # This retrieval attempt did not finish. Keep base
            # evidence, but disclose incomplete source coverage.
            state.source_failures.update(state.resolved_data_sources)
            raise
        except up._RAG_PROVIDER_FAILURES as exc:
            if rcfg.runtime.credential_runtime is None:
                raise
            state.latch_optional_provider_failure(component, exc)
            return []
        evidence_docs = getattr(evidence, "documents", None)
        return list(evidence_docs or [])
    state._execute_retrieval_variant = _execute_retrieval_variant

    effective_retrieval_plan = _effective_retrieval_plan_for_query(state.retrieval_query)

    retrieved_evidence = await state.resilient_call(
        "retrieval",
        up.execute_retrieval_phase,
        resolved_request=state.resolved_request,
        retrieval_plan=effective_retrieval_plan,
        retriever=state.retriever,
        retrieval_config=state.config,
        source_failures=state.source_failures,
        allowed_media_ids=rcfg.scope.include_media_ids,
        allowed_note_ids=rcfg.scope.include_note_ids,
        bypass_provider_failures=rcfg.runtime.credential_runtime is not None,
    )
    state.base_retrieval_executed = True
    state.documents = list(retrieved_evidence.documents)


async def merge_hyde_and_variants(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Media DB fallback, HyDE merge and query-variant expansion retrieval."""
    if (
        (not state.documents)
        and up._scope_includes_data_source(state.resolved_data_sources, up.DataSource.MEDIA_DB)
        and (state.media_db_path or rcfg.runtime.media_db is not None)
        and state.retrieval_search_mode in ("fts", "hybrid")
    ):
        try:
            from ..database_retrievers import MediaDBRetriever as _MDBR, RetrievalConfig as _RCfg  # noqa: I001 - moved from unified_pipeline.py (TASK-13950.24)

            fb_cfg = _RCfg(
                max_results=state.retrieval_top_k,
                min_score=state.retrieval_min_score,
                use_fts=True,
                use_vector=False,
                include_metadata=True,
                fts_level=state.fts_level,
            )
            fb_retriever = _MDBR(
                db_path=state.media_db_path,
                config=fb_cfg,
                user_id=str(rcfg.scope.user_id or "0"),
                media_db=rcfg.runtime.media_db,
            )
            fallback_docs = await fb_retriever.retrieve(
                query=state.retrieval_query,
                media_type=None,
                allowed_media_ids=rcfg.scope.include_media_ids,
            )
            if fallback_docs:
                state.documents = fallback_docs
                if isinstance(state.result.metadata, dict):
                    state.result.metadata.setdefault("fallbacks", {})
                    state.result.metadata["fallbacks"]["media_db_fts"] = True
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
            sqlite3.Error,
        ) as _fb_err:
            state.result.errors.append(f"Media DB fallback retrieval failed: {str(_fb_err)}")

    # Optionally run HyDE-enhanced media retrieval and merge
    if (
        state.enable_hyde
        and state.hyde_vector
        and state.retrieval_search_mode == "hybrid"
        and up._scope_includes_data_source(state.resolved_data_sources, up.DataSource.MEDIA_DB)
    ):
        try:
            media_retr = state.retriever.retrievers.get(up.DataSource.MEDIA_DB)
            if media_retr and hasattr(media_retr, "retrieve_hybrid"):
                hyde_docs = await media_retr.retrieve_hybrid(
                    query=state.retrieval_query,
                    alpha=state.hybrid_alpha,
                    index_namespace=state.retrieval_index_namespace,
                    query_vector=state.hyde_vector,
                )
                by_id: dict[str, up.Document] = {d.id: d for d in state.documents}
                for d in hyde_docs:
                    cur = by_id.get(d.id)
                    if cur is None or float(getattr(d, "score", 0.0)) > float(
                        getattr(cur, "score", 0.0)
                    ):
                        by_id[d.id] = d
                state.documents = sorted(by_id.values(), key=lambda x: getattr(x, "score", 0.0), reverse=True)
                state.result.metadata["hyde_merged_count"] = len(hyde_docs)
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"HyDE retrieval merge failed: {e}")

    # Optional: expand retrieval across query variants
    if state.expand_query and state.expanded_queries and len(state.expanded_queries) > 1:
        exp_start = time.time()
        try:
            extra_queries = [q for q in state.expanded_queries if q != state.retrieval_query]
            exp_docs: list[up.Document] = []
            for eq in extra_queries:
                try:
                    try:
                        expansion_top_k = max(1, int(state.retrieval_top_k or 1))
                        exp_cfg = replace(state.config, max_results=expansion_top_k)
                    except (TypeError, ValueError):
                        expansion_top_k = None
                        exp_cfg = state.config
                    eq_docs = await state._execute_retrieval_variant(
                        "retrieval_expansion",
                        eq,
                        retrieval_cfg=exp_cfg,
                        top_k_override=expansion_top_k,
                    )
                    if eq_docs:
                        exp_docs.extend(eq_docs)
                except (
                    AttributeError,
                    ConnectionError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    asyncio.TimeoutError,
                ) as _eq_err:
                    state.result.errors.append(f"Query expansion retrieval failed: {eq}: {str(_eq_err)}")
                    continue

            if exp_docs:
                by_id = {d.id: d for d in state.documents}
                added = 0
                for d in exp_docs:
                    cur = by_id.get(d.id)
                    if cur is None or float(getattr(d, "score", 0.0)) > float(
                        getattr(cur, "score", 0.0)
                    ):
                        if cur is None:
                            added += 1
                        by_id[d.id] = d
                state.documents = sorted(
                    by_id.values(),
                    key=lambda x: getattr(x, "score", 0.0),
                    reverse=True,
                )
                state.result.metadata.setdefault("query_expansion", {})
                state.result.metadata["query_expansion"]["retrieval_queries"] = len(extra_queries)
                state.result.metadata["query_expansion"]["retrieval_added"] = int(added)
            state.result.timings["query_expansion_retrieval"] = time.time() - exp_start
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as _exp_err:
            state.result.errors.append(f"Query expansion retrieval failed: {str(_exp_err)}")


async def run_prf(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Optional PRF second-pass retrieval to fill remaining slots."""
    if (
        rcfg.query_prep.enable_prf
        and up.apply_prf
        and up.PRFConfig
        and state.result.documents
        and len(state.result.documents) < state.retrieval_top_k
    ):
        try:
            prf_cfg = up.PRFConfig(
                max_terms=int(rcfg.query_prep.prf_terms or 0),
                sources=rcfg.query_prep.prf_sources or ["keywords", "entities", "numbers"],
                alpha=float(rcfg.query_prep.prf_alpha or 0.0),
                top_n=int(rcfg.query_prep.prf_top_n or 0),
            )
            prf_query, prf_meta = await up.apply_prf(
                state.retrieval_query,
                state.result.documents,
                prf_cfg,
            )
            state.result.metadata.setdefault("prf", {})
            state.result.metadata["prf"].update(prf_meta)

            # Only perform a second pass when PRF is enabled and query changed
            if prf_meta.get("enabled") and prf_query and prf_query != state.retrieval_query:
                remaining_slots = max(0, state.retrieval_top_k - len(state.result.documents))
                if remaining_slots > 0:
                    prf_docs = await state._execute_retrieval_variant(
                        "retrieval_prf",
                        prf_query,
                        top_k_override=remaining_slots,
                    )
                    prf_docs = prf_docs or []
                    existing_ids = {d.id for d in state.result.documents}
                    added = 0
                    for d in prf_docs:
                        if d.id not in existing_ids:
                            state.result.documents.append(d)
                            existing_ids.add(d.id)
                            added += 1
                            if len(state.result.documents) >= state.retrieval_top_k:
                                break
                    state.result.metadata["prf"]["second_pass_performed"] = True
                    state.result.metadata["prf"]["second_pass_added"] = int(added)
                else:
                    state.result.metadata["prf"]["second_pass_performed"] = False
                    state.result.metadata["prf"]["second_pass_added"] = 0
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as _prf_err:
            state.result.errors.append(f"PRF second-pass retrieval failed: {str(_prf_err)}")


async def run_decomposition(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Guided query decomposition with bounded concurrent sub-query retrieval."""
    if rcfg.retrieval.enable_query_decomposition:
        decomp_start = time.time()
        try:
            # Prefer agentic planner-style decomposition when available and
            # the query appears complex/compound; otherwise fall back to a
            # lightweight heuristic split.
            q_norm = (state.query or "").strip()
            subqueries: list[str] = []
            used_agentic = False

            # Use QueryAnalyzer (if available) to detect complex queries that
            # benefit from decomposition (comparative/causal/temporal/analytical).
            intent_val = state.analysis_intent_val
            comp_val = state.analysis_complexity_val
            # Try delegating to agentic-style decomposition when the query
            # is complex and of a multi-part intent.
            multi_intents = {"comparative", "causal", "analytical", "temporal"}
            if intent_val in multi_intents and comp_val == "complex":
                try:
                    from ..agentic_chunker import AgenticConfig as _ACfg, _decompose_query as _agentic_decompose  # noqa: I001 - moved from unified_pipeline.py (TASK-13950.24)

                    subgoal_max = rcfg.retrieval.max_subqueries if rcfg.retrieval.max_subqueries is not None else 3
                    acfg = _ACfg(enable_query_decomposition=True, subgoal_max=int(subgoal_max))
                    subqueries = _agentic_decompose(q_norm, acfg) or []
                    used_agentic = True
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                    used_agentic = False

            if not used_agentic:
                # Lightweight heuristic decomposition: split on common coordinators/punctuation.
                if q_norm:
                    parts = re.split(r"\b(?:and then|then|and|,|;|\?)\b", q_norm, flags=re.IGNORECASE)
                    subqueries = [p.strip() for p in parts if p and len(p.strip()) >= 3]
                if not subqueries:
                    subqueries = [q_norm] if q_norm else []

            # Ensure primary query is the first entry
            if q_norm:
                seen_norm = set()
                ordered = [q_norm]
                seen_norm.add(q_norm.lower())
                for sq in subqueries:
                    sq_norm = (sq or "").strip()
                    if not sq_norm:
                        continue
                    key = sq_norm.lower()
                    if key in seen_norm:
                        continue
                    seen_norm.add(key)
                    ordered.append(sq_norm)
                subqueries = ordered

            # Apply max_subqueries cap if provided (includes primary query implicitly)
            try:
                max_sub = int(rcfg.retrieval.max_subqueries or 0)
            except (TypeError, ValueError):
                max_sub = 0
            if max_sub and len(subqueries) > max_sub:
                subqueries = subqueries[:max_sub]

            meta_decomp: dict[str, Any] = {
                "enabled": True,
                "subqueries": [],
            }
            if intent_val is not None:
                meta_decomp["intent"] = intent_val
            if comp_val is not None:
                meta_decomp["complexity"] = comp_val
            time_budget = float(rcfg.retrieval.subquery_time_budget_sec) if rcfg.retrieval.subquery_time_budget_sec else None
            try:
                doc_budget = int(rcfg.retrieval.subquery_doc_budget) if rcfg.retrieval.subquery_doc_budget is not None else None
            except (TypeError, ValueError):
                doc_budget = None

            base_ids = {d.id for d in state.result.documents}
            total_added = 0

            # Only run additional retrievals for secondary subqueries
            if len(subqueries) > 1:
                try:
                    subquery_max_results = max(1, int(state.retrieval_top_k or 1))
                    if doc_budget is not None:
                        subquery_max_results = max(1, min(subquery_max_results, int(doc_budget)))
                except (TypeError, ValueError):
                    subquery_max_results = max(1, int(state.retrieval_top_k or 1))

                async def _fetch_subquery(sq: str) -> list[up.Document]:
                    try:
                        sq_cfg = replace(state.config, max_results=subquery_max_results)
                    except TypeError:
                        sq_cfg = state.config
                    res = await state._execute_retrieval_variant(
                        "retrieval_decomposition",
                        sq,
                        retrieval_cfg=sq_cfg,
                        top_k_override=subquery_max_results,
                    )
                    return res if isinstance(res, list) else []

                subquery_results: dict[str, Any] = {}
                subqueries_to_run = (
                    []
                    if state.latched_optional_provider_failure(
                        "retrieval_decomposition"
                    )
                    is not None
                    else list(subqueries[1:])
                )
                try:
                    max_workers = max(1, int(rcfg.retrieval.subquery_max_concurrency or 1))
                except (TypeError, ValueError):
                    max_workers = 3
                max_workers = min(max_workers, len(subqueries_to_run)) or 1
                sem = asyncio.Semaphore(max_workers)

                async def _run_subquery(sq: str) -> None:
                    async with sem:
                        subquery_results[sq] = await _fetch_subquery(sq)

                tasks = [asyncio.create_task(_run_subquery(sq)) for sq in subqueries_to_run]
                if tasks:
                    try:
                        if time_budget is not None:
                            remaining = max(0.0, time_budget - (time.time() - decomp_start))
                            done, pending = await asyncio.wait(tasks, timeout=remaining)
                            if pending:
                                state.source_failures.update(state.resolved_data_sources)
                        else:
                            done, _ = await asyncio.wait(tasks)
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    for task in done:
                        try:
                            task.result()
                        except (
                            AttributeError,
                            ConnectionError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            asyncio.TimeoutError,
                        ) as _sq_err:
                            state.result.errors.append(f"Decomposition subquery retrieval failed: {_sq_err}")

                for sq in subqueries_to_run:
                    if time_budget is not None and (time.time() - decomp_start) >= time_budget:
                        break
                    if doc_budget is not None and total_added >= doc_budget:
                        break
                    sq_docs = subquery_results.get(sq) or []
                    added_ids: list[str] = []
                    for d in sq_docs:
                        if d.id not in base_ids:
                            state.result.documents.append(d)
                            base_ids.add(d.id)
                            added_ids.append(d.id)
                            total_added += 1
                            if doc_budget is not None and total_added >= doc_budget:
                                break
                    meta_decomp["subqueries"].append(
                        {
                            "query": sq,
                            "added_doc_ids": added_ids,
                        }
                    )
                    if doc_budget is not None and total_added >= doc_budget:
                        break

                # Re-sort docs by score and cap to top_k
                try:
                    state.result.documents = sorted(
                        state.result.documents,
                        key=lambda d: getattr(d, "score", 0.0),
                        reverse=True,
                    )[:state.retrieval_top_k]
                except (TypeError, ValueError):
                    # Fallback: leave documents in current order
                    pass

            meta_decomp["total_added"] = int(total_added)
            meta_decomp["elapsed_sec"] = round(time.time() - decomp_start, 6)
            meta_decomp["time_budget_sec"] = float(time_budget) if time_budget is not None else None
            meta_decomp["doc_budget"] = int(doc_budget) if doc_budget is not None else None
            state.result.metadata["decomposition"] = meta_decomp
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as _dec_err:
            state.result.errors.append(f"Query decomposition failed: {str(_dec_err)}")


def record_retrieval_diagnostics(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Workspace filtering, retrieval guidance, sources and timing metrics."""
    state.apply_workspace_filtering_to_result()

    # Attach retrieval guidance prompt in metadata for downstream awareness/debugging
    try:
        _rg = up.load_prompt("rag", "retrieval_guidance")
        if _rg:
            state.result.metadata["retrieval_guidance"] = _rg
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        pass
    state.result.metadata["sources_searched"] = list(state.retrieval_sources)
    state.result.metadata["documents_retrieved"] = len(state.documents)

    state.result.timings["retrieval"] = time.time() - state.retrieval_start
    # Record phase duration with difficulty label
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram

        def _difficulty(docs: list) -> str:
            try:
                if not docs:
                    difficulty = "hard"
                else:
                    high = sum(
                        1 for d in docs if float(getattr(d, "score", 0.0)) >= max(rcfg.retrieval.min_score, 0.3)
                    )
                    if high >= max(3, int(0.3 * len(docs))):
                        difficulty = "easy"
                    elif high >= 1:
                        difficulty = "medium"
                    else:
                        difficulty = "hard"
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return "unknown"
            return difficulty

        observe_histogram(
            "rag_phase_duration_seconds",
            state.result.timings["retrieval"],
            labels={"phase": "retrieval", "difficulty": _difficulty(state.result.documents or [])},
        )
        # Also attach difficulty as OTEL attribute if span is active
        if state._otel_span is not None:
            try:
                state._otel_span.set_attribute("rag.query_difficulty", _difficulty(state.result.documents or []))
                state._otel_span.set_attribute("rag.doc_count", int(len(state.result.documents or [])))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    if state.metrics:
        state.metrics.retrieval_time = state.result.timings["retrieval"]


async def run_retrieval(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Document retrieval under the rag.retrieval span."""
    if not state.result.cache_hit and not state._skip_retrieval_stack and not state._skip_local_retrieval:
        state.retrieval_start = time.time()
        try:
            # --- OTEL: retrieval span ---
            _otel_cm = None
            state._otel_span = None
            if rcfg.observability.enable_observability and up.get_telemetry_manager:
                try:
                    _tm = up.get_telemetry_manager()
                    _tr = _tm.get_tracer("tldw.rag")
                    _attrs = {
                        "rag.phase": "retrieval",
                        "rag.search_mode": str(state.retrieval_search_mode),
                        "rag.top_k": int(state.retrieval_top_k or 0),
                        "rag.index_namespace": str(state.retrieval_index_namespace or ""),
                    }
                    _otel_cm = _tr.start_as_current_span("rag.retrieval")
                    state._otel_span = _otel_cm.__enter__()
                    for _k, _v in _attrs.items():
                        with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                            state._otel_span.set_attribute(_k, _v)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    _otel_cm = None
                    state._otel_span = None
            if up.MultiDatabaseRetriever and up.RetrievalConfig:
                # Set up database paths
                await prepare_retriever(rcfg, state)

                # Fallback: if no documents were retrieved via MultiDatabaseRetriever,
                # perform a direct Media DB FTS-only search. This guards against
                # configuration or adapter issues that can cause hybrid retrieval
                # to silently return an empty set even when media is present.
                await merge_hyde_and_variants(rcfg, state)

                state.result.documents = state.documents
                state.apply_workspace_filtering_to_result()
                # Optional PRF second-pass retrieval to fill remaining slots
                await run_prf(rcfg, state)

                # Optional: guided query decomposition to broaden recall for multi-part queries
                await run_decomposition(rcfg, state)

                record_retrieval_diagnostics(rcfg, state)

        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            Exception,  # noqa: BLE001 - moved from unified_pipeline.py (TASK-13950.24)
            TypeError,
            ValueError,
            asyncio.TimeoutError,
            sqlite3.Error,
        ) as e:
            if rcfg.runtime.credential_runtime is not None and isinstance(e, up._RAG_PROVIDER_FAILURES):
                raise
            state.result.errors.append("document_retrieval_failed")
            up.logger.error("Document retrieval failed")
            # Sample payload exemplar on retrieval failure
            try:
                from ..payload_exemplars import maybe_record_exemplar

                maybe_record_exemplar(
                    query=state.query,
                    documents=state.result.documents or [],
                    answer=state.result.generated_answer or "",
                    reason="retrieval_error",
                    user_id=rcfg.scope.user_id,
                    namespace=rcfg.retrieval.index_namespace,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                pass

            # On retrieval failure, attempt a best-effort Media DB FTS fallback.
            # This is especially important in local/test environments where
            # vector stores or adapters may be misconfigured but the Media DB
            # itself contains the uploaded content.
            if (
                (not state.result.documents)
                and up._scope_includes_data_source(state.resolved_data_sources, up.DataSource.MEDIA_DB)
                and (state.media_db_path or rcfg.runtime.media_db is not None)
                and state.retrieval_search_mode in ("fts", "hybrid")
            ):
                try:
                    from ..database_retrievers import MediaDBRetriever as _MDBR, RetrievalConfig as _RCfg  # noqa: I001 - moved from unified_pipeline.py (TASK-13950.24)

                    fb_cfg = _RCfg(
                        max_results=state.top_k,
                        min_score=rcfg.retrieval.min_score,
                        use_fts=True,
                        use_vector=False,
                        include_metadata=True,
                        fts_level=state.fts_level,
                    )
                    fb_retriever = _MDBR(
                        db_path=state.media_db_path,
                        config=fb_cfg,
                        user_id=str(rcfg.scope.user_id or "0"),
                        media_db=rcfg.runtime.media_db,
                    )
                    fallback_docs = await fb_retriever.retrieve(
                        query=state.query,
                        media_type=None,
                        allowed_media_ids=rcfg.scope.include_media_ids,
                    )
                    if fallback_docs:
                        state.result.documents = fallback_docs
                        if isinstance(state.result.metadata, dict):
                            state.result.metadata.setdefault("fallbacks", {})
                            state.result.metadata["fallbacks"]["media_db_fts_on_error"] = True
                except (
                    AttributeError,
                    ConnectionError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    asyncio.TimeoutError,
                    sqlite3.Error,
                ):
                    state.result.errors.append("media_db_fallback_failed")
                    up.logger.error("Media DB fallback retrieval failed")
        finally:
            # Ensure OTEL span is closed
            if _otel_cm is not None:
                with contextlib.suppress(ImportError, RuntimeError, TypeError, ValueError):
                    _otel_cm.__exit__(None, None, None)
    elif state._skip_retrieval_stack:
        state.result.timings["retrieval"] = 0.0
        state.result.metadata["retrieval_bypassed"] = {
            "reason": state._skip_retrieval_reason or "retrieval_bypassed",
            "documents_preserved": int(len(state.result.documents or [])),
        }
    elif state._skip_local_retrieval:
        state.result.timings["retrieval"] = 0.0
        state.result.metadata["retrieval_bypassed"] = {
            "reason": "classification_local_disabled",
            "documents_preserved": int(len(state.result.documents or [])),
        }
