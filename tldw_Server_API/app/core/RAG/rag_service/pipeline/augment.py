"""augment code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import time
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


async def augment_passages(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Multi-vector passages and the numeric/table-aware boost (pre-rerank)."""
    if state.enable_multi_vector_passages and state.result.documents:
        mv_start = time.time()
        try:
            used_precomputed = False
            if rcfg.retrieval.enable_precomputed_spans and up.apply_precomputed_spans and up.PrecomputedSpanConfig:
                try:
                    pcfg = up.PrecomputedSpanConfig()
                    pre_docs = await up.apply_precomputed_spans(
                        query=state.query,
                        documents=state.result.documents,
                        config=pcfg,
                        user_id=rcfg.scope.user_id,
                    )
                    state.result.metadata.setdefault("multi_vector", {})
                    # When implementation is available, pre_docs can override documents
                    if pre_docs:
                        state.result.documents = pre_docs[:state.top_k]
                        state.result.metadata["multi_vector"]["precomputed_spans"] = True
                        used_precomputed = True  # noqa: F841 - dead store moved from unified_pipeline.py (TASK-13950.24)
                    else:
                        state.result.metadata["multi_vector"]["precomputed_spans"] = False
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    # If precomputed path fails, fall back silently to on-the-fly spans
                    state.result.metadata.setdefault("multi_vector", {})
                    state.result.metadata["multi_vector"]["precomputed_spans"] = False

            if up.apply_multi_vector_passages and up.MultiVectorConfig:
                cfg = up.MultiVectorConfig(
                    span_chars=int(state.mv_span_chars or 300),
                    stride=int(state.mv_stride or 150),
                    max_spans_per_doc=int(state.mv_max_spans or 8),
                    flatten_to_spans=bool(rcfg.retrieval.mv_flatten_to_spans or False),
                )
                multi_vector_embedding_metadata: dict[str, Any] = {}
                multi_vector_kwargs: dict[str, Any] = {
                    "query": state.query,
                    "documents": state.result.documents,
                    "config": cfg,
                    "user_id": rcfg.scope.user_id,
                }
                if rcfg.runtime.credential_runtime is not None:
                    multi_vector_kwargs.update(
                        credential_runtime=rcfg.runtime.credential_runtime,
                        stage_metadata=multi_vector_embedding_metadata,
                    )
                mv_docs = await up.apply_multi_vector_passages(**multi_vector_kwargs)
                if multi_vector_embedding_metadata:
                    state.result.metadata.setdefault("multi_vector", {}).update(
                        multi_vector_embedding_metadata
                    )
                if mv_docs:
                    state.result.documents = mv_docs[:state.top_k]
                    state.result.metadata.setdefault("multi_vector", {})
                    state.result.metadata["multi_vector"].update(
                        {
                            "enabled": True,
                            "span_chars": cfg.span_chars,
                            "stride": cfg.stride,
                            "max_spans_per_doc": cfg.max_spans_per_doc,
                            "flattened": cfg.flatten_to_spans,
                        }
                    )
            else:
                state.result.errors.append("Multi-vector module not available")
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Multi-vector passages failed: {e}")
        finally:
            state.result.timings["multi_vector"] = time.time() - mv_start
            try:
                from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram

                observe_histogram(
                    "rag_phase_duration_seconds",
                    state.result.timings["multi_vector"],
                    labels={"phase": "multi_vector", "difficulty": str(state.result.metadata.get("query_intent", "na"))},
                )
            except (ImportError, RuntimeError, TypeError, ValueError):
                pass

    # ========== NUMERIC/TABLE-AWARE BOOST (optional, pre-rerank) ==========
    # Record the metadata block even when no documents are retrieved so tests can
    # confirm the feature path was considered for numeric queries.
    if rcfg.retrieval.enable_numeric_table_boost:
        try:
            import re as _re

            q_has_num = bool(_re.search(r"\d", state.query)) or bool(
                _re.search(
                    r"\b(percent|percentage|million|billion|thousand|\$|usd|eur|kg|g|lb|%|k|m|b)\b", state.query, _re.I
                )
            )
        except (TypeError, ValueError):
            q_has_num = False
        if q_has_num:
            affected = 0
            if state.result.documents:
                for d in state.result.documents:
                    try:
                        md = getattr(d, "metadata", None) or {}
                        chunk_type = str(md.get("chunk_type", "")).lower()
                        text = getattr(d, "content", "") or ""
                        numbers = sum(1 for _ in _re.finditer(r"\d", text))
                        looks_table = (chunk_type == "table") or (text.count("|") >= 3) or ("\t" in text)
                        if looks_table or numbers >= 6:
                            score_val = float(getattr(d, "score", 0.0) or 0.0)
                            # modest boost within [0,1]
                            d.score = min(1.0, score_val * 1.1 + 0.02)
                            md["numeric_table_boost"] = True
                            d.metadata = md
                            affected += 1
                    except (AttributeError, TypeError, ValueError):
                        continue
            # Always emit the metadata block when numeric intent is detected
            state.result.metadata["numeric_table_boost"] = {"enabled": True, "affected": int(affected)}


async def run_gap_analysis(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Gap analysis and follow-ups, then the keyword filter."""
    if rcfg.retrieval.enable_gap_analysis and state.result.documents:
        try:
            ga_start = time.time()
            followups: list[str] = []
            # Try a lightweight LLM to propose follow-ups
            try:
                from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze as llm_analyze

                # Determine default provider/model from config if available
                try:
                    from tldw_Server_API.app.core.config import load_and_log_configs

                    _cfg = load_and_log_configs() or {}
                    from ..generation_defaults import resolve_generation_defaults

                    _prov, _model = resolve_generation_defaults(_cfg)
                except (ImportError, AttributeError, OSError, TypeError, ValueError):
                    _prov, _model = "openai", "gpt-4o-mini"
                prompt = (
                    "You help a search system identify missing information.\n"
                    "Given the user query and several retrieved snippets, propose up to 2 concise follow-up search queries "
                    "that would likely fill important gaps. Return ONLY a JSON array of strings.\n\n"
                    f"Query: {state.query}\n\nSnippets:\n"
                )
                for d in state.result.documents[:5]:
                    snippet = (d.content or "")[:300].replace("\n", " ")
                    prompt += f"- {snippet}\n"
                prompt += "\nJSON:"
                gap_handle = None
                if rcfg.runtime.credential_runtime is not None:
                    gap_handle = await rcfg.runtime.credential_runtime.resolve(
                        _prov,
                        model=_model,
                    )
                gap_stream_state = {"content": False}

                def _call_gap_analyzer() -> Any:
                    response = llm_analyze(
                        api_name=_prov,
                        input_data=prompt,
                        custom_prompt_arg=None,
                        model_override=_model,
                        api_key=(gap_handle.api_key if gap_handle is not None else None),
                        app_config=(gap_handle.app_config if gap_handle is not None else None),
                        credentials_resolved=gap_handle is not None,
                        provider_credentials=gap_handle,
                        raise_on_error=gap_handle is not None,
                    )
                    return up._consume_bound_sgl_response(
                        response,
                        _prov,
                        on_content=lambda: gap_stream_state.__setitem__(
                            "content",
                            True,
                        ),
                        fail_closed=gap_handle is not None,
                    )

                async def _run_gap_analyzer() -> Any:
                    gap_completed = False
                    try:
                        value = await up.await_bounded_sync_call(
                            _call_gap_analyzer,
                            pool=up.SYNC_ADAPTER_CALL_POOL,
                            exhaustion_message=(
                                "RAG optional adapter capacity is exhausted"
                            ),
                        )
                        gap_completed = True
                        return value
                    finally:
                        if gap_handle is not None and (
                            gap_completed or gap_stream_state["content"]
                        ):
                            await rcfg.runtime.credential_runtime.mark_used(gap_handle)

                gap_operation = _run_gap_analyzer()
                llm_out = (
                    await up.await_owned_worker(gap_operation)
                    if gap_handle is not None
                    else await gap_operation
                )
                if isinstance(llm_out, str):
                    try:
                        parsed = up.parse_structured_output(
                            llm_out,
                            options=up.StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
                        )
                        if isinstance(parsed, list):
                            followups = [q for q in parsed if isinstance(q, str) and q.strip()]
                        elif isinstance(parsed, dict):
                            wrapped_followups = parsed.get("followups") or parsed.get("suggestions")
                            if isinstance(wrapped_followups, list):
                                followups = [q for q in wrapped_followups if isinstance(q, str) and q.strip()]
                    except (up.StructuredOutputParseError, TypeError, ValueError):
                        followups = [s.strip("- ") for s in llm_out.splitlines() if s.strip()]
            except (up.ByokResolutionError, up.SummaryProviderError, up.LLMQuotaExceededError) as exc:
                state.result.metadata["gap_analysis"] = {
                    "failure_code": (
                        exc.code
                        if isinstance(exc, up.ByokResolutionError)
                        else "provider_unavailable"
                    ),
                    "verification_available": False,
                }
                followups = []
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ):
                if rcfg.runtime.credential_runtime is not None:
                    state.result.metadata["gap_analysis"] = {
                        "failure_code": "provider_unavailable",
                        "verification_available": False,
                    }
                followups = [f"detailed {state.query}", f"examples {state.query}"]
            followups = [q for q in followups if isinstance(q, str) and q.strip()][:rcfg.retrieval.max_followup_searches]
            if followups:
                if state.latched_optional_provider_failure("retrieval_followup") is not None:
                    follow_results = []
                else:
                    # Calls already running when one fails finish normally; later
                    # optional callbacks observe the request-scoped latch.
                    followup_coroutines = [
                        state._execute_retrieval_variant("retrieval_followup", fq)
                        for fq in followups
                    ]
                    try:
                        follow_results = await asyncio.gather(*followup_coroutines)
                    except (
                        ConnectionError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        asyncio.TimeoutError,
                    ):
                        follow_results = []
                # Merge by id, keep higher score
                merged = {d.id: d for d in state.result.documents}
                for lst in follow_results:
                    for d in lst or []:
                        prev = merged.get(d.id)
                        if prev is None or float(getattr(d, "score", 0.0)) > float(getattr(prev, "score", 0.0)):
                            merged[d.id] = d
                state.result.documents = sorted(merged.values(), key=lambda x: getattr(x, "score", 0.0), reverse=True)[
                    :state.top_k
                ]
                state.result.metadata["followups"] = followups
            state.result.timings["gap_analysis"] = time.time() - ga_start
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Gap analysis failed: {e}")

    # ========== KEYWORD FILTERING ==========
    if rcfg.scope.keyword_filter and state.result.documents:
        filter_start = time.time()
        filtered_docs = []
        for doc in state.result.documents:
            content_lower = doc.content.lower()
            if any(keyword.lower() in content_lower for keyword in rcfg.scope.keyword_filter):
                filtered_docs.append(doc)

        state.result.metadata["pre_filter_count"] = len(state.result.documents)
        state.result.documents = filtered_docs
        state.result.metadata["post_filter_count"] = len(filtered_docs)
        state.result.timings["keyword_filter"] = time.time() - filter_start
