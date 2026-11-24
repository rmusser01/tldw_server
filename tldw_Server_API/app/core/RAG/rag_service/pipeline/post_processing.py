"""post_processing code moved from unified_pipeline.py (TASK-13950.24)."""

from types import SimpleNamespace
from typing import Any
import asyncio
import contextlib
import hashlib
import inspect
import time
import uuid
from typing import TYPE_CHECKING
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up
from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


async def finish_answer(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Evidence-chain refresh, utility grading, user feedback, result highlighting, cost tracking, workspace filtering."""
    # ========== POST-GENERATION EVIDENCE CHAINS UPDATE ==========
    # Now that we have the generated answer, rebuild evidence chains with claim extraction
    if rcfg.citations.enable_evidence_chains and state.result.generated_answer and state.result.documents and up.EvidenceChainBuilder:
        try:
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

            # Rebuild chains with the generated answer for claim extraction
            evidence_chain_result = await chain_builder.build_chains(
                query=state.query,
                documents=state.result.documents,
                generated_answer=state.result.generated_answer,
            )

            if evidence_chain_result:
                # Update metadata with full chain information
                chains_data = []
                for chain in evidence_chain_result.chains:
                    chains_data.append(
                        {
                            "hop_count": chain.hop_count,
                            "chain_confidence": chain.chain_confidence,
                            "source_documents": chain.get_source_documents(),
                            "root_claims": chain.root_claims,
                            "nodes_count": len(chain.nodes),
                        }
                    )

                state.result.metadata["evidence_chains"] = {
                    "total_chains": len(evidence_chain_result.chains),
                    "overall_confidence": evidence_chain_result.overall_confidence,
                    "multi_hop_detected": evidence_chain_result.multi_hop_detected,
                    "total_claims": evidence_chain_result.metadata.get("total_claims", 0),
                    "supported_claims": evidence_chain_result.metadata.get("supported_claims", 0),
                    "chains": chains_data[:5],  # Include top 5 chains
                    **(
                        {
                            key: evidence_chain_result.metadata[key]
                            for key in ("verification_available", "failure_code")
                            if key in evidence_chain_result.metadata
                        }
                    ),
                }

                # Optionally include full chain data for debugging
                if rcfg.observability.debug_mode:
                    state.result.metadata["evidence_chains_full"] = evidence_chain_result.to_dict()

        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Post-generation evidence chain update failed: {e}")
            up.logger.debug(f"Evidence chain update error: {e}")

    # ========== UTILITY GRADING (Self-Correcting RAG Stage 6) ==========
    # Rate response usefulness independent of factual grounding
    try:
        if rcfg.self_correct.enable_utility_grading and state.result.generated_answer and up.grade_utility:
            with up.otel_span("rag.utility_grading"):
                ug_result, ug_meta = await up.grade_utility(
                    query=state.query,
                    answer=state.result.generated_answer,
                    provider=rcfg.self_correct.utility_grading_provider,
                    model=rcfg.self_correct.utility_grading_model,
                    timeout_sec=rcfg.self_correct.utility_grading_timeout_sec,
                    credential_runtime=rcfg.runtime.credential_runtime,
                )
                state.result.metadata["utility_grade"] = ug_meta
    except (
        AttributeError,
        ConnectionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        asyncio.TimeoutError,
    ) as e:
        state.result.errors.append(f"Utility grading failed: {str(e)}")

    # ========== USER FEEDBACK ==========
    if rcfg.observability.collect_feedback:
        feedback_start = time.time()
        try:
            if up.UnifiedFeedbackSystem:
                collector = up.UnifiedFeedbackSystem()
                state.result.feedback_id = str(uuid.uuid4())
                state.result.metadata["feedback_enabled"] = True

                # Apply feedback boost if requested
                if rcfg.observability.apply_feedback_boost and state.result.documents:
                    try:
                        if up.UserPersonalizationStore:
                            store = up.UserPersonalizationStore(rcfg.scope.feedback_user_id or rcfg.scope.user_id)
                            state.result.documents = store.boost_documents(state.result.documents, corpus=rcfg.retrieval.index_namespace)
                    except ValueError as exc:
                        up.logger.debug(
                            f"Personalization boost disabled for user_id={rcfg.scope.feedback_user_id or rcfg.scope.user_id}: {exc}"
                        )
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                # Record anonymized search analytics
                with contextlib.suppress(
                    AttributeError,
                    ConnectionError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    asyncio.TimeoutError,
                ):
                    await collector.record_search(
                        query=state.query,
                        results_count=len(state.result.documents or []),
                        cache_hit=bool(state.result.cache_hit),
                        latency_ms=(time.time() - state.start_time) * 1000.0,
                    )

                state.result.timings["feedback"] = time.time() - feedback_start

        except ImportError:
            state.result.errors.append("Feedback module not available")
            up.logger.warning("Feedback requested but module not available")
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Feedback system failed: {str(e)}")
            up.logger.error(f"Feedback error: {e}")

    # ========== RESULT HIGHLIGHTING ==========
    if rcfg.observability.highlight_results and state.result.documents:
        highlight_start = time.time()
        try:
            if up.highlight_func:
                highlight_context = SimpleNamespace(
                    config={"highlighting": {"enabled": True}},
                    query=state.query if rcfg.observability.highlight_query_terms else "",
                    documents=state.result.documents,
                )
                highlighted = up.highlight_func(highlight_context)
                highlighted_context = await highlighted if inspect.isawaitable(highlighted) else highlighted
                if getattr(highlighted_context, "documents", None) is not None:
                    state.result.documents = highlighted_context.documents

                state.result.timings["highlighting"] = time.time() - highlight_start

        except ImportError:
            state.result.errors.append("Highlighting module not available")
            up.logger.warning("Highlighting requested but module not available")

    # ========== COST TRACKING ==========
    if rcfg.observability.track_cost:
        try:
            if up.track_llm_cost:
                # Calculate estimated cost
                total_tokens = sum(len(doc.content.split()) for doc in state.result.documents)
                cost = await up.track_llm_cost(
                    model=rcfg.generation.generation_model or "gpt-3.5-turbo",
                    input_tokens=total_tokens,
                    output_tokens=len(state.result.generated_answer.split()) if state.result.generated_answer else 0,
                )

                state.result.metadata["estimated_cost"] = cost

        except ImportError:
            state.result.errors.append("Cost tracking module not available")

    state.apply_workspace_filtering_to_result()


async def store_and_observe(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Cache storage, observability (retrieval metrics, faithfulness) and performance analysis."""
    # ========== CACHE STORAGE ==========
    if (
        state.retrieval_cache_eligible
        and not state.result.cache_hit
        and state.cache_retrieval_snapshot
    ):
        try:
            # Store in cache for future use
            cache = state.get_cache_instance()

            if cache:
                # Support both async/sync and set/add method names
                set_fn = getattr(cache, "set", None) or getattr(cache, "add", None)
                if set_fn:
                    wire_documents = [
                        up._serialize_result_document(document)
                        for document in state.cache_retrieval_snapshot
                    ]
                    cache_queries = [state.query]
                    cache_queries.extend([q for q in (state.result.expanded_queries or []) if isinstance(q, str)])
                    seen = set()
                    for cq in cache_queries:
                        if not isinstance(cq, str):
                            continue
                        cq = cq.strip()
                        if not cq or cq in seen:
                            continue
                        seen.add(cq)
                        cache_payload = {
                            "documents": up._clone_cached_documents(wire_documents),
                            "metadata": {
                                "kind": "retrieval_documents",
                                "schema_version": 1,
                            },
                        }
                        if callable(up._sanitize_cache_payload):
                            sanitized_cache_payload = up._sanitize_cache_payload(cache_payload)
                        else:
                            sanitized_cache_payload = cache_payload
                        if sanitized_cache_payload is None:
                            continue
                        await up._invoke_cache_callable(
                            set_fn,
                            cq,
                            sanitized_cache_payload,
                            ttl=rcfg.cache.cache_ttl,
                        )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            up.logger.error(
                "Cache storage failed (error_type={})",
                type(exc).__name__,
            )

    # ========== OBSERVABILITY ==========
    if rcfg.observability.enable_observability:
        try:
            if up.Tracer:
                tracer = up.Tracer()
                await tracer.trace(
                    trace_id=rcfg.observability.trace_id or str(uuid.uuid4()),
                    operation="unified_rag_pipeline",
                    query=state.query,
                    timings=state.result.timings,
                    metadata=state.result.metadata,
                )

        except ImportError:
            state.result.errors.append("Observability module not available")

    # ========== PERFORMANCE ANALYSIS ==========
    if rcfg.observability.enable_performance_analysis:
        try:
            if up.PerformanceMonitor:
                monitor = up.PerformanceMonitor()
                state.analysis = await monitor.analyze(
                    timings=state.result.timings, document_count=len(state.result.documents), cache_hit=state.result.cache_hit
                )

                state.result.metadata["performance_analysis"] = state.analysis

        except ImportError:
            state.result.errors.append("Performance monitor not available")


async def finalize_run(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """The pipeline's `finally` body: total time, metrics, observability and debug logging."""
    # Calculate total time
    state.result.total_time = time.time() - state.start_time
    state.result.timings["total"] = state.result.total_time

    # Finalize metrics if monitoring
    if state.metrics:
        state.metrics.total_duration = state.result.total_time
        state.metrics.cache_hit = state.result.cache_hit
        state.metrics.documents_retrieved = len(state.result.documents)

        if rcfg.observability.enable_monitoring:
            try:
                collector = up.MetricsCollector()
                collector.end_query(state.metrics)
            except (RuntimeError, TypeError, ValueError) as e:
                up.logger.error(f"Metrics recording error: {e}")

    # ---- Retrieval quality metrics (optional) ----
    if rcfg.observability.ground_truth_doc_ids and state.result.documents:
        try:
            from ..retrieval_metrics import evaluate_retrieval as _eval_retrieval

            _retrieved_ids = [str(getattr(d, "id", "")) for d in state.result.documents]
            _ret_metrics = _eval_retrieval(
                _retrieved_ids,
                [str(gid) for gid in rcfg.observability.ground_truth_doc_ids],
                k=rcfg.observability.metrics_k,
            )
            state.result.metadata["retrieval_metrics"] = _ret_metrics.to_dict()
            # Emit to Prometheus
            try:
                from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram

                observe_histogram("rag_retrieval_precision", _ret_metrics.precision)
                observe_histogram("rag_retrieval_recall", _ret_metrics.recall)
                observe_histogram("rag_retrieval_mrr", _ret_metrics.mrr)
                observe_histogram("rag_retrieval_ndcg", _ret_metrics.ndcg)
                if hasattr(_ret_metrics, "f1"):
                    observe_histogram("rag_retrieval_f1", _ret_metrics.f1)
            except (ImportError, AttributeError, TypeError):
                pass
        except Exception as _rm_err:  # noqa: BLE001 - moved from unified_pipeline.py (TASK-13950.24)
            up.logger.warning(f"Retrieval metrics computation failed: {_rm_err}")
            state.result.errors.append(f"Retrieval metrics failed: {_rm_err}")

    # ---- Faithfulness evaluation (optional) ----
    if rcfg.verification.enable_faithfulness_eval and state.result.generated_answer:
        try:
            from ..faithfulness import FaithfulnessEvaluator as _FaithEval

            # Build context from retrieved documents
            _ctx_parts = [getattr(d, "content", "") for d in (state.result.documents or [])]
            _ctx_text = "\n\n".join(p for p in _ctx_parts if p)
            if _ctx_text:
                # Attempt to find an LLM callable from kwargs or auto-construct one
                _llm_obj = rcfg.runtime.faithfulness_llm
                if _llm_obj is None:
                    # Auto-construct an LLM adapter from the pipeline's config
                    try:
                        from tldw_Server_API.app.core.config import load_and_log_configs as _load_cfg
                        from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import (
                            analyze as _sgl_analyze,
                        )

                        _f_cfg = _load_cfg() or {}
                        from ..generation_defaults import resolve_generation_defaults

                        _f_prov, _f_model = resolve_generation_defaults(_f_cfg, rcfg.generation.generation_provider, rcfg.generation.generation_model)
                        _f_handle = None
                        if rcfg.runtime.credential_runtime is not None:
                            _f_handle = await rcfg.runtime.credential_runtime.resolve(
                                _f_prov,
                                model=_f_model,
                            )
                        _f_state = {"marked": False, "content": False}

                        class _FaithfulnessLLMAdapter:
                            """Wraps analyze() to satisfy the LLMCallable protocol."""

                            async def generate(self, prompt: str) -> str:
                                def _call_faithfulness() -> Any:
                                    response = _sgl_analyze(
                                        api_name=_f_prov,
                                        input_data=prompt,
                                        custom_prompt_arg=None,
                                        model_override=_f_model,
                                        api_key=(
                                            _f_handle.api_key
                                            if _f_handle is not None
                                            else None
                                        ),
                                        app_config=(
                                            _f_handle.app_config
                                            if _f_handle is not None
                                            else None
                                        ),
                                        credentials_resolved=_f_handle is not None,
                                        provider_credentials=_f_handle,
                                        raise_on_error=_f_handle is not None,
                                    )
                                    return up._consume_bound_sgl_response(
                                        response,
                                        _f_prov,
                                        on_content=lambda: _f_state.__setitem__(
                                            "content",
                                            True,
                                        ),
                                        fail_closed=_f_handle is not None,
                                    )

                                async def _run_faithfulness() -> Any:
                                    faithfulness_completed = False
                                    try:
                                        value = await up.await_bounded_sync_call(
                                            _call_faithfulness,
                                            pool=up.SYNC_ADAPTER_CALL_POOL,
                                            exhaustion_message=(
                                                "RAG optional adapter capacity is exhausted"
                                            ),
                                        )
                                        faithfulness_completed = True
                                        return value
                                    finally:
                                        if (
                                            _f_handle is not None
                                            and not _f_state["marked"]
                                            and (
                                                faithfulness_completed
                                                or _f_state["content"]
                                            )
                                        ):
                                            await rcfg.runtime.credential_runtime.mark_used(
                                                _f_handle
                                            )
                                            _f_state["marked"] = True

                                faithfulness_operation = _run_faithfulness()
                                result_text = (
                                    await up.await_owned_worker(
                                        faithfulness_operation
                                    )
                                    if _f_handle is not None
                                    else await faithfulness_operation
                                )
                                return str(result_text) if result_text else ""

                        _llm_obj = _FaithfulnessLLMAdapter()
                    except (ImportError, AttributeError, TypeError) as _auto_err:
                        up.logger.debug(f"Could not auto-construct faithfulness LLM: {_auto_err}")

                if _llm_obj is not None:
                    _faith_eval = _FaithEval(
                        _llm_obj,
                        propagate_errors=rcfg.runtime.credential_runtime is not None,
                    )
                    _faith_result = await _faith_eval.evaluate_detailed(
                        state.result.generated_answer, _ctx_text
                    )
                    state.result.metadata["faithfulness"] = _faith_result.to_dict()
                    # Emit faithfulness score to Prometheus
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import set_gauge

                        _f_score = _faith_result.to_dict().get("faithfulness_score")
                        if _f_score is not None:
                            set_gauge("rag_eval_faithfulness_score", float(_f_score), labels={"dataset": "online"})
                    except (ImportError, AttributeError, TypeError, ValueError):
                        pass
                else:
                    up.logger.debug(
                        "Faithfulness eval requested but no LLM available"
                    )
        except (up.ByokResolutionError, up.SummaryProviderError) as exc:
            state.result.metadata["faithfulness"] = {
                "failure_code": (
                    exc.code
                    if isinstance(exc, up.ByokResolutionError)
                    else "provider_unavailable"
                ),
                "verification_available": False,
            }
            up.logger.warning("Faithfulness evaluation provider unavailable")
        except Exception as _fe_err:  # noqa: BLE001 - moved from unified_pipeline.py (TASK-13950.24)
            if rcfg.runtime.credential_runtime is not None:
                state.result.metadata["faithfulness"] = {
                    "failure_code": "provider_unavailable",
                    "verification_available": False,
                }
                up.logger.warning("Faithfulness evaluation provider unavailable")
            else:
                up.logger.warning(f"Faithfulness evaluation failed: {_fe_err}")
                state.result.errors.append(f"Faithfulness eval failed: {_fe_err}")

    # Debug output if requested
    if rcfg.observability.debug_mode:
        try:
            _qh = hashlib.md5((state.query or "").encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
            up.logger.debug(f"Query hash={_qh} len={len(state.query or '')}")
        except (AttributeError, TypeError, ValueError):
            up.logger.debug("Received query (hash unavailable)")
        up.logger.debug(f"Documents found: {len(state.result.documents)}")
        up.logger.debug(f"Cache hit: {state.result.cache_hit}")
        up.logger.debug(f"Timings: {state.result.timings}")
        up.logger.debug(f"Errors: {state.result.errors}")
