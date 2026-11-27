"""generation code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import contextlib
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


def prepare_generation(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Calibration gating, the effective enable flag and the retrieval-only result."""
    # ========== ANSWER GENERATION ==========
    # Honor reranking calibration gating if present (e.g., TwoTier strategy)
    try:
        _reranking_status = state.result.metadata.get("reranking", {})
        if (
            isinstance(_reranking_status, dict)
            and _reranking_status.get("verification_available") is False
        ):
            state.result.metadata.pop("reranking_calibration", None)
            _cal = None
        else:
            _cal = state.result.metadata.get("reranking_calibration")
        state.gated_generation = bool(_cal.get("gated")) if isinstance(_cal, dict) else False
        # When calibration metadata is present, ensure fused_score/version/decision
        # are wired for observability.
        if isinstance(_cal, dict):
            if "fused_score" not in _cal and "top_doc_prob" in _cal:
                with contextlib.suppress(TypeError, ValueError):
                    _cal["fused_score"] = float(_cal.get("top_doc_prob") or 0.0)
            if rcfg.rerank.enable_learned_fusion:
                _cal["enabled"] = True
            if rcfg.rerank.calibrator_version:
                _cal.setdefault("version", rcfg.rerank.calibrator_version)
            # Decision will be finalized in the gated branch below
            state.result.metadata["reranking_calibration"] = _cal
    except (AttributeError, TypeError, ValueError):
        state.gated_generation = False

    state.effective_enable_generation = state.resolve_effective_enable_generation()

    if (
        state.effective_enable_generation
        and not state.result.documents
        and state._skip_retrieval_reason != "classification_skip_search"
    ):
        state.result.generated_answer = None
        state.result.metadata["answer_generation_skipped"] = "no_documents"
        state.effective_enable_generation = False

    if not state.effective_enable_generation:
        retrieval_only_result = up.build_retrieval_only_result(
            resolved_request=state.resolved_request,
            retrieval_plan=state.retrieval_plan,
            retrieval_result=up.RetrievedEvidence(
                documents=list(state.result.documents or []),
                metadata=dict(state.result.metadata or {}),
            ),
        )
        retrieval_only_result = up.coordinate_standard_result_evidence(
            retrieval_only_result,
            state.resolved_request,
            retrieval_plan=state.retrieval_plan,
        )
        state.standard_evidence_coordinated = True
        state.result.documents = list(retrieval_only_result.documents)
        state.result.metadata.update(dict(retrieval_only_result.metadata or {}))


async def synthesize_multi_turn(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Draft, critique and refine the answer within a time budget."""
    # Strict budget control
    t0 = time.time()
    budget = float(rcfg.generation.synthesis_time_budget_sec) if rcfg.generation.synthesis_time_budget_sec else None
    aborted = False

    # Draft
    draft_tokens = int(rcfg.generation.synthesis_draft_tokens or min(rcfg.generation.max_generation_tokens, 400))
    d_start = time.time()
    draft_out = await state.generator.generate(
        query=state.query,
        context=state.context,
        prompt_template=state.generation_prompt,
        max_tokens=draft_tokens,
    )
    d_ans = draft_out.get("answer") if isinstance(draft_out, dict) else draft_out
    if isinstance(d_ans, str):
        d_ans_text = d_ans
    elif d_ans is None:
        d_ans_text = ""
    else:
        try:
            d_ans_text = "".join(str(x) for x in d_ans)
        except (TypeError, ValueError):
            d_ans_text = str(d_ans)
    d_dt = time.time() - d_start

    # Critique
    c_text = None
    c_dt = 0.0
    try:
        import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

        # Construct a compact critique prompt using small snippets
        snippets = []
        for d in state.context_docs[:3]:
            s = (getattr(d, "content", "") or "")[:250].replace("\n", " ")
            if s:
                snippets.append(f"- {s}")
        crit_prompt = (
            "You are a careful reviewer.\n"
            "Given the user query, retrieved snippets, and the draft answer, list the top 3 issues (missing facts or unsupported claims).\n"
            f"Query: {state.query}\nSnippets:\n"
            + "\n".join(snippets)
            + f"\n\nDraft:\n{d_ans_text}\n\nIssues:"
        )
        c_start = time.time()
        critique_handle = None
        if rcfg.runtime.credential_runtime is not None:
            critique_handle = await rcfg.runtime.credential_runtime.resolve("openai")
        critique_stream_state = {"content": False}

        def _call_critique() -> Any:
            response = sgl.analyze(
                api_name="openai",
                input_data=crit_prompt,
                custom_prompt_arg=None,
                model_override=None,
                api_key=(
                    critique_handle.api_key
                    if critique_handle is not None
                    else None
                ),
                app_config=(
                    critique_handle.app_config
                    if critique_handle is not None
                    else None
                ),
                credentials_resolved=critique_handle is not None,
                provider_credentials=critique_handle,
                raise_on_error=critique_handle is not None,
                llm_surface="rag.auxiliary",
            )
            return up._consume_bound_sgl_response(
                response,
                "openai",
                on_content=lambda: critique_stream_state.__setitem__(
                    "content",
                    True,
                ),
                fail_closed=critique_handle is not None,
            )

        async def _run_critique() -> Any:
            critique_completed = False
            try:
                value = await up.await_bounded_sync_call(
                    _call_critique,
                    pool=up.SYNC_ADAPTER_CALL_POOL,
                    exhaustion_message=(
                        "RAG optional adapter capacity is exhausted"
                    ),
                )
                critique_completed = True
                return value
            finally:
                if critique_handle is not None and (
                    critique_completed
                    or critique_stream_state["content"]
                ):
                    await rcfg.runtime.credential_runtime.mark_used(
                        critique_handle
                    )

        critique_operation = _run_critique()
        c_text = (
            await up.await_owned_worker(critique_operation)
            if critique_handle is not None
            else await critique_operation
        )
        c_dt = time.time() - c_start
    except (up.ByokResolutionError, up.SummaryProviderError) as exc:
        state.result.metadata.setdefault("synthesis", {})
        state.result.metadata["synthesis"].update(
            {
                "failure_code": (
                    exc.code
                    if isinstance(exc, up.ByokResolutionError)
                    else "provider_unavailable"
                ),
                "verification_available": False,
            }
        )
        c_text = "- Ensure claims are supported by provided snippets."
    except (ImportError, ConnectionError, OSError, RuntimeError, TypeError, ValueError, asyncio.TimeoutError):
        if rcfg.runtime.credential_runtime is not None:
            state.result.metadata.setdefault("synthesis", {})
            state.result.metadata["synthesis"].update(
                {
                    "failure_code": "provider_unavailable",
                    "verification_available": False,
                }
            )
        c_text = "- Ensure claims are supported by provided snippets.\n- Add missing specifics.\n- Clarify ambiguous statements."
    if isinstance(c_text, str):
        c_text_val = c_text
    elif c_text is None:
        c_text_val = ""
    else:
        try:
            c_text_val = "".join(str(x) for x in c_text)
        except (TypeError, ValueError):
            c_text_val = str(c_text)

    # Check budget
    if budget is not None and (time.time() - t0) >= budget:
        aborted = True  # noqa: F841 - moved from unified_pipeline.py (TASK-13950.24)
        state.result.generated_answer = d_ans_text
        state.result.metadata.setdefault("synthesis", {})
        state.result.metadata["synthesis"].update(
            {
                "enabled": True,
                "aborted": True,
                "durations": {"draft": d_dt, "critique": c_dt, "refine": 0.0},
            }
        )
    else:
        # Refine
        refine_tokens = int(rcfg.generation.synthesis_refine_tokens or rcfg.generation.max_generation_tokens)
        r_ctx = state.context + "\n\nCRITIQUE:\n" + c_text_val
        r_start = time.time()
        r_out = await state.generator.generate(
            query=state.query,
            context=r_ctx,
            prompt_template=state.generation_prompt,
            max_tokens=refine_tokens,
        )
        r_ans = r_out.get("answer") if isinstance(r_out, dict) else r_out
        r_dt = time.time() - r_start
        state.result.generated_answer = r_ans
        state.result.metadata.setdefault("synthesis", {})
        state.result.metadata["synthesis"].update(
            {
                "enabled": True,
                "aborted": False,
                "durations": {"draft": d_dt, "critique": c_dt, "refine": r_dt},
            }
        )


async def generate_standard_answer(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Run the generation phase and fold its result into the pipeline result."""
    async def _generate_standard_answer(
        *,
        query: str,
        context: str,
        generation_prompt: Optional[str] = None,
        max_generation_tokens: Optional[int] = None,
    ) -> Any:
        return await state.resilient_call(
            "generation",
            state.generator.generate,
            query=query,
            context=context,
            prompt_template=generation_prompt,
            max_tokens=max_generation_tokens,
        )

    state.resolved_request.payload["generation_prompt"] = state.generation_prompt
    state.resolved_request.payload["max_generation_tokens"] = rcfg.generation.max_generation_tokens
    generation_phase_kwargs: dict[str, Any] = {
        "resolved_request": state.resolved_request,
        "retrieval_plan": state.retrieval_plan,
        "derived_evidence": SimpleNamespace(
            documents=list(state.context_docs),
            metadata=dict(state.result.metadata),
            citations=list((state.result.metadata or {}).get("chunk_citations", []) or []),
            verification_report=(state.result.metadata or {}).get("verification_report"),
        ),
        "generate_answer_fn": _generate_standard_answer,
        "generation_context": state.context,
    }
    generation_result = await up.execute_generation_phase(**generation_phase_kwargs)
    if isinstance(generation_result, dict):
        generation_result = up.coordinate_standard_result_evidence(
            generation_result,
            state.resolved_request,
            retrieval_plan=state.retrieval_plan,
        )
    else:
        generation_result = up.coordinate_standard_result_evidence(
            generation_result,
            state.resolved_request,
            retrieval_plan=state.retrieval_plan,
        )
    state.standard_evidence_coordinated = True
    if isinstance(generation_result, dict):
        state.result.generated_answer = generation_result.get(
            "generated_answer",
            generation_result.get("answer"),
        )
        generation_documents = generation_result.get("documents") or generation_result.get(
            "sources"
        )
        if generation_documents is not None:
            state.result.documents = list(generation_documents)
        state.result.metadata.update(dict(generation_result.get("metadata") or {}))
        if generation_result.get("chunk_citations") is not None:
            state.result.metadata["chunk_citations"] = list(
                generation_result.get("chunk_citations") or []
            )
        if generation_result.get("verification_report") is not None:
            state.result.metadata["verification_report"] = generation_result.get("verification_report")
    else:
        state.result.generated_answer = generation_result.generated_answer
        state.result.metadata.update(dict(generation_result.metadata or {}))
        state.result.metadata["chunk_citations"] = list(generation_result.chunk_citations or [])
        if generation_result.verification_report is not None:
            state.result.metadata["verification_report"] = generation_result.verification_report


async def finish_generation(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Timings, metrics, the span attribute and follow-up suggestions."""
    state.result.timings["answer_generation"] = time.time() - state.generation_start
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram

        observe_histogram(
            "rag_phase_duration_seconds",
            state.result.timings["answer_generation"],
            labels={
                "phase": "generation",
                "difficulty": str(state.result.metadata.get("query_intent", "na")),
            },
        )
        if rcfg.generation.enable_multi_turn_synthesis:
            observe_histogram(
                "rag_phase_duration_seconds",
                state.result.timings["answer_generation"],
                labels={
                    "phase": "synthesis",
                    "difficulty": str(state.result.metadata.get("query_intent", "na")),
                },
            )
        if state._otel_span_gen is not None:
            try:
                _ans_len = len(state.result.generated_answer or "")
                state._otel_span_gen.set_attribute("rag.answer_length", int(_ans_len))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    if state.metrics:
        state.metrics.generation_time = state.result.timings["answer_generation"]

    # Follow-up suggestions generation (bounded-latency, best effort)
    if rcfg.generation.enable_suggestions and up.generate_suggestions is not None and state.result.generated_answer:
        try:
            _suggestions_start = time.time()
            _suggestion_stage_metadata: dict[str, Any] = {}
            _suggestions = await up.generate_suggestions(
                query=state.query,
                response_text=str(state.result.generated_answer),
                chat_history=rcfg.query_prep.chat_history,
                llm_provider=rcfg.research.classifier_provider or rcfg.generation.generation_provider or "openai",
                llm_model=rcfg.research.classifier_model or rcfg.generation.generation_model,
                num_suggestions=rcfg.generation.num_suggestions,
                llm_timeout_sec=3.0,
                **(
                    {
                        "credential_runtime": rcfg.runtime.credential_runtime,
                        "stage_metadata": _suggestion_stage_metadata,
                    }
                    if rcfg.runtime.credential_runtime is not None
                    else {}
                ),
            )
            state.result.metadata["suggestions"] = _suggestions
            if _suggestion_stage_metadata:
                state.result.metadata["suggestion_generation"] = dict(
                    _suggestion_stage_metadata
                )
            state.result.timings["suggestions"] = time.time() - _suggestions_start
        except Exception as _sug_exc:  # noqa: BLE001 - moved from unified_pipeline.py (TASK-13950.24)
            up.logger.debug(f"Suggestion generation failed: {_sug_exc!r}")


def answer_when_gated(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Record the gate, apply the abstention policy."""
    # Record a metadata entry and bump a metric for observability
    state.result.metadata.setdefault("generation_gate", {})
    state.result.metadata["generation_gate"].update(
        {
            "reason": "low_relevance_probability",
            "at": time.time(),
        }
    )
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

        increment_counter("rag_generation_gated_total", 1, labels={"strategy": "two_tier"})
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    # Sample payload exemplar when generation is gated
    try:
        from ..payload_exemplars import maybe_record_exemplar

        maybe_record_exemplar(
            query=state.query,
            documents=state.result.documents or [],
            answer=state.result.generated_answer or "",
            reason="generation_gated",
            user_id=rcfg.scope.user_id,
            namespace=rcfg.retrieval.index_namespace,
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass

    # Decide abstention policy for calibration-based gating.
    # Priority:
    #   1) enable_learned_fusion + abstention_policy
    #   2) enable_abstention + abstention_behavior
    #   3) default "continue" (no replacement answer)
    effective_policy: str = "continue"
    if rcfg.rerank.enable_learned_fusion:
        effective_policy = rcfg.generation.abstention_policy or "continue"
    elif rcfg.generation.enable_abstention:
        effective_policy = rcfg.generation.abstention_behavior or "continue"

    # Record the decision in reranking calibration metadata when present
    try:
        if isinstance(state.result.metadata, dict):
            cal = state.result.metadata.get("reranking_calibration")
            if isinstance(cal, dict):
                cal["decision"] = effective_policy
                state.result.metadata["reranking_calibration"] = cal
    except (AttributeError, TypeError, ValueError):
        pass

    # Abstention / clarifying question path based on effective policy
    if effective_policy in {"ask", "decline"}:
        try:
            if effective_policy == "ask":
                clar_q = None
                # Form a concise clarifying question using query analysis if available
                if up.QueryAnalyzer:
                    try:
                        domain = state.analysis_domain
                        clar_q = f"Please clarify: what specific aspect of '{state.query}' should I focus on{f' in {domain}' if domain else ''}?"
                    except (AttributeError, TypeError, ValueError):
                        clar_q = None
                if not clar_q:
                    clar_q = f"Could you clarify which specific details about '{state.query}' you need?"
                state.result.generated_answer = clar_q
            elif effective_policy == "decline":
                state.result.generated_answer = "I don’t have sufficient grounded evidence to answer confidently. Please clarify your question or provide more context."
        except (AttributeError, TypeError, ValueError):
            pass


async def run_generation(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Answer generation under the rag.generation span, or the gated path."""
    if state.effective_enable_generation and not state.gated_generation:
        state.result.metadata["generation_executed"] = True
        state.generation_start = time.time()
        try:
            # --- OTEL: generation span ---
            _otel_cm_gen = None
            state._otel_span_gen = None
            if rcfg.observability.enable_observability and up.get_telemetry_manager:
                try:
                    _tm = up.get_telemetry_manager()
                    _tr = _tm.get_tracer("tldw.rag")
                    _attrs = {
                        "rag.phase": "generation",
                        "rag.model": str(rcfg.generation.generation_model or ""),
                        "rag.provider": str(rcfg.generation.generation_provider or ""),
                        "rag.multi_turn": bool(rcfg.generation.enable_multi_turn_synthesis),
                    }
                    _otel_cm_gen = _tr.start_as_current_span("rag.generation")
                    state._otel_span_gen = _otel_cm_gen.__enter__()
                    for _k, _v in _attrs.items():
                        with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                            state._otel_span_gen.set_attribute(_k, _v)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    _otel_cm_gen = None
                    state._otel_span_gen = None
            # Strict extractive path: assemble answer from retrieved spans only
            if bool(state.strict_extractive):
                try:
                    # Simple extractive assembly: pick top sentences from top documents
                    max_sents = 6
                    chosen: list[str] = []
                    import re as _re

                    q_terms = [t.lower() for t in _re.findall(r"[A-Za-z0-9_-]{3,}", state.query or "")][:10]
                    for doc in (state.result.documents or [])[: min(5, len(state.result.documents or []))]:
                        text = (getattr(doc, "content", "") or "").strip()
                        if not text:
                            continue
                        sents = [s.strip() for s in _re.split(r"(?<=[\.!?])\s+", text) if s.strip()]
                        # prefer a sentence containing a query term
                        hit = None
                        for s in sents:
                            low = s.lower()
                            if any(t in low for t in q_terms):
                                hit = s
                                break
                        if not hit and sents:
                            hit = sents[0]
                        if hit and hit not in chosen:
                            chosen.append(hit)
                            if len(chosen) >= max_sents:
                                break
                    state.result.generated_answer = " ".join(chosen).strip()
                except (AttributeError, TypeError, ValueError) as _se:
                    state.result.errors.append(f"Strict extractive assembly failed: {_se}")
                    state.result.generated_answer = None
            elif up.AnswerGenerator:
                state.generator = up.AnswerGenerator(
                    model=rcfg.generation.generation_model,
                    provider=rcfg.generation.generation_provider,
                    credential_runtime=rcfg.runtime.credential_runtime,
                )

                # Prepare base context from top documents
                state.context_docs = state.result.documents[:5] if state.result.documents else []

                # Structured response writer: XML-tagged context with citation rules
                if (
                    rcfg.generation.enable_structured_response
                    and up.format_context_xml is not None
                    and up.build_writer_system_prompt is not None
                ):
                    _writer_chunks = []
                    for _wd in state.context_docs:
                        _wd_meta = getattr(_wd, "metadata", {}) or {}
                        _writer_chunks.append(
                            {
                                "content": getattr(_wd, "content", str(_wd)),
                                "title": _wd_meta.get("title", ""),
                                "url": _wd_meta.get("url", "") or _wd_meta.get("source", ""),
                                "metadata": _wd_meta,
                            }
                        )
                    state.context = up.format_context_xml(_writer_chunks)
                    _writer_mode = rcfg.request.search_depth_mode or "balanced"
                    _writer_policy = None
                    if up.get_writer_depth_policy is not None:
                        with contextlib.suppress(TypeError, ValueError):
                            _writer_policy = up.get_writer_depth_policy(
                                mode=_writer_mode,
                                max_generation_tokens=rcfg.generation.max_generation_tokens,
                            )
                    _writer_sys = up.build_writer_system_prompt(
                        mode=_writer_mode,
                        max_generation_tokens=rcfg.generation.max_generation_tokens,
                    )
                    # Override generation_prompt with structured writer prompt
                    if not state.generation_prompt:
                        state.generation_prompt = _writer_sys
                    else:
                        state.generation_prompt = f"{state.generation_prompt}\n\n{_writer_sys}"
                    if up.build_writer_user_prompt is not None:
                        state.context = up.build_writer_user_prompt(query=state.query, context_xml=state.context)
                    state.result.metadata.setdefault("structured_writer", {})
                    state.result.metadata["structured_writer"].update(
                        {
                            "enabled": True,
                            "mode": _writer_mode,
                            "context_results": len(_writer_chunks),
                            "max_generation_tokens": int(rcfg.generation.max_generation_tokens),
                        }
                    )
                    if isinstance(_writer_policy, dict):
                        state.result.metadata["structured_writer"]["depth_policy"] = _writer_policy
                else:
                    state.context = "\n\n".join([getattr(doc, "content", str(doc)) for doc in state.context_docs])

                if rcfg.generation.enable_multi_turn_synthesis:
                    await synthesize_multi_turn(rcfg, state)
                else:

                    await generate_standard_answer(rcfg, state)
                await finish_generation(rcfg, state)

        except ImportError:
            state.result.errors.append("Generation module not available")
            up.logger.warning("Answer generation requested but module not available")
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ):
            state.result.errors.append("answer_generation_failed")
            up.logger.error("Answer generation failed")
            try:
                from ..payload_exemplars import maybe_record_exemplar

                maybe_record_exemplar(
                    query=state.query,
                    documents=state.result.documents or [],
                    answer=state.result.generated_answer or "",
                    reason="generation_error",
                    user_id=rcfg.scope.user_id,
                    namespace=rcfg.retrieval.index_namespace,
                )
            except (ImportError, RuntimeError, TypeError, ValueError):
                pass
        finally:
            if _otel_cm_gen is not None:
                with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                    _otel_cm_gen.__exit__(None, None, None)
    elif state.effective_enable_generation and state.gated_generation:
        answer_when_gated(rcfg, state)
