# completion_pipeline.py
# Description: Coordinator boundary for Chat completion execution.
"""Coordinator object for Chat completion execution paths."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING
import asyncio
import contextlib
import functools

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Chat.call_state import NonStreamCall


__all__ = ["run_non_stream_call"]


async def _record_provider_use(cst: "NonStreamCall", provider_name: str) -> bool:
    """Best-effort, once-only accounting immediately after provider success."""

    if not callable(cst.on_success):
        return False
    async with cst.provider_use_lock:
        if cst.provider_use_recorded:
            return True
        try:
            persisted = await cst.on_success(provider_name)
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            return False
        if persisted is False:
            return False
        cst.provider_use_recorded = True
        return True


async def _record_dict_response_usage(
    cst: "NonStreamCall",
    *,
    response: dict[str, Any],
    messages: list[dict[str, Any]],
    content: Any,
    provider_name: str,
    model_name: str,
    started_at: float,
) -> None:
    """Record token and durable usage for one validated dict response."""

    choices = response.get("choices")
    usage = cs.usage_with_service_tier(response)  # OpenAI reports service_tier beside usage
    if usage:
        try:
            normalized_usage = cs.normalize_llm_usage(
                provider=provider_name,
                usage=usage if isinstance(usage, dict) else None,
                choices=choices if isinstance(choices, list) else None,
            )
            prompt_tokens = normalized_usage.input_tokens
            completion_tokens = normalized_usage.output_tokens
            cst.metrics.track_tokens(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model_name,
                provider=provider_name,
            )
            user_id = None
            api_key_id = None
            try:
                if cst.request is not None and hasattr(cst.request, "state"):
                    user_id = getattr(cst.request.state, "user_id", None)
                    api_key_id = getattr(cst.request.state, "api_key_id", None)
            except cs._CHAT_NONCRITICAL_EXCEPTIONS:
                pass
            await cs.log_llm_usage(
                user_id=user_id,
                key_id=api_key_id,
                request=cst.request,
                endpoint=(
                    f"{cst.request.method}:{cst.request.url.path}"
                    if cst.request
                    else "POST:/api/v1/chat/completions"
                ),
                operation="chat",
                provider=provider_name,
                model=model_name,
                status=200,
                latency_ms=int((cs.time.time() - started_at) * 1000),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=normalized_usage.total_tokens,
                request_id=(
                    cst.request.headers.get("X-Request-ID") if cst.request else None
                )
                or (cs.get_request_id() or None),
                conversation_id=(
                    str(cst.final_conversation_id)
                    if cst.final_conversation_id is not None
                    else None
                ),
                usage_metadata=usage if isinstance(usage, dict) else None,
                choice_count=normalized_usage.choice_count,
                estimate_source=normalized_usage.estimate_source,
                credential_source=cs.call_credential_source(cst.cleaned_args),
            )
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        return

    try:
        try:
            prompt_tokens = cs._estimate_tokens_from_messages(messages)
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            prompt_tokens = 0
        response_choices = cs.collect_non_stream_choices(response)
        completion_tokens = (
            cs.estimate_completion_tokens_from_choices(response_choices)
            if response_choices
            else max(0, len(cs._extract_text_from_content(content)) // 4)
        )
        cst.metrics.track_tokens(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            model=model_name,
            provider=provider_name,
        )
        user_id = None
        api_key_id = None
        try:
            if cst.request is not None and hasattr(cst.request, "state"):
                user_id = getattr(cst.request.state, "user_id", None)
                api_key_id = getattr(cst.request.state, "api_key_id", None)
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        await cs.log_llm_usage(
            user_id=user_id,
            key_id=api_key_id,
            request=cst.request,
            endpoint=(
                f"{cst.request.method}:{cst.request.url.path}"
                if cst.request
                else "POST:/api/v1/chat/completions"
            ),
            operation="chat",
            provider=provider_name,
            model=model_name,
            status=200,
            latency_ms=int((cs.time.time() - started_at) * 1000),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens + completion_tokens),
            request_id=(cst.request.headers.get("X-Request-ID") if cst.request else None)
            or (cs.get_request_id() or None),
            conversation_id=(
                str(cst.final_conversation_id)
                if cst.final_conversation_id is not None
                else None
            ),
            estimated=True,
            estimate_source="missing_usage",
            credential_source=cs.call_credential_source(cst.cleaned_args),
        )
    except cs._CHAT_NONCRITICAL_EXCEPTIONS:
        pass


async def _record_provider_use_after_cancel(cst: "NonStreamCall", result: Any) -> None:
    """Record only usable late output before cancellation closes its runtime."""

    if cs._nonstream_provider_result_is_usable(result):
        await _record_provider_use(cst, cst.selected_provider)


def _queued_processor(cst: "NonStreamCall"):
    local_start = cs.time.time()
    try:
        result = cst.llm_call_func()
        cs._require_usable_nonstream_provider_result(result)
        latency = cs.time.time() - local_start
        cst.metrics.track_llm_call(cst.selected_provider, cst.model, latency, success=True)
        if cst.provider_manager:
            cst.provider_manager.record_success(cst.selected_provider, latency)
        if cst.selected_provider != cst.provider:
            with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
                cst.metrics.track_provider_fallback_success(
                    requested_provider=cst.provider,
                    selected_provider=cst.selected_provider,
                    streaming=False,
                    queued=True,
                )
        return result
    except (cs.LLMQuotaExceededError, cs.LLMPolicyRefusal):
        raise  # the user's limit or a policy refusal, not the provider's health: no failure, no fallback
    except cs._CHAT_NONCRITICAL_EXCEPTIONS as proc_error:
        latency = cs.time.time() - local_start
        cst.metrics.track_llm_call(
            cst.selected_provider,
            cst.model,
            latency,
            success=False,
            error_type=type(proc_error).__name__,
        )
        if cst.provider_manager:
            cst.provider_manager.record_failure(
                cst.selected_provider,
                cs.sanitized_provider_stream_exception(proc_error),
            )
        cst.queue_failure_recorded = True
        bounded_error = cs._bound_provider_invocation_exception(proc_error)
        cs.raise_detached_error(bounded_error)


def _invoke_non_stream_provider(cst: "NonStreamCall") -> Any:
    try:
        return cst.llm_call_func()
    except asyncio.CancelledError as provider_cancel:
        cs.raise_detached_error(
            cs.sanitized_provider_stream_exception(provider_cancel)
        )


async def _record_fallback_use_after_cancel(cst: "NonStreamCall", result: Any) -> None:
    if cs._nonstream_provider_result_is_usable(result):
        await _record_provider_use(cst, cst.fallback_provider)


def prepare_non_stream_request(cst: "NonStreamCall") -> None:
    """Structured-response preparation, the tool-autoexec guard and the prompt-cost guardrail."""
    try:
        if cst.structured_request_context is None:
            cst.structured_request_context = cs.prepare_structured_response_request(
                provider=cst.selected_provider,
                response_format=cst.cleaned_args.get("response_format"),
                requested_response_format=cst.cleaned_args.get("_structured_requested_response_format"),
            )
        cs.apply_structured_response_request(
            cleaned_args=cst.cleaned_args,
            structured_request_context=cst.structured_request_context,
        )
        cs.ensure_tool_autoexec_supports_request(
            cleaned_args=cst.cleaned_args,
            should_run_tool_autoexec=cs.should_run_legacy_tool_autoexec,
        )
        cst.prompt_guardrail_decision = cs._evaluate_chat_prompt_cost_guardrails(
            cleaned_args=cst.cleaned_args,
            templated_llm_payload=cst.templated_llm_payload,
            selected_provider=cst.selected_provider,
            model=cst.model,
            streaming=False,
            continuation_metadata=cst.normalized_continuation_metadata,
        )
    except (
        cs.StructuredGenerationCapabilityError,
        cs.StructuredGenerationParseError,
        cs.StructuredGenerationSchemaError,
    ) as structured_exc:
        cs.raise_detached_error(cs.build_structured_http_exception(structured_exc))


async def call_non_stream_provider(cst: "NonStreamCall") -> None:
    """Call the provider through the request queue or a bounded worker."""
    if cst.queue_enabled:
        est_tokens_for_queue = cs.estimate_tokens_from_json(cst.request_json)
        try:
            fut = await cst.queue_for_exec.enqueue(
                request_id=(cs.get_request_id() or "unknown"),
                request_data={"endpoint": "/api/v1/chat/completions", "mode": "non-stream"},
                client_id=str(cst.client_id),
                priority=cs.RequestPriority.NORMAL,
                estimated_tokens=est_tokens_for_queue,
                processor=functools.partial(_queued_processor, cst),
                processor_args=(),
                processor_kwargs={},
                streaming=False,
                stream_channel=None,
            )
        except (ValueError, TimeoutError) as admission_error:
            with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
                cst.metrics.track_rate_limit(str(cst.client_id))
            detail = str(admission_error) or "Service busy. Please retry."
            status_code = (
                cs.status.HTTP_429_TOO_MANY_REQUESTS
                if "rate limit" in detail.lower()
                else cs.status.HTTP_503_SERVICE_UNAVAILABLE
            )
            queue_exc = cs.HTTPException(status_code=status_code, detail=detail)
            raise cs._mark_trusted_local_chat_signal(
                queue_exc,
                "queue_admission",
            ) from admission_error
        try:
            cst.llm_response = await cs.await_owned_worker(
                fut,
                on_cancel_result=functools.partial(_record_provider_use_after_cancel, cst),
            )
            cs._require_usable_nonstream_provider_result(cst.llm_response)
            await cs.await_owned_worker(
                _record_provider_use(cst, cst.selected_provider)
            )
        except Exception as future_error:  # noqa: BLE001 - queue/provider trust boundary
            cst.bounded_error = cs._bound_provider_invocation_exception(future_error)
            cs.raise_detached_error(cst.bounded_error)
        cst.metrics_recorded = True
    else:
        # Execute provided LLM call function in a worker to avoid blocking the loop.
        # llm_call_func is a sync callable (partial of perform_chat_api_call or a mock).
        try:
            cst.llm_response = await cs.await_owned_worker(
                cs.await_bounded_sync_call(
                    functools.partial(_invoke_non_stream_provider, cst),
                    pool=cs.SYNC_ADAPTER_CALL_POOL,
                    exhaustion_message="Provider adapter capacity is exhausted",
                ),
                on_cancel_result=functools.partial(_record_provider_use_after_cancel, cst),
            )
            cs._require_usable_nonstream_provider_result(cst.llm_response)
            await cs.await_owned_worker(
                _record_provider_use(cst, cst.selected_provider)
            )
        except Exception as provider_error:  # noqa: BLE001 - adapter trust boundary
            cst.provider_failure_type = type(provider_error).__name__
            cst.bounded_error = cs._bound_provider_invocation_exception(provider_error)
            cs.raise_detached_error(cst.bounded_error)


async def run_non_stream_call(cst: "NonStreamCall") -> dict[str, Any]:
    """Run one non-streaming chat completion: queue, failover, moderation and persistence."""

    cst.llm_start_time = cs.time.time()
    cst.structured_metadata: dict[str, Any] | None = None
    cst.normalized_continuation_metadata = (
        dict(cst.continuation_metadata)
        if isinstance(cst.continuation_metadata, dict) and cst.continuation_metadata
        else None
    )
    cst.run_first_metric_context = cs._emit_chat_run_first_rollout_metrics(
        cst.metrics,
        cleaned_args=cst.cleaned_args,
        provider=cst.selected_provider,
        model=cst.model,
        streaming=False,
    )
    cst.llm_response = None
    cst.metrics_recorded = False
    cst.queue_failure_recorded = False
    cst.queue_enabled = False
    cst.provider_use_recorded = False
    cst.provider_use_lock = asyncio.Lock()
    cst.provider_failure_type: str | None = None
    cst.pending_assistant_payload: dict[str, Any] | None = None
    cst.pending_tool_messages: list[dict[str, Any]] = []
    cst.prompt_guardrail_decision: cs.PromptCostGuardrailDecision | None = None

    try:
        prepare_non_stream_request(cst)
        cst.queue_for_exec = None
        try:
            cst.queue_for_exec = cs.get_request_queue()
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            cst.queue_for_exec = None
        cst.queue_enabled = (
            cst.queue_execution_enabled
            and cst.queue_for_exec is not None
            and cs.queue_is_active(cst.queue_for_exec)
        )

        await call_non_stream_provider(cst)
        llm_latency = cs.time.time() - cst.llm_start_time
        if not cst.metrics_recorded:
            cst.metrics.track_llm_call(cst.selected_provider, cst.model, llm_latency, success=True)
            if cst.provider_manager:
                cst.provider_manager.record_success(cst.selected_provider, llm_latency)
            if cst.selected_provider != cst.provider:
                with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
                    cst.metrics.track_provider_fallback_success(
                        requested_provider=cst.provider,
                        selected_provider=cst.selected_provider,
                        streaming=False,
                        queued=False,
                    )
    except cs.HTTPException as he:
        cs._emit_chat_run_first_completion_metric(
            cst.metrics,
            context=cst.run_first_metric_context,
            outcome="error",
        )
        if cs._is_trusted_local_chat_http_exception(he):
            raise
        cs.raise_detached_error(cs.sanitized_provider_stream_exception(he))
    except cs._CHAT_NONCRITICAL_EXCEPTIONS as e:
        llm_latency = cs.time.time() - cst.llm_start_time
        if not cst.queue_failure_recorded:
            cst.metrics.track_llm_call(
                cst.selected_provider,
                cst.model,
                llm_latency,
                success=False,
                error_type=cst.provider_failure_type or type(e).__name__,
            )
            if cst.provider_manager:
                cst.provider_manager.record_failure(
                    cst.selected_provider,
                    cs.sanitized_provider_stream_exception(e),
                )

        if cst.provider_manager:
            if (
                cst.enable_provider_fallback
                and isinstance(e, (cs.ChatProviderError, cs.ChatAPIError))
                and not cs._is_client_like_error(e)
                and cs.provider_stream_error_allows_replay(e)
            ):
                cst.fallback_provider = cst.provider_manager.get_available_provider(exclude=[cst.selected_provider])
                if cst.fallback_provider:
                    cs.logger.warning(f"Trying fallback provider {cst.fallback_provider} after {cst.selected_provider} failed")
                    try:
                        refreshed = cst.refresh_provider_params(cst.fallback_provider)
                        if hasattr(refreshed, "__await__"):
                            refreshed_args, refreshed_model = await refreshed  # type: ignore[misc]
                        else:
                            refreshed_args, refreshed_model = refreshed
                        # Validate refreshed params have required fields before proceeding
                        if not isinstance(refreshed_args, dict) or "messages_payload" not in refreshed_args:
                            raise ValueError(
                                f"Invalid refreshed params for {cst.fallback_provider}: missing required fields"
                            )
                        try:
                            cst.structured_request_context = cs.prepare_structured_response_request(
                                provider=cst.fallback_provider,
                                response_format=refreshed_args.get("response_format"),
                                requested_response_format=(
                                    cst.structured_request_context.requested_response_format
                                    if cst.structured_request_context is not None
                                    else refreshed_args.get("_structured_requested_response_format")
                                ),
                            )
                        except (
                            cs.StructuredGenerationCapabilityError,
                            cs.StructuredGenerationParseError,
                            cs.StructuredGenerationSchemaError,
                        ) as structured_exc:
                            cs.raise_detached_error(
                                cs.build_structured_http_exception(structured_exc)
                            )
                        cs.apply_structured_response_request(
                            cleaned_args=refreshed_args,
                            structured_request_context=cst.structured_request_context,
                        )
                    except cs._CHAT_PROVIDER_REFRESH_EXCEPTIONS as refresh_error:
                        if not isinstance(refresh_error, cs.ByokResolutionError):
                            cst.provider_manager.record_failure(
                                cst.fallback_provider,
                                cs.sanitized_provider_stream_exception(refresh_error),
                            )
                        cs._emit_chat_run_first_completion_metric(
                            cst.metrics,
                            context=cst.run_first_metric_context,
                            outcome="error",
                        )
                        cst.bounded_error = cs._bound_provider_http_exception(refresh_error)
                        cs.raise_detached_error(cst.bounded_error)
                    try:
                        cs._check_fallback_prompt_cost(refreshed_args, cst.fallback_provider, refreshed_model or cst.model, streaming=False)
                        cs.ensure_tool_autoexec_supports_request(
                            cleaned_args=refreshed_args,
                            should_run_tool_autoexec=cs.should_run_legacy_tool_autoexec,
                        )
                    except cs.HTTPException:
                        cs._emit_chat_run_first_completion_metric(
                            cst.metrics,
                            context=cst.run_first_metric_context,
                            outcome="error",
                        )
                        raise
                    cst.cleaned_args = refreshed_args
                    cst.model = refreshed_model or cst.model
                    fallback_start_time = cs.time.time()

                    try:
                        cst.llm_response = await cs.await_owned_worker(
                            cs.perform_chat_api_call_async(**cst.cleaned_args),
                            on_cancel_result=functools.partial(_record_fallback_use_after_cancel, cst),
                        )
                        cs._require_usable_nonstream_provider_result(cst.llm_response)
                        await cs.await_owned_worker(
                            _record_provider_use(cst, cst.fallback_provider)
                        )
                        fallback_latency = cs.time.time() - fallback_start_time
                        cst.provider_manager.record_success(cst.fallback_provider, fallback_latency)
                        cst.metrics.track_llm_call(cst.fallback_provider, cst.model, fallback_latency, success=True)
                        cst.selected_provider = cst.fallback_provider
                        cst.run_first_metric_context = cs._emit_chat_run_first_rollout_metrics(
                            cst.metrics,
                            cleaned_args=cst.cleaned_args,
                            provider=cst.selected_provider,
                            model=cst.model,
                            streaming=False,
                        )
                        cst.metrics_recorded = True
                        with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
                            cst.metrics.track_provider_fallback_success(
                                requested_provider=cst.provider,
                                selected_provider=cst.fallback_provider,
                                streaming=False,
                                queued=False,
                            )
                    except (cs.LLMQuotaExceededError, cs.LLMPolicyRefusal):
                        cs._emit_chat_run_first_completion_metric(
                            cst.metrics,
                            context=cst.run_first_metric_context,
                            outcome="error",
                        )
                        raise  # the user's refusal, not the fallback provider's health (TASK-13955)
                    except Exception as fallback_error:  # noqa: BLE001 - adapter trust boundary
                        fallback_latency = cs.time.time() - fallback_start_time
                        cst.bounded_error = cs._bound_provider_invocation_exception(
                            fallback_error
                        )
                        cst.metrics.track_llm_call(
                            cst.fallback_provider,
                            cst.model,
                            fallback_latency,
                            success=False,
                            error_type=type(cst.bounded_error).__name__,
                        )
                        cst.provider_manager.record_failure(
                            cst.fallback_provider,
                            cs.sanitized_provider_stream_exception(cst.bounded_error),
                        )
                        cs._emit_chat_run_first_completion_metric(
                            cst.metrics,
                            context=cst.run_first_metric_context,
                            outcome="error",
                        )
                        cs.raise_detached_error(cst.bounded_error)
                else:
                    cs._emit_chat_run_first_completion_metric(
                        cst.metrics,
                        context=cst.run_first_metric_context,
                        outcome="error",
                    )
                    raise
            else:
                cs._emit_chat_run_first_completion_metric(
                    cst.metrics,
                    context=cst.run_first_metric_context,
                    outcome="error",
                )
                raise
        else:
            cs._emit_chat_run_first_completion_metric(
                cst.metrics,
                context=cst.run_first_metric_context,
                outcome="error",
            )
            raise

    await _response_processor.collect_non_stream_reply(cst)

    _refresh_primary_choice_state = functools.partial(_response_processor._refresh_primary_choice_state, cst)

    _apply_output_safety_to_current_choices = functools.partial(_response_processor._apply_output_safety_to_current_choices, cst)

    await _apply_output_safety_to_current_choices()

    await _persistence_service.save_first_non_stream_reply(cst)

    await _tool_execution_service.run_non_stream_tool_autoexec(cst)

    await _persistence_service.flush_deferred_non_stream_persistence(cst)

    await _response_processor.encode_non_stream_payload(cst)

    # Audit success
    if cst.audit_service and cst.audit_context:
        with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
            await cst.audit_service.log_event(
                event_type=cs.AuditEventType.API_RESPONSE,
                context=cst.audit_context,
                action="chat_completion_success",
                result="success",
                metadata={
                    "conversation_id": cst.final_conversation_id,
                    "provider": cst.selected_provider,
                    "model": cst.model,
                    "streaming": False,
                },
            )

    # Retry best-effort accounting if an earlier touch failed transiently.
    await _record_provider_use(cst, cst.selected_provider)

    return cst.encoded_payload


# late imports: moved code reads these at call time (TASK-13950.25 R5)
from tldw_Server_API.app.core.Chat import persistence_service as _persistence_service
from tldw_Server_API.app.core.Chat import response_processor as _response_processor
from tldw_Server_API.app.core.Chat import tool_execution_service as _tool_execution_service
from tldw_Server_API.app.core.Chat import chat_service as cs
