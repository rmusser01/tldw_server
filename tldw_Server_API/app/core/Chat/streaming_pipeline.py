# streaming_pipeline.py
# Description: Thin streaming response assembly wrapper for Chat completions.
"""
Compatibility wrapper for constructing Chat streaming responses.

The streaming implementation remains in ``streaming_utils``. This module gives
``chat_service`` a narrower assembly boundary without changing SSE behavior.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable
from functools import partial
from tldw_Server_API.app.core.Chat.streaming_utils import StopStreamWithError
from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from typing import TYPE_CHECKING
import asyncio
import contextlib
import functools
import os
import uuid as _uuid
from collections.abc import AsyncIterator
from collections.abc import Iterator

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Chat.call_state import StreamCall


@dataclass(frozen=True)
class StreamingPipelineRequest:
    """Inputs needed to construct the Chat SSE stream."""

    stream: Any
    conversation_id: str
    model_name: str
    save_callback: Callable[..., Any] | None = None
    finalize_callback: Callable[..., Any] | None = None
    on_first_output: Callable[..., Any] | None = None
    idle_timeout: float | None = None
    heartbeat_interval: float | None = None
    text_transform: Callable[[str], str | Awaitable[str]] | None = None
    before_success_callback: Callable[..., Any] | None = None
    system_message_id: str | None = None
    continuation_metadata: dict[str, Any] | None = None
    history_persistence_ack: bool | None = None
    user_message_id: str | None = None
    partial_save_callback: Callable[..., Any] | None = None
    forward_usage_chunk: bool | None = None
    capture_provider_reasoning: bool | None = None
    retain_reasoning: bool | None = None
    defer_provider_finish: bool | None = None


def create_chat_streaming_response(
    *,
    request: StreamingPipelineRequest,
    stream_factory: Callable[..., Any],
) -> Any:
    """Create a streaming response through the injected stream factory."""
    factory_kwargs: dict[str, Any] = {
        "stream": request.stream,
        "conversation_id": request.conversation_id,
        "model_name": request.model_name,
    }
    optional_kwargs = {
        "save_callback": request.save_callback,
        "finalize_callback": request.finalize_callback,
        "on_first_output": request.on_first_output,
        "idle_timeout": request.idle_timeout,
        "heartbeat_interval": request.heartbeat_interval,
        "text_transform": request.text_transform,
        "before_success_callback": request.before_success_callback,
        "system_message_id": request.system_message_id,
        "user_message_id": request.user_message_id,
        "continuation_metadata": request.continuation_metadata,
        "history_persistence_ack": request.history_persistence_ack,
        "partial_save_callback": request.partial_save_callback,
        "forward_usage_chunk": request.forward_usage_chunk,
        "capture_provider_reasoning": request.capture_provider_reasoning,
        "retain_reasoning": request.retain_reasoning,
        "defer_provider_finish": request.defer_provider_finish,
    }
    factory_kwargs.update(
        {
            key: value
            for key, value in optional_kwargs.items()
            if value is not None
        }
    )
    return stream_factory(**factory_kwargs)


__all__ = [
    "StreamingPipelineRequest",
    "create_chat_streaming_response",
]


def _emit_with_holdback(cst: "StreamCall", text: str) -> str:
    if cst.stream_buffer_limit <= 0:
        cst.stream_holdback = ""
        return text
    if not text:
        return ""
    if len(text) <= cst.stream_buffer_limit:
        cst.stream_holdback = text
        return ""
    emit = text[:-cst.stream_buffer_limit]
    cst.stream_holdback = text[-cst.stream_buffer_limit:]
    return emit


def _flush_holdback(cst: "StreamCall") -> str:
    if not cst.stream_holdback:
        return ""
    out = cst.stream_holdback
    cst.stream_holdback = ""
    return out


async def _await_mandatory_moderation_audits(cst: "StreamCall") -> None:
    pending_tasks = list(cst.mandatory_moderation_audit_tasks)
    cst.mandatory_moderation_audit_tasks.clear()
    failures: list[BaseException] = []
    if pending_tasks:
        results = await asyncio.gather(
            *pending_tasks,
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        for failure in failures:
            cs.logger.error(
                "Mandatory moderation audit task failed for {} error_type={}",
                cst.final_conversation_id,
                type(failure).__name__,
            )
        raise StopStreamWithError(
            message="Mandatory audit persistence unavailable",
            error_type="audit_persistence_failure",
        ) from failures[0]
    pending_stopper = cst.stream_mod_state.pop("pending_stop_error", None)
    if pending_stopper is not None:
        raise pending_stopper


async def _record_provider_output_once(cst: "StreamCall") -> bool:
    """Record validated provider output once, retaining a clean-end retry."""

    if not callable(cst.on_provider_output):
        return False
    async with cst.provider_output_lock:
        if cst.provider_output_recorded:
            return True
        try:
            persisted = await cst.on_provider_output(cst.selected_provider)
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            return False
        if persisted is not False:
            cst.provider_output_recorded = True
        return cst.provider_output_recorded


async def _record_clean_success_once(cst: "StreamCall") -> bool:
    """Invoke the clean-completion callback at most once per stream."""

    if not callable(cst.on_success):
        return False
    async with cst.clean_success_lock:
        if cst.clean_success_recorded:
            return True
        try:
            persisted = await cst.on_success(cst.selected_provider)
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            return False
        if persisted is not False:
            cst.clean_success_recorded = True
        return cst.clean_success_recorded


def _retain_stream_reasoning(cst: "StreamCall") -> bool:
    return cst.keep_partial_replies and cst.native_persistence == "retain_reasoning" and cst.structured_request_context is None


async def partial_save_callback(cst: "StreamCall", full_reply: str, *, generation: dict[str, Any]) -> str | None:
    """Keep an unfinished reply after a disconnect, stop or provider failure (D7 S3-1).

        Only the text is kept: partial tool calls are incomplete, and tool
        execution, success accounting and audit success belong to complete
        replies. A reply blocked by output policy is never saved.
        """
    if not cst.keep_partial_replies or not cst.final_conversation_id:
        return None
    if cst.stream_mod_state.get("pending_stop_error") is not None:
        return None
    full_reply_to_save, post_stream_blocked, _ = await cs.prepare_saved_stream_reply(
        full_reply, functools.partial(_moderation_pipeline._apply_stream_output_policies, cst), retain_reasoning=_retain_stream_reasoning(cst),
    )
    if post_stream_blocked or not isinstance(full_reply_to_save, str) or not full_reply_to_save.strip():
        return None
    message_payload = cs.build_assistant_message_payload(
        character_card_for_context=cst.character_card_for_context,
        assistant_parent_message_id=cst.assistant_parent_message_id,
        content=full_reply_to_save,
        tool_calls=None,
        function_call=None,
        generation_metadata=cs._stream_generation_metadata(generation, model=cst.model, provider=cst.selected_provider),
    )
    return await cs.save_assistant_message(
        chat_db=cst.chat_db,
        conversation_id=cst.final_conversation_id,
        save_message_fn=cst.save_message_fn,
        payload=message_payload,
    )


def _stream_usage_recorder(
    cst: "StreamCall",
    stream_usage: cs._StreamUsage | None, full_reply: str = ""
) -> Callable[[], Awaitable[None]] | None:
    """The write of this stream's one usage row and token commit, or None once it is claimed (TASK-13953)."""
    if cst.stream_usage_recorded:
        return None
    cst.stream_usage_recorded = True
    return partial(
        cs._record_stream_usage,
        request=cst.request,
        provider=cst.selected_provider,
        model=cst.model,
        templated_llm_payload=cst.templated_llm_payload,
        conversation_id=cst.final_conversation_id,
        llm_start_time=cst.llm_start_time,
        credential_source=cst.usage_credential_source,
        stream_usage=stream_usage,
        full_reply=full_reply,
        rg_commit_cb=cst.rg_commit_cb,
    )


async def save_callback(
    cst: "StreamCall",
    full_reply: str,
    tool_calls: list[dict[str, Any]] | None,
    function_call: dict[str, Any] | None,
    *,
    generation: dict[str, Any] | None = None,
    stream_usage: cs._StreamUsage | None = None,
    provider_reasoning: str | None = None,
):
    # The usage row and the token commit, once per stream, detached: a client drop or a failure during the
    # save below must not lose it (an early end records in _finalize_stream)
    record = _stream_usage_recorder(cst, stream_usage, full_reply)
    if record is not None:
        cs.submit_accounting(record)
    saved_message_id: str | None = None
    tool_execution_payload: list[dict[str, Any]] | None = None
    structured_events: list[dict[str, Any]] = []
    full_reply_to_save, post_stream_blocked, moonshot_reasoning = await cs.prepare_saved_stream_reply(
        full_reply, functools.partial(_moderation_pipeline._apply_stream_output_policies, cst), retain_reasoning=_retain_stream_reasoning(cst),
        provider_reasoning=provider_reasoning
        if provider_reasoning and cs.official_moonshot_chat_base_url(cst.selected_provider, cst.cleaned_args) else None,
    )

    if cst.structured_request_context is not None:
        try:
            structured_metadata = cs.validate_structured_response(
                raw_text=full_reply_to_save,
                structured_request_context=cst.structured_request_context,
            )
            structured_events.append(
                {
                    "event": "structured_result",
                    "data": structured_metadata,
                }
            )
        except (
            cs.StructuredGenerationCapabilityError,
            cs.StructuredGenerationParseError,
            cs.StructuredGenerationSchemaError,
        ) as structured_exc:
            structured_events.append(
                {
                    "event": "structured_error",
                    "data": cs._structured_error_detail(structured_exc),
                }
            )

    if callable(cst.on_stream_full_reply):
        try:
            maybe_result = cst.on_stream_full_reply(cs.strip_reasoning_blocks(full_reply))
            if hasattr(maybe_result, "__await__"):
                await maybe_result  # type: ignore[misc]
        except cs._CHAT_NONCRITICAL_EXCEPTIONS as stream_callback_err:
            cs.logger.debug(
                "on_stream_full_reply callback skipped error_type={}",
                type(stream_callback_err).__name__,
            )

    if not cst.stream_metrics_recorded:
        try:
            latency = cs.time.time() - cst.llm_start_time
            cst.metrics.track_llm_call(cst.selected_provider, cst.model, latency, success=True)
            if cst.provider_manager:
                cst.provider_manager.record_success(cst.selected_provider, latency)
            cst.stream_metrics_recorded = True
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            pass

    cs._emit_chat_run_first_tool_path_metrics(
        cst.metrics,
        context=cst.run_first_metric_context,
        tool_calls=tool_calls,
        function_call=function_call,
    )
    cs._emit_chat_run_first_completion_metric(
        cst.metrics,
        context=cst.run_first_metric_context,
        outcome="blocked" if post_stream_blocked else "success",
    )

    if moonshot_reasoning and cst.client_moonshot_history:
        structured_events.append(cs.moonshot_reasoning_event(moonshot_reasoning))

    if cst.should_persist and cst.final_conversation_id and not post_stream_blocked and (
        full_reply_to_save or tool_calls or function_call
    ):
        message_payload = cs.build_assistant_message_payload(
            character_card_for_context=cst.character_card_for_context,
            assistant_parent_message_id=cst.assistant_parent_message_id,
            content=full_reply_to_save,
            tool_calls=tool_calls,
            function_call=function_call,
            generation_metadata=cs._stream_generation_metadata(generation, model=cst.model, provider=cst.selected_provider),
            moonshot_reasoning=moonshot_reasoning,
        )
        saved_message_id = await cs.save_assistant_message(
            chat_db=cst.chat_db,
            conversation_id=cst.final_conversation_id,
            save_message_fn=cst.save_message_fn,
            payload=message_payload,
        )

    if (
        not post_stream_blocked
        and cs.should_run_legacy_tool_autoexec(cst.cleaned_args)
        and isinstance(tool_calls, list)
        and tool_calls
    ):
        try:
            req_user_id = None
            try:
                if cst.request is not None and hasattr(cst.request, "state"):
                    req_user_id = getattr(cst.request.state, "user_id", None)
            except cs._CHAT_NONCRITICAL_EXCEPTIONS:
                req_user_id = None

            autoexec_result = await cs.execute_assistant_tool_calls(
                tool_calls=tool_calls,
                user_id=str(req_user_id) if req_user_id is not None else None,
                client_id=cst.client_id,
                max_tool_calls=cs.get_chat_max_tool_calls(),
                timeout_ms=cs.get_chat_tool_timeout_ms(),
                allow_catalog=cs._resolve_chat_autoexec_allow_catalog(cst.cleaned_args),
                attach_idempotency=cs.should_attach_tool_idempotency(),
                idempotency_seed=str(cst.final_conversation_id) if cst.final_conversation_id else None,
            )
            tool_execution_payload = autoexec_result.event_payload().get("tool_results", [])

            if cst.should_persist and cst.final_conversation_id:
                await cs.save_tool_messages(
                    chat_db=cst.chat_db,
                    conversation_id=cst.final_conversation_id,
                    save_message_fn=cst.save_message_fn,
                    tool_messages=autoexec_result.tool_messages(),
                )
        except cs._CHAT_NONCRITICAL_EXCEPTIONS as autoexec_err:
            cs.logger.warning(
                "Streaming chat tool auto-execution skipped error_type={}",
                type(autoexec_err).__name__,
            )
    # Audit success
    try:
        if cst.audit_service and cst.audit_context:
            await cst.audit_service.log_event(
                event_type=cs.AuditEventType.API_RESPONSE,
                context=cst.audit_context,
                action="chat_completion_success",
                result="success",
                metadata={
                    "conversation_id": cst.final_conversation_id,
                    "provider": cst.selected_provider,
                    "model": cst.model,
                    "streaming": True,
                },
            )
    except cs._CHAT_NONCRITICAL_EXCEPTIONS:
        pass
    # Retry any failed first-output accounting at the last clean boundary,
    # then notify consumers that the complete stream succeeded.
    try:
        await _record_provider_output_once(cst)
    except cs._CHAT_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        await _record_clean_success_once(cst)
    except cs._CHAT_NONCRITICAL_EXCEPTIONS:
        pass
    events: list[dict[str, Any]] = list(structured_events)
    if tool_execution_payload is not None:
        events.append(
            {
                "event": "tool_results",
                "data": {"tool_results": tool_execution_payload},
            }
        )
        if cst.loop_compat_enabled:
            seq = 1
            events.append(
                {
                    "event": "run_started",
                    "data": {"run_id": cst.loop_compat_run_id, "seq": seq},
                }
            )
            seq += 1
            if isinstance(tool_calls, list) and tool_calls:
                events.append(
                    {
                        "event": "tool_proposed",
                        "data": {"run_id": cst.loop_compat_run_id, "seq": seq, "tool_calls": tool_calls},
                    }
                )
                seq += 1
            events.append(
                {
                    "event": "tool_finished",
                    "data": {"run_id": cst.loop_compat_run_id, "seq": seq, "tool_results": tool_execution_payload},
                }
            )
            seq += 1
            events.append(
                {
                    "event": "run_complete",
                    "data": {"run_id": cst.loop_compat_run_id, "seq": seq},
                }
            )
    elif cst.loop_compat_enabled:
        events.extend(
            [
                {
                    "event": "run_started",
                    "data": {"run_id": cst.loop_compat_run_id, "seq": 1},
                },
                {
                    "event": "run_complete",
                    "data": {"run_id": cst.loop_compat_run_id, "seq": 2},
                },
            ]
        )

    if events:
        return {
            "saved_message_id": saved_message_id,
            "events": events,
        }
    return saved_message_id


async def tracked_streaming_generator(cst: "StreamCall"):
    async with cst.metrics.track_streaming(cst.final_conversation_id) as stream_tracker:
        _get_mod = cst.moderation_getter or cs.get_moderation_service
        cst.moderation = _get_mod()
        cst.req_user_id = None
        try:
            if cst.request is not None and hasattr(cst.request, "state"):
                cst.req_user_id = getattr(cst.request.state, "user_id", None)
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            cst.req_user_id = None
        cst.eff_policy = cst.moderation.get_effective_policy(str(cst.req_user_id) if cst.req_user_id is not None else cst.client_id, surface="chat.completions")

        cst.mandatory_moderation_audit_tasks: list[asyncio.Task[Any]] = []
        cst.stream_id = _uuid.uuid4().hex
        cst.chunk_seq = 0

        cst.stream_holdback = ""
        if cs.CHAT_STREAM_INCLUDE_METADATA and cst.final_conversation_id:
            metadata_payload = {
                "event": "tldw_metadata",
                "conversation_id": cst.final_conversation_id,
                "tldw_conversation_id": cst.final_conversation_id,
            }
            if cst.user_message_id:
                metadata_payload["tldw_user_message_id"] = cst.user_message_id
            if cst.system_message_id:
                metadata_payload["tldw_system_message_id"] = cst.system_message_id
            if cst.normalized_continuation_metadata:
                metadata_payload["tldw_continuation"] = cst.normalized_continuation_metadata
            yield cs.sse_data(metadata_payload)
        try:
            cst.stream_buffer_limit = int(os.getenv("MODERATION_STREAM_BUFFER_CHARS", "1024"))
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            cst.stream_buffer_limit = 1024
        if cst.stream_buffer_limit < 0:
            cst.stream_buffer_limit = 0

        _out_transform = functools.partial(_moderation_pipeline._out_transform, cst)

        # Allow streaming handler to flush any held-back tail at stream end
        with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
            _out_transform.flush = functools.partial(_flush_holdback, cst)

        async def _finalize_stream(
            *, success: bool, cancelled: bool, error: bool, stream_usage: cs._StreamUsage | None = None
        ) -> None:
            if not success and not cst.stream_failure_recorded and not cst.stream_metrics_recorded:
                try:
                    latency = cs.time.time() - cst.llm_start_time
                    error_type = "stream_cancelled" if cancelled else "stream_error" if error else "stream_incomplete"
                    cst.metrics.track_llm_call(
                        cst.selected_provider,
                        cst.model,
                        latency,
                        success=False,
                        error_type=error_type,
                    )
                    if (
                        cst.provider_manager and error and not cancelled
                        and not isinstance(cst.stream_mod_state.get("pending_stop_error"), ModerationExecutionError)
                    ):
                        cst.provider_manager.record_failure(cst.selected_provider, RuntimeError(error_type))
                    cst.stream_failure_recorded = True
                except cs._CHAT_NONCRITICAL_EXCEPTIONS:
                    pass
            if success:
                return
            if not (stream_usage and (stream_usage.usage is not None or stream_usage.chars)):
                # Nothing came back (the endpoint may be retrying this attempt): refund, as before (TASK-13953)
                await cs._maybe_refund_streaming_rg(cst.rg_refund_cb, cancelled=cancelled, error=error)
                return
            # Recorded and committed, never refunded: the stream was read (TASK-13953). Detached, because a
            # cancelled stream's teardown may cut any await here.
            record = _stream_usage_recorder(cst, stream_usage)
            if record is not None:
                cs.submit_accounting(record)

        generator = cs.create_chat_streaming_response(
            request=cs.StreamingPipelineRequest(
                stream=cst.raw_stream_iter,  # type: ignore[arg-type]
                conversation_id=cst.final_conversation_id,
                model_name=cst.model,
                save_callback=functools.partial(save_callback, cst),
                partial_save_callback=functools.partial(partial_save_callback, cst) if cst.keep_partial_replies else None,
                finalize_callback=_finalize_stream,
                on_first_output=functools.partial(_record_provider_output_once, cst),
                idle_timeout=cs.CHAT_IDLE_TIMEOUT,
                heartbeat_interval=cs.CHAT_HEARTBEAT_INTERVAL,
                text_transform=_out_transform,
                before_success_callback=functools.partial(_await_mandatory_moderation_audits, cst),
                system_message_id=cst.system_message_id,
                user_message_id=cst.user_message_id,
                continuation_metadata=cst.normalized_continuation_metadata,
                history_persistence_ack=bool(cst.native_persistence) or None,
                forward_usage_chunk=cst.forward_usage_chunk,
                capture_provider_reasoning=cst.capture_moonshot_reasoning or None,
                retain_reasoning=_retain_stream_reasoning(cst) or None,
                defer_provider_finish=True,
            ),
            stream_factory=cs.create_streaming_response_with_timeout,
        )
        try:
            async for chunk in generator:
                if "heartbeat" in chunk:
                    stream_tracker.add_heartbeat()
                else:
                    stream_tracker.add_chunk()
                yield chunk
        finally:
            if cst.mandatory_moderation_audit_tasks:
                await asyncio.gather(*cst.mandatory_moderation_audit_tasks, return_exceptions=True)


def _remaining_provider_factory_timeout(cst: "StreamCall") -> float:
    now = cs._provider_factory_monotonic()
    if cst.provider_factory_deadline is None:
        cst.provider_factory_deadline = now + cst.factory_budget_seconds
    return max(0.0, cst.provider_factory_deadline - now)


def _public_stream_error_payload(cst: "StreamCall", value: Any) -> dict[str, Any]:
    if isinstance(value, cs.UnsupportedMultiChoiceToolAutoexecError):
        payload: dict[str, Any] = {
            "error": {
                "code": value.error_code,
                "type": value.error_code,
                "message": value.public_message,
            }
        }
    else:
        payload = cs.provider_stream_error_payload(value)
    if cs.CHAT_STREAM_INCLUDE_METADATA and cst.final_conversation_id:
        payload["conversation_id"] = cst.final_conversation_id
        payload["tldw_conversation_id"] = cst.final_conversation_id
        if cst.user_message_id:
            payload["tldw_user_message_id"] = cst.user_message_id
        if cst.system_message_id:
            payload["tldw_system_message_id"] = cst.system_message_id
        if cst.normalized_continuation_metadata:
            payload["tldw_continuation"] = cst.normalized_continuation_metadata
    return payload


async def _terminal_stream_error(cst: "StreamCall", payload: dict[str, Any]) -> AsyncIterator[str]:
    yield cs.sse_data(payload)
    yield cs.sse_done()


def _refresh_params_sync(cst: "StreamCall", target_provider: str) -> tuple[dict[str, Any], str | None]:
    """Resolve provider params inside a sync queue processor."""
    return cs._refresh_provider_params_bounded_sync(
        cst.refresh_provider_params,
        target_provider,
        functools.partial(_remaining_provider_factory_timeout, cst),
        cst.current_loop,
    )


def _queued_processor(cst: "StreamCall"):
    local_start = cs.time.time()
    try:
        result = cs._call_stream_factory_from_isolated_worker(
            cst.llm_call_func,
            functools.partial(_remaining_provider_factory_timeout, cst),
        )
        if not hasattr(result, "__aiter__") and hasattr(result, "__iter__"):
            result = cs.wrap_sync_stream(result)
        if cst.selected_provider != cst.provider:
            with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
                cst.metrics.track_provider_fallback_success(
                    requested_provider=cst.provider,
                    selected_provider=cst.selected_provider,
                    streaming=True,
                    queued=True,
                )
        return result
    except (cs.LLMQuotaExceededError, cs.LLMPolicyRefusal):
        raise  # the user's limit or a policy refusal, not the provider's health: no failure, no fallback
    except Exception as proc_error:  # noqa: BLE001 - adapter trust boundary
        latency = cs.time.time() - local_start
        cst.metrics.track_llm_call(
            cst.selected_provider,
            cst.model,
            latency,
            success=False,
            error_type=type(proc_error).__name__,
        )
        cst.stream_failure_recorded = True
        if cst.provider_manager:
            cst.provider_manager.record_failure(
                cst.selected_provider,
                cs.sanitized_provider_stream_exception(proc_error),
            )
            if (
                cst.enable_provider_fallback
                and isinstance(proc_error, (cs.ChatProviderError, cs.ChatAPIError))
                and not cs._is_client_like_error(proc_error)
                and cs.provider_stream_error_allows_replay(proc_error)
            ):
                fallback_provider = cst.provider_manager.get_available_provider(exclude=[cst.selected_provider])
                if fallback_provider:
                    cs.logger.warning(
                        f"Trying fallback provider {fallback_provider} after {cst.selected_provider} failed (queued)"
                    )
                    try:
                        refreshed_args, refreshed_model = _refresh_params_sync(cst, fallback_provider)
                        if not isinstance(refreshed_args, dict) or "messages_payload" not in refreshed_args:
                            raise ValueError(
                                f"Invalid refreshed params for {fallback_provider}: missing required fields"
                            )
                        try:
                            cst.structured_request_context = cs.prepare_structured_response_request(
                                provider=fallback_provider,
                                response_format=(
                                    refreshed_args.get("response_format")
                                ),
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
                                fallback_provider,
                                cs.sanitized_provider_stream_exception(refresh_error),
                            )
                        cs.raise_detached_error(
                            cs._bound_provider_http_exception(refresh_error)
                        )
                    try:
                        cs.ensure_tool_autoexec_supports_request(
                            cleaned_args=refreshed_args,
                            should_run_tool_autoexec=cs.should_run_legacy_tool_autoexec,
                        )
                    except cs.UnsupportedMultiChoiceToolAutoexecError as guard_error:
                        return iter(
                            [
                                cs.trusted_local_stream_error_frame(
                                    guard_error.error_code
                                )
                            ]
                        )
                    cs._check_fallback_prompt_cost(refreshed_args, fallback_provider, refreshed_model or cst.model, streaming=True)
                    cst.model = refreshed_model or cst.model
                    def llm_call_func_fb():
                        return cs.perform_chat_api_call(**refreshed_args, llm_surface="chat.completions")
                    try:
                        result = cs._call_stream_factory_from_isolated_worker(
                            llm_call_func_fb,
                            functools.partial(_remaining_provider_factory_timeout, cst),
                        )
                        if not hasattr(result, "__aiter__") and hasattr(result, "__iter__"):
                            result = cs.wrap_sync_stream(result)
                        cst.cleaned_args = refreshed_args
                        cst.selected_provider = fallback_provider
                        cst.usage_credential_source = cs.call_credential_source(refreshed_args)
                        # Update run-first metric context to reflect fallback provider/model
                        if cst.run_first_metric_context is not None:
                            cst.run_first_metric_context["provider"] = str(fallback_provider or "").strip() or "unknown"
                            cst.run_first_metric_context["model"] = str(cst.model or "").strip() or "unknown"
                        cst.llm_call_func = llm_call_func_fb
                        with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
                            cst.metrics.track_provider_fallback_success(
                                requested_provider=cst.provider,
                                selected_provider=fallback_provider,
                                streaming=True,
                                queued=True,
                            )
                        return result
                    except (cs.LLMQuotaExceededError, cs.LLMPolicyRefusal):
                        raise  # the user's refusal, not the fallback provider's health (TASK-13955)
                    except cs._CHAT_NONCRITICAL_EXCEPTIONS as fallback_error:
                        cst.provider_manager.record_failure(
                            fallback_provider,
                            cs.sanitized_provider_stream_exception(fallback_error),
                        )
                        cs.raise_detached_error(
                            cs._bound_provider_invocation_exception(fallback_error)
                        )
        cs.raise_detached_error(cs._bound_provider_invocation_exception(proc_error))


async def _channel_stream(cst: "StreamCall"):
    graceful_end = False
    try:
        while True:
            try:
                # Add timeout to prevent indefinite hang if producer crashes
                # without sending the sentinel None value
                item = await asyncio.wait_for(
                    cst.stream_channel.get(),
                    timeout=float(cs.CHAT_IDLE_TIMEOUT)
                )
            except asyncio.TimeoutError:
                cs.logger.warning(
                    "Stream channel get timed out after {}s - "
                    "producer may have crashed without sending sentinel",
                    cs.CHAT_IDLE_TIMEOUT
                )
                try:
                    if cst.queue_future is not None and not cst.queue_future.done():
                        cst.queue_future.cancel()
                except cs._CHAT_NONCRITICAL_EXCEPTIONS:
                    pass
                try:
                    while True:
                        cst.stream_channel.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                yield cs.sse_data(_public_stream_error_payload(cst, 'provider_unavailable'))
                break
            if item is None:
                graceful_end = True
                break
            if isinstance(item, cs.QueueStreamTerminalError):
                yield (
                    cs.sse_data(_public_stream_error_payload(cst, item.code))
                )
                yield cs.sse_done()
                break
            yield item
    except asyncio.CancelledError:
        cs.logger.info(
            "Queued stream consumer cancelled for request {}; cancelling queued job",
            cst.queue_request_id,
        )
        raise
    finally:
        # Pragmatic disconnect handling: cancel queued producer immediately so
        # workers do not remain blocked waiting on a full channel.
        if not graceful_end:
            with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
                if cst.queue_future is not None and not cst.queue_future.done():
                    cst.queue_future.cancel()
            with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
                while True:
                    try:
                        cst.stream_channel.get_nowait()
                    except asyncio.QueueEmpty:
                        break


async def _produce(cst: "StreamCall"):
    try:
        async for ln in cst.streaming_generator:
            if not ln:
                continue
            if cs.is_done_line(ln):
                # Suppress provider DONE; emit unified DONE immediately and stop producing
                if not cst.done_seen:
                    await cst.sse_stream.done()
                    cst.done_seen = True
                break
            if cs.is_trusted_local_stream_frame(ln):
                # This marker is created only after raw provider frames pass
                # through the adapter sanitizer. Preserve its identity across
                # the unified queue; ASGI encoding removes it before delivery.
                await cst.sse_stream._enqueue(ln)
            else:
                await cst.sse_stream.send_raw_sse_line(ln)
        if not cst.done_seen:
            await cst.sse_stream.done()
    except asyncio.CancelledError:
        raise
    except cs._CHAT_NONCRITICAL_EXCEPTIONS as e:
        # As a safeguard; tracked_streaming_generator typically yields error frames itself
        normalized_error = cs.normalize_provider_stream_error(e)
        code = normalized_error.code if normalized_error else "provider_unavailable"
        await cst.sse_stream.error(
            code,
            cs.provider_stream_error_payload(code)["error"]["message"],
            force=True,
        )


async def _gen(cst: "StreamCall"):
    prod = asyncio.create_task(_produce(cst))
    try:
        async for line in cst.sse_stream.iter_sse():
            yield line
    finally:
        await cs.cancel_stream_tasks_bounded([prod])


def prepare_stream_request(cst: "StreamCall") -> None:
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
        cs._evaluate_chat_prompt_cost_guardrails(
            cleaned_args=cst.cleaned_args,
            templated_llm_payload=cst.templated_llm_payload,
            selected_provider=cst.selected_provider,
            model=cst.model,
            streaming=True,
            continuation_metadata=cst.normalized_continuation_metadata,
        )
    except (
        cs.StructuredGenerationCapabilityError,
        cs.StructuredGenerationParseError,
        cs.StructuredGenerationSchemaError,
    ) as structured_exc:
        cs.raise_detached_error(cs.build_structured_http_exception(structured_exc))


async def start_queued_stream(cst: "StreamCall") -> None:
    try:
        _maxsz_raw = os.getenv("CHAT_STREAM_CHANNEL_MAXSIZE")
        if not _maxsz_raw:
            from tldw_Server_API.app.core.config import load_comprehensive_config
            _cp = load_comprehensive_config()
            _maxsz_raw = _cp.get('Chat-Module', 'chat_stream_channel_maxsize', fallback='100') if _cp else '100'
        stream_channel_maxsize = int(str(_maxsz_raw))
    except cs._CHAT_NONCRITICAL_EXCEPTIONS:
        stream_channel_maxsize = 100
    cst.stream_channel: asyncio.Queue = cs.QueueStreamChannel(
        maxsize=stream_channel_maxsize
    )
    est_tokens_for_queue = cs.estimate_tokens_from_json(cst.request_json)

    cst.queue_request_id = cst.queue_request_id or cs.get_request_id() or "unknown"
    try:
        cst.queue_future = await cst.queue_for_exec.enqueue(
            request_id=cst.queue_request_id,
            request_data={"endpoint": "/api/v1/chat/completions", "mode": "stream"},
            client_id=str(cst.client_id),
            priority=cs.RequestPriority.HIGH,
            estimated_tokens=est_tokens_for_queue,
            processor=functools.partial(_queued_processor, cst),
            processor_args=(),
            processor_kwargs={},
            streaming=True,
            stream_channel=cst.stream_channel,
            stream_factory_timeout=cst.factory_budget_seconds,
        )
        cs._attach_queue_future_logger(cst.queue_future, cst.queue_request_id)
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

    cst.raw_stream_iter = _channel_stream(cst)


def record_stream_http_error(cst: "StreamCall", he, local_signal_kind) -> None:
    cst.metrics.track_llm_call(
        cst.selected_provider,
        cst.model,
        cs.time.time() - cst.llm_start_time,
        success=False,
        error_type=type(he).__name__,
    )
    cst.error_payload = _public_stream_error_payload(cst, he)
    if (
        he.status_code == cs.status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        and local_signal_kind == "prompt_cost_guardrail"
        and isinstance(he.detail, dict)
        and he.detail.get("code") == "prompt_cost_guardrail_block"
        and he.detail.get("type") == "prompt_cost_guardrail_block"
    ):
        cst.error_payload["error"] = {
            "code": "prompt_cost_guardrail_block",
            "type": "prompt_cost_guardrail_block",
            "message": "Prompt cost guardrail blocked request before provider dispatch.",
        }


def build_stream_error_response(cst: "StreamCall", he, local_signal_kind) -> None:
    cst.response = cs.StreamingResponse(
        _terminal_stream_error(cst, cst.error_payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
    if (
        he.status_code == cs.status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        and local_signal_kind == "prompt_cost_guardrail"
    ):
        cs._mark_trusted_local_chat_signal(cst.response, "prompt_cost_guardrail")


async def run_streaming_call(cst: "StreamCall") -> cs.StreamingResponse:
    """Run one streaming call."""

    cst.llm_start_time = cs.time.time()
    cst.usage_credential_source = cs.call_credential_source(cst.cleaned_args)  # whose key the reply ran on, for the usage row
    cst.factory_budget_seconds = (
        cs.PROVIDER_STREAM_FACTORY_TIMEOUT_SECONDS
        if cst.provider_factory_timeout is None
        else max(0.0, float(cst.provider_factory_timeout))
    )
    cst.provider_factory_deadline: float | None = None

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
        streaming=True,
    )
    cst.stream_metrics_recorded = False
    cst.stream_failure_recorded = False
    cst.stream_usage_recorded = False  # one usage row per stream, however it ends (TASK-13953)
    cst.raw_stream_iter: AsyncIterator[str] | Iterator[str] | None = None
    cst.queue_for_exec = None
    cst.queue_future: asyncio.Future[Any] | None = None
    queue_enabled = False

    try:
        prepare_stream_request(cst)
        try:
            cst.queue_for_exec = cs.get_request_queue()
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            cst.queue_for_exec = None
        queue_enabled = (
            cst.queue_execution_enabled
            and cst.queue_for_exec is not None
            and cs.queue_is_active(cst.queue_for_exec)
        )

        if queue_enabled:
            # Submit streaming job to the queue and bridge chunks via channel
            # Bounded per-request channel for queued streaming; size is configurable via config.txt
            await start_queued_stream(cst)
        else:
            # Execute provided LLM call function in a worker to avoid blocking the loop.
            # llm_call_func is a sync callable (partial of perform_chat_api_call or a mock).
            cst.raw_stream_iter = await cs._call_stream_factory_bounded(
                cst.llm_call_func,
                timeout=_remaining_provider_factory_timeout(cst),
            )
            if cst.selected_provider != cst.provider:
                with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
                    cst.metrics.track_provider_fallback_success(
                        requested_provider=cst.provider,
                        selected_provider=cst.selected_provider,
                        streaming=True,
                        queued=False,
                    )
    except asyncio.CancelledError:
        raise
    except cs.HTTPException as he:
        cs._emit_chat_run_first_completion_metric(
            cst.metrics,
            context=cst.run_first_metric_context,
            outcome="error",
        )
        local_signal_kind = cs.trusted_local_chat_signal_kind(he)
        if local_signal_kind == "queue_admission":
            raise
        record_stream_http_error(cst, he, local_signal_kind)
        await cs._maybe_refund_streaming_rg(cst.rg_refund_cb, cancelled=False, error=True)
        if isinstance(he, (cs.LLMQuotaExceededError, cs.LLMPolicyRefusal)):
            raise  # quota (§4b) or policy refusal: the caller gets its 402/400, not a provider-style error frame
        build_stream_error_response(cst, he, local_signal_kind)
        return cst.response
    except cs._CHAT_NONCRITICAL_EXCEPTIONS as e:
        cst.metrics.track_llm_call(
            cst.selected_provider,
            cst.model,
            cs.time.time() - cst.llm_start_time,
            success=False,
            error_type=type(e).__name__,
        )
        cst._fallback_stream_ok = False
        if cst.provider_manager and not queue_enabled:
            cst.provider_manager.record_failure(
                cst.selected_provider,
                cs.sanitized_provider_stream_exception(e),
            )
            # Only fallback on upstream/server errors; skip fallback for client/config errors
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
                        cst.refreshed_args, cst.refreshed_model = await cs._refresh_provider_params_bounded(
                            cst.refresh_provider_params,
                            cst.fallback_provider,
                            functools.partial(_remaining_provider_factory_timeout, cst),
                        )
                        # Validate refreshed params have required fields before proceeding
                        if not isinstance(cst.refreshed_args, dict) or "messages_payload" not in cst.refreshed_args:
                            raise ValueError(f"Invalid refreshed params for {cst.fallback_provider}: missing required fields")
                        try:
                            cst.structured_request_context = cs.prepare_structured_response_request(
                                provider=cst.fallback_provider,
                                response_format=cst.refreshed_args.get("response_format"),
                                requested_response_format=(
                                    cst.structured_request_context.requested_response_format
                                    if cst.structured_request_context is not None
                                    else cst.refreshed_args.get("_structured_requested_response_format")
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
                            cleaned_args=cst.refreshed_args,
                            structured_request_context=cst.structured_request_context,
                        )
                    except cs._CHAT_NONCRITICAL_EXCEPTIONS as refresh_error:
                        cst.provider_manager.record_failure(
                            cst.fallback_provider,
                            cs.sanitized_provider_stream_exception(refresh_error),
                        )
                        raise
                    try:
                        cs._check_fallback_prompt_cost(cst.refreshed_args, cst.fallback_provider, cst.refreshed_model or cst.model, streaming=True)
                        cs.ensure_tool_autoexec_supports_request(
                            cleaned_args=cst.refreshed_args,
                            should_run_tool_autoexec=cs.should_run_legacy_tool_autoexec,
                        )
                    except cs.HTTPException as guard_error:
                        cs._emit_chat_run_first_completion_metric(
                            cst.metrics,
                            context=cst.run_first_metric_context,
                            outcome="error",
                        )
                        await cs._maybe_refund_streaming_rg(cst.rg_refund_cb, cancelled=False, error=True)
                        return cs.StreamingResponse(
                            _terminal_stream_error(
                                cst,
                                _public_stream_error_payload(cst, guard_error)
                            ),
                            media_type="text/event-stream",
                            headers={
                                "Cache-Control": "no-cache, no-transform",
                                "X-Accel-Buffering": "no",
                            },
                        )
                    cst.cleaned_args = cst.refreshed_args
                    cst.usage_credential_source = cs.call_credential_source(cst.cleaned_args)
                    cst.model = cst.refreshed_model or cst.model
                    fallback_start_time = cs.time.time()
                    def llm_call_func_fb():
                        return cs.perform_chat_api_call(**cst.cleaned_args, llm_surface="chat.completions")
                    try:
                        cst.raw_stream_iter = await cs._call_stream_factory_bounded(
                            llm_call_func_fb,
                            timeout=_remaining_provider_factory_timeout(cst),
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
                            streaming=True,
                        )
                        cst.llm_call_func = llm_call_func_fb
                        cst._fallback_stream_ok = True
                        # Explicit telemetry for direct (non-queued) streaming fallback success
                        with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
                            cst.metrics.track_provider_fallback_success(
                                requested_provider=cst.provider,
                                selected_provider=cst.fallback_provider,
                                streaming=True,
                                queued=False,
                            )
                    except (cs.LLMQuotaExceededError, cs.LLMPolicyRefusal):
                        raise  # the user's refusal, not the fallback provider's health (TASK-13955)
                    except cs._CHAT_NONCRITICAL_EXCEPTIONS as fallback_error:
                        cst.provider_manager.record_failure(
                            cst.fallback_provider,
                            cs.sanitized_provider_stream_exception(fallback_error),
                        )
                        raise

        if not cst._fallback_stream_ok:
            cs._emit_chat_run_first_completion_metric(
                cst.metrics,
                context=cst.run_first_metric_context,
                outcome="error",
            )

            cst.error_payload = _public_stream_error_payload(cst, e)
            await cs._maybe_refund_streaming_rg(cst.rg_refund_cb, cancelled=False, error=True)
            return cs.StreamingResponse(
                _terminal_stream_error(cst, cst.error_payload),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                },
            )

    if not (hasattr(cst.raw_stream_iter, "__aiter__") or hasattr(cst.raw_stream_iter, "__iter__")):
        raise cs.HTTPException(status_code=cs.status.HTTP_502_BAD_GATEWAY, detail="Provider did not return a valid stream.")

    cst.stream_mod_state = {
        "block_logged": False,
        "redact_logged": False,
        "block_capture_logged": False,
        "redact_capture_logged": False,
        "warn_capture_logged": False,
    }
    cst.provider_output_recorded = False
    cst.provider_output_lock = asyncio.Lock()
    cst.clean_success_recorded = False
    cst.clean_success_lock = asyncio.Lock()

    cst.loop_compat_enabled = False
    cst.loop_compat_run_id = str(cst.final_conversation_id or f"run_{_uuid.uuid4().hex[:12]}")
    try:
        if cst.request is not None:
            cst.loop_compat_enabled = cs._shared_is_truthy(cst.request.headers.get("X-TLDW-Loop-Compat"))
    except cs._CHAT_NONCRITICAL_EXCEPTIONS:
        cst.loop_compat_enabled = False

    # Partial replies are kept only for turns admitted through native history
    # selection, where the server owns settlement. Other save_to_db clients
    # retry an unanswered turn with tldw_retry_failed_turn, and that contract
    # needs the failed turn to have no saved reply.
    cst.keep_partial_replies = bool(cst.should_persist and cst.native_persistence)

    cst.client_moonshot_history = not cst.should_persist and cst.cleaned_args.get("_chat_moonshot_reasoning_history") is True
    cst.capture_moonshot_reasoning = bool(cst.should_persist and cst.final_conversation_id) or cst.client_moonshot_history

    tracked_streaming_generator = functools.partial(cs._streaming_pipeline.tracked_streaming_generator, cst)

    cst.streaming_generator = tracked_streaming_generator()

    # Feature-flagged: route through unified SSE abstraction for pilot
    try:
        use_unified = cs._shared_is_truthy(os.getenv("STREAMS_UNIFIED", "0"))
    except cs._CHAT_NONCRITICAL_EXCEPTIONS:
        use_unified = False

    if use_unified:
        # Use SSEStream to standardize lifecycle + metrics; forward lines from the
        # existing tracked generator, filtering provider [DONE] and emitting our own.
        from tldw_Server_API.app.core.Streaming.streams import SSEStream

        cst.sse_stream = SSEStream(
            labels={"component": "chat", "endpoint": "chat_completions_stream"},
            provider_control_passthru=True,
        )
        cst.done_seen = False

        return cs.StreamingResponse(
            _gen(cst),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    # Legacy path: return the tracked generator directly
    return cs.StreamingResponse(
        cst.streaming_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# late imports: moved code reads these at call time (TASK-13950.25 R5)
from tldw_Server_API.app.core.Chat import moderation_pipeline as _moderation_pipeline
from tldw_Server_API.app.core.Chat import chat_service as cs
