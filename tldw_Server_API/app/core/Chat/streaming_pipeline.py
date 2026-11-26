# streaming_pipeline.py
# Description: Thin streaming response assembly wrapper for Chat completions.
"""
Compatibility wrapper for constructing Chat streaming responses.

The streaming implementation remains in ``streaming_utils``. This module gives
``chat_service`` a narrower assembly boundary without changing SSE behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from collections.abc import Awaitable
from functools import partial
from tldw_Server_API.app.core.Chat.streaming_utils import StopStreamWithError
from typing import TYPE_CHECKING
import asyncio
import contextlib
import functools
import os
import uuid as _uuid

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
    text_transform: Callable[[str], str] | None = None
    before_success_callback: Callable[..., Any] | None = None
    system_message_id: str | None = None
    continuation_metadata: dict[str, Any] | None = None
    history_persistence_ack: bool | None = None
    user_message_id: str | None = None
    partial_save_callback: Callable[..., Any] | None = None
    forward_usage_chunk: bool | None = None
    capture_provider_reasoning: bool | None = None
    retain_reasoning: bool | None = None


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
        cst.eff_policy = cst.moderation.get_effective_policy(str(cst.req_user_id) if cst.req_user_id is not None else cst.client_id)

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
                    if cst.provider_manager and error and not cancelled:
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


# late imports: moved code reads these at call time (TASK-13950.25 R5)
from tldw_Server_API.app.core.Chat import moderation_pipeline as _moderation_pipeline
from tldw_Server_API.app.core.Chat import chat_service as cs
