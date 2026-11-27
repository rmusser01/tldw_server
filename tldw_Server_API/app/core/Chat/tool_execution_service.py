"""Request guardrails for local chat tool auto-execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from fastapi import HTTPException, status
from typing import TYPE_CHECKING
import functools

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Chat.call_state import NonStreamCall

_ABSENT_CHOICE_STRINGS = {"", "none", "false", "null"}


class UnsupportedMultiChoiceToolAutoexecError(HTTPException):
    """Canonical local request error for unsupported tool auto-execution."""

    error_code = "unsupported_multi_choice_tool_autoexec"
    public_message = "Local tool auto-execution supports one assistant choice per request."

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": self.error_code,
                "message": self.public_message,
            },
        )


def request_choice_count(cleaned_args: dict[str, Any] | None) -> int:
    """Return the requested assistant choice count, defaulting invalid input to one."""

    if not isinstance(cleaned_args, dict):
        return 1
    raw_n = cleaned_args.get("n", 1)
    try:
        return max(1, int(raw_n))
    except (TypeError, ValueError):
        return 1


def _non_empty_request_value(value: Any) -> bool:
    """Return whether a request option declares meaningful user intent."""

    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _ABSENT_CHOICE_STRINGS
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value)
    return bool(value)


def request_declares_local_tool_use(cleaned_args: dict[str, Any] | None) -> bool:
    """Return whether the request includes local tools or legacy functions."""

    if not isinstance(cleaned_args, dict):
        return False
    return _non_empty_request_value(cleaned_args.get("tools")) or _non_empty_request_value(
        cleaned_args.get("functions")
    )


def ensure_tool_autoexec_supports_request(
    *,
    cleaned_args: dict[str, Any] | None,
    should_run_tool_autoexec: Callable[[dict[str, Any] | None], bool],
) -> None:
    """Reject ambiguous multi-choice requests when local tool auto-execution is active."""

    if (
        should_run_tool_autoexec(cleaned_args)
        and request_declares_local_tool_use(cleaned_args)
        and request_choice_count(cleaned_args) > 1
    ):
        raise UnsupportedMultiChoiceToolAutoexecError()


async def run_non_stream_tool_autoexec(cst: "NonStreamCall") -> None:
    """Run local tool autoexec and the auto-continue turns."""
    if cs.should_run_legacy_tool_autoexec(cst.cleaned_args) and isinstance(cst.tool_calls_to_save, list) and cst.tool_calls_to_save:
        try:
            req_user_id = None
            try:
                if cst.request is not None and hasattr(cst.request, "state"):
                    req_user_id = getattr(cst.request.state, "user_id", None)
            except cs._CHAT_NONCRITICAL_EXCEPTIONS:
                req_user_id = None

            autoexec_result = await cs.execute_assistant_tool_calls(
                tool_calls=cst.tool_calls_to_save,
                user_id=str(req_user_id) if req_user_id is not None else None,
                client_id=cst.client_id,
                max_tool_calls=cs.get_chat_max_tool_calls(),
                timeout_ms=cs.get_chat_tool_timeout_ms(),
                allow_catalog=cs._resolve_chat_autoexec_allow_catalog(cst.cleaned_args),
                attach_idempotency=cs.should_attach_tool_idempotency(),
                idempotency_seed=str(cst.final_conversation_id) if cst.final_conversation_id else None,
            )
            cst.tool_execution_payload = autoexec_result.event_payload().get("tool_results", [])
            tool_messages = autoexec_result.tool_messages()

            failed_results = len([r for r in autoexec_result.results if not r.ok])
            timed_out_results = len([r for r in autoexec_result.results if r.timed_out])
            cs.logger.info(
                "Chat tool auto-exec summary (conv={} requested={} attempts={} executed={} failed={} timed_out={} truncated={})",
                cst.final_conversation_id,
                autoexec_result.requested_calls,
                autoexec_result.execution_attempts,
                autoexec_result.executed_calls,
                failed_results,
                timed_out_results,
                autoexec_result.truncated,
            )

            if cst.should_persist and cst.final_conversation_id:
                if cst.defer_structured_persistence:
                    cst.pending_tool_messages = list(tool_messages)
                else:
                    await cs.save_tool_messages(
                        chat_db=cst.chat_db,
                        conversation_id=cst.final_conversation_id,
                        save_message_fn=cst.save_message_fn,
                        tool_messages=list(tool_messages),
                    )

            if cs.should_auto_continue_tools_once() and tool_messages:
                cst.tool_auto_continue_meta = {"attempted": True, "succeeded": False}
                try:
                    continuation_messages: list[dict[str, Any]] = list(cst.templated_llm_payload)
                    assistant_context_message = cs._tool_continuation_assistant_message(
                        cst.first_choice, cst.selected_provider, content=cst.content_to_save,
                        tool_calls=cst.tool_calls_to_save, function_call=cst.function_call_to_save,
                    )
                    if len(assistant_context_message) > 1:
                        continuation_messages.append(assistant_context_message)
                    continuation_messages.extend(tool_messages)

                    continuation_args = dict(cst.cleaned_args)
                    continuation_args["messages_payload"] = continuation_messages
                    continuation_args["streaming"] = False
                    continuation_start_time = cs.time.time()
                    try:
                        continuation_response = await cs.await_owned_worker(
                            cs.perform_chat_api_call_async(**continuation_args, llm_surface="chat.completions"),
                            on_cancel_result=functools.partial(_completion_pipeline._record_provider_use_after_cancel, cst),
                        )
                        cs._require_usable_nonstream_provider_result(
                            continuation_response
                        )
                    except Exception as continuation_error:  # noqa: BLE001 - adapter trust boundary
                        continuation_latency = cs.time.time() - continuation_start_time
                        cst.bounded_error = cs._bound_provider_invocation_exception(
                            continuation_error
                        )
                        cst.metrics.track_llm_call(
                            cst.selected_provider,
                            cst.model,
                            continuation_latency,
                            success=False,
                            error_type=type(cst.bounded_error).__name__,
                        )
                        if cst.provider_manager and not isinstance(continuation_error, cs.LLMPolicyRefusal):
                            cst.provider_manager.record_failure(
                                cst.selected_provider,
                                cs.sanitized_provider_stream_exception(cst.bounded_error),
                            )
                        cs.raise_detached_error(cst.bounded_error)
                    await cs.await_owned_worker(
                        _completion_pipeline._record_provider_use(cst, cst.selected_provider)
                    )
                    continuation_latency = cs.time.time() - continuation_start_time
                    cst.metrics.track_llm_call(
                        cst.selected_provider,
                        cst.model,
                        continuation_latency,
                        success=True,
                    )
                    if cst.provider_manager:
                        cst.provider_manager.record_success(
                            cst.selected_provider,
                            continuation_latency,
                        )
                    if isinstance(continuation_response, str) and cs.should_force_normalize_string_responses():
                        continuation_response = cs._wrap_raw_string_response(continuation_response, cst.model)

                    continuation_content: Any | None = None
                    if continuation_response and isinstance(continuation_response, dict):
                        continuation_choices = continuation_response.get("choices")
                        if continuation_choices and isinstance(continuation_choices, list):
                            continuation_message = continuation_choices[0].get("message") or {}
                            if isinstance(continuation_message, dict):
                                continuation_content = continuation_message.get("content")
                    elif isinstance(continuation_response, str):
                        continuation_content = continuation_response

                    if isinstance(continuation_response, dict):
                        await _completion_pipeline._record_dict_response_usage(
                            cst,
                            response=continuation_response,
                            messages=continuation_messages,
                            content=continuation_content,
                            provider_name=cst.selected_provider,
                            model_name=cst.model,
                            started_at=continuation_start_time,
                        )
                    elif isinstance(continuation_response, str):
                        await _completion_pipeline._record_dict_response_usage(
                            cst,
                            response={},
                            messages=continuation_messages,
                            content=continuation_content,
                            provider_name=cst.selected_provider,
                            model_name=cst.model,
                            started_at=continuation_start_time,
                        )

                    if continuation_response is not None:
                        cst.llm_response = continuation_response
                        cst.processed_choices = cs.collect_non_stream_choices(cst.llm_response)
                        cst.first_choice = cs.primary_choice(cst.processed_choices)
                        cst.content_to_save = cst.first_choice.content if cst.first_choice else None
                        cst.tool_calls_to_save = cst.first_choice.tool_calls if cst.first_choice else None
                        cst.function_call_to_save = cst.first_choice.function_call if cst.first_choice else None
                        if cst.first_choice is None and isinstance(cst.llm_response, str):
                            cst.content_to_save = cst.llm_response
                        cst.content_text_for_usage = cs._extract_text_from_content(cst.content_to_save)
                        await _response_processor._apply_output_safety_to_current_choices(cst)
                        if not cst.processed_choices and isinstance(cst.llm_response, str):
                            cst.llm_response = cst.content_to_save
                        try:
                            continuation_structured_metadata = cs.validate_structured_choices(
                                choices=cst.processed_choices,
                                structured_request_context=cst.structured_request_context,
                                validate_structured_response=cs.validate_structured_response,
                                fallback_content=cst.content_to_save,
                            )
                        except (
                            cs.StructuredGenerationCapabilityError,
                            cs.StructuredGenerationParseError,
                            cs.StructuredGenerationSchemaError,
                        ) as structured_exc:
                            cs._emit_chat_run_first_completion_metric(
                                cst.metrics,
                                context=cst.run_first_metric_context,
                                outcome="error",
                            )
                            cs.raise_detached_error(
                                cs.build_structured_http_exception(structured_exc)
                            )
                        if continuation_structured_metadata is not None:
                            cst.structured_metadata = continuation_structured_metadata

                        if (
                            cst.should_persist
                            and cst.final_conversation_id
                            and (cst.content_to_save or cst.tool_calls_to_save or cst.function_call_to_save)
                        ):
                            if cst.pending_assistant_payload is not None:
                                cst.assistant_message_id = await cs.save_assistant_message(
                                    chat_db=cst.chat_db,
                                    conversation_id=cst.final_conversation_id,
                                    save_message_fn=cst.save_message_fn,
                                    payload=cst.pending_assistant_payload,
                                )
                                cst.pending_assistant_payload = None
                            if cst.pending_tool_messages:
                                await cs.save_tool_messages(
                                    chat_db=cst.chat_db,
                                    conversation_id=cst.final_conversation_id,
                                    save_message_fn=cst.save_message_fn,
                                    tool_messages=cst.pending_tool_messages,
                                )
                                cst.pending_tool_messages = []
                            continuation_payload = cs._build_assistant_message_payload(
                                character_card_for_context=cst.character_card_for_context,
                                assistant_parent_message_id=cst.assistant_parent_message_id,
                                content=cst.content_to_save,
                                tool_calls=cst.tool_calls_to_save,
                                function_call=cst.function_call_to_save,
                                moonshot_reasoning=await _response_processor.approve_non_stream_moonshot_reasoning(cst),
                                generation_metadata=cs._nonstream_generation_metadata(
                                    llm_response=cst.llm_response,
                                    choice=cst.first_choice,
                                    model=cst.model,
                                    provider=cst.selected_provider,
                                ),
                            )
                            continuation_message_id = await cs.save_assistant_message(
                                chat_db=cst.chat_db,
                                conversation_id=cst.final_conversation_id,
                                save_message_fn=cst.save_message_fn,
                                payload=continuation_payload,
                            )
                            if continuation_message_id:
                                cst.assistant_message_id = continuation_message_id
                        cst.tool_auto_continue_meta["succeeded"] = True
                except cs.HTTPException:
                    raise
                except cs._CHAT_NONCRITICAL_EXCEPTIONS as continue_err:
                    cs.logger.warning(
                        "Chat tool auto-continue skipped due to error: {}",
                        cs.exception_summary(continue_err),
                    )
        except cs.HTTPException:
            raise
        except cs._CHAT_NONCRITICAL_EXCEPTIONS as autoexec_err:
            cs.logger.warning(
                "Chat tool auto-execution skipped due to error: {}",
                cs.exception_summary(autoexec_err),
            )


# late imports: moved code reads these at call time (TASK-13950.25 R5)
from tldw_Server_API.app.core.Chat import completion_pipeline as _completion_pipeline
from tldw_Server_API.app.core.Chat import response_processor as _response_processor
from tldw_Server_API.app.core.Chat import chat_service as cs
