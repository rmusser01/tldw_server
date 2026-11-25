"""Per-call state for the chat executors (TASK-13950.25 R6).

``NonStreamCall`` holds every parameter and every local of ``_execute_non_stream_call_impl`` that crosses a
function boundary once the executor is decomposed. The executor reads and writes them as ``cst.<name>``.

Stdlib only. An unset slot raises ``UnboundLocalError``, as the local it replaces does; that is not in
``chat_service._CHAT_NONCRITICAL_EXCEPTIONS``, whereas a plain ``AttributeError`` is and would be swallowed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PARAMS: tuple[str, ...] = (
    "current_loop",
    "cleaned_args",
    "selected_provider",
    "provider",
    "model",
    "request_json",
    "request",
    "metrics",
    "provider_manager",
    "templated_llm_payload",
    "should_persist",
    "final_conversation_id",
    "character_card_for_context",
    "chat_db",
    "save_message_fn",
    "system_message_id",
    "user_message_id",
    "audit_service",
    "audit_context",
    "client_id",
    "queue_execution_enabled",
    "enable_provider_fallback",
    "llm_call_func",
    "refresh_provider_params",
    "structured_request_context",
    "moderation_getter",
    "on_success",
    "self_monitoring_service",
    "assistant_parent_message_id",
    "continuation_metadata",
)

LOCALS: tuple[str, ...] = (
    "assistant_message_id",
    "content_text_for_usage",
    "content_to_save",
    "defer_structured_persistence",
    "encoded_payload",
    "fallback_provider",
    "first_choice",
    "first_turn_function_call",
    "first_turn_tool_calls",
    "function_call_to_save",
    "llm_response",
    "llm_start_time",
    "normalized_continuation_metadata",
    "pending_assistant_payload",
    "pending_tool_messages",
    "processed_choices",
    "prompt_guardrail_decision",
    "provider_failure_type",
    "provider_use_lock",
    "provider_use_recorded",
    "queue_enabled",
    "queue_failure_recorded",
    "queue_for_exec",
    "run_first_metric_context",
    "structured_metadata",
    "tool_auto_continue_meta",
    "tool_calls_to_save",
    "tool_execution_payload",
)


class NonStreamCall:
    """The state of one non-streaming chat call: one slot per name, no defaults."""

    PARAMS = PARAMS
    LOCALS = LOCALS
    __slots__ = (*PARAMS, *LOCALS, "__weakref__")

    @classmethod
    def from_call_locals(cls, values: Mapping[str, Any], /) -> NonStreamCall:
        """Build the state from the executor's ``locals()``, taken as its first statement (parameters only)."""
        self = cls()
        for name in PARAMS:
            setattr(self, name, values[name])
        return self

    def __getattr__(self, name: str) -> Any:
        if name in type(self).__slots__:
            raise UnboundLocalError(f"cannot access local variable '{name}' where it is not associated with a value")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __copy__(self) -> NonStreamCall:
        """Copy the set slots only: ``copyreg`` reads every slot, and an unset one would raise ``UnboundLocalError``."""
        new = type(self)()
        for name in PARAMS + LOCALS:
            try:
                setattr(new, name, object.__getattribute__(self, name))
            except AttributeError:
                pass
        return new
