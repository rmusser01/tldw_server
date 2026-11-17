"""Normalize an already byte-bounded provider envelope into text-only output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

from tldw_Server_API.app.core.AuthNZ.repos.provider_usage_reservations_repo import MAX_INT
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionFailure,
    ModelCompletionRequest,
    ModelFailureDomain,
)


@dataclass(frozen=True, slots=True)
class NormalizedModelCompletion:
    """Internal text and independently trusted counts, never raw provider usage."""

    content: str
    input_tokens: int | None
    output_tokens: int | None


def _invalid_output() -> NoReturn:
    """Reject without retaining a provider body or active private exception."""
    raise_detached_error(ModelCompletionFailure("invalid_model_output", ModelFailureDomain.REQUEST))


def _usage_count(value: object, *, maximum: int = MAX_INT) -> int | None:
    """Trust only exact JSON integers within the conservative accounting bound."""
    if type(value) is int and 0 <= value <= maximum:
        return value
    return None


def normalize_model_completion_response(envelope: object, request: ModelCompletionRequest) -> NormalizedModelCompletion:
    """Validate one parsed, bounded JSON completion without trimming or truncation.

    The transport must bound decompressed bytes before JSON parsing. This pure
    boundary enforces the validated request's character and UTF-8 byte ceilings
    on the entire line-ending-normalized content. Invalid optional usage is
    unknown, not a reason to reject otherwise valid text.
    """
    if type(envelope) is not dict:
        _invalid_output()
    choices = envelope.get("choices")
    if type(choices) is not list or len(choices) != 1:
        _invalid_output()
    choice = choices[0]
    if type(choice) is not dict:
        _invalid_output()
    message = choice.get("message")
    if type(message) is not dict:
        _invalid_output()
    if any("tool_calls" in part or "function_call" in part for part in (envelope, choice, message)):
        _invalid_output()
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and type(finish_reason) is not str:
        _invalid_output()
    if finish_reason in ("tool_calls", "function_call"):
        _invalid_output()
    content = message.get("content")
    if type(content) is not str:
        _invalid_output()

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    if not content.strip():
        _invalid_output()
    if any((ord(char) < 0x20 and char not in "\t\n") or 0x7F <= ord(char) <= 0x9F for char in content):
        _invalid_output()
    if len(content) > request.max_output_chars:
        _invalid_output()
    try:
        encoded = content.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        _invalid_output()
    if len(encoded) > request.max_output_bytes:
        _invalid_output()

    input_tokens = None
    output_tokens = None
    usage = envelope.get("usage")
    if type(usage) is dict:
        input_tokens = _usage_count(usage.get("prompt_tokens"))
        output_tokens = _usage_count(usage.get("completion_tokens"), maximum=min(MAX_INT, request.max_output_tokens))
    return NormalizedModelCompletion(content=content, input_tokens=input_tokens, output_tokens=output_tokens)
