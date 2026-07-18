"""API v1 message endpoints and Anthropic conversion helpers."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, status
from loguru import logger
from starlette.responses import JSONResponse, StreamingResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, check_rate_limit, get_request_user
from tldw_Server_API.app.api.v1.schemas.anthropic_messages import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.core.AuthNZ.byok_helpers import derive_trusted_credential_scope
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    record_byok_missing_credentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    capture_provider_override_call_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    ProviderCredentialRuntime,
    mark_provider_credential_used,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import await_owned_worker
from tldw_Server_API.app.core.Chat.chat_service import (
    perform_chat_api_call_async,
    resolve_provider_and_model,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    invoke_stream_close_bounded,
    normalize_provider_stream_error,
)
from tldw_Server_API.app.core.config import loaded_config_data
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.http_client import create_async_client as async_http_client_factory
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import provider_auth_is_resolved
from tldw_Server_API.app.core.LLM_Calls.anthropic_messages import (
    anthropic_messages_to_openai,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    openai_response_to_anthropic,
    openai_stream_to_anthropic,
)
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
from tldw_Server_API.app.core.testing import is_test_mode, is_truthy

router = APIRouter()
public_router = APIRouter()

# Backward-compatible symbol used by legacy tests.
http_client_factory = async_http_client_factory

MESSAGES_NATIVE_PROVIDERS = {"anthropic", "llama.cpp"}
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_MESSAGES_SSE_ACCOUNTING_BUFFER_LIMIT = 1024 * 1024
_MESSAGES_SSE_CONTROL_EVENTS = {"heartbeat", "keepalive", "ping", "pong"}
_MESSAGES_SSE_STRUCTURAL_EVENTS = {
    "content_block_stop",
    "message_delta",
    "message_start",
    "message_stop",
}
_MESSAGES_SSE_CONTENT_EVENTS = {"content_block_delta", "content_block_start"}
_MESSAGES_SSE_KNOWN_EVENTS = (
    _MESSAGES_SSE_CONTROL_EVENTS
    | _MESSAGES_SSE_STRUCTURAL_EVENTS
    | _MESSAGES_SSE_CONTENT_EVENTS
    | {"error"}
)
_MESSAGES_CREDENTIAL_ERROR_MESSAGES = {
    "invalid_provider_credentials": "The selected provider credentials are invalid.",
    "credential_store_unavailable": "Provider credential storage is temporarily unavailable.",
    "credential_scope_revoked": "The selected provider credential scope is no longer available.",
}

logger.debug(
    "messages module initialized; router={}, public_router={}, native_providers={}, default_anthropic_version={}",
    router,
    public_router,
    sorted(MESSAGES_NATIVE_PROVIDERS),
    DEFAULT_ANTHROPIC_VERSION,
)


def _openai_response_has_structural_error(response: Any) -> bool:
    """Return whether an OpenAI response contains an explicit error envelope.

    Assistant text and tool arguments are domain data, even when they contain
    words such as ``provider_unavailable`` or JSON with an ``error`` key. Only
    protocol-owned response containers and typed content blocks are inspected.
    """

    if not isinstance(response, dict):
        return False
    if normalize_provider_stream_error(response) is not None:
        return True
    choices = response.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        if normalize_provider_stream_error(choice) is not None:
            return True
        for field in ("message", "delta"):
            container = choice.get(field)
            if not isinstance(container, dict):
                continue
            if normalize_provider_stream_error(container) is not None:
                return True
            content = container.get("content")
            if isinstance(content, list) and any(
                isinstance(block, dict)
                and normalize_provider_stream_error(block) is not None
                for block in content
            ):
                return True
    return False


def _convert_semantic_openai_messages_response(
    response: Any,
    *,
    model: str | None,
) -> dict[str, Any] | None:
    """Convert an OpenAI response only when it contains usable assistant output."""

    if not isinstance(response, dict) or _openai_response_has_structural_error(
        response
    ):
        return None
    converted = openai_response_to_anthropic(response, model=model)
    return converted if _anthropic_message_payload_is_semantic(converted) else None


def _is_nonnegative_int(value: Any) -> bool:
    """Return whether *value* is an integer token count, excluding booleans."""

    return type(value) is int and value >= 0


def _anthropic_message_payload_is_semantic(data: Any) -> bool:
    """Validate one final non-stream Anthropic message response."""

    if not isinstance(data, dict) or normalize_provider_stream_error(data) is not None:
        return False
    if data.get("type") != "message" or data.get("role") != "assistant":
        return False
    if not isinstance(data.get("id"), str) or not data["id"].strip():
        return False
    if not isinstance(data.get("model"), str) or not data["model"].strip():
        return False

    usage = data.get("usage")
    if not isinstance(usage, dict) or not all(
        _is_nonnegative_int(usage.get(field))
        for field in ("input_tokens", "output_tokens")
    ):
        return False

    stop_reason = data.get("stop_reason")
    if stop_reason not in {"end_turn", "max_tokens", "refusal", "tool_use"}:
        return False
    content = data.get("content")
    if not isinstance(content, list):
        return False
    if not content:
        return stop_reason == "refusal" and data.get("stop_sequence") is None
    has_semantic_output = False
    has_tool_output = False
    for block in content:
        if not isinstance(block, dict):
            return False
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                return False
            has_semantic_output = has_semantic_output or bool(text.strip())
        elif block_type == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            if not isinstance(tool_id, str) or not tool_id.strip():
                return False
            if not isinstance(name, str) or not name.strip():
                return False
            if not isinstance(block.get("input"), dict):
                return False
            has_tool_output = True
            has_semantic_output = True
        elif block_type == "image":
            source = block.get("source")
            if not isinstance(source, dict):
                return False
            has_semantic_output = has_semantic_output or any(
                isinstance(source.get(field), str) and source[field].strip()
                for field in ("url", "data")
            )
    if (stop_reason == "tool_use") != has_tool_output:
        return False
    return has_semantic_output and data.get("stop_sequence") is None


_MESSAGES_NATIVE_STOP_REASONS = {
    "compaction",
    "end_turn",
    "max_tokens",
    "model_context_window_exceeded",
    "pause_turn",
    "refusal",
    "stop_sequence",
    "tool_use",
}
_MESSAGES_NATIVE_MESSAGE_FIELDS = {
    "container",
    "content",
    "context_management",
    "diagnostics",
    "id",
    "model",
    "role",
    "stop_details",
    "stop_reason",
    "stop_sequence",
    "type",
    "usage",
}
_MESSAGES_NATIVE_USAGE_FIELDS = {
    "cache_creation",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "inference_geo",
    "input_tokens",
    "iterations",
    "output_tokens",
    "output_tokens_details",
    "server_tool_use",
    "service_tier",
    "speed",
    "thinking_tokens",
}
_MESSAGES_NATIVE_DELTA_USAGE_FIELDS = {
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "input_tokens",
    "iterations",
    "output_tokens",
    "output_tokens_details",
    "server_tool_use",
}
_MESSAGES_NATIVE_RESULT_BLOCK_TYPES = {
    "advisor_tool_result",
    "bash_code_execution_tool_result",
    "code_execution_tool_result",
    "computer_tool_result",
    "mcp_tool_result",
    "text_editor_code_execution_tool_result",
    "tool_search_tool_result",
    "web_fetch_tool_result",
    "web_search_tool_result",
}
_MESSAGES_NATIVE_TOOL_BLOCK_TYPES = {
    "mcp_tool_use",
    "server_tool_use",
    "tool_use",
}


def _has_only_fields(
    value: dict[str, Any],
    allowed: set[str],
    *,
    required: set[str] | None = None,
) -> bool:
    """Return whether a protocol mapping has only known and required fields."""

    keys = set(value)
    return keys <= allowed and (required is None or required <= keys)


def _is_nonempty_string(value: Any) -> bool:
    """Return whether *value* is a non-empty string."""

    return isinstance(value, str) and bool(value.strip())


def _native_stop_details_are_valid(value: Any, stop_reason: Any) -> bool:
    """Validate current refusal metadata without interpreting its explanation."""

    if value is None:
        return True
    if stop_reason != "refusal" or not isinstance(value, dict):
        return False
    if not _has_only_fields(
        value,
        {
            "category",
            "explanation",
            "fallback_credit_token",
            "fallback_has_prefill_claim",
            "recommended_model",
            "type",
        },
        required={"type"},
    ):
        return False
    if value.get("type") != "refusal" or value.get("category") not in {
        None,
        "bio",
        "cyber",
        "frontier_llm",
        "reasoning_extraction",
    }:
        return False
    if value.get("explanation") is not None and not isinstance(
        value["explanation"], str
    ):
        return False
    for field in ("fallback_credit_token", "recommended_model"):
        if value.get(field) is not None and not isinstance(value[field], str):
            return False
    claim = value.get("fallback_has_prefill_claim")
    return claim is None or type(claim) is bool


def _native_container_is_valid(value: Any) -> bool:
    """Validate current stable and beta container response metadata."""

    if value is None:
        return True
    if not isinstance(value, dict) or not _has_only_fields(
        value,
        {"expires_at", "id", "skills"},
        required={"expires_at", "id"},
    ):
        return False
    if not _is_nonempty_string(value.get("id")) or not _is_nonempty_string(
        value.get("expires_at")
    ):
        return False
    skills = value.get("skills")
    return skills is None or (
        isinstance(skills, list)
        and all(isinstance(skill, dict) for skill in skills)
    )


def _native_usage_is_valid(
    value: Any,
    *,
    require_input: bool,
    require_output: bool,
) -> bool:
    """Validate documented usage accounting while rejecting diagnostic siblings."""

    allowed_fields = (
        _MESSAGES_NATIVE_USAGE_FIELDS
        if require_input
        else _MESSAGES_NATIVE_DELTA_USAGE_FIELDS
    )
    if not isinstance(value, dict) or not _has_only_fields(
        value,
        allowed_fields,
        required=(
            ({"input_tokens"} if require_input else set())
            | ({"output_tokens"} if require_output else set())
        ),
    ):
        return False
    for field in {
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "input_tokens",
        "output_tokens",
        "thinking_tokens",
    }:
        if (
            field in value
            and value[field] is not None
            and not _is_nonnegative_int(value[field])
        ):
            return False

    cache_creation = value.get("cache_creation")
    if cache_creation is not None:
        if not isinstance(cache_creation, dict) or not _has_only_fields(
            cache_creation,
            {"ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens"},
            required={
                "ephemeral_1h_input_tokens",
                "ephemeral_5m_input_tokens",
            },
        ):
            return False
        if not all(_is_nonnegative_int(count) for count in cache_creation.values()):
            return False

    server_tools = value.get("server_tool_use")
    if server_tools is not None:
        if not isinstance(server_tools, dict):
            return False
        for name, count in server_tools.items():
            if (
                not isinstance(name, str)
                or len(name) > 64
                or not name.endswith("_requests")
                or not name.replace("_", "").isalnum()
                or not name.islower()
                or not _is_nonnegative_int(count)
            ):
                return False

    service_tier = value.get("service_tier")
    if service_tier is not None and service_tier not in {
        "batch",
        "priority",
        "standard",
    }:
        return False
    inference_geo = value.get("inference_geo")
    if inference_geo is not None and not isinstance(inference_geo, str):
        return False

    iterations = value.get("iterations")
    if iterations is not None:
        if not isinstance(iterations, list):
            return False
        common_iteration_fields = {
            "cache_creation",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "input_tokens",
            "output_tokens",
            "type",
        }
        for iteration in iterations:
            if not isinstance(iteration, dict):
                return False
            iteration_type = iteration.get("type")
            if iteration_type not in {
                "advisor_message",
                "compaction",
                "fallback_message",
                "message",
            }:
                return False
            allowed_iteration_fields = set(common_iteration_fields)
            if iteration_type != "compaction":
                allowed_iteration_fields.add("model")
            if not _has_only_fields(
                iteration,
                allowed_iteration_fields,
                required=common_iteration_fields
                | ({"model"} if iteration_type != "compaction" else set()),
            ):
                return False
            if iteration_type != "compaction" and not _is_nonempty_string(
                iteration.get("model")
            ):
                return False
            if any(
                not _is_nonnegative_int(iteration[field])
                for field in {
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                    "input_tokens",
                    "output_tokens",
                }
            ):
                return False
            iteration_cache = iteration.get("cache_creation")
            if iteration_cache is not None and (
                not isinstance(iteration_cache, dict)
                or not _has_only_fields(
                    iteration_cache,
                    {"ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens"},
                    required={
                        "ephemeral_1h_input_tokens",
                        "ephemeral_5m_input_tokens",
                    },
                )
                or not all(
                    _is_nonnegative_int(count)
                    for count in iteration_cache.values()
                )
            ):
                return False

    output_details = value.get("output_tokens_details")
    if output_details is not None and (
        not isinstance(output_details, dict)
        or not _has_only_fields(
            output_details,
            {"thinking_tokens"},
            required={"thinking_tokens"},
        )
        or not _is_nonnegative_int(output_details.get("thinking_tokens"))
        or output_details["thinking_tokens"] > value.get("output_tokens", 0)
    ):
        return False
    speed = value.get("speed")
    return speed is None or speed in {"fast", "standard"}


def _native_context_management_is_valid(value: Any) -> bool:
    """Validate context-management response metadata without leaking extensions."""

    if value is None:
        return True
    if not isinstance(value, dict) or not _has_only_fields(
        value,
        {"applied_edits"},
        required={"applied_edits"},
    ):
        return False
    edits = value.get("applied_edits")
    if not isinstance(edits, list):
        return False
    for edit in edits:
        if not isinstance(edit, dict):
            return False
        edit_type = edit.get("type")
        count_field = {
            "clear_thinking_20251015": "cleared_thinking_turns",
            "clear_tool_uses_20250919": "cleared_tool_uses",
        }.get(edit_type)
        if count_field is None or not _has_only_fields(
            edit,
            {"cleared_input_tokens", count_field, "type"},
            required={"cleared_input_tokens", count_field, "type"},
        ):
            return False
        if not _is_nonnegative_int(
            edit.get("cleared_input_tokens")
        ) or not _is_nonnegative_int(edit.get(count_field)):
            return False
    return True


def _native_diagnostics_are_valid(value: Any) -> bool:
    """Validate the bounded prompt-cache diagnostic response union."""

    if value is None:
        return True
    if not isinstance(value, dict) or set(value) != {"cache_miss_reason"}:
        return False
    reason = value.get("cache_miss_reason")
    if reason is None:
        return True
    if not isinstance(reason, dict):
        return False
    reason_type = reason.get("type")
    if reason_type in {"previous_message_not_found", "unavailable"}:
        return set(reason) == {"type"}
    if reason_type not in {
        "messages_changed",
        "model_changed",
        "system_changed",
        "tools_changed",
    }:
        return False
    return set(reason) == {"cache_missed_input_tokens", "type"} and (
        _is_nonnegative_int(reason.get("cache_missed_input_tokens"))
    )


_MESSAGES_NATIVE_CITATION_FIELDS = {
    "char_location": {
        "cited_text",
        "document_index",
        "document_title",
        "end_char_index",
        "file_id",
        "start_char_index",
        "type",
    },
    "content_block_location": {
        "cited_text",
        "document_index",
        "document_title",
        "end_block_index",
        "file_id",
        "start_block_index",
        "type",
    },
    "page_location": {
        "cited_text",
        "document_index",
        "document_title",
        "end_page_number",
        "file_id",
        "start_page_number",
        "type",
    },
    "search_result_location": {
        "cited_text",
        "end_block_index",
        "search_result_index",
        "source",
        "start_block_index",
        "title",
        "type",
    },
    "web_search_result_location": {
        "cited_text",
        "encrypted_index",
        "title",
        "type",
        "url",
    },
}
_MESSAGES_NATIVE_CITATION_OPTIONAL_FIELDS = {
    "char_location": {"document_title", "file_id"},
    "content_block_location": {"document_title", "file_id"},
    "page_location": {"document_title", "file_id"},
    "search_result_location": {"source", "title"},
    "web_search_result_location": {"title"},
}


def _native_citation_is_valid(value: Any) -> bool:
    """Validate one documented citation union member."""

    if not isinstance(value, dict):
        return False
    citation_type = value.get("type")
    allowed = _MESSAGES_NATIVE_CITATION_FIELDS.get(citation_type)
    optional = _MESSAGES_NATIVE_CITATION_OPTIONAL_FIELDS.get(citation_type, set())
    if allowed is None or not _has_only_fields(
        value,
        allowed,
        required=allowed - optional,
    ):
        return False
    numeric_fields = {
        "document_index",
        "end_block_index",
        "end_char_index",
        "end_page_number",
        "search_result_index",
        "start_block_index",
        "start_char_index",
        "start_page_number",
    }
    for field, field_value in value.items():
        if field == "type":
            continue
        if field in numeric_fields:
            if not _is_nonnegative_int(field_value):
                return False
        elif field in optional and field_value is None:
            continue
        elif not isinstance(field_value, str):
            return False
    return True


def _native_caller_is_valid(value: Any) -> bool:
    """Validate the current tool caller union."""

    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    caller_type = value.get("type")
    if caller_type == "direct":
        return _has_only_fields(value, {"type"}, required={"type"})
    if caller_type in {
        "code_execution_20250825",
        "code_execution_20260120",
        "code_execution_20260521",
    }:
        return _has_only_fields(
            value,
            {"tool_id", "type"},
            required={"tool_id", "type"},
        ) and _is_nonempty_string(value.get("tool_id"))
    return False


def _native_content_block_kind(value: Any) -> str | None:
    """Validate one native response block and classify semantic output."""

    if not isinstance(value, dict):
        return None
    block_type = value.get("type")
    if block_type in {"connector_text", "text"}:
        if not _has_only_fields(
            value,
            {"citations", "text", "type"},
            required={"text", "type"},
        ) or not isinstance(value.get("text"), str):
            return None
        citations = value.get("citations")
        if citations is not None and (
            not isinstance(citations, list)
            or not all(_native_citation_is_valid(citation) for citation in citations)
        ):
            return None
        return "output" if value["text"] or citations else "control"
    if block_type == "thinking":
        if not _has_only_fields(
            value,
            {"signature", "thinking", "type"},
            required={"thinking", "type"},
        ):
            return None
        thinking = value.get("thinking")
        signature = value.get("signature")
        if not isinstance(thinking, str) or (
            signature is not None and not isinstance(signature, str)
        ):
            return None
        return "output" if thinking or signature else "control"
    if block_type == "redacted_thinking":
        if not _has_only_fields(
            value,
            {"data", "type"},
            required={"data", "type"},
        ) or not isinstance(value.get("data"), str):
            return None
        return "output" if value["data"] else "control"
    if block_type == "fallback":
        if not _has_only_fields(
            value,
            {"from", "to", "trigger", "type"},
            required={"from", "to", "trigger", "type"},
        ):
            return None
        for target in (value.get("from"), value.get("to")):
            if not isinstance(target, dict) or not _has_only_fields(
                target,
                {"model"},
                required={"model"},
            ) or not _is_nonempty_string(target.get("model")):
                return None
        trigger = value.get("trigger")
        if not isinstance(trigger, dict) or not _has_only_fields(
            trigger,
            {"category", "type"},
            required={"type"},
        ):
            return None
        if trigger.get("type") != "refusal" or trigger.get("category") not in {
            None,
            "bio",
            "cyber",
            "frontier_llm",
            "reasoning_extraction",
        }:
            return None
        return "control"
    if block_type == "compaction":
        if not _has_only_fields(
            value,
            {"content", "encrypted_content", "type"},
            required={"content", "encrypted_content", "type"},
        ):
            return None
        content = value.get("content")
        encrypted = value.get("encrypted_content")
        if (content is not None and not isinstance(content, str)) or (
            encrypted is not None and not isinstance(encrypted, str)
        ):
            return None
        return "output" if content or encrypted else "control"
    if block_type == "container_upload":
        if not _has_only_fields(
            value,
            {"file_id", "type"},
            required={"file_id", "type"},
        ) or not _is_nonempty_string(value.get("file_id")):
            return None
        return "output"
    if block_type in _MESSAGES_NATIVE_TOOL_BLOCK_TYPES:
        allowed = {"caller", "id", "input", "name", "type"}
        if block_type == "mcp_tool_use":
            allowed.add("server_name")
        if not _has_only_fields(
            value,
            allowed,
            required={"id", "input", "name", "type"},
        ):
            return None
        if (
            not _is_nonempty_string(value.get("id"))
            or not _is_nonempty_string(value.get("name"))
            or not isinstance(value.get("input"), dict)
            or not _native_caller_is_valid(value.get("caller"))
            or (
                block_type == "mcp_tool_use"
                and not _is_nonempty_string(value.get("server_name"))
            )
        ):
            return None
        return "output"
    if block_type in _MESSAGES_NATIVE_RESULT_BLOCK_TYPES:
        if not _has_only_fields(
            value,
            {"caller", "content", "is_error", "tool_use_id", "type"},
            required={"content", "tool_use_id", "type"},
        ):
            return None
        if (
            not _is_nonempty_string(value.get("tool_use_id"))
            or not _native_caller_is_valid(value.get("caller"))
            or ("is_error" in value and type(value["is_error"]) is not bool)
            or (block_type == "mcp_tool_result" and "is_error" not in value)
        ):
            return None
        # Result content is domain data. It is intentionally opaque here so
        # typed tool failures are not confused with transport failures.
        return "output"
    return None


def _native_message_payload_is_semantic(data: Any) -> bool:
    """Validate one non-stream native Anthropic message response."""

    if not isinstance(data, dict) or normalize_provider_stream_error(data) is not None:
        return False
    if not _has_only_fields(
        data,
        _MESSAGES_NATIVE_MESSAGE_FIELDS,
        required={
            "content",
            "id",
            "model",
            "role",
            "stop_reason",
            "stop_sequence",
            "type",
            "usage",
        },
    ):
        return False
    if (
        data.get("type") != "message"
        or data.get("role") != "assistant"
        or not _is_nonempty_string(data.get("id"))
        or not _is_nonempty_string(data.get("model"))
        or data.get("stop_reason") not in _MESSAGES_NATIVE_STOP_REASONS
        or not _native_stop_details_are_valid(
            data.get("stop_details"),
            data.get("stop_reason"),
        )
        or not _native_container_is_valid(data.get("container"))
        or not _native_context_management_is_valid(data.get("context_management"))
        or not _native_diagnostics_are_valid(data.get("diagnostics"))
        or not _native_usage_is_valid(
            data.get("usage"),
            require_input=True,
            require_output=True,
        )
    ):
        return False
    stop_sequence = data.get("stop_sequence")
    if data.get("stop_reason") == "stop_sequence":
        if not _is_nonempty_string(stop_sequence):
            return False
    elif stop_sequence is not None:
        return False
    content = data.get("content")
    if not isinstance(content, list):
        return False
    if not content:
        return data.get("stop_reason") == "refusal"
    kinds = [_native_content_block_kind(block) for block in content]
    if not all(kind is not None for kind in kinds) or "output" not in kinds:
        return False
    has_client_tool = any(
        isinstance(block, dict) and block.get("type") == "tool_use"
        for block in content
    )
    stop_reason = data.get("stop_reason")
    return not (
        (has_client_tool and stop_reason not in {"max_tokens", "tool_use"})
        or (not has_client_tool and stop_reason == "tool_use")
    )


def _native_count_payload_is_semantic(data: Any) -> bool:
    """Validate one non-stream native count_tokens response."""

    if not isinstance(data, dict) or normalize_provider_stream_error(data) is not None:
        return False
    if not _has_only_fields(
        data,
        {"context_management", "input_tokens"},
        required={"input_tokens"},
    ) or not _is_nonnegative_int(data.get("input_tokens")):
        return False
    context_management = data.get("context_management")
    return context_management is None or (
        isinstance(context_management, dict)
        and set(context_management) == {"original_input_tokens"}
        and _is_nonnegative_int(context_management.get("original_input_tokens"))
    )


def _config_default_llm_provider() -> str | None:
    """Return the default LLM provider from loaded config sections."""
    cfg = loaded_config_data
    def _extract(section: str) -> str | None:
        """Extract default_api from a config section if present."""
        try:
            data = cfg.get(section)
        except (AttributeError, TypeError, KeyError):
            data = None
        if isinstance(data, dict):
            default_api = data.get("default_api")
            if isinstance(default_api, str):
                value = default_api.strip()
                if value:
                    return value
        return None
    return _extract("llm_api_settings") or _extract("API")


def _get_default_provider() -> str:
    """Resolve the default provider using config, env, and test fallbacks."""
    cfg_default = _config_default_llm_provider()
    if cfg_default:
        return cfg_default
    env_val = os.getenv("DEFAULT_LLM_PROVIDER")
    if env_val:
        return env_val
    if is_test_mode():
        return "local-llm"
    return DEFAULT_LLM_PROVIDER


def _resolve_messages_base_url(provider: str, app_config: dict[str, Any] | None) -> str:
    """Resolve the base URL for a messages-native provider."""
    config_is_authoritative = app_config is not None
    cfg = app_config if config_is_authoritative else loaded_config_data
    if provider == "anthropic":
        base = None
        try:
            anth = cfg.get("anthropic_api")
        except (AttributeError, TypeError, KeyError):
            anth = None
        if isinstance(anth, dict):
            base = anth.get("api_base_url")
        if not base:
            base = (
                "https://api.anthropic.com/v1"
                if config_is_authoritative
                else os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
            )
        return str(base)
    if provider == "llama.cpp":
        base = None
        try:
            llama = cfg.get("llama_api")
        except (AttributeError, TypeError, KeyError):
            llama = None
        if isinstance(llama, dict):
            base = llama.get("api_ip") or llama.get("api_base_url")
        if not base:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Llama.cpp API URL/IP is required but not configured.",
            )
        normalized = _normalize_llamacpp_base_url(str(base))
        if not normalized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Llama.cpp API URL/IP is required but not configured.",
            )
        return normalized
    raise HTTPException(status_code=400, detail=f"Provider '{provider}' is not messages-native.")


def _join_messages_endpoint(base_url: str, suffix: str) -> str:
    """Join a base URL with a Messages endpoint suffix."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}{suffix}"
    return f"{base}/v1{suffix}"


def _normalize_llamacpp_base_url(base_url: str) -> str:
    """Strip known completion suffixes from a llama.cpp base URL."""
    normalized = base_url.strip().rstrip("/")
    lowered = normalized.lower()
    for suffix in (
        "/v1/messages/count_tokens",
        "/v1/messages",
        "/messages/count_tokens",
        "/messages",
        "/v1/chat/completions",
        "/v1/completions",
        "/chat/completions",
        "/completions",
        "/completion",
    ):
        if lowered.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _resolve_provider_and_model_for_request(request_data: Any) -> tuple[str, str]:
    """Resolve provider/model pair from the request payload."""
    _, metrics_model, selected_provider, selected_model, _debug = resolve_provider_and_model(
        request_data=request_data,
        metrics_default_provider=DEFAULT_LLM_PROVIDER,
        normalize_default_provider=_get_default_provider(),
    )
    provider = selected_provider
    model = selected_model or metrics_model or getattr(request_data, "model", None)
    return provider, model


def _messages_credential_http_exception(exc: ByokResolutionError) -> HTTPException:
    """Map typed credential failures to bounded Messages API responses."""
    code = getattr(exc, "policy_code", exc.code)
    if code in {"provider_disabled", "model_not_allowed"}:
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": code,
                "message": "The selected provider or model is disabled by administrator policy.",
            },
        )
    if code == "credential_scope_revoked":
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": code,
                "message": _MESSAGES_CREDENTIAL_ERROR_MESSAGES[code],
            },
        )
    message = _MESSAGES_CREDENTIAL_ERROR_MESSAGES.get(code)
    if message is None:
        code = "invalid_provider_credentials"
        message = _MESSAGES_CREDENTIAL_ERROR_MESSAGES[code]
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error_code": code, "message": message},
    )


def _resolve_llamacpp_api_key(app_config: dict[str, Any] | None) -> str | None:
    """Resolve the llama.cpp API key from app config fallbacks."""
    def _from_cfg(cfg: Any) -> str | None:
        """Extract the llama.cpp API key from a config mapping."""
        try:
            llama = cfg.get("llama_api")
        except (AttributeError, TypeError, KeyError):
            llama = None
        if isinstance(llama, dict):
            key = llama.get("api_key")
            if isinstance(key, str) and key.strip():
                return key.strip()
        return None

    return _from_cfg(app_config if app_config is not None else loaded_config_data)


def _resolve_native_timeout(provider: str, app_config: dict[str, Any] | None) -> float | None:
    """Resolve timeout values for native Messages providers."""
    cfg = app_config if app_config is not None else loaded_config_data
    section = "anthropic_api" if provider == "anthropic" else "llama_api"
    default_timeout = 60.0 if provider == "anthropic" else 120.0
    try:
        section_cfg = cfg.get(section)
    except (AttributeError, TypeError, KeyError):
        section_cfg = None
    if not isinstance(section_cfg, dict):
        return default_timeout
    raw = section_cfg.get("api_timeout")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default_timeout


def _build_native_headers(
    provider: str,
    api_key: str | None,
    *,
    anthropic_version: str | None,
    anthropic_beta: str | None,
) -> dict[str, str]:
    """Build headers for native Messages provider calls."""
    headers = {"Content-Type": "application/json"}
    if provider == "anthropic":
        if api_key:
            headers["x-api-key"] = api_key
        headers["anthropic-version"] = anthropic_version or DEFAULT_ANTHROPIC_VERSION
        if anthropic_beta:
            headers["anthropic-beta"] = anthropic_beta
        return headers
    if provider == "llama.cpp":
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if anthropic_version:
            headers["anthropic-version"] = anthropic_version
        if anthropic_beta:
            headers["anthropic-beta"] = anthropic_beta
        return headers
    return headers


def _extract_stream_flag(raw: Any) -> bool:
    """Normalize a stream flag from user-provided values."""
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return is_truthy(raw)
    return bool(raw)


def _build_openai_call_params(
    *,
    request_data: AnthropicMessagesRequest,
    provider: str,
    model: str,
    app_config: dict[str, Any] | None,
    api_key: str | None,
    credentials: ProviderCallCredentials,
) -> dict[str, Any]:
    """Build OpenAI-compatible call parameters from Messages input."""
    messages_payload, system_message = anthropic_messages_to_openai(
        [m.model_dump(exclude_none=True) for m in request_data.messages],
        request_data.system,
    )
    tools = anthropic_tools_to_openai(
        [t.model_dump(exclude_none=True) for t in request_data.tools]
    ) if request_data.tools else None
    tool_choice = anthropic_tool_choice_to_openai(request_data.tool_choice)

    call_params: dict[str, Any] = {
        "api_provider": provider,
        "model": model,
        "messages": messages_payload,
        "system_message": system_message,
        "api_key": api_key,
        "app_config": app_config,
        "credentials_resolved": True,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: credentials,
        "stream": _extract_stream_flag(request_data.stream),
        "temperature": request_data.temperature,
        "top_p": request_data.top_p,
        "top_k": request_data.top_k,
        "max_tokens": request_data.max_tokens,
        "stop": request_data.stop_sequences,
        "tools": tools,
        "tool_choice": tool_choice,
    }
    return call_params


def _prepare_native_payload(request_data: Any, *, model: str) -> dict[str, Any]:
    """Prepare payload for native Messages providers."""
    payload = request_data.model_dump(exclude_none=True)
    payload.pop("api_provider", None)
    payload["model"] = model
    return payload


def _map_native_upstream_exception(
    exc: Exception,
    *,
    provider: str,
    operation: str,
) -> HTTPException:
    """Translate upstream HTTP/network errors into stable API errors."""

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int) or not (400 <= status_code <= 599):
        status_code = status.HTTP_502_BAD_GATEWAY
    elif status_code in {
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN,
    }:
        # These credentials belong to the upstream provider, not this API.
        # Returning 401/403 would make clients treat a provider rejection as
        # an expired or unauthorized tldw session.
        status_code = status.HTTP_502_BAD_GATEWAY

    detail: dict[str, Any] = {
        "error_code": "upstream_provider_error",
        "provider": provider,
        "operation": operation,
        "message": f"Upstream provider '{provider}' request failed.",
    }

    return HTTPException(status_code=status_code, detail=detail)


async def _native_post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float | None,
    provider: str,
    operation: str,
) -> Any:
    """Execute a native provider POST and map upstream failures consistently."""
    try:
        async with async_http_client_factory(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - provider clients may raise arbitrary errors
        raise_detached_error(
            _map_native_upstream_exception(
                exc,
                provider=provider,
                operation=operation,
            )
        )


async def _prepare_native_stream_iterator(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float | None,
    provider: str,
    operation: str,
) -> AsyncIterator[bytes]:
    """Open a native provider stream and preflight status before returning an iterator."""
    client_cm = None
    client_entered = False
    stream_cm = None
    stream_entered = False
    response = None

    async def _close_entered_contexts() -> None:
        nonlocal client_entered, stream_entered
        try:
            if stream_entered and stream_cm is not None:
                stream_entered = False
                with contextlib.suppress(Exception):
                    await stream_cm.__aexit__(None, None, None)
        finally:
            if client_entered and client_cm is not None:
                client_entered = False
                with contextlib.suppress(Exception):
                    await client_cm.__aexit__(None, None, None)

    try:
        client_cm = async_http_client_factory(timeout=timeout)
        client = await client_cm.__aenter__()
        client_entered = True
        stream_cm = client.stream("POST", url, headers=headers, json=payload)
        response = await stream_cm.__aenter__()
        stream_entered = True
        response.raise_for_status()
    except asyncio.CancelledError:
        await _close_entered_contexts()
        raise
    except Exception as exc:  # noqa: BLE001 - provider clients may raise arbitrary errors
        await _close_entered_contexts()
        raise_detached_error(
            _map_native_upstream_exception(
                exc,
                provider=provider,
                operation=operation,
            )
        )

    async def _iter() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_raw():  # type: ignore[union-attr]
                if chunk:
                    yield chunk
        finally:
            await _close_entered_contexts()

    return _iter()


class _MessagesSSEState:
    """Request-local lifecycle state for one native Messages stream."""

    __slots__ = (
        "message_delta_seen",
        "message_started",
        "message_stopped",
        "next_block_index",
        "open_block_index",
        "open_block_type",
        "require_message_lifecycle",
        "saw_client_tool",
        "semantic_output_seen",
    )

    def __init__(self, *, require_message_lifecycle: bool = True) -> None:
        self.message_started = False
        self.message_delta_seen = False
        self.message_stopped = False
        self.next_block_index = 0
        self.open_block_index: int | None = None
        self.open_block_type: str | None = None
        self.require_message_lifecycle = require_message_lifecycle
        self.saw_client_tool = False
        self.semantic_output_seen = False


_MESSAGES_SSE_PAYLOAD_FIELDS = {
    "content_block_delta": {"delta", "index", "type"},
    "content_block_start": {"content_block", "index", "type"},
    "content_block_stop": {"index", "type"},
    "message_delta": {"context_management", "delta", "type", "usage"},
    "message_start": {"message", "type"},
    "message_stop": {"type"},
}


def _native_message_start_is_valid(message: Any) -> bool:
    """Validate the nested Message carried by a message_start event."""

    if not isinstance(message, dict) or not _has_only_fields(
        message,
        _MESSAGES_NATIVE_MESSAGE_FIELDS,
        required={
            "content",
            "id",
            "model",
            "role",
            "stop_reason",
            "stop_sequence",
            "type",
            "usage",
        },
    ):
        return False
    return (
        message.get("type") == "message"
        and message.get("role") == "assistant"
        and _is_nonempty_string(message.get("id"))
        and _is_nonempty_string(message.get("model"))
        and message.get("content") == []
        and message.get("stop_reason") is None
        and message.get("stop_sequence") is None
        and _native_stop_details_are_valid(message.get("stop_details"), None)
        and _native_container_is_valid(message.get("container"))
        and _native_context_management_is_valid(message.get("context_management"))
        and _native_diagnostics_are_valid(message.get("diagnostics"))
        and _native_usage_is_valid(
            message.get("usage"),
            require_input=True,
            require_output=True,
        )
    )


def _native_message_delta_is_valid(
    delta: Any,
    usage: Any,
    context_management: Any,
) -> bool:
    """Validate the protocol-owned fields of a terminal message_delta event."""

    if not isinstance(delta, dict) or not _has_only_fields(
        delta,
        {
            "container",
            "stop_details",
            "stop_reason",
            "stop_sequence",
        },
        required={"stop_reason", "stop_sequence"},
    ):
        return False
    stop_reason = delta.get("stop_reason")
    stop_sequence = delta.get("stop_sequence")
    if stop_reason not in _MESSAGES_NATIVE_STOP_REASONS:
        return False
    if stop_reason == "stop_sequence":
        if not _is_nonempty_string(stop_sequence):
            return False
    elif stop_sequence is not None:
        return False
    return (
        _native_stop_details_are_valid(delta.get("stop_details"), stop_reason)
        and _native_container_is_valid(delta.get("container"))
        and _native_context_management_is_valid(context_management)
        and _native_usage_is_valid(
            usage,
            require_input=False,
            require_output=True,
        )
    )


def _native_content_delta_kind(delta: Any, block_type: str | None) -> str | None:
    """Validate one content delta and classify whether it carries output."""

    if not isinstance(delta, dict):
        return None
    delta_type = delta.get("type")
    if delta_type == "signature_delta":
        if (
            block_type not in {None, "thinking"}
            or not _has_only_fields(
                delta,
                {"signature", "type"},
                required={"signature", "type"},
            )
            or not isinstance(delta.get("signature"), str)
        ):
            return None
        return "output" if delta["signature"] else "control"
    if delta_type == "text_delta":
        if (
            block_type not in {None, "connector_text", "text"}
            or not _has_only_fields(
                delta,
                {"text", "type"},
                required={"text", "type"},
            )
            or not isinstance(delta.get("text"), str)
        ):
            return None
        return "output" if delta["text"] else "control"
    if delta_type == "thinking_delta":
        if (
            block_type not in {None, "thinking"}
            or not _has_only_fields(
                delta,
                {"thinking", "type"},
                required={"thinking", "type"},
            )
            or not isinstance(delta.get("thinking"), str)
        ):
            return None
        return "output" if delta["thinking"] else "control"
    if delta_type == "citations_delta":
        if block_type not in {None, "text"} or not _has_only_fields(
            delta,
            {"citation", "type"},
            required={"citation", "type"},
        ):
            return None
        return "output" if _native_citation_is_valid(delta.get("citation")) else None
    if delta_type == "input_json_delta":
        if (
            block_type not in ({None} | _MESSAGES_NATIVE_TOOL_BLOCK_TYPES)
            or not _has_only_fields(
                delta,
                {"partial_json", "type"},
                required={"partial_json", "type"},
            )
            or not isinstance(delta.get("partial_json"), str)
        ):
            return None
        return "output" if delta["partial_json"] else "control"
    if delta_type == "tool_use_delta":
        if (
            block_type not in ({None} | _MESSAGES_NATIVE_TOOL_BLOCK_TYPES)
            or not _has_only_fields(
                delta,
                {"name", "type"},
                required={"name", "type"},
            )
            or not _is_nonempty_string(delta.get("name"))
        ):
            return None
        return "output"
    if delta_type == "compaction_delta":
        if block_type not in {None, "compaction"} or not _has_only_fields(
            delta,
            {"content", "encrypted_content", "type"},
            required={"content", "encrypted_content", "type"},
        ):
            return None
        content = delta.get("content")
        encrypted = delta.get("encrypted_content")
        if (content is not None and not isinstance(content, str)) or (
            encrypted is not None and not isinstance(encrypted, str)
        ):
            return None
        return "output" if content or encrypted else "control"
    return None


def _classify_messages_sse_frame(
    frame: str,
    *,
    state: _MessagesSSEState | None = None,
) -> str:
    """Classify one complete SSE frame as output, error, terminal, or control."""

    event_type = "message"
    data_lines: list[str] = []
    plain_lines: list[str] = []
    saw_sse_field = False
    saw_event_field = False
    for raw_line in frame.splitlines():
        if not raw_line:
            continue
        if raw_line.startswith(":"):
            saw_sse_field = True
            comment = raw_line[1:].strip().lower()
            if comment and comment not in _MESSAGES_SSE_CONTROL_EVENTS:
                plain_lines.append(raw_line)
            continue
        field, separator, value = raw_line.partition(":")
        if not separator:
            plain_lines.append(raw_line)
            continue
        field = field.strip().lower()
        if field not in {"data", "event", "id", "retry"}:
            plain_lines.append(raw_line)
            continue
        saw_sse_field = True
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_type = value.strip().lower()
            saw_event_field = True
        elif field == "data":
            data_lines.append(value)

    if not saw_sse_field:
        return "output" if any(line.strip() for line in plain_lines) else "control"
    if plain_lines:
        return "invalid"
    if event_type == "error":
        return "error"
    if not data_lines:
        return (
            "control"
            if not saw_event_field or event_type in _MESSAGES_SSE_CONTROL_EVENTS
            else "invalid"
        )

    data = "\n".join(data_lines).strip()
    if not data:
        return "control" if event_type in _MESSAGES_SSE_CONTROL_EVENTS else "invalid"
    if data.lower() == "[done]":
        return "control"
    try:
        payload = json.loads(data)
    except (TypeError, ValueError):
        return "invalid"
    if not isinstance(payload, dict):
        return "invalid"

    payload_type = str(payload.get("type") or "").strip().lower()
    if normalize_provider_stream_error(payload) is not None:
        return "error"
    if not payload_type:
        return "invalid"
    if saw_event_field and event_type not in {"message", payload_type}:
        return "invalid"
    if payload_type not in _MESSAGES_SSE_KNOWN_EVENTS:
        return "ignored"
    if payload_type in _MESSAGES_SSE_CONTROL_EVENTS:
        return "control" if set(payload) <= {"type"} else "invalid"
    allowed_fields = _MESSAGES_SSE_PAYLOAD_FIELDS.get(payload_type)
    if allowed_fields is None or not set(payload) <= allowed_fields:
        return "invalid"

    if payload_type == "message_start":
        if state is not None and (
            state.message_started or state.message_delta_seen or state.message_stopped
        ):
            return "invalid"
        if not _native_message_start_is_valid(payload.get("message")):
            return "invalid"
        if state is not None:
            state.message_started = True
        return "control"

    if payload_type == "message_delta":
        if state is not None and (
            (state.require_message_lifecycle and not state.message_started)
            or state.message_stopped
            or state.open_block_index is not None
        ):
            return "invalid"
        if not _native_message_delta_is_valid(
            payload.get("delta"),
            payload.get("usage"),
            payload.get("context_management"),
        ):
            return "invalid"
        delta = payload.get("delta")
        stop_reason = delta.get("stop_reason")
        if state is not None and (
            (
                state.saw_client_tool
                and stop_reason not in {"max_tokens", "tool_use"}
            )
            or (not state.saw_client_tool and stop_reason == "tool_use")
        ):
            return "invalid"
        if state is not None:
            state.message_delta_seen = True
        kind = "output" if delta.get("stop_reason") == "refusal" else "control"
        if state is not None and kind == "output":
            state.semantic_output_seen = True
        return kind

    if payload_type == "message_stop":
        if state is not None:
            if (
                (
                    state.require_message_lifecycle
                    and (
                        not state.message_started
                        or not state.message_delta_seen
                        or not state.semantic_output_seen
                    )
                )
                or state.message_stopped
                or state.open_block_index is not None
            ):
                return "invalid"
            state.message_stopped = True
        return "terminal"

    index = payload.get("index")
    if type(index) is not int or index < 0:
        return "invalid"

    if payload_type == "content_block_stop":
        if state is not None:
            if (
                (state.require_message_lifecycle and not state.message_started)
                or state.message_delta_seen
                or state.message_stopped
                or state.open_block_index != index
            ):
                return "invalid"
            state.open_block_index = None
            state.open_block_type = None
            state.next_block_index += 1
        return "control"

    if payload_type == "content_block_start":
        if state is not None and (
            (state.require_message_lifecycle and not state.message_started)
            or state.message_delta_seen
            or state.message_stopped
            or state.open_block_index is not None
            or index != state.next_block_index
        ):
            return "invalid"
        content_block = payload.get("content_block")
        kind = _native_content_block_kind(content_block)
        if kind is None:
            return "invalid"
        block_type = str(content_block.get("type") or "").strip().lower()
        if state is not None:
            state.open_block_index = index
            state.open_block_type = block_type
            if block_type == "tool_use":
                state.saw_client_tool = True
            if kind == "output":
                state.semantic_output_seen = True
        return kind

    if state is not None and (
        (state.require_message_lifecycle and not state.message_started)
        or state.message_delta_seen
        or state.message_stopped
        or state.open_block_index != index
    ):
        return "invalid"
    block_type = state.open_block_type if state is not None else None
    kind = _native_content_delta_kind(payload.get("delta"), block_type)
    if state is not None and kind == "output":
        state.semantic_output_seen = True
    return kind if kind is not None else "invalid"


def _messages_upstream_error_sse() -> str:
    """Return the bounded public error event for a provider stream failure."""
    return (
        "event: error\n"
        + "data: "
        + json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "The upstream provider returned an error.",
                },
            },
            ensure_ascii=True,
        )
        + "\n\n"
    )


def _messages_upstream_http_exception() -> HTTPException:
    """Return the bounded public error for a converted provider failure."""
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Upstream provider request failed.",
    )


def _is_native_messages_sse_frame(frame: str) -> bool:
    """Return whether a frame contains at least one recognized SSE field."""
    for raw_line in frame.splitlines():
        line = raw_line.lstrip()
        if line.startswith(":"):
            return True
        field, separator, _value = line.partition(":")
        if separator and field.strip().lower() in {"data", "event", "id", "retry"}:
            return True
    return False


def _sanitize_native_messages_stream(stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Validate one complete native lifecycle and bound provider diagnostics."""

    async def _iter() -> AsyncIterator[str]:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buffer = ""
        state = _MessagesSSEState()
        try:
            async for item in stream:
                if isinstance(item, (bytes, bytearray)):
                    text = decoder.decode(bytes(item), final=False)
                else:
                    text = str(item)
                buffer = (buffer + text).replace("\r\n", "\n")
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    frame_kind = _classify_messages_sse_frame(frame, state=state)
                    if not _is_native_messages_sse_frame(frame) or frame_kind in {
                        "error",
                        "invalid",
                    }:
                        yield _messages_upstream_error_sse()
                        return
                    if frame_kind == "ignored":
                        continue
                    if frame_kind == "terminal":
                        # A terminal event must be the final buffered event. This
                        # prevents a provider from smuggling diagnostics after it
                        # in the same transport chunk.
                        if buffer.strip():
                            yield _messages_upstream_error_sse()
                            return
                        yield f"{frame}\n\n"
                        return
                    yield f"{frame}\n\n"
                if len(buffer) > _MESSAGES_SSE_ACCOUNTING_BUFFER_LIMIT:
                    yield _messages_upstream_error_sse()
                    return

            buffer += decoder.decode(b"", final=True)
            # Native success requires an explicit, complete message_stop. Any
            # partial frame or otherwise premature EOF is a bounded failure.
            if buffer.strip() or not state.message_stopped:
                yield _messages_upstream_error_sse()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - provider transports may raise arbitrary errors
            yield _messages_upstream_error_sse()
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    await invoke_stream_close_bounded(close)

    return _MessagesStreamUsagePolicy(_iter())


class _MessagesStreamUsagePolicy:
    """Async-iterator wrapper carrying a request-local usage policy."""

    defer_usage_until_terminal = True

    def __init__(self, stream: AsyncIterator[Any]) -> None:
        self._stream = stream

    def __aiter__(self) -> _MessagesStreamUsagePolicy:
        return self

    async def __anext__(self) -> Any:
        return await self._stream.__anext__()

    async def aclose(self) -> None:
        close = getattr(self._stream, "aclose", None)
        if callable(close):
            await close()


def _sanitize_converted_messages_stream(
    stream: AsyncIterator[Any],
) -> AsyncIterator[Any]:
    """Replace converted failures and defer usage until terminal validation."""

    async def _iter() -> AsyncIterator[Any]:
        try:
            async for item in stream:
                yield item
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - arbitrary adapter failures are untrusted
            yield _messages_upstream_error_sse()
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    await invoke_stream_close_bounded(close)

    return _MessagesStreamUsagePolicy(_iter())


async def _touch_on_first_stream_output(
    stream: AsyncIterator[Any],
    credential_runtime: ProviderCredentialRuntime,
    credentials: ProviderCallCredentials,
) -> AsyncIterator[Any]:
    """Record credential use after the first complete, non-control SSE event."""

    touched = False
    saw_output = False
    accounting_buffer = ""
    accounting_disabled = False
    defer_usage_until_terminal = bool(
        getattr(stream, "defer_usage_until_terminal", False)
    )
    state = _MessagesSSEState(
        require_message_lifecycle=defer_usage_until_terminal
    )
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    async def _inspect_complete_frames(text: str) -> None:
        nonlocal accounting_buffer, accounting_disabled, saw_output, touched
        if touched or accounting_disabled:
            return
        accounting_buffer = (accounting_buffer + text).replace("\r\n", "\n")
        while "\n\n" in accounting_buffer:
            frame, accounting_buffer = accounting_buffer.split("\n\n", 1)
            frame_kind = _classify_messages_sse_frame(
                frame,
                state=state,
            )
            if frame_kind in {"error", "invalid"}:
                accounting_buffer = ""
                accounting_disabled = True
                return
            if frame_kind == "terminal":
                if defer_usage_until_terminal and saw_output:
                    touched = await await_owned_worker(
                        _mark_messages_credential_used(
                            credential_runtime,
                            credentials,
                        )
                    )
                accounting_buffer = ""
                return
            if frame_kind == "output":
                saw_output = True
                if not defer_usage_until_terminal:
                    touched = await await_owned_worker(
                        _mark_messages_credential_used(
                            credential_runtime,
                            credentials,
                        )
                    )
                    accounting_buffer = ""
                    return
        if len(accounting_buffer) > _MESSAGES_SSE_ACCOUNTING_BUFFER_LIMIT:
            accounting_buffer = ""
            accounting_disabled = True

    try:
        async for item in stream:
            if isinstance(item, (bytes, bytearray)):
                item_text = decoder.decode(bytes(item), final=False)
            else:
                item_text = str(item)
            await _inspect_complete_frames(item_text)
            yield item
        if not defer_usage_until_terminal and not touched and not accounting_disabled:
            accounting_buffer += decoder.decode(b"", final=True)
            if (
                accounting_buffer
                and _classify_messages_sse_frame(
                    accounting_buffer,
                    state=state,
                )
                == "output"
            ):
                await await_owned_worker(
                    _mark_messages_credential_used(
                        credential_runtime,
                        credentials,
                    )
                )
    finally:
        try:
            close = getattr(stream, "aclose", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    await await_owned_worker(invoke_stream_close_bounded(close))
        finally:
            await await_owned_worker(credential_runtime.close())


def _new_messages_credential_runtime(
    current_user: User,
    request: Request,
) -> ProviderCredentialRuntime:
    """Build one Messages runtime from trusted authenticated request state."""
    try:
        user_id, team_ids, org_ids, trusted_base_url_override = (
            derive_trusted_credential_scope(request, current_user)
        )
    except ByokResolutionError as exc:
        raise_detached_error(_messages_credential_http_exception(exc))
    try:
        return ProviderCredentialRuntime(
            user_id=user_id,
            team_ids=team_ids,
            org_ids=org_ids,
            trusted_base_url_override=trusted_base_url_override,
            override_snapshot_resolver=capture_provider_override_call_snapshot,
        )
    except (RuntimeError, ValueError):
        raise_detached_error(
            HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "invalid_provider_credentials",
                    "message": _MESSAGES_CREDENTIAL_ERROR_MESSAGES[
                        "invalid_provider_credentials"
                    ],
                },
            )
        )


async def _mark_messages_credential_used(
    credential_runtime: ProviderCredentialRuntime,
    credentials: ProviderCallCredentials,
) -> bool:
    """Persist one Messages credential touch, retrying one explicit failure."""

    return await mark_provider_credential_used(credential_runtime, credentials)


async def _resolve_messages_credentials(
    credential_runtime: ProviderCredentialRuntime,
    provider: str,
    model: str,
    *,
    operation: str,
) -> ProviderCallCredentials:
    """Resolve and validate one runtime-owned Messages credential handle."""
    try:
        credentials = await credential_runtime.resolve(provider, model=model)
    except ByokResolutionError as exc:
        raise_detached_error(_messages_credential_http_exception(exc))
    except RuntimeError:
        raise_detached_error(
            HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "invalid_provider_credentials",
                    "message": _MESSAGES_CREDENTIAL_ERROR_MESSAGES[
                        "invalid_provider_credentials"
                    ],
                },
            )
        )

    if provider_requires_api_key(provider) and not provider_auth_is_resolved(
        provider,
        api_key=credentials.api_key,
        app_config=credentials.app_config,
        credentials_resolved=credentials.credentials_resolved,
    ):
        record_byok_missing_credentials(provider, operation=operation)
        raise_detached_error(
            HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "missing_provider_credentials",
                    "message": f"Provider '{provider}' requires an API key. Please configure credentials.",
                },
            )
        )
    return credentials


async def _handle_messages(
    request_data: AnthropicMessagesRequest,
    *,
    current_user: User,
    request: Request,
    anthropic_version: str | None,
    anthropic_beta: str | None,
) -> JSONResponse | StreamingResponse:
    """Handle an Anthropic-compatible Messages request."""
    provider, model = _resolve_provider_and_model_for_request(request_data)
    credential_runtime = _new_messages_credential_runtime(current_user, request)
    runtime_owned_by_stream = False
    try:
        credentials = await _resolve_messages_credentials(
            credential_runtime,
            provider,
            model,
            operation="messages",
        )
        api_key = credentials.api_key
        app_config_override = credentials.app_config or {}
        if provider == "llama.cpp" and not api_key:
            api_key = _resolve_llamacpp_api_key(app_config_override)

        if provider in MESSAGES_NATIVE_PROVIDERS:
            base_url = _resolve_messages_base_url(provider, app_config_override)
            url = _join_messages_endpoint(base_url, "/messages")
            payload = _prepare_native_payload(request_data, model=model)
            timeout = _resolve_native_timeout(provider, app_config_override)
            headers = _build_native_headers(
                provider=provider,
                api_key=api_key,
                anthropic_version=anthropic_version,
                anthropic_beta=anthropic_beta,
            )
            stream = _extract_stream_flag(request_data.stream)
            if stream:
                stream_iter = await _prepare_native_stream_iterator(
                    url,
                    headers,
                    payload,
                    timeout=timeout,
                    provider=provider,
                    operation="messages.stream",
                )
                response = StreamingResponse(
                    _touch_on_first_stream_output(
                        _sanitize_native_messages_stream(stream_iter),
                        credential_runtime,
                        credentials,
                    ),
                    media_type="text/event-stream",
                )
                runtime_owned_by_stream = True
                return response
            data = await _native_post_json(
                url,
                headers,
                payload,
                timeout=timeout,
                provider=provider,
                operation="messages",
            )
            if not _native_message_payload_is_semantic(data):
                raise_detached_error(_messages_upstream_http_exception())
            response = JSONResponse(data)
            await await_owned_worker(
                _mark_messages_credential_used(
                    credential_runtime,
                    credentials,
                )
            )
            return response

        # Non-native providers: convert to OpenAI-compatible request
        call_params = _build_openai_call_params(
            request_data=request_data,
            provider=provider,
            model=model,
            app_config=app_config_override,
            api_key=api_key,
            credentials=credentials,
        )

        stream = bool(call_params.get("stream"))
        if stream:
            try:
                stream_iter = await perform_chat_api_call_async(**call_params)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - arbitrary adapter failures are untrusted
                raise_detached_error(_messages_upstream_http_exception())
            response = StreamingResponse(
                _touch_on_first_stream_output(
                    _sanitize_converted_messages_stream(
                        openai_stream_to_anthropic(stream_iter, model=model)
                    ),
                    credential_runtime,
                    credentials,
                ),
                media_type="text/event-stream",
            )
            runtime_owned_by_stream = True
            return response

        try:
            async def _mark_late_valid_response(result: Any) -> None:
                if _convert_semantic_openai_messages_response(result, model=model) is not None:
                    await _mark_messages_credential_used(
                        credential_runtime,
                        credentials,
                    )

            response = await await_owned_worker(
                perform_chat_api_call_async(**call_params),
                on_cancel_result=_mark_late_valid_response,
            )
            converted_payload = _convert_semantic_openai_messages_response(
                response,
                model=model,
            )
            if converted_payload is None:
                raise ValueError("Invalid converted-provider response")
            converted_response = JSONResponse(converted_payload)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - arbitrary adapter failures are untrusted
            raise_detached_error(_messages_upstream_http_exception())
        await await_owned_worker(
            _mark_messages_credential_used(
                credential_runtime,
                credentials,
            )
        )
        return converted_response
    finally:
        if not runtime_owned_by_stream:
            await await_owned_worker(credential_runtime.close())


async def _handle_count_tokens(
    request_data: AnthropicCountTokensRequest,
    *,
    current_user: User,
    request: Request,
    anthropic_version: str | None,
    anthropic_beta: str | None,
) -> JSONResponse:
    """Handle an Anthropic-compatible count_tokens request."""
    provider, model = _resolve_provider_and_model_for_request(request_data)
    credential_runtime = _new_messages_credential_runtime(current_user, request)
    try:
        credentials = await _resolve_messages_credentials(
            credential_runtime,
            provider,
            model,
            operation="messages.count_tokens",
        )
        api_key = credentials.api_key
        app_config_override = credentials.app_config or {}
        if provider == "llama.cpp" and not api_key:
            api_key = _resolve_llamacpp_api_key(app_config_override)

        if provider not in MESSAGES_NATIVE_PROVIDERS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="count_tokens is only supported for Anthropic-compatible providers.",
            )

        base_url = _resolve_messages_base_url(provider, app_config_override)
        url = _join_messages_endpoint(base_url, "/messages/count_tokens")
        payload = _prepare_native_payload(request_data, model=model)
        timeout = _resolve_native_timeout(provider, app_config_override)
        headers = _build_native_headers(
            provider,
            api_key,
            anthropic_version=anthropic_version,
            anthropic_beta=anthropic_beta,
        )
        data = await _native_post_json(
            url,
            headers,
            payload,
            timeout=timeout,
            provider=provider,
            operation="messages.count_tokens",
        )
        if not _native_count_payload_is_semantic(data):
            raise_detached_error(_messages_upstream_http_exception())
        response = JSONResponse(data)
        await await_owned_worker(
            _mark_messages_credential_used(
                credential_runtime,
                credentials,
            )
        )
        return response
    finally:
        await await_owned_worker(credential_runtime.close())


@router.post(
    "/messages",
    summary="Anthropic-compatible Messages API",
    dependencies=[Depends(check_rate_limit)],
    responses={
        status.HTTP_200_OK: {
            "description": "Anthropic-compatible messages JSON response or SSE stream.",
            "content": {
                "application/json": {},
                "text/event-stream": {},
            },
        },
    },
)
async def create_messages(
    request: Request,
    request_data: AnthropicMessagesRequest = Body(...),
    current_user: User = Depends(get_request_user),
    anthropic_version: str | None = Header(None, alias="anthropic-version"),
    anthropic_beta: str | None = Header(None, alias="anthropic-beta"),
):
    """Create an Anthropic-compatible Messages response."""
    return await _handle_messages(
        request_data,
        current_user=current_user,
        request=request,
        anthropic_version=anthropic_version,
        anthropic_beta=anthropic_beta,
    )


@public_router.post(
    "/v1/messages",
    summary="Anthropic-compatible Messages API",
    dependencies=[Depends(check_rate_limit)],
    responses={
        status.HTTP_200_OK: {
            "description": "Anthropic-compatible messages JSON response or SSE stream.",
            "content": {
                "application/json": {},
                "text/event-stream": {},
            },
        },
    },
)
async def create_messages_public(
    request: Request,
    request_data: AnthropicMessagesRequest = Body(...),
    current_user: User = Depends(get_request_user),
    anthropic_version: str | None = Header(None, alias="anthropic-version"),
    anthropic_beta: str | None = Header(None, alias="anthropic-beta"),
):
    """Public endpoint for Anthropic-compatible Messages."""
    return await _handle_messages(
        request_data,
        current_user=current_user,
        request=request,
        anthropic_version=anthropic_version,
        anthropic_beta=anthropic_beta,
    )


@router.post(
    "/messages/count_tokens",
    summary="Anthropic-compatible Messages count_tokens",
    dependencies=[Depends(check_rate_limit)],
)
async def count_tokens(
    request: Request,
    request_data: AnthropicCountTokensRequest = Body(...),
    current_user: User = Depends(get_request_user),
    anthropic_version: str | None = Header(None, alias="anthropic-version"),
    anthropic_beta: str | None = Header(None, alias="anthropic-beta"),
):
    """Return token counts for Messages inputs."""
    return await _handle_count_tokens(
        request_data,
        current_user=current_user,
        request=request,
        anthropic_version=anthropic_version,
        anthropic_beta=anthropic_beta,
    )


@public_router.post(
    "/v1/messages/count_tokens",
    summary="Anthropic-compatible Messages count_tokens",
    dependencies=[Depends(check_rate_limit)],
)
async def count_tokens_public(
    request: Request,
    request_data: AnthropicCountTokensRequest = Body(...),
    current_user: User = Depends(get_request_user),
    anthropic_version: str | None = Header(None, alias="anthropic-version"),
    anthropic_beta: str | None = Header(None, alias="anthropic-beta"),
):
    """Public endpoint for Anthropic-compatible count_tokens."""
    return await _handle_count_tokens(
        request_data,
        current_user=current_user,
        request=request,
        anthropic_version=anthropic_version,
        anthropic_beta=anthropic_beta,
    )
