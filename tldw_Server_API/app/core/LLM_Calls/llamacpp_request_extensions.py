from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.exceptions import ChatBadRequestError


_LLAMA_CPP_EXTENSION_FIELDS = (
    "thinking_budget_tokens",
    "grammar_mode",
    "grammar_id",
    "grammar_inline",
    "grammar_override",
)


def _coerce_bool(value: Any) -> bool:
    """Normalize permissive truthy inputs used in llama.cpp runtime settings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return `value` as a mapping, falling back to an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def _has_llamacpp_extension_values(request_fields: Mapping[str, Any]) -> bool:
    """Report whether a request carries any non-empty llama.cpp extension fields."""
    grammar_mode = request_fields.get("grammar_mode")
    if grammar_mode not in (None, "", "none"):
        return True
    return any(request_fields.get(field) is not None for field in _LLAMA_CPP_EXTENSION_FIELDS if field != "grammar_mode")


def resolve_llamacpp_runtime_caps(
    *,
    app_config: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve deployment-specific llama.cpp capability metadata.

    The returned mapping captures whether strict OpenAI compatibility is active,
    whether a per-request thinking-budget key is configured, and which raw
    `extra_body` keys the UI should reserve for first-class llama.cpp controls.
    """
    context = dict(runtime_context or {})
    llama_settings = _as_mapping(_as_mapping(app_config).get("llama_api"))
    local_api_settings = _as_mapping(_as_mapping(app_config).get("local_api"))

    strict_override = context.get("strict_openai_compat")
    strict_env = os.getenv("LOCAL_LLM_STRICT_OPENAI_COMPAT")
    if strict_override is not None:
        strict_mode = _coerce_bool(strict_override)
    elif strict_env is not None:
        strict_mode = _coerce_bool(strict_env)
    else:
        strict_mode = _coerce_bool(llama_settings.get("strict_openai_compat"))

    request_key = str(
        context.get("thinking_budget_request_key")
        or os.getenv("LLAMA_CPP_THINKING_BUDGET_PARAM")
        or llama_settings.get("llama_cpp_thinking_budget_param")
        or llama_settings.get("thinking_budget_request_key")
        or local_api_settings.get("llama_cpp_thinking_budget_param")
        or ""
    ).strip()

    return {
        "strict_openai_compat": strict_mode,
        "thinking_budget": {
            "supported": bool(request_key),
            "request_key": request_key or None,
        },
        "reserved_extra_body_keys": ["grammar", *([request_key] if request_key else [])],
    }


def _resolve_grammar_text(
    *,
    request_fields: Mapping[str, Any],
    grammar_record: Mapping[str, Any] | None,
) -> str | None:
    """Choose the final grammar text for a llama.cpp request, if any."""
    grammar_mode = request_fields.get("grammar_mode")
    if grammar_mode == "library":
        if not isinstance(grammar_record, Mapping):
            raise ChatBadRequestError(provider="llama.cpp", message="Saved grammar could not be resolved")
        grammar_text = request_fields.get("grammar_override") or grammar_record.get("grammar_text")
        if not grammar_text:
            raise ChatBadRequestError(provider="llama.cpp", message="Saved grammar is missing grammar text")
        return str(grammar_text)
    if grammar_mode == "inline":
        grammar_text = request_fields.get("grammar_inline")
        return str(grammar_text) if grammar_text is not None else None
    return None


def resolve_llamacpp_request_extensions(
    *,
    request_fields: Mapping[str, Any],
    provider: str,
    grammar_record: Mapping[str, Any] | None,
    runtime_caps: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate app-level llama.cpp controls into provider `extra_body` fields.

    This resolver applies provider guards, strict-compatibility checks, saved
    grammar resolution, and thinking-budget mapping in one place so every chat
    caller produces a consistent outbound llama.cpp payload.
    """
    raw_extra_body = request_fields.get("extra_body")
    has_original_extra_body = isinstance(raw_extra_body, Mapping)
    extra_body = dict(raw_extra_body) if has_original_extra_body else {}
    has_extension_values = _has_llamacpp_extension_values(request_fields)

    if provider != "llama.cpp":
        if has_extension_values:
            raise ChatBadRequestError(
                provider=provider or None,
                message="llama.cpp extension fields require the resolved target provider to be 'llama.cpp'",
            )
        return {"extra_body": extra_body if has_original_extra_body else None}

    if runtime_caps.get("strict_openai_compat") and has_extension_values:
        raise ChatBadRequestError(
            provider=provider,
            message="llama.cpp advanced controls are disabled by strict_openai_compat",
        )

    grammar_text = _resolve_grammar_text(request_fields=request_fields, grammar_record=grammar_record)
    if grammar_text:
        extra_body["grammar"] = grammar_text

    thinking_budget_tokens = request_fields.get("thinking_budget_tokens")
    if thinking_budget_tokens is not None:
        thinking_caps = _as_mapping(runtime_caps.get("thinking_budget"))
        request_key = str(
            thinking_caps.get("request_key") or runtime_caps.get("thinking_budget_request_key") or ""
        ).strip()
        if not thinking_caps.get("supported") or not request_key:
            raise ChatBadRequestError(
                provider=provider,
                message="thinking_budget_tokens is not supported by this deployment",
            )
        extra_body[request_key] = int(thinking_budget_tokens)

    if extra_body or has_original_extra_body:
        return {"extra_body": extra_body}
    return {"extra_body": None}


__all__ = [
    "resolve_llamacpp_request_extensions",
    "resolve_llamacpp_runtime_caps",
]
