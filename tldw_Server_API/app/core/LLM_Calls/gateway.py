"""The one entrypoint for LLM provider calls, and the policy hook around them (ADR-025, TASK-13950.13)."""

from __future__ import annotations

import json as _json
import os
from typing import Any

from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.core import config_llm_providers
from tldw_Server_API.app.core.exceptions import (
    ChatAPIError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_Server_API.app.core.LLM_Calls.error_utils import exception_summary

_GATEWAY_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ChatAPIError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    HTTPException,
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    _json.JSONDecodeError,
)


def resolve_provider_api_key(
    provider: str,
    *,
    prefer_module_keys_in_tests: bool = True,
    include_override: bool = True,
) -> tuple[str | None, dict[str, Any]]:
    """
    Resolve a provider's API key: the LLM provider override store first, then env and config through
    `config_llm_providers.get_api_keys()`. Returns `(normalized_key, debug_info)`.
    """
    # prefer_module_keys_in_tests is accepted for its existing callers and has no effect.
    provider_key = (provider or "").strip().lower()
    debug_info: dict[str, Any] = {
        "provider": provider_key or provider,
        "selected_source": "missing",
        "raw_value_provided": False,
        "raw_value_was_empty": False,
        "dynamic_value_present": False,
        "override_value_present": False,
    }

    try:
        dynamic_keys = config_llm_providers.get_api_keys() or {}
    except _GATEWAY_NONCRITICAL_EXCEPTIONS as _err:
        # API key loading errors should be visible in production
        logger.warning("resolve_provider_api_key failed to load dynamic keys: {}", exception_summary(_err))
        dynamic_keys = {}

    override_value = None
    if include_override:
        try:
            from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import get_llm_provider_override

            override = get_llm_provider_override(provider_key)
            override_value = override.api_key if override else None
        except _GATEWAY_NONCRITICAL_EXCEPTIONS:
            override_value = None

    debug_info["override_value_present"] = override_value is not None

    dynamic_value = dynamic_keys.get(provider_key)
    debug_info["dynamic_value_present"] = dynamic_value is not None

    def _normalize(value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    raw_value = dynamic_value
    if override_value is not None:
        raw_value = override_value
        debug_info["selected_source"] = "override"
    elif raw_value is not None:
        env_var = (
            f"{provider_key.upper().replace('.', '_').replace('-', '_')}_API_KEY"
            if provider_key
            else None
        )
        debug_info["selected_source"] = "env" if env_var and os.getenv(env_var) is not None else "config"

    debug_info["raw_value_provided"] = raw_value is not None
    debug_info["raw_value_was_empty"] = isinstance(raw_value, str) and raw_value.strip() == ""
    normalized_value = _normalize(raw_value)
    return normalized_value, debug_info
