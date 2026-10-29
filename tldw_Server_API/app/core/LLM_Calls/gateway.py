"""The one entrypoint for LLM provider calls, and the policy hook around them (ADR-025, TASK-13950.13)."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json as _json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Iterator, Mapping
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, NoReturn

from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.core import config_llm_providers
from tldw_Server_API.app.core.custom_openai_providers import custom_openai_provider_number
from tldw_Server_API.app.core.exceptions import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatModelNotAvailableError,
    ChatProviderError,
    EgressPolicyError,
    raise_detached_error,
)
from tldw_Server_API.app.core.LLM_Calls import adapter_registry as _adapter_registry
from tldw_Server_API.app.core.LLM_Calls.anthropic_model_ids import canonical_anthropic_model_id
from tldw_Server_API.app.core.LLM_Calls.bounded_daemon import (
    SYNC_ADAPTER_CALL_POOL,
    DaemonCapacityError,
    await_bounded_sync_call,
)
from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.LLM_Calls.error_utils import (
    exception_summary,
    privacy_safe_chat_error,
    provider_error_privacy_scope,
)
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.LLM_Calls.provider_model_inventory import (
    discover_provider_models,
    has_model_listing,
    inventory_endpoint_scope,
    resolve_provider_model_selection,
    resolve_provider_models_base_url,
)
from tldw_Server_API.app.core.LLM_Calls.provider_readiness import normalize_catalog_provider_for_chat
from tldw_Server_API.app.core.LLM_Calls.provider_results import sanitized_provider_stream_exception
from tldw_Server_API.app.core.LLM_Calls.routing.models import RoutingDecision
from tldw_Server_API.app.core.LLM_Calls.streaming import wrap_sync_stream
from tldw_Server_API.app.core.testing import is_truthy as _shared_is_truthy
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
from tldw_Server_API.app.core.Usage.llm_quota_gate import check_llm_call_async, check_llm_call_sync
from tldw_Server_API.app.core.Usage.pricing_catalog import get_pricing_catalog, list_provider_models

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

_INLINE_MODEL_PROVIDER_NAMES = frozenset(
    normalize_catalog_provider_for_chat(provider_name)
    for provider_name in _adapter_registry.ChatProviderRegistry.DEFAULT_ADAPTERS
)
_NAMESPACED_MODEL_PROVIDERS = {"openrouter", "huggingface", "novita", "poe", "together"}


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


# --- Cached helpers (module scope) -------------------------------------------
@lru_cache(maxsize=16)
def _load_models_with_case_cached(provider: str) -> list[str]:
    """Load provider models preserving original key casing when possible.

    Attempts to read the raw model_pricing.json to preserve the original
    casing of model identifiers. Falls back to the normalized catalog
    via list_provider_models (lowercase keys) if needed.
    """
    try:
        cfg_path = Path(__file__).resolve().parents[3] / "Config_Files" / "model_pricing.json"
        if cfg_path.exists():
            data = _json.loads(cfg_path.read_text())
            prov_block = data.get(provider) or data.get(provider.capitalize()) or data.get(provider.upper())
            if isinstance(prov_block, dict):
                return list(prov_block.keys())
    except (OSError, ValueError, KeyError, AttributeError) as _e:
        # Expected file/format issues: fall back to normalized catalog
        logger.debug(
            "Model catalog raw load fallback for provider '{}': {}",
            provider,
            exception_summary(_e),
        )
    except _GATEWAY_NONCRITICAL_EXCEPTIONS as _ue:
        # Unexpected exceptions should be visible in production logs
        pass
        logger.warning(
            "Unexpected error loading model catalog for provider '{}': {}",
            provider,
            exception_summary(_ue),
        )

    # Fallback: use the normalized list (lowercase keys)
    return list_provider_models(provider) or []


@lru_cache(maxsize=1)
def _load_alias_overrides_cached() -> dict[str, dict[str, str]]:
    """Load model alias overrides from env and pricing catalog.

    Resolution order:
    1) ENV var CHAT_MODEL_ALIAS_OVERRIDES (JSON)
    2) Keys in Config_Files/model_pricing.json: model_aliases/aliases/alias_map
    """
    # 1) ENV
    try:
        raw = os.getenv("CHAT_MODEL_ALIAS_OVERRIDES")
        if raw:
            data = _json.loads(raw)
            if isinstance(data, dict):
                return {
                    str(k).lower(): {str(ak).lower(): str(av) for ak, av in (v or {}).items()}
                    for k, v in data.items()
                    if isinstance(v, dict)
                }
    except (ValueError, TypeError) as _e:
        logger.debug("CHAT_MODEL_ALIAS_OVERRIDES parse failed: {}", exception_summary(_e))
    except _GATEWAY_NONCRITICAL_EXCEPTIONS as _ue:
        # Unexpected exceptions should be visible in production logs
        pass
        logger.warning("Unexpected error parsing CHAT_MODEL_ALIAS_OVERRIDES: {}", exception_summary(_ue))

    # 2) File keys in pricing catalog
    try:
        cfg_path = Path(__file__).resolve().parents[3] / "Config_Files" / "model_pricing.json"
        if cfg_path.exists():
            data = _json.loads(cfg_path.read_text())
            for key in ("model_aliases", "aliases", "alias_map"):
                block = data.get(key)
                if isinstance(block, dict):
                    return {
                        str(k).lower(): {str(ak).lower(): str(av) for ak, av in (v or {}).items()}
                        for k, v in block.items()
                        if isinstance(v, dict)
                    }
    except (OSError, ValueError, KeyError, AttributeError) as _e:
        logger.debug("Alias overrides load fallback: {}", exception_summary(_e))
    except _GATEWAY_NONCRITICAL_EXCEPTIONS as _ue:
        # Unexpected exceptions should be visible in production logs
        pass
        logger.warning("Unexpected error loading alias overrides: {}", exception_summary(_ue))

    return {}


@lru_cache(maxsize=2048)
def _provider_has_model_cached(provider: str, model: str) -> bool:
    """Return True when provider catalog contains the given model (exact, case-insensitive)."""
    provider_key = (provider or "").strip().lower()
    model_key = (model or "").strip().lower()
    if not provider_key or not model_key:
        return False
    try:
        for candidate in _load_models_with_case_cached(provider_key):
            if str(candidate).strip().lower() == model_key:
                return True
    except _GATEWAY_NONCRITICAL_EXCEPTIONS:
        return False
    return False


@lru_cache(maxsize=4096)
def _find_catalog_providers_for_model_cached(model: str) -> tuple[str, ...]:
    """Return providers whose pricing catalog contains the model (exact, case-insensitive)."""
    model_key = (model or "").strip().lower()
    if not model_key:
        return ()

    matches: set[str] = set()
    try:
        catalog = getattr(get_pricing_catalog(), "_catalog", {})
        if isinstance(catalog, dict):
            for provider_name, provider_models in catalog.items():
                if not isinstance(provider_models, dict):
                    continue
                for model_name, model_meta in provider_models.items():
                    if str(model_name).strip().lower() != model_key:
                        continue
                    if isinstance(model_meta, dict) and model_meta.get("placeholder"):
                        continue
                    provider_key = str(provider_name).strip().lower()
                    if provider_key:
                        matches.add(provider_key)
                    break
    except _GATEWAY_NONCRITICAL_EXCEPTIONS:
        return ()

    return tuple(sorted(matches))


def infer_provider_from_model_catalog(
    *,
    provider: str,
    model: str | None,
    provider_explicit: bool,
) -> tuple[str, dict[str, Any]]:
    """Infer provider from model catalog when the current provider likely mismatches.

    Safety rules:
    - Never override an explicit provider.
    - Only switch when the model has a unique provider match in catalog.
    - Keep current provider on ambiguous matches.
    """
    current_provider = (provider or "").strip().lower()
    model_value = (model or "").strip()
    debug: dict[str, Any] = {
        "attempted": False,
        "provider_explicit": provider_explicit,
        "current_provider": current_provider,
        "model": model_value,
        "current_provider_has_model": None,
        "candidates": [],
        "selected_provider": current_provider,
        "reason": "skipped",
    }

    if provider_explicit:
        debug["reason"] = "explicit_provider"
        return current_provider, debug
    if not current_provider or not model_value:
        debug["reason"] = "missing_provider_or_model"
        return current_provider, debug
    if not _is_local_chat_provider(current_provider):
        debug["reason"] = "commercial_inventory_requires_credentials"
        return current_provider, debug
    if "/" in model_value:
        debug["reason"] = "inline_model_namespace"
        return current_provider, debug

    debug["attempted"] = True
    # A saved dotted name (claude-sonnet-4.5) finds the catalog entry of its dashed API ID; the name as written
    # is tried first, so a provider that lists the dotted name itself keeps it.
    lookup_model = canonical_anthropic_model_id(model_value)
    has_current_match = _provider_has_model_cached(current_provider, model_value) or _provider_has_model_cached(
        current_provider, lookup_model
    )
    debug["current_provider_has_model"] = has_current_match
    if has_current_match:
        debug["reason"] = "current_provider_match"
        return current_provider, debug

    candidates = [
        name for name in _find_catalog_providers_for_model_cached(model_value)
        if _is_local_chat_provider(name)
    ] or [
        name for name in _find_catalog_providers_for_model_cached(lookup_model)
        if _is_local_chat_provider(name)
    ]
    debug["candidates"] = candidates
    if not candidates:
        debug["reason"] = "no_catalog_match"
        return current_provider, debug

    if len(candidates) == 1:
        selected = candidates[0]
        debug["selected_provider"] = selected
        debug["reason"] = "unique_catalog_match"
        return selected, debug

    if current_provider in candidates:
        debug["reason"] = "ambiguous_kept_current"
        return current_provider, debug

    debug["reason"] = "ambiguous_no_change"
    return current_provider, debug


def _split_inline_provider_model(
    model_str: str, api_provider: str | None = None
) -> tuple[str | None, str | None, str | None]:
    """Split recognized provider prefixes without changing opaque model IDs."""
    model_value = (model_str or "").strip()
    if not model_value:
        return None, None, None
    explicit_provider = (
        canonical_provider_name(normalize_catalog_provider_for_chat(api_provider))
        if api_provider else None
    )
    preserve_local_namespace = (
        explicit_provider in _adapter_registry.ChatProviderRegistry.DEFAULT_LOCAL_PROVIDERS
    )

    if "/" in model_value:
        model_provider, model_name = model_value.split("/", 1)
        normalized_provider = canonical_provider_name(
            normalize_catalog_provider_for_chat(model_provider)
        )
        if normalized_provider in _INLINE_MODEL_PROVIDER_NAMES and (
            not preserve_local_namespace or normalized_provider == explicit_provider
        ):
            return model_provider.strip(), model_name.strip(), "/"
        # Model identifiers can be paths or repository names. Only registered
        # provider prefixes qualify for stripping; a colon prefix may precede
        # a path (for example, "llamacpp:../../models/model.gguf").

    if ":" not in model_value:
        return None, None, None

    model_provider, model_name = model_value.split(":", 1)
    normalized_provider = canonical_provider_name(normalize_catalog_provider_for_chat(model_provider))
    if normalized_provider not in _INLINE_MODEL_PROVIDER_NAMES or (
        preserve_local_namespace and normalized_provider != explicit_provider
    ):
        return None, None, None
    return model_provider.strip(), model_name.strip(), ":"


def invalidate_model_alias_caches() -> None:
    """Invalidate cached model list and alias overrides for hot-reload.

    Clears lru_cache for both `_load_models_with_case_cached` and
    `_load_alias_overrides_cached`. Safe to call at runtime (no side effects
    beyond cache flush). Subsequent requests will repopulate from sources.
    """
    with contextlib.suppress(_GATEWAY_NONCRITICAL_EXCEPTIONS):
        _load_models_with_case_cached.cache_clear()
    with contextlib.suppress(_GATEWAY_NONCRITICAL_EXCEPTIONS):
        _load_alias_overrides_cached.cache_clear()
    with contextlib.suppress(_GATEWAY_NONCRITICAL_EXCEPTIONS):
        _provider_has_model_cached.cache_clear()
    with contextlib.suppress(_GATEWAY_NONCRITICAL_EXCEPTIONS):
        _find_catalog_providers_for_model_cached.cache_clear()


def parse_provider_model_for_metrics(
    request_data: Any,
    default_provider: str,
) -> tuple[str, str]:
    """Parse provider and model for metrics logging without mutating request_data.

    Accepts model strings like "anthropic/claude-opus-4.1" and an optional
    api_provider on the request, falling back to the server default.

    Returns (provider, model_for_metrics).
    """
    raw_model = getattr(request_data, "model", None)
    model_str = (
        raw_model.strip()
        if isinstance(raw_model, str) and raw_model.strip()
        else "unknown"
    )
    api_provider = getattr(request_data, "api_provider", None)
    model_provider, model_name, _ = _split_inline_provider_model(model_str, api_provider)
    if model_provider is not None and model_name is not None:
        raw_provider = api_provider or model_provider
        provider = normalize_catalog_provider_for_chat((raw_provider or "").strip().lower())
        model = model_name
    else:
        raw_provider = api_provider or default_provider
        provider = normalize_catalog_provider_for_chat((raw_provider or "").strip().lower())
        model = model_str
    return provider, model


def normalize_request_provider_and_model(
    request_data: Any,
    default_provider: str,
) -> str:
    """Normalize provider and model on the request.

    If the request's model contains a provider prefix (e.g., "groq/llama-3"),
    update request_data.model in-place to only the model component and return
    the selected provider name. This mirrors the behavior already present in
    the endpoint and avoids duplication.
    """
    model_str = getattr(request_data, "model", None) or ""
    api_provider = getattr(request_data, "api_provider", None)

    # --- Alias resolution via configurable catalog (model_pricing.json) ---
    # Prefer provider-agnostic model keys and avoid hardcoded, versioned IDs.
    # Alias resolution is strict: only explicit overrides or exact model matches
    # are applied. This avoids silently picking a default model.
    try:
        # Determine provider context for alias resolution.
        # If model includes an inline provider (e.g., "anthropic/claude-sonnet"),
        # prefer that; else honor api_provider, then default_provider.
        inline_provider: str | None = None
        inline_model_part: str | None = None
        inline_separator: str | None = None
        inline_provider, inline_model_part, inline_separator = _split_inline_provider_model(model_str, api_provider)
        provider_for_mapping = normalize_catalog_provider_for_chat(
            ((inline_provider or api_provider or default_provider) or "").strip().lower()
        )


        def _resolve_alias(provider: str, raw_model: str) -> str | None:
            if not _is_local_chat_provider(provider):
                return None
            m = (raw_model or "").strip()
            if not m:
                return None
            models = _load_models_with_case_cached(provider)
            if not models:
                return None
            m_lower = m.lower()
            # Strict resolution: only exact (case-insensitive) matches.
            for cand in models:
                if cand.lower() == m_lower:
                    return cand
            return None

        # Optional alias overrides allow cross-provider or provider-agnostic mappings
        # without changing code. Sources (first match wins):
        #  - ENV JSON CHAT_MODEL_ALIAS_OVERRIDES = { provider: { alias: concrete_model } }
        #  - Config_Files/model_pricing.json key "model_aliases" or "aliases"
        # Explicit aliases select a destination; only scoped inventory can
        # authorize that destination for a commercial provider at dispatch.
        alias_overrides = _load_alias_overrides_cached()

        # First apply alias override (supports cross-provider targets)
        ov_map = alias_overrides.get(provider_for_mapping, {})
        target_model_part = inline_model_part if inline_model_part is not None else model_str
        override_target = ov_map.get((target_model_part or "").strip().lower())
        if override_target:
            resolved = override_target
        else:
            # Resolve against known models when no explicit override present
            resolved = _resolve_alias(provider_for_mapping, target_model_part)
        if resolved and resolved != model_str:
            allow_cross = _shared_is_truthy(os.getenv("CHAT_ALLOW_CROSS_PROVIDER_ALIASING", "0"))
            if inline_provider:
                # Preserve inline provider prefix until final normalization below
                combined = f"{inline_provider}{inline_separator or '/'}{resolved}"
                request_data.model = combined
                model_str = combined
            else:
                # Prevent accidental provider flips unless explicitly allowed.
                if provider_for_mapping == "openrouter":
                    request_data.model = resolved
                    model_str = resolved
                elif "/" in resolved and not allow_cross:
                    resolved = None
                else:
                    request_data.model = resolved
                    model_str = request_data.model
                if not resolved:
                    resolved = None  # leave model_str untouched, no override applied
    except (AttributeError, KeyError, ValueError):
        # Expected lookup/attr issues: do not block request
        pass
    except _GATEWAY_NONCRITICAL_EXCEPTIONS as _unexpected:
        # Unexpected exceptions should be visible in production logs
        logger.warning(
            "Unexpected error during model alias resolution: {}",
            exception_summary(_unexpected),
        )
    provider = normalize_catalog_provider_for_chat(
        ((api_provider or default_provider) or "").strip().lower()
    )
    model_provider, actual_model, inline_separator = _split_inline_provider_model(model_str, api_provider)
    if model_provider is not None and actual_model is not None:
        inline_provider_lower = normalize_catalog_provider_for_chat(
            model_provider.strip().lower()
        )
        # If the api_provider was not explicitly set, allow inline provider to select it
        if not api_provider:
            provider = inline_provider_lower
            # In this case, strip the inline provider prefix from the model
            request_data.model = actual_model
        else:
            # api_provider is explicitly set on the request. For some providers,
            # many valid model IDs include a namespace
            # (e.g., "openai/gpt-4o-mini", "z-ai/glm-4.6"). Preserve slash
            # namespaces unless the inline namespace is the provider itself.
            if provider in _NAMESPACED_MODEL_PROVIDERS and inline_separator == "/":
                if provider == "openrouter" and inline_provider_lower == "openrouter":
                    request_data.model = actual_model
                else:
                    request_data.model = model_str
            else:
                # Non-namespaced providers do not use provider-qualified model ids; strip prefix
                request_data.model = actual_model
    return provider


def resolve_provider_and_model(
    request_data: Any,
    metrics_default_provider: str,
    normalize_default_provider: str,
    routing_decision: RoutingDecision | None = None,
) -> tuple[str, str, str, str | None, dict[str, Any]]:
    """Resolve provider/model for metrics and execution and record the decision path.

    Returns a 5-tuple:
        (metrics_provider, metrics_model, selected_provider, selected_model, debug_info)

    - metrics_provider/metrics_model mirror the legacy behavior of
      `parse_provider_model_for_metrics` (used for request-level metrics and audit).
    - selected_provider/selected_model reflect the normalized provider/model that
      downstream logic should use (after alias resolution and inline prefixes).
    - debug_info is a JSON-serializable dict describing the decision path.

    The function may update `request_data.model` in-place via
    `normalize_request_provider_and_model`, matching existing behavior.
    """
    raw_model = getattr(request_data, "model", None)
    raw_api_provider = getattr(request_data, "api_provider", None)
    provider_explicit = bool(str(raw_api_provider or "").strip())

    if routing_decision is not None and routing_decision.canonical:
        request_data.api_provider = routing_decision.provider
        request_data.model = routing_decision.model

        debug_info: dict[str, Any] = {
            "raw": {
                "api_provider": raw_api_provider,
                "model": raw_model,
                "provider_explicit": provider_explicit,
            },
            "routing": {
                "canonical": True,
                "decision_source": routing_decision.decision_source,
                "provider": routing_decision.provider,
                "model": routing_decision.model,
            },
            "metrics": {
                "default_provider": metrics_default_provider,
                "provider": routing_decision.provider,
                "model": routing_decision.model,
            },
            "selected": {
                "provider": routing_decision.provider,
                "model": routing_decision.model,
            },
        }
        return (
            routing_decision.provider,
            routing_decision.model,
            routing_decision.provider,
            routing_decision.model,
            debug_info,
        )

    # Step 1: derive metrics-facing provider/model without mutating the request
    metrics_provider, metrics_model = parse_provider_model_for_metrics(
        request_data, metrics_default_provider
    )

    selected_provider = metrics_provider
    selected_model = (
        raw_model.strip()
        if isinstance(raw_model, str) and raw_model.strip()
        else None
    )

    # Step 2: normalize provider/model for execution (may mutate request_data.model)
    try:
        normalized_provider = normalize_request_provider_and_model(
            request_data, normalize_default_provider
        )
        selected_provider = normalized_provider
        new_model = getattr(request_data, "model", None)
        selected_model = (
            new_model.strip()
            if isinstance(new_model, str) and new_model.strip()
            else None
        )
    except _GATEWAY_NONCRITICAL_EXCEPTIONS as exc:
        # Keep the pre-normalization execution values if optional normalization fails.
        pass
        logger.debug(
            "resolve_provider_and_model: normalization failed, "
            "keeping pre-normalization execution values. Error={}",
            exc,
        )

    catalog_inference_debug: dict[str, Any] | None = None
    try:
        selected_provider, catalog_inference_debug = infer_provider_from_model_catalog(
            provider=selected_provider,
            model=selected_model,
            provider_explicit=provider_explicit,
        )
    except _GATEWAY_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "resolve_provider_and_model: catalog inference failed; keeping selected provider. Error={}",
            exc,
        )

    debug_info: dict[str, Any] = {
        "raw": {
            "api_provider": raw_api_provider,
            "model": raw_model,
            "provider_explicit": provider_explicit,
        },
        "metrics": {
            "default_provider": metrics_default_provider,
            "provider": metrics_provider,
            "model": metrics_model,
        },
        "normalized": {
            "default_provider": normalize_default_provider,
            "provider": selected_provider,
            "model": selected_model,
        },
        "changed": {
            "provider_changed": metrics_provider != selected_provider,
            "model_changed": metrics_model != selected_model,
        },
        "catalog_inference": catalog_inference_debug,
    }

    return metrics_provider, metrics_model, selected_provider, selected_model, debug_info


def _resolve_base_url_override(provider: str, chat_args: dict[str, Any]) -> str | None:
    base_url = chat_args.get("base_url")
    if base_url is None:
        base_url = chat_args.get("api_base_url")
    if base_url is None:
        return None
    provider_key = (provider or "").strip().lower()
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
        is_trusted_base_url_request,
        resolve_byok_base_url_allowlist,
        validate_base_url_override,
    )

    allowlist = _resolve_chat_provider_allowlist(resolve_byok_base_url_allowlist())
    if provider_key not in allowlist:
        raise ChatBadRequestError(
            provider=provider_key or None,
            message=f"base_url override is not enabled for provider '{provider_key}'",
        )
    trusted_override = bool(chat_args.get("trusted_base_url_override"))
    if not trusted_override:
        request_obj = chat_args.get("request") or chat_args.get("caller_request")
        principal = chat_args.get("principal")
        user = chat_args.get("auth_user")
        trusted_override = is_trusted_base_url_request(request_obj, principal=principal, user=user)
    if not trusted_override:
        raise ChatBadRequestError(
            provider=provider_key or None,
            message="base_url override requires a trusted caller",
        )
    try:
        return validate_base_url_override(base_url)
    except ValueError as exc:
        raise ChatBadRequestError(provider=provider_key or None, message=str(exc)) from exc


def _resolve_chat_provider_name(provider: str | None) -> str:
    """Return the canonical chat provider key used for adapter dispatch."""
    provider_key = (provider or "").strip().lower()
    if provider_key == "local":
        return "local-llm"

    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry

    return get_registry().resolve_provider_name(provider_key)


def _resolve_chat_provider_allowlist(providers: set[str]) -> set[str]:
    """Canonicalize provider names and aliases used in BYOK allowlists."""
    return {
        resolved
        for provider in providers
        if (resolved := _resolve_chat_provider_name(provider))
    }


def _is_local_chat_provider(provider: str | None) -> bool:
    """Return whether a chat provider name or alias resolves to a local adapter."""
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry

    return get_registry().is_local_provider_name(provider)


def _find_local_request_url_override_keys(chat_args: dict[str, Any]) -> list[str]:
    """Return non-null request keys that would override a local endpoint URL."""
    return sorted(
        key
        for key, value in chat_args.items()
        if value is not None
        and (
            key in {"api_url", "base_url", "api_base_url"}
            or key.endswith("_api_url")
        )
    )


def _reject_local_request_url_overrides(provider: str, chat_args: dict[str, Any]) -> None:
    """Reject request-level endpoint URL overrides for local chat providers."""
    provider_key = (provider or "").strip().lower()
    if not _is_local_chat_provider(provider_key):
        return

    override_keys = _find_local_request_url_override_keys(chat_args)
    if not override_keys:
        return

    raise ChatBadRequestError(
        provider=provider_key,
        message=(
            "Local provider endpoint URLs are config-only; request-level "
            f"override keys are not allowed: {', '.join(override_keys)}"
        ),
    )


def _build_adapter_request_from_chat_args(chat_args: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Translate chat_api_call-style args into an adapter request payload."""
    from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
        bind_provider_call_credentials,
        ensure_app_config,
        normalize_provider,
        provider_auth_is_resolved,
        resolve_provider_api_key_from_config,
        resolve_provider_model,
        resolve_provider_section,
    )
    from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key

    provider = normalize_provider(
        chat_args.get("api_endpoint") or chat_args.get("api_provider") or chat_args.get("provider")
    )
    provider = _resolve_chat_provider_name(provider)
    if not provider:
        raise ChatConfigurationError(provider=str(chat_args.get("api_endpoint")), message="LLM provider is required.")

    chat_args, provider_credentials = bind_provider_call_credentials(
        provider,
        chat_args,
        consume=False,
    )

    _reject_local_request_url_overrides(provider, chat_args)
    is_local_provider = _is_local_chat_provider(provider)

    credentials_resolved = provider_credentials is not None
    explicit_app_config = (
        provider_credentials.app_config
        if provider_credentials is not None
        else chat_args.get("app_config")
    )
    if credentials_resolved:
        app_config = copy.deepcopy(explicit_app_config or {})
        provider_section = resolve_provider_section(provider)
        if provider_section:
            app_config.setdefault(provider_section, {})
    else:
        app_config = (
            explicit_app_config
            if explicit_app_config is not None
            else (None if is_local_provider else ensure_app_config(None))
        )
    model = chat_args.get("model")
    if model is None and app_config is not None:
        if credentials_resolved:
            provider_config = app_config.get(resolve_provider_section(provider)) or {}
            model = provider_config.get("model") or provider_config.get("model_id")
            if provider == "mlx":
                model = model or provider_config.get("model_path") or provider_config.get("mlx_model_path")
        else:
            model = resolve_provider_model(provider, app_config)
    if not model and not is_local_provider:
        raise ChatConfigurationError(provider=provider, message="Model is required for provider.")

    if credentials_resolved:
        api_key = (
            provider_credentials.api_key
            if provider_credentials is not None
            else chat_args.get("api_key")
        )
        if provider_requires_api_key(provider) and not provider_auth_is_resolved(
            provider,
            api_key=api_key,
            app_config=app_config,
            credentials_resolved=True,
        ):
            raise ChatConfigurationError(
                provider=provider,
                message="API key is required for provider.",
            )
    else:
        api_key = chat_args.get("api_key") or resolve_provider_api_key_from_config(provider, app_config)
    messages_payload = chat_args.get("messages_payload") or chat_args.get("messages") or []

    request: dict[str, Any] = {
        "messages": messages_payload,
        "system_message": chat_args.get("system_message"),
        "model": model,
        "api_key": api_key,
        "app_config": app_config,
    }
    if credentials_resolved:
        request["credentials_resolved"] = True
    if provider_credentials is not None:
        request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = provider_credentials
    base_url_override = _resolve_base_url_override(provider, chat_args)
    if base_url_override:
        request["base_url"] = base_url_override

    stream_value = chat_args.get("stream")
    if stream_value is None:
        stream_value = chat_args.get("streaming")
    if stream_value is not None:
        request["stream"] = bool(stream_value)

    skip_keys = {
        "api_endpoint",
        "api_provider",
        "provider",
        "messages_payload",
        "messages",
        "system_message",
        "model",
        "api_key",
        "app_config",
        "credentials_resolved",
        "base_url",
        "api_base_url",
        "request",
        "caller_request",
        "principal",
        "auth_user",
        "trusted_base_url_override",
        "configured_endpoint_base_url",
        "configured_endpoint_scope",
        "endpoint_provenance",
        "http_client_factory",
        "http_fetcher",
        "http_streamer",
        "timeout",
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
        "_structured_requested_response_format",
        "stream",
        "streaming",
        "history_message_limit",
        "history_message_order",
        "slash_command_injection_mode",
    }
    for key, value in chat_args.items():
        if key in skip_keys or key.startswith("_chat_") or key == "_endpoint_provenance" or value is None:
            continue
        if key not in request:
            request[key] = value

    provenance = chat_args.get("_endpoint_provenance")
    if (
        custom_openai_provider_number(provider) is not None
        and provenance in {"server_config", "byok", "request_override"}
    ):
        request["_endpoint_provenance"] = provenance

    timeout = chat_args.get("timeout")
    internal = {"timeout": timeout} if timeout is not None else {}
    return provider, request, internal


def _attach_internal_http_hooks(adapter: Any, request: dict[str, Any], internal: dict[str, Any]) -> None:
    """Compatibility no-op: transport hooks are adapter-owned, never request-owned."""
    return None


def _get_llm_registry():
    """Resolve the adapter registry at call time to honor test monkeypatching."""
    return _adapter_registry.get_registry()


def _map_provider_egress_error(
    provider: str,
    exc: EgressPolicyError,
) -> ChatAPIError:
    """Map typed policy failures without exposing target or resolver details."""
    if exc.reason_code == "dns_unresolved":
        return ChatProviderError(
            provider=provider,
            message="Provider endpoint is unreachable.",
            status_code=503,
        )
    return ChatConfigurationError(
        provider=provider,
        message="Provider endpoint is blocked by the outbound security policy.",
    )


def _map_sync_stream_egress_errors(provider: str, stream: Any) -> Iterator[Any]:
    """Map policy failures raised lazily while consuming a sync stream."""
    try:
        yield from stream
    except EgressPolicyError as exc:
        raise _map_provider_egress_error(provider, exc) from exc


async def _map_async_stream_egress_errors(
    provider: str,
    stream: AsyncIterator[Any],
) -> AsyncIterator[Any]:
    """Map policy failures raised lazily while consuming an async stream."""
    try:
        async for item in stream:
            yield item
    except EgressPolicyError as exc:
        raise _map_provider_egress_error(provider, exc) from exc


def _privacy_safe_provider_errors_enabled(request: dict[str, Any]) -> bool:
    """Return whether this opt-in call requires metadata-only provider errors."""

    call_policy = request.get("call_policy")
    return bool(
        isinstance(call_policy, ProviderCallPolicy)
        and call_policy.privacy_safe_errors
    )


def _validate_audited_call_policy_transport(
    registry: Any,
    provider: str,
    request: dict[str, Any],
) -> ProviderCallPolicy | None:
    """Fail closed when an opt-in strict policy lacks adapter guarantees."""
    policy = request.get("call_policy")
    if not isinstance(policy, ProviderCallPolicy) or (
        policy.maximum_timeout_seconds is None
        and policy.required_endpoint_scope is None
    ):
        return None
    transport = registry.get_audited_call_policy_transport(provider)
    if (
        policy.maximum_timeout_seconds is None
        or policy.required_endpoint_scope is None
        or transport is None
        or policy.max_transport_attempts is None
        or policy.max_transport_attempts > transport.maximum_transport_attempts
        or not transport.enforces_configured_endpoint_scope
        or not transport.enforces_maximum_timeout
    ):
        raise ChatConfigurationError(
            provider=provider,
            message="LLM adapter does not support the required call policy.",
        )
    return policy


async def _await_with_call_policy_deadline(
    awaitable: Awaitable[Any],
    policy: ProviderCallPolicy | None,
) -> Any:
    if policy is None:
        return await awaitable
    return await asyncio.wait_for(
        awaitable,
        timeout=policy.maximum_timeout_seconds,
    )


def _provider_inventory_endpoint(
    provider: str, request: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Resolve the discovery base and the discovery options carrying generation's egress decision for it.

    Unavailable scoped configuration never falls back to global discovery. A call policy's scope wins, as it
    does for generation; otherwise ``inventory_endpoint_scope`` decides from the endpoint's provenance.
    """
    from tldw_Server_API.app.core.LLM_Calls.adapter_utils import PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY

    base_url = resolve_provider_models_base_url(
        provider, request.get("app_config"),
        credentials_resolved=request.get("credentials_resolved") is True,
        base_url=request.get("base_url"),
    )
    if not base_url:
        raise ChatConfigurationError(
            provider=provider, message="Provider model inventory endpoint is unavailable.",
        )
    policy = request.get("call_policy")
    if isinstance(policy, ProviderCallPolicy):
        return base_url, {"call_policy": policy}
    credentials = request.get(PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY)
    provenance = request.get("_endpoint_provenance") or getattr(credentials, "endpoint_provenance", None)
    scope = inventory_endpoint_scope(provider, base_url, endpoint_provenance=provenance)
    return base_url, ({"configured_endpoint": scope} if scope is not None else {})


_INVENTORY_OUTAGES = frozenset({"unreachable", "server_error"})


def _raise_inventory_failure(provider: str, status: str, *, refreshable_oauth: bool = False) -> NoReturn:
    """Raise the typed error the endpoints already map for a model list that is not ready.

    auth_failed is the provider rejecting the credential, so it is an authentication error. Discovery runs
    before generation, so a refreshable OAuth credential may refresh and retry (review I3). An outage gets its
    own public code (review M1); anything else is a configuration problem.
    """
    if status == "auth_failed":
        error = ChatAuthenticationError(
            provider=provider, message="The provider rejected the credential while listing its models.",
        )
        if refreshable_oauth:
            error.credential_refresh_retry_safe = True
        raise error
    if status in _INVENTORY_OUTAGES:
        raise ChatConfigurationError(
            provider=provider,
            message="Provider model inventory is temporarily unavailable.",
            error_code="provider_model_inventory_unavailable",
        )
    raise ChatConfigurationError(provider=provider, message="Provider model inventory is not supported.")


def _validate_provider_model_selection(
    provider: str, request: dict[str, Any], *, refreshable_oauth: bool | None = None,
) -> None:
    """Authorize current IDs, confirmed aliases and explicit router fallback IDs."""
    from tldw_Server_API.app.core.LLM_Calls.adapter_utils import PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY

    if not has_model_listing(provider):
        return  # no listing: the provider itself rejects a bad model
    if refreshable_oauth is None:
        credentials = request.get(PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY)
        refreshable_oauth = getattr(credentials, "auth_source", None) == "oauth"
    selected_models: list[Any] = [request.get("model")]
    extra_body = request.get("extra_body")
    fallback_models = None
    if provider == "openrouter" and isinstance(extra_body, Mapping) and "models" in extra_body:
        raw_models = extra_body["models"]
        if (
            not isinstance(raw_models, list)
            or not raw_models
            or any(
                not isinstance(model, str) or not model.strip() or model != model.strip()
                for model in raw_models
            )
        ):
            raise ChatBadRequestError(
                provider=provider, message="Fallback models must be a non-empty list of model IDs.",
            )
        fallback_models = list(raw_models)
        selected_models.extend(fallback_models)
    policy = request.get("call_policy")
    discovery_started = time.monotonic()
    base_url, discovery_options = _provider_inventory_endpoint(provider, request)
    result = discover_provider_models(
        provider, request.get("api_key"), base_url=base_url, **discovery_options,
    )
    if result.status != "ready":
        _raise_inventory_failure(provider, result.status, refreshable_oauth=refreshable_oauth)
    for model in selected_models:
        if model in result.models:
            continue
        selection_policy = policy if isinstance(policy, ProviderCallPolicy) else None
        if selection_policy is not None and selection_policy.maximum_timeout_seconds is not None:
            remaining = selection_policy.maximum_timeout_seconds - (time.monotonic() - discovery_started)
            if remaining <= 0:
                _raise_inventory_failure(provider, "unreachable")
            selection_policy = replace(selection_policy, maximum_timeout_seconds=remaining)
        selection_options = {"call_policy": selection_policy} if selection_policy is not None else discovery_options
        selection = resolve_provider_model_selection(
            provider, request.get("api_key"), model,
            base_url=base_url, inventory=result, **selection_options,
        )
        if selection.status == "not_found":
            raise ChatModelNotAvailableError(provider, model)
        if selection.status != "ready":
            _raise_inventory_failure(provider, selection.status, refreshable_oauth=refreshable_oauth)
    # Freeze the resolved endpoint so a config/env rotation between discovery
    # and generation cannot validate one server's models and call another.
    request["base_url"] = base_url
    if fallback_models is not None:
        request["extra_body"] = {**(extra_body or {}), "models": fallback_models}


def _call_policy_after_model_discovery(
    provider: str, request: dict[str, Any], started_at: float,
) -> ProviderCallPolicy | None:
    policy = request.get("call_policy")
    if not isinstance(policy, ProviderCallPolicy) or policy.maximum_timeout_seconds is None:
        return None
    remaining = policy.maximum_timeout_seconds - (time.monotonic() - started_at)
    if remaining <= 0:
        _raise_inventory_failure(provider, "unreachable")
    remaining_policy = replace(policy, maximum_timeout_seconds=remaining)
    request["call_policy"] = remaining_policy
    return remaining_policy


def perform_chat_api_call(**kwargs: Any) -> Any:
    """Adapter-backed replacement for chat_orchestrator.chat_api_call."""
    # LLM usage accounting spec §4b: refuse before the try below, which can rebuild any error as a provider error.
    checked = check_llm_call_sync()
    provider, request, internal = _build_adapter_request_from_chat_args(kwargs)
    registry = _get_llm_registry()
    adapter = registry.get_adapter(provider)
    if adapter is None:
        raise ChatConfigurationError(provider=provider, message="LLM adapter unavailable.")
    _attach_internal_http_hooks(adapter, request, internal)
    timeout = internal.get("timeout")
    timeout_kwargs = {"timeout": timeout} if timeout is not None else {}
    privacy_safe_errors = _privacy_safe_provider_errors_enabled(request)
    # A checked call's adapter method is not checked again; one that could not be checked here is.
    admitted = llm_call_context(prechecked=True) if checked else contextlib.nullcontext()
    try:
        with provider_error_privacy_scope(privacy_safe_errors), admitted:
            call_started = time.monotonic()
            _validate_audited_call_policy_transport(registry, provider, request)
            _validate_provider_model_selection(provider, request)
            if has_model_listing(provider):
                _call_policy_after_model_discovery(provider, request, call_started)
            if request.get("stream"):
                return _map_sync_stream_egress_errors(
                    provider,
                    adapter.stream(request, **timeout_kwargs),
                )
            return adapter.chat(request, **timeout_kwargs)
    except EgressPolicyError as exc:
        raise _map_provider_egress_error(provider, exc) from exc
    except Exception as exc:
        if privacy_safe_errors:
            raise privacy_safe_chat_error(provider, exc) from None
        raise


async def perform_chat_api_call_async(**kwargs: Any) -> Any:
    """Async adapter-backed replacement for chat_orchestrator.chat_api_call_async."""
    # LLM usage accounting spec §4b: refuse before the try below, which can rebuild any error as a provider error.
    await check_llm_call_async()
    provider, request, internal = _build_adapter_request_from_chat_args(kwargs)
    registry = _get_llm_registry()
    adapter = registry.get_adapter(provider)
    if adapter is None:
        raise ChatConfigurationError(provider=provider, message="LLM adapter unavailable.")
    _attach_internal_http_hooks(adapter, request, internal)
    timeout = internal.get("timeout")
    timeout_kwargs = {"timeout": timeout} if timeout is not None else {}
    privacy_safe_errors = _privacy_safe_provider_errors_enabled(request)

    try:
        with provider_error_privacy_scope(privacy_safe_errors), llm_call_context(prechecked=True):
            call_started = time.monotonic()
            strict_policy = _validate_audited_call_policy_transport(
                registry, provider, request
            )
            if has_model_listing(provider):
                await await_bounded_sync_call(
                    partial(_validate_provider_model_selection, provider, request),
                    pool=SYNC_ADAPTER_CALL_POOL,
                    exhaustion_message="Provider model discovery capacity is exhausted",
                )
                strict_policy = _call_policy_after_model_discovery(provider, request, call_started)
            if request.get("stream"):
                try:
                    stream_iter = adapter.astream(request, **timeout_kwargs)
                    if inspect.isawaitable(stream_iter):
                        stream_iter = await stream_iter
                    return _map_async_stream_egress_errors(provider, stream_iter)
                except NotImplementedError:
                    stream_iter = adapter.stream(request, **timeout_kwargs)
                    return _map_async_stream_egress_errors(
                        provider,
                        wrap_sync_stream(stream_iter),
                    )

            if getattr(adapter, "async_chat_is_native", None) is True:
                try:
                    return await _await_with_call_policy_deadline(
                        adapter.achat(request, **timeout_kwargs),
                        strict_policy,
                    )
                except NotImplementedError:
                    pass
            return await _await_with_call_policy_deadline(
                await_bounded_sync_call(
                    partial(adapter.chat, request, **timeout_kwargs),
                    pool=SYNC_ADAPTER_CALL_POOL,
                    exhaustion_message="Provider adapter capacity is exhausted",
                ),
                strict_policy,
            )
    except EgressPolicyError as exc:
        raise _map_provider_egress_error(provider, exc) from exc
    except DaemonCapacityError:
        raise_detached_error(
            sanitized_provider_stream_exception("provider_unavailable")
        )
    except asyncio.CancelledError as provider_cancel:
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            raise
        raise_detached_error(
            sanitized_provider_stream_exception(provider_cancel)
        )
    except Exception as exc:
        if privacy_safe_errors:
            raise privacy_safe_chat_error(provider, exc) from None
        raise
