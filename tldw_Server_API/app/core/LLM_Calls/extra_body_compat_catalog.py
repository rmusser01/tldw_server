from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_provider_number,
    custom_openai_section_name,
)

_TRUE_SET = {"1", "true", "yes", "on"}

_MIKUPAD_KNOWN_PARAMS = [
    "dynatemp_range",
    "dynatemp_exponent",
    "repeat_penalty",
    "repeat_last_n",
    "penalize_nl",
    "ignore_eos",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "typical_p",
    "min_p",
    "tfs_z",
    "xtc_threshold",
    "xtc_probability",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "dry_sequence_breakers",
    "banned_tokens",
    "grammar",
    "logit_bias",
]

_PARAM_GROUPS = ["sampling", "penalties", "constraints"]

_DEFAULT = {
    "supported": False,
    "effective_reason": "unsupported for deployment/runtime configuration",
    "known_params": [],
    "param_groups": [],
    "notes": "No extra_body compatibility metadata registered for this provider/model.",
    "example": {"extra_body": {}},
    "source": "catalog+runtime",
}

_OPENAI_COMPAT_BASE = {
    "supported": True,
    "effective_reason": "supported in current deployment",
    "known_params": _MIKUPAD_KNOWN_PARAMS,
    "param_groups": _PARAM_GROUPS,
    "notes": "Provider-specific advanced params should be sent under extra_body.",
    "example": {"extra_body": {"mirostat": 2, "mirostat_tau": 5, "mirostat_eta": 0.1}},
    "source": "catalog+runtime",
}

_LLAMA_COMPAT_BASE = {
    **_OPENAI_COMPAT_BASE,
    "known_params": [*_MIKUPAD_KNOWN_PARAMS, "post_sampling_probs"],
}

_CATALOG: dict[str, dict[str, Any]] = {
    "openai": {"provider_default": _OPENAI_COMPAT_BASE, "models": {}},
    "custom_openai_api": {"provider_default": _OPENAI_COMPAT_BASE, "models": {}},
    "custom_openai_api_2": {"provider_default": _OPENAI_COMPAT_BASE, "models": {}},
    "llama": {"provider_default": _LLAMA_COMPAT_BASE, "models": {}},
    "kobold": {"provider_default": _OPENAI_COMPAT_BASE, "models": {}},
    "ooba": {"provider_default": _OPENAI_COMPAT_BASE, "models": {}},
    "tabby": {"provider_default": _OPENAI_COMPAT_BASE, "models": {}},
    "vllm": {"provider_default": _OPENAI_COMPAT_BASE, "models": {}},
    "aphrodite": {"provider_default": _OPENAI_COMPAT_BASE, "models": {}},
    "ollama": {"provider_default": _OPENAI_COMPAT_BASE, "models": {}},
}

_PROVIDER_ALIASES: dict[str, str] = {
    "custom-openai-api": "custom_openai_api",
    "custom_openai_api": "custom_openai_api",
    "custom-openai-api-2": "custom_openai_api_2",
    "custom_openai_api_2": "custom_openai_api_2",
    "custom_openai_api2": "custom_openai_api_2",
    "llama.cpp": "llama",
    "llama_cpp": "llama",
    "llamacpp": "llama",
    "koboldcpp": "kobold",
    "oobabooga": "ooba",
    "tabbyapi": "tabby",
}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_SET
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _normalize_provider(provider: str | None) -> str:
    raw = str(provider or "").strip().lower()
    if not raw:
        return ""
    custom_number = custom_openai_provider_number(raw)
    if custom_number is not None:
        return custom_openai_section_name(custom_number)
    compact = raw.replace(" ", "")
    normalized = compact.replace("-", "_")
    return _PROVIDER_ALIASES.get(compact) or _PROVIDER_ALIASES.get(normalized) or normalized


def _normalize_model(model: str | None) -> str:
    return str(model or "").strip()


def _default_runtime_context() -> dict[str, Any]:
    return {
        "strict_openai_compat": _coerce_bool(os.getenv("LOCAL_LLM_STRICT_OPENAI_COMPAT", False)),
        "thinking_budget_request_key": (os.getenv("LLAMA_CPP_THINKING_BUDGET_PARAM") or "").strip() or None,
    }


def _resolved_runtime_context(runtime_context: dict[str, Any] | None) -> dict[str, Any]:
    merged = _default_runtime_context()
    if isinstance(runtime_context, dict):
        merged.update(runtime_context)
    return merged


def _apply_runtime_overrides(payload: dict[str, Any], runtime_context: dict[str, Any] | None) -> dict[str, Any]:
    out = deepcopy(payload)
    context = _resolved_runtime_context(runtime_context)
    if _coerce_bool(context.get("strict_openai_compat")) and bool(out.get("supported")):
        out["supported"] = False
        out["effective_reason"] = "disabled by strict_openai_compat runtime setting"
    return out


def _fallback_payload() -> dict[str, Any]:
    return deepcopy(_DEFAULT)


def _provider_payload(provider: str) -> dict[str, Any]:
    entry = _CATALOG.get(provider)
    if entry is None and custom_openai_provider_number(provider) is not None:
        entry = _CATALOG.get("custom_openai_api")
    if not entry:
        return _fallback_payload()
    provider_default = entry.get("provider_default")
    if not isinstance(provider_default, dict):
        return _fallback_payload()
    return deepcopy(provider_default)


def get_provider_extra_body_compat(provider: str, *, runtime_context: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized_provider = _normalize_provider(provider)
    payload = _provider_payload(normalized_provider)
    return _apply_runtime_overrides(payload, runtime_context)


def get_model_extra_body_compat(
    provider: str,
    model: str,
    *,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_provider = _normalize_provider(provider)
    normalized_model = _normalize_model(model)
    provider_entry = _CATALOG.get(normalized_provider) or {}
    model_map = provider_entry.get("models")
    payload: dict[str, Any] | None = None
    if isinstance(model_map, dict) and normalized_model:
        raw = model_map.get(normalized_model)
        if isinstance(raw, dict):
            payload = deepcopy(raw)
    if payload is None:
        payload = _provider_payload(normalized_provider)
    return _apply_runtime_overrides(payload, runtime_context)


def list_known_extra_body_params(provider: str, model: str | None = None) -> list[str]:
    if model and str(model).strip():
        payload = get_model_extra_body_compat(provider, str(model))
    else:
        payload = get_provider_extra_body_compat(provider)
    params = payload.get("known_params")
    if isinstance(params, list):
        return [str(item) for item in params if isinstance(item, str)]
    return []
