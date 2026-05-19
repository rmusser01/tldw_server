from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.llm_provider_overrides_repo import (
    AuthnzLLMProviderOverridesRepo,
)
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    decrypt_byok_payload,
    loads_envelope,
    normalize_provider_name,
)


@dataclass(frozen=True)
class LLMProviderOverride:
    provider: str
    is_enabled: bool | None = None
    allowed_models: list[str] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    credential_fields: dict[str, Any] = field(default_factory=dict)
    api_key_hint: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


_OVERRIDE_CACHE: dict[str, LLMProviderOverride] = {}
_OVERRIDE_LOCK = threading.Lock()


def _parse_json_value(raw: Any) -> Any | None:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


def _normalize_models(raw: Any | None) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [v.strip() for v in raw.split(",")]
    if not isinstance(raw, list):
        return None
    cleaned = [str(v).strip() for v in raw if isinstance(v, (str, int, float)) and str(v).strip()]
    return cleaned or None


def _normalize_optional_bool(raw: Any) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if is_truthy(lowered):
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _parse_override_row(row: dict[str, Any]) -> LLMProviderOverride:
    provider = normalize_provider_name(row.get("provider"))
    allowed_models = _normalize_models(_parse_json_value(row.get("allowed_models")))
    config = _parse_json_value(row.get("config_json")) or {}
    if not isinstance(config, dict):
        config = {}

    api_key: str | None = None
    credential_fields: dict[str, Any] = {}
    secret_blob = row.get("secret_blob")
    if secret_blob:
        try:
            payload = decrypt_byok_payload(loads_envelope(secret_blob))
            api_key = payload.get("api_key")
            credential_fields_raw = payload.get("credential_fields")
            if isinstance(credential_fields_raw, dict):
                credential_fields = credential_fields_raw
        except Exception:
            logger.warning("Provider override decrypt failed")

    return LLMProviderOverride(
        provider=provider,
        is_enabled=_normalize_optional_bool(row.get("is_enabled")),
        allowed_models=allowed_models,
        config=config,
        api_key=api_key,
        credential_fields=credential_fields,
        api_key_hint=row.get("api_key_hint"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def get_llm_provider_override(provider: str) -> LLMProviderOverride | None:
    provider_norm = normalize_provider_name(provider)
    with _OVERRIDE_LOCK:
        return _OVERRIDE_CACHE.get(provider_norm)


def get_llm_provider_overrides_snapshot() -> dict[str, LLMProviderOverride]:
    with _OVERRIDE_LOCK:
        return dict(_OVERRIDE_CACHE)


def set_llm_provider_overrides_cache_for_tests(overrides: dict[str, LLMProviderOverride] | None) -> None:
    with _OVERRIDE_LOCK:
        _OVERRIDE_CACHE.clear()
        if overrides:
            _OVERRIDE_CACHE.update(overrides)


def apply_llm_provider_overrides_to_listing(payload: dict[str, Any]) -> dict[str, Any]:
    overrides = get_llm_provider_overrides_snapshot()
    if not overrides:
        return payload

    providers = payload.get("providers", [])
    if not isinstance(providers, list):
        return payload

    updated_providers = []
    for entry in providers:
        if not isinstance(entry, dict):
            updated_providers.append(entry)
            continue
        provider_name = normalize_provider_name(entry.get("name"))
        override = overrides.get(provider_name)
        if not override:
            entry.setdefault("enabled", True)
            updated_providers.append(entry)
            continue

        merged = dict(entry)
        merged["enabled"] = override.is_enabled if override.is_enabled is not None else merged.get("enabled", True)

        models = list(merged.get("models") or [])
        config_models = override.config.get("models")
        if isinstance(config_models, list):
            models = [str(v).strip() for v in config_models if str(v).strip()]
        if override.allowed_models:
            models = [m for m in models if m in override.allowed_models] if models else list(override.allowed_models)

        preferred_order = get_override_model_priority(provider_name, "highest_quality")
        if preferred_order:
            order_index = {name: index for index, name in enumerate(preferred_order)}
            models = sorted(
                models,
                key=lambda model_name: (order_index.get(model_name, len(order_index)), model_name),
            )
        merged["models"] = models

        models_info = merged.get("models_info")
        if isinstance(models_info, list) and (override.allowed_models or isinstance(config_models, list)):
            filtered_info = []
            for mi in models_info:
                if not isinstance(mi, dict):
                    continue
                if mi.get("name") in models:
                    filtered_info.append(mi)
            merged["models_info"] = filtered_info
        if isinstance(merged.get("models_info"), list) and preferred_order:
            order_index = {name: index for index, name in enumerate(preferred_order)}
            models_info_entries = [
                model_info for model_info in merged["models_info"]
                if isinstance(model_info, dict)
            ]
            merged["models_info"] = sorted(
                models_info_entries,
                key=lambda model_info: (
                    order_index.get(str(model_info.get("name") or ""), len(order_index)),
                    str(model_info.get("name") or ""),
                ),
            )

        default_model = override.config.get("default_model")
        if isinstance(default_model, str) and default_model.strip():
            merged["default_model"] = default_model.strip()

        merged["override"] = {
            "is_enabled": override.is_enabled,
            "allowed_models": override.allowed_models,
            "config": override.config,
            "credential_fields": override.credential_fields,
            "has_api_key": bool(override.api_key),
            "api_key_hint": override.api_key_hint,
        }
        updated_providers.append(merged)

    updated = dict(payload)
    updated["providers"] = updated_providers
    return updated


def validate_provider_override(provider: str, model: str | None) -> dict[str, str] | None:
    override = get_llm_provider_override(provider)
    if not override:
        return None
    if override.is_enabled is False:
        return {
            "error_code": "provider_disabled",
            "message": f"Provider '{provider}' is disabled by admin override.",
        }
    if override.allowed_models and model and model not in override.allowed_models:
        return {
            "error_code": "model_not_allowed",
            "message": f"Model '{model}' is not allowed for provider '{provider}'.",
        }
    return None


def get_override_model_priority(provider: str, objective: str = "highest_quality") -> list[str] | None:
    override = get_llm_provider_override(provider)
    if not override:
        return None

    config = override.config if isinstance(override.config, dict) else {}

    routing_config = config.get("routing")
    if isinstance(routing_config, dict):
        model_rankings = routing_config.get("model_rankings")
        if isinstance(model_rankings, dict):
            ranked_models = _normalize_models(model_rankings.get(objective))
            if ranked_models:
                return ranked_models

    model_rankings = config.get("model_rankings")
    if isinstance(model_rankings, dict):
        ranked_models = _normalize_models(model_rankings.get(objective))
        if ranked_models:
            return ranked_models

    return None


def get_override_default_model(provider: str) -> str | None:
    override = get_llm_provider_override(provider)
    if not override:
        return None
    default_model = override.config.get("default_model")
    if isinstance(default_model, str) and default_model.strip():
        return default_model.strip()
    return None


def get_override_credentials(provider: str) -> dict[str, Any] | None:
    override = get_llm_provider_override(provider)
    if not override:
        return None
    if not override.api_key and not override.credential_fields:
        return None
    return {
        "api_key": override.api_key,
        "credential_fields": override.credential_fields,
    }


async def refresh_llm_provider_overrides(pool: DatabasePool | None = None) -> dict[str, LLMProviderOverride]:
    try:
        db_pool = pool or await get_db_pool()
        repo = AuthnzLLMProviderOverridesRepo(db_pool)
        await repo.ensure_tables()
        rows = await repo.list_overrides()
    except Exception:
        logger.warning("Failed to load provider overrides")
        rows = []

    overrides: dict[str, LLMProviderOverride] = {}
    for row in rows:
        try:
            override = _parse_override_row(row)
            overrides[override.provider] = override
        except Exception:
            logger.warning("Failed to parse provider override row")

    with _OVERRIDE_LOCK:
        _OVERRIDE_CACHE.clear()
        _OVERRIDE_CACHE.update(overrides)
    return overrides
