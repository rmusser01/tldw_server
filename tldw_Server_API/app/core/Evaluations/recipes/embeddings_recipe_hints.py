"""Candidate hints and apply previews for the embeddings model selection recipe."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus

RECIPE_ID = "embeddings_model_selection"
APPLY_AUDIT_METADATA_KEY = "embedding_recipe_apply_audit"
ConfigUpdater = Callable[..., object]
_EMBEDDING_POLICY_HELPER_MODULE = (
    "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced"
)
_EMBEDDING_CONFIG_OVERRIDE_ENV_KEYS = (
    "EMBEDDINGS_DEFAULT_PROVIDER",
    "EMBEDDINGS_PROVIDER",
    "EMBEDDINGS_DEFAULT_MODEL",
    "EMBEDDINGS_MODEL",
)
_LOCALISH_PROVIDERS = {
    "local",
    "local_api",
    "huggingface",
    "onnx",
    "llamacpp",
    "sentence-transformers",
}
_REMOTE_PROVIDERS_REQUIRING_KEYS = {
    "openai",
    "cohere",
    "voyage",
    "jina",
    "mistral",
    "google",
    "azure",
}
_PROVIDER_ENV_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "cohere": ("COHERE_API_KEY",),
    "voyage": ("VOYAGE_API_KEY", "VOYAGEAI_API_KEY"),
    "jina": ("JINA_API_KEY", "JINAAI_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "azure": ("AZURE_OPENAI_API_KEY",),
}


def get_simplified_embeddings_config() -> dict[str, Any]:
    """Return the embeddings simplified config in a stable, secret-free shape."""
    from tldw_Server_API.app.core.Embeddings.simplified_config import get_config

    config = get_config()
    if hasattr(config, "to_dict"):
        raw = config.to_dict()
    elif isinstance(config, dict):
        raw = dict(config)
    elif is_dataclass(config):
        raw = asdict(config)
    else:
        raw = {
            "default_provider": getattr(config, "default_provider", None),
            "default_model": getattr(config, "default_model", None),
            "providers": getattr(config, "providers", []),
        }

    providers = []
    for provider in raw.get("providers") or []:
        provider_data = _object_to_dict(provider)
        api_key = provider_data.get("api_key")
        providers.append(
            {
                "name": str(provider_data.get("name") or "").strip(),
                "enabled": bool(provider_data.get("enabled", True)),
                "models": [
                    str(model).strip()
                    for model in provider_data.get("models") or []
                    if str(model).strip()
                ],
                "api_key_configured": bool(api_key),
                "api_url": provider_data.get("api_url"),
                "priority": provider_data.get("priority"),
            }
        )

    return {
        "default_provider": str(raw.get("default_provider") or "").strip(),
        "default_model": str(raw.get("default_model") or "").strip(),
        "providers": providers,
    }


def get_allowed_embedding_providers() -> list[str] | None:
    """Return the configured embedding provider allowlist from the production API policy."""
    helper = _load_embedding_policy_helper("_get_allowed_providers")
    if helper is None:
        return None
    return helper()


def get_allowed_embedding_models() -> list[str] | None:
    """Return the configured embedding model allowlist from the production API policy."""
    helper = _load_embedding_policy_helper("_get_allowed_models")
    if helper is None:
        return None
    return helper()


def should_enforce_embedding_policy(user: object | None = None) -> bool:
    """Return whether embedding allowlists should be enforced for this request."""
    helper = _load_embedding_policy_helper("_should_enforce_policy")
    if helper is None:
        return False
    return bool(helper(user))


def _load_embedding_policy_helper(helper_name: str) -> Callable[..., Any] | None:
    try:
        module = import_module(_EMBEDDING_POLICY_HELPER_MODULE)
    except (ImportError, RuntimeError) as exc:
        logger.warning(
            "Embedding policy helper {} unavailable: {}",
            helper_name,
            type(exc).__name__,
        )
        return None

    helper = getattr(module, helper_name, None)
    if not callable(helper):
        logger.warning("Embedding policy helper {} unavailable: missing callable", helper_name)
        return None
    return helper


def get_current_embedding_config() -> dict[str, str | None]:
    """Return the current default embedding provider/model without secrets."""
    config = get_simplified_embeddings_config()
    return {
        "provider": _clean_optional(config.get("default_provider")),
        "model": _clean_optional(config.get("default_model")),
    }


def build_embedding_recipe_candidate_hints(*, user: object | None) -> dict[str, object]:
    """Build candidate readiness hints from the current embeddings configuration."""
    config = get_simplified_embeddings_config()
    current_provider = str(config.get("default_provider") or "").strip()
    current_model = str(config.get("default_model") or "").strip()
    current = None
    if current_provider or current_model:
        current = _candidate_from_provider_model(
            current_provider,
            current_model,
            default=True,
            config=config,
            user=user,
        )

    candidates = _collect_configured_candidates(config, user=user)
    return {
        "recipe_id": RECIPE_ID,
        "current": current,
        "candidates": _dedupe_with_current_first(current, candidates),
        "warnings": [],
    }


def build_embedding_recipe_apply_preview(
    service: object,
    run_id: str,
    slot_name: str,
    candidate_run_id: str | None = None,
    live_apply_supported: bool = False,
) -> dict[str, object]:
    """Return a secret-free copy-config preview for an embeddings recipe recommendation."""
    report = service.get_report(run_id)
    normalized = _normalize_report(report)
    run = normalized["run"]
    recipe_id = str(run.get("recipe_id") or "")
    response = _preview_response_base(
        run_id=str(run.get("run_id") or run_id),
        recipe_id=recipe_id,
        slot_name=slot_name,
        candidate_run_id=candidate_run_id,
    )

    if recipe_id != RECIPE_ID:
        response["blocked_reason"] = (
            f"Recipe run is '{recipe_id or 'unknown'}', not '{RECIPE_ID}'."
        )
        return response

    status_value = _status_value(run.get("status"))
    if status_value != RunStatus.COMPLETED.value:
        response["blocked_reason"] = "Recipe run must be completed before its recommendation can be previewed."
        return response

    slot = normalized["recommendation_slots"].get(slot_name)
    if slot is None:
        response["blocked_reason"] = f"Recommendation slot '{slot_name}' was not found."
        return response

    slot_candidate_run_id = _clean_optional(slot.get("candidate_run_id"))
    response["candidate_run_id"] = slot_candidate_run_id
    if candidate_run_id and slot_candidate_run_id != candidate_run_id:
        response["blocked_reason"] = (
            f"Requested candidate_run_id '{candidate_run_id}' does not match slot candidate_run_id "
            f"'{slot_candidate_run_id or 'none'}'."
        )
        return response
    if not slot_candidate_run_id:
        response["blocked_reason"] = "Recommendation slot metadata is missing candidate_run_id."
        return response

    metadata = _object_to_dict(slot.get("metadata") or {})
    provider = _clean_optional(metadata.get("provider"))
    model = _clean_optional(metadata.get("model"))
    if not provider or not model:
        missing = "provider" if not provider else "model"
        response["blocked_reason"] = f"Recommendation slot metadata is missing {missing}."
        return response

    warnings = [str(item) for item in metadata.get("apply_warnings") or []]
    metadata_eligible = metadata.get("apply_eligible", True)
    apply_eligible = bool(metadata_eligible) and not any(
        warning in {"missing_candidate_run_id", "missing_provider", "missing_model"}
        for warning in warnings
    )
    if not apply_eligible:
        response["blocked_reason"] = "Recommendation slot is not eligible for apply preview."

    apply_available = bool(live_apply_supported and apply_eligible)
    override_keys = get_embedding_config_override_env_keys()
    if apply_available and override_keys:
        apply_available = False
        warnings.append(
            "Live apply is unavailable because environment variable(s) override "
            f"the effective embeddings provider/model: {', '.join(override_keys)}."
        )

    response.update(
        {
            "apply_eligible": apply_eligible,
            "apply_available": apply_available,
            "warnings": warnings,
            "current": get_current_embedding_config(),
            "proposed": {"provider": provider, "model": model},
            "affected_config": {
                "section": "Embeddings",
                "provider_key": "embedding_provider",
                "model_key": "embedding_model",
            },
            "copy_config": {
                "Embeddings": {
                    "embedding_provider": provider,
                    "embedding_model": model,
                }
            },
            "reindex_required": True,
        }
    )
    return response


def apply_embedding_recipe_recommendation(
    service: object,
    *,
    run_id: str,
    slot_name: str,
    candidate_run_id: str | None,
    confirmed_provider: str,
    confirmed_model: str,
    config_updater: ConfigUpdater,
) -> dict[str, object]:
    """Apply a completed embeddings recipe recommendation to the RAG embeddings config."""
    preview = build_embedding_recipe_apply_preview(
        service,
        run_id=run_id,
        slot_name=slot_name,
        candidate_run_id=candidate_run_id,
        live_apply_supported=True,
    )
    if preview.get("blocked_reason"):
        raise ValueError(str(preview["blocked_reason"]))
    if not preview.get("apply_eligible"):
        raise ValueError("Recommendation slot is not eligible for live apply.")

    proposed = _object_to_dict(preview.get("proposed") or {})
    proposed_provider = _clean_optional(proposed.get("provider"))
    proposed_model = _clean_optional(proposed.get("model"))
    confirmed_provider_clean = _clean_optional(confirmed_provider)
    confirmed_model_clean = _clean_optional(confirmed_model)
    if not confirmed_provider_clean or not confirmed_model_clean:
        raise ValueError("confirmed_provider and confirmed_model are required.")
    if (
        confirmed_provider_clean != proposed_provider
        or confirmed_model_clean != proposed_model
    ):
        raise ValueError("Confirmed provider/model do not match the recommendation preview.")

    override_keys = get_embedding_config_override_env_keys()
    if override_keys:
        raise ValueError(
            "Cannot apply recommendation because environment variable(s) override "
            f"the effective embeddings provider/model: {', '.join(override_keys)}."
        )

    updates = {
        "Embeddings": {
            "embedding_provider": proposed_provider,
            "embedding_model": proposed_model,
        }
    }
    audit_ref = APPLY_AUDIT_METADATA_KEY
    applied_at = datetime.now(UTC).isoformat()
    audit = {
        "timestamp": applied_at,
        "run_id": str(preview.get("run_id") or run_id),
        "slot": slot_name,
        "candidate_run_id": preview.get("candidate_run_id"),
        "previous": preview.get("current") or {},
        "current": preview.get("current") or {},
        "proposed": proposed,
        "new": None,
        "backup_path": None,
        "status": "pending",
    }
    _persist_embedding_apply_audit(service, run_id, audit)

    try:
        backup_path = config_updater(updates, create_backup=True)
    except Exception:
        failed_audit = {
            **audit,
            "status": "failed",
            "failed_at": datetime.now(UTC).isoformat(),
            "failure": "config_update_failed",
        }
        _persist_embedding_apply_audit(service, run_id, failed_audit)
        raise

    backup_path_str = str(backup_path) if backup_path is not None else None
    final_audit = {
        **audit,
        "new": proposed,
        "backup_path": backup_path_str,
        "status": "applied",
        "applied_at": datetime.now(UTC).isoformat(),
    }
    _persist_embedding_apply_audit(service, run_id, final_audit)

    logger.info(
        "Applied embeddings recipe recommendation run_id={} slot={} candidate_run_id={}",
        str(preview.get("run_id") or run_id),
        slot_name,
        preview.get("candidate_run_id"),
    )
    preview.update(
        {
            "applied": True,
            "backup_path": backup_path_str,
            "audit_ref": audit_ref,
        }
    )
    return preview


def _persist_embedding_apply_audit(
    service: object,
    run_id: str,
    audit: dict[str, object],
) -> None:
    record = service.get_run(run_id)
    metadata = dict(getattr(record, "metadata", {}) or {})
    metadata[APPLY_AUDIT_METADATA_KEY] = audit
    updated = getattr(service, "db").update_recipe_run(run_id, metadata=metadata)
    if not updated:
        raise RuntimeError("Failed to record embeddings recommendation apply audit metadata.")


def get_embedding_config_override_env_keys() -> list[str]:
    """Return env keys that override effective embeddings provider/model defaults."""
    return [key for key in _EMBEDDING_CONFIG_OVERRIDE_ENV_KEYS if os.getenv(key)]


def _candidate_from_provider_model(
    provider: str,
    model: str,
    *,
    default: bool,
    config: dict[str, Any],
    user: object | None,
) -> dict[str, object]:
    provider_name = provider.strip()
    model_name = model.strip()
    provider_config = _find_provider_config(config, provider_name)
    status, reason = _classify_candidate(
        provider_name,
        model_name,
        provider_config=provider_config,
        user=user,
    )
    return {
        "provider": provider_name,
        "model": model_name,
        "is_local": _is_localish_provider(provider_name),
        "default": default,
        "status": status,
        "status_reason": reason,
        "dimensions": _positive_int(provider_config.get("dimensions") if provider_config else None),
        "revision": _clean_optional(provider_config.get("revision") if provider_config else None),
        "cost_hint": _clean_optional(provider_config.get("cost_hint") if provider_config else None),
    }


def _collect_configured_candidates(config: dict[str, Any], *, user: object | None) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for provider_config in config.get("providers") or []:
        provider_data = _object_to_dict(provider_config)
        provider = str(provider_data.get("name") or "").strip()
        if not provider:
            continue
        for model in provider_data.get("models") or []:
            model_name = str(model).strip()
            if not model_name:
                continue
            candidates.append(
                _candidate_from_provider_model(
                    provider,
                    model_name,
                    default=False,
                    config=config,
                    user=user,
                )
            )
    return candidates


def _classify_candidate(
    provider: str,
    model: str,
    *,
    provider_config: dict[str, Any] | None,
    user: object | None,
) -> tuple[str, str | None]:
    if not provider or not model:
        return "unknown", "Provider and model are required."

    policy_enforced = should_enforce_embedding_policy(user)
    allowed_providers = get_allowed_embedding_providers()
    if policy_enforced and allowed_providers is not None and provider.lower() not in {
        str(item).lower() for item in allowed_providers
    }:
        return "disallowed_provider", f"Provider '{provider}' is not allowed by embedding policy."

    allowed_models = get_allowed_embedding_models()
    if policy_enforced and allowed_models is not None and not _model_allowed(model, allowed_models):
        return "disallowed_model", f"Model '{model}' is not allowed by embedding policy."

    if _remote_provider_requires_key(provider) and not _provider_has_key(provider, provider_config):
        return "missing_key", f"Provider '{provider}' requires an API key."

    return "ready", None


def _dedupe_with_current_first(
    current: dict[str, object] | None,
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    if current is None:
        return candidates

    current_key = (
        str(current.get("provider") or "").lower(),
        str(current.get("model") or ""),
    )
    matching_candidate = next(
        (
            candidate
            for candidate in candidates
            if (
                str(candidate.get("provider") or "").lower(),
                str(candidate.get("model") or ""),
            )
            == current_key
        ),
        None,
    )
    current_candidate = matching_candidate or current
    current_candidate["default"] = True
    ordered: list[dict[str, object]] = [current_candidate]
    seen: set[tuple[str, str]] = set()
    for candidate in ordered + candidates:
        if candidate is None:
            continue
        key = (
            str(candidate.get("provider") or "").lower(),
            str(candidate.get("model") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        if candidate is not current_candidate:
            ordered.append(candidate)
    return ordered


def _normalize_report(report: object) -> dict[str, Any]:
    if hasattr(report, "model_dump"):
        payload = report.model_dump(mode="python")
    else:
        payload = _object_to_dict(report)
    run = _object_to_dict(payload.get("run") or {})
    slots = {
        str(slot_name): _object_to_dict(slot_value)
        for slot_name, slot_value in (payload.get("recommendation_slots") or {}).items()
    }
    return {"run": run, "recommendation_slots": slots}


def _preview_response_base(
    *,
    run_id: str,
    recipe_id: str,
    slot_name: str,
    candidate_run_id: str | None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "recipe_id": recipe_id,
        "slot_name": slot_name,
        "candidate_run_id": candidate_run_id,
        "apply_eligible": False,
        "apply_available": False,
        "blocked_reason": None,
        "warnings": [],
        "current": {},
        "proposed": {},
        "affected_config": {},
        "copy_config": {},
        "reindex_required": True,
    }


def _find_provider_config(config: dict[str, Any], provider: str) -> dict[str, Any] | None:
    provider_lower = provider.lower()
    for provider_config in config.get("providers") or []:
        provider_data = _object_to_dict(provider_config)
        if str(provider_data.get("name") or "").lower() == provider_lower:
            return provider_data
    return None


def _provider_has_key(provider: str, provider_config: dict[str, Any] | None) -> bool:
    if provider_config:
        if provider_config.get("api_key_configured"):
            return True
        if provider_config.get("api_key"):
            return True
    return any(os.getenv(env_name) for env_name in _env_names_for_provider(provider))


def _env_names_for_provider(provider: str) -> tuple[str, ...]:
    normalized = provider.lower().replace("-", "_")
    return _PROVIDER_ENV_KEYS.get(normalized, (f"{normalized.upper()}_API_KEY",))


def _remote_provider_requires_key(provider: str) -> bool:
    provider_lower = provider.lower()
    return provider_lower in _REMOTE_PROVIDERS_REQUIRING_KEYS and not _is_localish_provider(provider)


def _is_localish_provider(provider: str) -> bool:
    provider_lower = provider.lower()
    return provider_lower in _LOCALISH_PROVIDERS


def _model_allowed(model: str, allowed_models: list[str]) -> bool:
    for pattern in allowed_models:
        normalized_pattern = str(pattern)
        if normalized_pattern.endswith("*") and model.startswith(normalized_pattern[:-1]):
            return True
        if model == normalized_pattern:
            return True
    return False


def _object_to_dict(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if is_dataclass(value):
        return asdict(value)
    data = getattr(value, "__dict__", None)
    return dict(data) if isinstance(data, dict) else {}


def _status_value(value: object) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def _clean_optional(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
