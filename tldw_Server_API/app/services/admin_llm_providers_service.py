from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    LLMProviderOverrideListResponse,
    LLMProviderOverrideRequest,
    LLMProviderOverrideResponse,
    LLMProviderTestRequest,
    LLMProviderTestResponse,
)
from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
    validate_base_url_override,
    validate_credential_fields,
)
from tldw_Server_API.app.core.AuthNZ.byok_testing import test_provider_credentials
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    get_llm_provider_override,
    get_llm_provider_overrides_snapshot,
    refresh_llm_provider_overrides,
)
from tldw_Server_API.app.core.AuthNZ.repos.llm_provider_overrides_repo import (
    AuthnzLLMProviderOverridesRepo,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
    normalize_provider_name,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError


async def get_llm_provider_overrides_repo() -> AuthnzLLMProviderOverridesRepo:
    """Initialize provider overrides repository and ensure schema exists."""
    try:
        pool = await get_db_pool()
        repo = AuthnzLLMProviderOverridesRepo(pool)
        await repo.ensure_tables()
        return repo
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to initialize LLM provider overrides repository")
        raise HTTPException(
            status_code=500,
            detail="Provider overrides infrastructure is not available",
        ) from exc


def normalize_allowed_models(raw: list[str] | None) -> list[str] | None:
    if raw is None:
        return None
    cleaned = [str(v).strip() for v in raw if isinstance(v, (str, int, float)) and str(v).strip()]
    return cleaned or None


def build_override_response(override: Any) -> LLMProviderOverrideResponse:
    return LLMProviderOverrideResponse(
        provider=override.provider,
        is_enabled=override.is_enabled,
        allowed_models=override.allowed_models,
        config=override.config or None,
        credential_fields=override.credential_fields or None,
        has_api_key=bool(override.api_key or override.api_key_hint),
        api_key_hint=override.api_key_hint,
        created_at=override.created_at,
        updated_at=override.updated_at,
    )


def _normalize_credential_fields(
    provider: str,
    fields: dict[str, Any] | None,
) -> dict[str, Any]:
    provider_norm = normalize_provider_name(provider)
    credential_fields = validate_credential_fields(
        provider_norm,
        fields,
        allow_base_url=True,
    )
    if "base_url" in credential_fields:
        credential_fields["base_url"] = validate_base_url_override(
            credential_fields["base_url"]
        )
    return credential_fields


async def list_overrides(
    provider: str | None,
) -> LLMProviderOverrideListResponse:
    await refresh_llm_provider_overrides()
    overrides = get_llm_provider_overrides_snapshot()
    provider_norm = normalize_provider_name(provider) if provider else None

    items: list[LLMProviderOverrideResponse] = []
    for name in sorted(overrides.keys()):
        if provider_norm and name != provider_norm:
            continue
        items.append(build_override_response(overrides[name]))

    return LLMProviderOverrideListResponse(items=items)


async def get_override(provider: str) -> LLMProviderOverrideResponse:
    await refresh_llm_provider_overrides()
    override = get_llm_provider_override(provider)
    if not override:
        raise HTTPException(status_code=404, detail="Provider override not found")
    return build_override_response(override)


async def upsert_override(
    provider: str,
    payload: LLMProviderOverrideRequest,
) -> LLMProviderOverrideResponse:
    provider_norm = normalize_provider_name(provider)

    if (
        payload.is_enabled is None
        and payload.allowed_models is None
        and payload.config is None
        and payload.api_key is None
        and payload.credential_fields is None
        and not payload.clear_api_key
    ):
        raise HTTPException(status_code=400, detail="No override fields supplied")

    repo = await get_llm_provider_overrides_repo()
    existing = await repo.fetch_override(provider_norm)
    is_enabled = existing.get("is_enabled") if existing else None
    allowed_models_json = existing.get("allowed_models") if existing else None
    config_json = existing.get("config_json") if existing else None
    secret_blob = existing.get("secret_blob") if existing else None
    api_key_hint = existing.get("api_key_hint") if existing else None

    if payload.is_enabled is not None:
        is_enabled = payload.is_enabled

    if payload.allowed_models is not None:
        normalized_models = normalize_allowed_models(payload.allowed_models)
        allowed_models_json = json.dumps(normalized_models) if normalized_models else None

    if payload.config is not None:
        if not isinstance(payload.config, dict):
            raise HTTPException(status_code=400, detail="config must be an object")
        config_json = json.dumps(payload.config) if payload.config else None

    credential_fields: dict[str, Any] | None = None
    if payload.credential_fields is not None:
        try:
            credential_fields = _normalize_credential_fields(provider_norm, payload.credential_fields)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid provider credential fields") from exc

    if payload.clear_api_key:
        secret_blob = None
        api_key_hint = None

    if payload.api_key is not None:
        api_key = payload.api_key.strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="api_key cannot be empty")

        if credential_fields is None and secret_blob:
            try:
                payload_existing = decrypt_byok_payload(loads_envelope(secret_blob))
                existing_fields = payload_existing.get("credential_fields")
                if isinstance(existing_fields, dict):
                    credential_fields = existing_fields
            except Exception:
                credential_fields = credential_fields or None

        secret_payload = build_secret_payload(api_key, credential_fields or None)
        try:
            envelope = encrypt_byok_payload(secret_payload)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="BYOK encryption is not configured") from exc
        secret_blob = dumps_envelope(envelope)
        api_key_hint = key_hint_for_api_key(api_key)
    elif credential_fields is not None:
        if not secret_blob:
            raise HTTPException(status_code=400, detail="credential_fields require an existing api_key")
        try:
            payload_existing = decrypt_byok_payload(loads_envelope(secret_blob))
            existing_key = payload_existing.get("api_key")
            if not existing_key:
                raise ValueError("Existing api_key is missing")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Failed to load existing api_key") from exc
        secret_payload = build_secret_payload(existing_key, credential_fields or None)
        try:
            envelope = encrypt_byok_payload(secret_payload)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="BYOK encryption is not configured") from exc
        secret_blob = dumps_envelope(envelope)
        api_key_hint = key_hint_for_api_key(existing_key)

    now = datetime.now(timezone.utc)
    try:
        await repo.upsert_override(
            provider=provider_norm,
            is_enabled=is_enabled,
            allowed_models=allowed_models_json,
            config_json=config_json,
            secret_blob=secret_blob,
            api_key_hint=api_key_hint,
            updated_at=now,
        )
    except Exception as exc:
        logger.error("Failed to store provider override")
        raise HTTPException(status_code=500, detail="Failed to store provider override") from exc

    await refresh_llm_provider_overrides()
    override = get_llm_provider_override(provider_norm)
    if not override:
        raise HTTPException(status_code=500, detail="Failed to load provider override")
    return build_override_response(override)


async def delete_override(provider: str) -> None:
    repo = await get_llm_provider_overrides_repo()
    provider_norm = normalize_provider_name(provider)
    try:
        deleted = await repo.delete_override(provider_norm)
    except Exception as exc:
        logger.error("Failed to delete provider override")
        raise HTTPException(status_code=500, detail="Failed to delete provider override") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Provider override not found")
    await refresh_llm_provider_overrides()


async def test_provider(
    payload: LLMProviderTestRequest,
) -> LLMProviderTestResponse:
    provider_norm = normalize_provider_name(payload.provider)
    await refresh_llm_provider_overrides()

    api_key = (payload.api_key or "").strip()
    credential_fields = payload.credential_fields
    model = payload.model

    if payload.use_override and (not api_key or credential_fields is None or model is None):
        override = get_llm_provider_override(provider_norm)
        if override:
            if not api_key:
                api_key = override.api_key or api_key
            if credential_fields is None:
                credential_fields = override.credential_fields or None
            if model is None:
                model = override.config.get("default_model")

    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    if credential_fields is not None:
        try:
            credential_fields = _normalize_credential_fields(provider_norm, credential_fields)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid provider credential fields") from exc

    try:
        model_used = await test_provider_credentials(
            provider=provider_norm,
            api_key=api_key,
            credential_fields=credential_fields,
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Provider credential validation failed") from exc
    except ChatAPIError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Provider test call failed") from exc

    return LLMProviderTestResponse(
        provider=provider_norm,
        status="valid",
        model=model_used,
    )
