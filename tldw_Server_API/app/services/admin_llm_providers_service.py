from __future__ import annotations

import asyncio
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
    load_server_config_snapshot,
    validate_base_url_override,
    validate_credential_fields,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ServerFallbackCredentials,
    resolve_byok_credentials,
    resolve_static_server_fallback_from_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.byok_testing import (
    provider_validation_public_error,
    test_provider_credentials,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    LLMProviderOverridesRefreshError,
    ProviderOverrideCallSnapshot,
    capture_provider_override_call_snapshot,
    get_llm_provider_override,
    get_llm_provider_overrides_snapshot,
    get_override_default_model,
    get_override_server_fallback,  # noqa: F401 - compatibility patch seam
    refresh_llm_provider_overrides,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    configured_provider_model_from_snapshot,
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
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import (
    canonical_builtin_llm_provider_name,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import provider_auth_is_resolved
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key


async def _refresh_overrides_or_503(*, force: bool = False) -> None:
    """Refresh admin-visible overrides or surface a sanitized availability error."""
    refresh_failed = False
    try:
        await refresh_llm_provider_overrides(force=force)
    except asyncio.CancelledError:
        raise
    except LLMProviderOverridesRefreshError:
        refresh_failed = True
    except Exception:
        refresh_failed = True
    if refresh_failed:
        raise HTTPException(
            status_code=503,
            detail="Provider credential storage is temporarily unavailable",
        ) from None


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
        del exc
        raise_detached_error(
            HTTPException(
                status_code=500,
                detail="Provider overrides infrastructure is not available",
            )
        )


def normalize_allowed_models(raw: list[str] | None) -> list[str] | None:
    if raw is None:
        return None
    cleaned = [str(v).strip() for v in raw if isinstance(v, (str, int, float)) and str(v).strip()]
    return cleaned or None


def _canonical_override_provider_or_400(provider: str) -> str:
    """Validate one admin override identity against built-in LLM adapters."""
    try:
        return canonical_builtin_llm_provider_name(provider)
    except ValueError as exc:
        del exc
        raise_detached_error(
            HTTPException(status_code=400, detail="Unsupported LLM provider")
        )


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
    provider_norm = canonical_provider_name(provider)
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


def _override_test_model(
    provider: str,
    *,
    override_snapshot: ProviderOverrideCallSnapshot | None = None,
) -> str | None:
    """Return the configured override model in admin-test precedence order."""
    if override_snapshot is not None:
        return override_snapshot.test_model()
    default_model = get_override_default_model(provider)
    if default_model:
        return default_model
    override = get_llm_provider_override(provider)
    if override and override.allowed_models:
        first_model = override.allowed_models[0]
        if isinstance(first_model, str) and first_model.strip():
            return first_model.strip()
    return None


async def list_overrides(
    provider: str | None,
) -> LLMProviderOverrideListResponse:
    provider_norm = (
        _canonical_override_provider_or_400(provider)
        if provider is not None
        else None
    )
    await _refresh_overrides_or_503()
    overrides = get_llm_provider_overrides_snapshot()

    items: list[LLMProviderOverrideResponse] = []
    for name in sorted(overrides.keys()):
        if provider_norm and name != provider_norm:
            continue
        items.append(build_override_response(overrides[name]))

    return LLMProviderOverrideListResponse(items=items)


async def get_override(provider: str) -> LLMProviderOverrideResponse:
    provider_norm = _canonical_override_provider_or_400(provider)
    await _refresh_overrides_or_503()
    override = get_llm_provider_override(provider_norm)
    if not override:
        raise HTTPException(status_code=404, detail="Provider override not found")
    return build_override_response(override)


async def upsert_override(
    provider: str,
    payload: LLMProviderOverrideRequest,
) -> LLMProviderOverrideResponse:
    provider_norm = _canonical_override_provider_or_400(provider)

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
    requested_credential_fields: dict[str, Any] | None = None
    if payload.credential_fields is not None:
        try:
            requested_credential_fields = _normalize_credential_fields(
                provider_norm,
                payload.credential_fields,
            )
        except ValueError as exc:
            del exc
            raise_detached_error(
                HTTPException(status_code=400, detail="Invalid provider credential fields")
            )

    api_key: str | None = None
    if payload.api_key is not None:
        api_key = payload.api_key.strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="api_key cannot be empty")

    for _attempt in range(3):
        try:
            existing = await repo.fetch_override(provider_norm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Failed to load provider override")
            del exc
            raise_detached_error(
                HTTPException(
                    status_code=500,
                    detail="Failed to load provider override",
                )
            )
        secret_blob = existing.get("secret_blob") if existing else None
        credential_fields = (
            dict(requested_credential_fields)
            if requested_credential_fields is not None
            else None
        )
        patch_fields: dict[str, Any] = {}
        compare_secret_blob = False
        expected_secret_blob: str | None = None

        if payload.is_enabled is not None:
            patch_fields["is_enabled"] = payload.is_enabled

        if payload.allowed_models is not None:
            normalized_models = normalize_allowed_models(payload.allowed_models)
            patch_fields["allowed_models"] = (
                json.dumps(normalized_models) if normalized_models else None
            )

        if payload.config is not None:
            if not isinstance(payload.config, dict):
                raise HTTPException(status_code=400, detail="config must be an object")
            patch_fields["config_json"] = (
                json.dumps(payload.config) if payload.config else None
            )

        if payload.clear_api_key:
            secret_blob = None
            patch_fields.update(secret_blob=None, api_key_hint=None)

        if api_key is not None:
            if credential_fields is None and secret_blob:
                compare_secret_blob = True
                expected_secret_blob = secret_blob
                try:
                    payload_existing = decrypt_byok_payload(loads_envelope(secret_blob))
                    existing_fields = payload_existing.get("credential_fields")
                    if isinstance(existing_fields, dict):
                        credential_fields = existing_fields
                except Exception:
                    credential_fields = None

            secret_payload = build_secret_payload(api_key, credential_fields or None)
            try:
                envelope = encrypt_byok_payload(secret_payload)
            except ValueError as exc:
                del exc
                raise_detached_error(
                    HTTPException(
                        status_code=500,
                        detail="BYOK encryption is not configured",
                    )
                )
            secret_blob = dumps_envelope(envelope)
            patch_fields.update(
                secret_blob=secret_blob,
                api_key_hint=key_hint_for_api_key(api_key),
            )
        elif credential_fields is not None:
            if not secret_blob:
                raise HTTPException(
                    status_code=400,
                    detail="credential_fields require an existing api_key",
                )
            compare_secret_blob = True
            expected_secret_blob = secret_blob
            try:
                payload_existing = decrypt_byok_payload(loads_envelope(secret_blob))
                existing_key = payload_existing.get("api_key")
                if not existing_key:
                    raise ValueError("Existing api_key is missing")
            except Exception as exc:
                del exc
                raise_detached_error(
                    HTTPException(
                        status_code=400,
                        detail="Failed to load existing api_key",
                    )
                )
            secret_payload = build_secret_payload(existing_key, credential_fields or None)
            try:
                envelope = encrypt_byok_payload(secret_payload)
            except ValueError as exc:
                del exc
                raise_detached_error(
                    HTTPException(
                        status_code=500,
                        detail="BYOK encryption is not configured",
                    )
                )
            secret_blob = dumps_envelope(envelope)
            patch_fields.update(
                secret_blob=secret_blob,
                api_key_hint=key_hint_for_api_key(existing_key),
            )

        try:
            stored = await repo.patch_override(
                provider=provider_norm,
                fields=patch_fields,
                updated_at=datetime.now(timezone.utc),
                compare_secret_blob=compare_secret_blob,
                expected_secret_blob=expected_secret_blob,
            )
        except Exception as exc:
            logger.error("Failed to store provider override")
            del exc
            raise_detached_error(
                HTTPException(
                    status_code=500,
                    detail="Failed to store provider override",
                )
            )
        if stored is not None:
            break
    else:
        raise HTTPException(
            status_code=409,
            detail="Provider override changed concurrently; retry request",
        )

    await _refresh_overrides_or_503(force=True)
    override = get_llm_provider_override(provider_norm)
    if not override:
        raise HTTPException(status_code=500, detail="Failed to load provider override")
    return build_override_response(override)


async def delete_override(provider: str) -> None:
    provider_norm = _canonical_override_provider_or_400(provider)
    repo = await get_llm_provider_overrides_repo()
    try:
        deleted = await repo.delete_override(provider_norm)
    except Exception as exc:
        logger.error("Failed to delete provider override")
        del exc
        raise_detached_error(
            HTTPException(status_code=500, detail="Failed to delete provider override")
        )
    if not deleted:
        raise HTTPException(status_code=404, detail="Provider override not found")
    await _refresh_overrides_or_503(force=True)


async def test_provider(
    payload: LLMProviderTestRequest,
    *,
    refresh_overrides: bool = True,
    timeout_seconds: float | None = None,
) -> LLMProviderTestResponse:
    provider_norm = _canonical_override_provider_or_400(payload.provider)
    if refresh_overrides and payload.use_override:
        await _refresh_overrides_or_503()

    try:
        override_snapshot = (
            capture_provider_override_call_snapshot(provider_norm)
            if payload.use_override
            else None
        )
        api_key = (payload.api_key or "").strip() or None
        credential_fields = payload.credential_fields
        model = payload.model
        if model is None and payload.use_override:
            model = _override_test_model(
                provider_norm,
                override_snapshot=override_snapshot,
            )
        if override_snapshot is not None:
            override_snapshot.enforce(model)
        if credential_fields is not None:
            try:
                credential_fields = _normalize_credential_fields(
                    provider_norm,
                    credential_fields,
                )
            except ValueError as exc:
                del exc
                raise_detached_error(
                    HTTPException(
                        status_code=400,
                        detail="Invalid provider credential fields",
                    )
                )

        server_config_snapshot = load_server_config_snapshot()
        static_fallback = resolve_static_server_fallback_from_snapshot(
            provider_norm,
            server_config_snapshot,
        )
        fallback = (
            override_snapshot.server_fallback(static_fallback)
            if override_snapshot is not None
            else static_fallback
        ) or static_fallback

        if api_key is not None or credential_fields is not None:
            fallback = ServerFallbackCredentials(
                api_key=api_key or fallback.api_key,
                credential_fields=(
                    credential_fields
                    if credential_fields is not None
                    else dict(fallback.credential_fields)
                ),
                auth_source=None if api_key else fallback.auth_source,
                app_config=(
                    dict(fallback.app_config)
                    if fallback.app_config is not None
                    else {}
                ),
            )

        resolved = await resolve_byok_credentials(
            provider_norm,
            user_id=None,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=True,
            fallback_override=fallback,
            server_config_snapshot=server_config_snapshot,
        )
        if override_snapshot is not None:
            override_snapshot.ensure_healthy()
        if model is None:
            model = configured_provider_model_from_snapshot(
                provider_norm,
                resolved.app_config,
            )
        if override_snapshot is not None:
            override_snapshot.enforce(model)
        if provider_requires_api_key(provider_norm) and not provider_auth_is_resolved(
            provider_norm,
            api_key=resolved.api_key,
            app_config=resolved.app_config,
            credentials_resolved=True,
        ):
            raise_detached_error(
                HTTPException(
                    status_code=400,
                    detail="Provider credentials are not configured",
                )
            )

        model_used = await test_provider_credentials(
            provider=provider_norm,
            api_key=resolved.api_key,
            app_config=resolved.app_config,
            model=model,
            include_override_model=False,
            timeout_seconds=timeout_seconds,
        )
    except asyncio.CancelledError:
        raise
    except HTTPException:
        raise
    except ByokResolutionError as exc:
        detail = (
            "Provider credential storage is temporarily unavailable"
            if exc.code == "credential_store_unavailable"
            else "Provider test call failed"
        )
        status_code = 503 if exc.code == "credential_store_unavailable" else 502
        del exc
        raise_detached_error(HTTPException(status_code=status_code, detail=detail))
    except ValueError as exc:
        del exc
        raise_detached_error(
            HTTPException(status_code=400, detail="Provider credential validation failed")
        )
    except ChatAPIError as exc:
        public_error = provider_validation_public_error(exc)
        raise_detached_error(
            HTTPException(
                status_code=public_error.status_code,
                detail=public_error.message,
            )
        )
    except Exception as exc:
        del exc
        raise_detached_error(
            HTTPException(status_code=502, detail="Provider test call failed")
        )

    return LLMProviderTestResponse(
        provider=provider_norm,
        status="valid",
        model=model_used,
    )
