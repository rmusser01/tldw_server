from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, NoReturn

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.user_keys import (
    AdminUserKeysResponse,
    AdminUserKeyStatusItem,
    SharedProviderKeyResponse,
    SharedProviderKeysResponse,
    SharedProviderKeyStatusItem,
    SharedProviderKeyTestRequest,
    SharedProviderKeyTestResponse,
    SharedProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
    is_byok_enabled,
    is_provider_allowlisted,
    resolve_byok_allowlist,
    validate_base_url_override,
    validate_credential_fields,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    openai_credential_mutation_lock,
)
from tldw_Server_API.app.core.AuthNZ.byok_testing import (
    provider_validation_public_error,
    test_provider_credentials,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseLockError,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    MembershipAuthority,
    MembershipAuthorizationError,
    MembershipScopeNotFound,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.transaction_policy import (
    get_authnz_transaction_policy,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    ProviderCredentialAliasConflictError,
    build_secret_payload,
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.services import admin_scope_service


def _raise_provider_validation_http_error(exc: ChatAPIError) -> NoReturn:
    """Raise one detached, bounded credential-validation HTTP error."""
    public_error = provider_validation_public_error(exc)
    raise_detached_error(
        HTTPException(
            status_code=public_error.status_code,
            detail=public_error.message,
        )
    )


async def get_user_byok_repo() -> AuthnzUserProviderSecretsRepo:
    """Initialize user BYOK repository and ensure schema exists."""
    try:
        pool = await get_db_pool()
        repo = AuthnzUserProviderSecretsRepo(pool)
        await repo.ensure_tables()
        return repo
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to initialize user BYOK repository")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BYOK infrastructure is not available",
        ) from exc


async def get_shared_byok_repo() -> AuthnzOrgProviderSecretsRepo:
    """Initialize shared BYOK repository and ensure schema exists."""
    try:
        pool = await get_db_pool()
        repo = AuthnzOrgProviderSecretsRepo(pool)
        await repo.ensure_tables()
        return repo
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to initialize shared BYOK repository")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BYOK infrastructure is not available",
        ) from exc


def require_byok_enabled() -> None:
    if not is_byok_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="BYOK is disabled in this deployment",
        )


def _platform_admin_membership_context(
    principal: AuthPrincipal,
) -> ActorMembershipWriteContext:
    if type(principal.user_id) is not int or principal.user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to manage shared BYOK keys",
        )
    return ActorMembershipWriteContext(
        actor_user_id=principal.user_id,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )


_SHARED_KEY_CONTROL_ERRORS = (
    MembershipAuthorizationError,
    MembershipScopeNotFound,
    ConnectionPoolExhaustedError,
    DatabaseLockError,
    TimeoutError,
)


def _shared_key_control_http_exception(
    exc: (
        MembershipAuthorizationError
        | MembershipScopeNotFound
        | ConnectionPoolExhaustedError
        | DatabaseLockError
        | TimeoutError
    ),
) -> HTTPException:
    if isinstance(exc, MembershipAuthorizationError):
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to manage shared BYOK keys",
        )
    if isinstance(exc, MembershipScopeNotFound):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Shared BYOK scope not found",
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Authentication database is busy. Please retry shortly.",
        headers={
            "Retry-After": str(
                get_authnz_transaction_policy().busy_retry_after_seconds
            ),
        },
    )


def normalize_credential_fields(
    provider: str,
    fields: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize credential fields; base_url is allowlisted per provider and egress-validated."""
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


async def touch_shared_last_used_if_match(
    repo: AuthnzOrgProviderSecretsRepo,
    *,
    scope_type: str,
    scope_id: int,
    provider: str,
    api_key: str,
) -> None:
    row = await repo.fetch_secret(scope_type, scope_id, provider)
    if not row:
        return
    encrypted_blob = row.get("encrypted_blob")
    if not encrypted_blob:
        return
    try:
        payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
    except (ValueError, KeyError, TypeError):
        logger.debug("BYOK: failed to decrypt shared secret")
        return
    if payload.get("api_key") != api_key:
        return
    await repo.touch_last_used(scope_type, scope_id, provider, datetime.now(timezone.utc))


async def list_user_keys(
    principal: AuthPrincipal,
    user_id: int,
) -> AdminUserKeysResponse:
    require_byok_enabled()
    await admin_scope_service.enforce_admin_user_scope(
        principal,
        user_id,
        require_hierarchy=False,
    )
    repo = await get_user_byok_repo()
    try:
        rows = await repo.list_secrets_for_user(user_id)
    except ProviderCredentialAliasConflictError as exc:
        raise HTTPException(status_code=409, detail="Conflicting provider credential aliases") from exc
    except Exception as exc:
        logger.error("Failed to list user BYOK keys")
        raise HTTPException(status_code=500, detail="Failed to list user BYOK keys") from exc
    allowlist = resolve_byok_allowlist()
    items = [
        AdminUserKeyStatusItem(
            provider=row.get("provider"),
            key_hint=row.get("key_hint"),
            last_used_at=row.get("last_used_at"),
            allowed=str(row.get("provider")) in allowlist,
        )
        for row in rows
    ]
    return AdminUserKeysResponse(user_id=user_id, items=items)


async def revoke_user_key(
    principal: AuthPrincipal,
    user_id: int,
    provider: str,
) -> None:
    require_byok_enabled()
    await admin_scope_service.enforce_admin_user_scope(
        principal,
        user_id,
        require_hierarchy=True,
    )
    repo = await get_user_byok_repo()
    provider_norm = canonical_provider_name(provider)
    try:
        if provider_norm == "openai":
            async with openai_credential_mutation_lock(
                user_id=user_id,
                provider=provider_norm,
            ) as locked_repo:
                deleted = await (locked_repo or repo).delete_secret(
                    user_id,
                    provider_norm,
                    revoked_by=principal.user_id,
                )
        else:
            deleted = await repo.delete_secret(
                user_id,
                provider_norm,
                revoked_by=principal.user_id,
            )
    except ByokResolutionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BYOK credential storage is temporarily unavailable",
        ) from None
    except ProviderCredentialAliasConflictError as exc:
        raise HTTPException(status_code=409, detail="Conflicting provider credential aliases") from exc
    except Exception as exc:
        logger.error("Failed to revoke user BYOK key")
        raise HTTPException(status_code=500, detail="Failed to revoke user BYOK key") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Key not found")


async def upsert_shared_key(
    principal: AuthPrincipal,
    payload: SharedProviderKeyUpsertRequest,
) -> SharedProviderKeyResponse:
    require_byok_enabled()
    authorization_context = _platform_admin_membership_context(principal)
    provider_norm = canonical_provider_name(payload.provider)
    if not is_provider_allowlisted(provider_norm):
        raise HTTPException(status_code=403, detail="Provider not allowed for BYOK")

    api_key = (payload.api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    try:
        credential_fields = normalize_credential_fields(provider_norm, payload.credential_fields)
    except ValueError as exc:
        del exc
        raise_detached_error(
            HTTPException(status_code=400, detail="Invalid provider credential fields")
        )

    try:
        await test_provider_credentials(
            provider=provider_norm,
            api_key=api_key,
            credential_fields=credential_fields,
            model=None,
        )
    except ValueError as exc:
        del exc
        raise_detached_error(
            HTTPException(status_code=400, detail="Provider credential validation failed")
        )
    except ChatAPIError as exc:
        _raise_provider_validation_http_error(exc)
    except Exception as exc:
        del exc
        raise_detached_error(
            HTTPException(status_code=502, detail="Provider test call failed")
        )

    secret_payload = build_secret_payload(api_key, credential_fields or None)
    try:
        envelope = encrypt_byok_payload(secret_payload)
    except ValueError as exc:
        del exc
        raise_detached_error(
            HTTPException(status_code=500, detail="BYOK encryption is not configured")
        )

    repo = await get_shared_byok_repo()
    now = datetime.now(timezone.utc)
    try:
        row = await repo.upsert_secret(
            scope_type=payload.scope_type,
            scope_id=payload.scope_id,
            provider=provider_norm,
            encrypted_blob=dumps_envelope(envelope),
            key_hint=key_hint_for_api_key(api_key),
            metadata=payload.metadata,
            updated_at=now,
            created_by=principal.user_id,
            updated_by=principal.user_id,
            authorization_context=authorization_context,
        )
    except ProviderCredentialAliasConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail="Conflicting provider credential aliases",
        ) from exc
    except _SHARED_KEY_CONTROL_ERRORS as exc:
        raise _shared_key_control_http_exception(exc) from exc
    except Exception as exc:
        logger.error("Failed to store shared BYOK key")
        raise HTTPException(status_code=500, detail="Failed to store shared BYOK key") from exc
    return SharedProviderKeyResponse(
        scope_type=payload.scope_type,
        scope_id=payload.scope_id,
        provider=provider_norm,
        key_hint=row.get("key_hint") or key_hint_for_api_key(api_key),
        updated_at=row.get("updated_at") or now,
    )


async def test_shared_key(
    principal: AuthPrincipal,
    payload: SharedProviderKeyTestRequest,
) -> SharedProviderKeyTestResponse:
    require_byok_enabled()
    provider_norm = canonical_provider_name(payload.provider)
    if not is_provider_allowlisted(provider_norm):
        raise HTTPException(status_code=403, detail="Provider not allowed for BYOK")

    repo = await get_shared_byok_repo()
    try:
        row = await repo.fetch_secret(payload.scope_type, payload.scope_id, provider_norm)
    except ProviderCredentialAliasConflictError as exc:
        raise HTTPException(status_code=409, detail="Conflicting provider credential aliases") from exc
    if not row:
        raise HTTPException(status_code=404, detail="Key not found")
    encrypted_blob = row.get("encrypted_blob")
    if not encrypted_blob:
        raise HTTPException(status_code=404, detail="Key not found")
    try:
        stored_payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
    except Exception:
        raise HTTPException(status_code=404, detail="Key not found") from None

    api_key = (stored_payload.get("api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=404, detail="Key not found")

    try:
        credential_fields = normalize_credential_fields(
            provider_norm,
            stored_payload.get("credential_fields") or {},
        )
    except ValueError as exc:
        del exc
        raise_detached_error(
            HTTPException(status_code=400, detail="Invalid provider credential fields")
        )

    try:
        model_used = await test_provider_credentials(
            provider=provider_norm,
            api_key=api_key,
            credential_fields=credential_fields,
            model=payload.model,
        )
    except ValueError as exc:
        del exc
        raise_detached_error(
            HTTPException(status_code=400, detail="Provider credential validation failed")
        )
    except ChatAPIError as exc:
        _raise_provider_validation_http_error(exc)
    except Exception as exc:
        del exc
        raise_detached_error(
            HTTPException(status_code=502, detail="Provider test call failed")
        )

    try:
        await touch_shared_last_used_if_match(
            repo,
            scope_type=payload.scope_type,
            scope_id=payload.scope_id,
            provider=provider_norm,
            api_key=api_key,
        )
    except ProviderCredentialAliasConflictError as exc:
        raise HTTPException(status_code=409, detail="Conflicting provider credential aliases") from exc

    return SharedProviderKeyTestResponse(
        scope_type=payload.scope_type,
        scope_id=payload.scope_id,
        provider=provider_norm,
        status="valid",
        model=model_used,
    )


async def list_shared_keys(
    principal: AuthPrincipal,
    *,
    scope_type: Literal["org", "team"] | None,
    scope_id: int | None,
    provider: str | None,
) -> SharedProviderKeysResponse:
    require_byok_enabled()
    repo = await get_shared_byok_repo()
    try:
        rows = await repo.list_secrets(
            scope_type=scope_type,
            scope_id=scope_id,
            provider=provider,
        )
    except ProviderCredentialAliasConflictError as exc:
        raise HTTPException(status_code=409, detail="Conflicting provider credential aliases") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid shared BYOK key query") from exc
    except Exception as exc:
        logger.error("Failed to list shared BYOK keys")
        raise HTTPException(status_code=500, detail="Failed to list shared BYOK keys") from exc
    items = [
        SharedProviderKeyStatusItem(
            scope_type=row.get("scope_type"),
            scope_id=row.get("scope_id"),
            provider=row.get("provider"),
            key_hint=row.get("key_hint"),
            last_used_at=row.get("last_used_at"),
        )
        for row in rows
    ]
    return SharedProviderKeysResponse(items=items)


async def delete_shared_key(
    principal: AuthPrincipal,
    scope_type: Literal["org", "team"],
    scope_id: int,
    provider: str,
) -> None:
    require_byok_enabled()
    authorization_context = _platform_admin_membership_context(principal)
    repo = await get_shared_byok_repo()
    provider_norm = canonical_provider_name(provider)
    try:
        deleted = await repo.delete_secret(
            scope_type,
            scope_id,
            provider_norm,
            revoked_by=principal.user_id,
            authorization_context=authorization_context,
        )
    except ProviderCredentialAliasConflictError as exc:
        raise HTTPException(status_code=409, detail="Conflicting provider credential aliases") from exc
    except _SHARED_KEY_CONTROL_ERRORS as exc:
        raise _shared_key_control_http_exception(exc) from exc
    except Exception as exc:
        logger.error("Failed to delete shared BYOK key")
        raise HTTPException(status_code=500, detail="Failed to delete shared BYOK key") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Key not found")
