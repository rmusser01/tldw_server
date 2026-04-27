from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.schemas.user_keys import (
    ProviderKeyTestRequest,
    SharedProviderKeyResponse,
    SharedProviderKeysResponse,
    SharedProviderKeyStatusItem,
    SharedProviderKeyTestResponse,
    UserProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
    is_byok_enabled,
    is_provider_allowlisted,
    is_trusted_base_url_request,
    validate_base_url_override,
    validate_credential_fields,
)
from tldw_Server_API.app.core.AuthNZ.byok_testing import test_provider_credentials
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_org_members, list_team_members
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
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

router = APIRouter(prefix="", tags=["org-team-keys"])
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


async def _get_shared_byok_repo() -> AuthnzOrgProviderSecretsRepo:
    pool = await get_db_pool()
    repo = AuthnzOrgProviderSecretsRepo(pool)
    await repo.ensure_tables()
    return repo


def _is_manager(role: str | None) -> bool:
    if not role:
        return False
    return str(role).lower() in {"owner", "admin", "lead"}


def _is_admin_principal(principal: AuthPrincipal) -> bool:
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if "admin" in roles:
        return True
    permissions = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


def _principal_user_id(principal: AuthPrincipal) -> int:
    raw_id = principal.user_id
    try:
        user_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid user context") from exc
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user context")
    return user_id


async def _require_org_manager(principal: AuthPrincipal, org_id: int) -> None:
    if _is_admin_principal(principal):
        return
    if principal.user_id is None:
        raise HTTPException(status_code=403, detail="Org manager role required")
    try:
        members = await list_org_members(org_id=org_id, limit=1000, offset=0)
        uid = int(principal.user_id)
        for m in members:
            if int(m.get("user_id")) == uid and _is_manager(m.get("role")):
                return
    except Exception:
        logger.debug("Org manager check failed")
    raise HTTPException(status_code=403, detail="Org manager role required")


async def _require_team_manager(principal: AuthPrincipal, team_id: int) -> None:
    if _is_admin_principal(principal):
        return
    if principal.user_id is None:
        raise HTTPException(status_code=403, detail="Team manager role required")
    try:
        members = await list_team_members(team_id)
        uid = int(principal.user_id)
        for m in members:
            if int(m.get("user_id")) == uid and _is_manager(m.get("role")):
                return
    except Exception:
        logger.debug("Team manager check failed")
    raise HTTPException(status_code=403, detail="Team manager role required")


def _require_byok_enabled() -> None:
    if not is_byok_enabled():
        raise HTTPException(status_code=403, detail="BYOK is disabled in this deployment")


async def _touch_shared_last_used_if_match(
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
    except Exception:
        return
    if payload.get("api_key") != api_key:
        return
    await repo.touch_last_used(scope_type, scope_id, provider, datetime.now(timezone.utc))


@router.post(
    "/orgs/{org_id}/keys/shared",
    response_model=SharedProviderKeyResponse,
    status_code=status.HTTP_200_OK,
)
async def upsert_org_shared_key(
    org_id: int,
    payload: UserProviderKeyUpsertRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> SharedProviderKeyResponse:
    _require_byok_enabled()
    await _require_org_manager(principal, org_id)

    provider_norm = normalize_provider_name(payload.provider)
    if not is_provider_allowlisted(provider_norm):
        raise HTTPException(status_code=403, detail="Provider not allowed for BYOK")

    api_key = (payload.api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    allow_base_url = is_trusted_base_url_request(request, principal=principal)
    raw_fields = payload.credential_fields or {}
    if isinstance(raw_fields, dict) and "base_url" in raw_fields and not allow_base_url:
        raise HTTPException(
            status_code=400,
            detail="base_url override requires admin or service principal",
        )

    try:
        credential_fields = validate_credential_fields(
            provider_norm,
            payload.credential_fields,
            allow_base_url=allow_base_url,
        )
        if "base_url" in credential_fields:
            credential_fields["base_url"] = validate_base_url_override(
                credential_fields["base_url"]
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid provider credential fields") from exc

    try:
        await test_provider_credentials(
            provider=provider_norm,
            api_key=api_key,
            credential_fields=credential_fields,
            model=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Provider credential validation failed") from exc
    except ChatAPIError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Provider test call failed") from exc

    secret_payload = build_secret_payload(api_key, credential_fields or None)
    try:
        envelope = encrypt_byok_payload(secret_payload)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="BYOK encryption is not configured") from exc

    repo = await _get_shared_byok_repo()
    now = datetime.now(timezone.utc)
    actor_id = _principal_user_id(principal)
    row = await repo.upsert_secret(
        scope_type="org",
        scope_id=org_id,
        provider=provider_norm,
        encrypted_blob=dumps_envelope(envelope),
        key_hint=key_hint_for_api_key(api_key),
        metadata=payload.metadata,
        updated_at=now,
        created_by=actor_id,
        updated_by=actor_id,
    )
    return SharedProviderKeyResponse(
        scope_type="org",
        scope_id=org_id,
        provider=provider_norm,
        key_hint=row.get("key_hint") or key_hint_for_api_key(api_key),
        updated_at=row.get("updated_at") or now,
    )


@router.get("/orgs/{org_id}/keys/shared", response_model=SharedProviderKeysResponse)
async def list_org_shared_keys(
    org_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> SharedProviderKeysResponse:
    _require_byok_enabled()
    await _require_org_manager(principal, org_id)
    repo = await _get_shared_byok_repo()
    rows = await repo.list_secrets(scope_type="org", scope_id=org_id)
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


@router.post(
    "/orgs/{org_id}/keys/shared/test",
    response_model=SharedProviderKeyTestResponse,
    status_code=status.HTTP_200_OK,
)
async def test_org_shared_key(
    org_id: int,
    payload: ProviderKeyTestRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> SharedProviderKeyTestResponse:
    _require_byok_enabled()
    await _require_org_manager(principal, org_id)

    provider_norm = normalize_provider_name(payload.provider)
    if not is_provider_allowlisted(provider_norm):
        raise HTTPException(status_code=403, detail="Provider not allowed for BYOK")

    repo = await _get_shared_byok_repo()
    row = await repo.fetch_secret("org", org_id, provider_norm)
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

    allow_base_url = is_trusted_base_url_request(request, principal=principal)
    credential_fields_raw = stored_payload.get("credential_fields") or {}
    if isinstance(credential_fields_raw, dict) and "base_url" in credential_fields_raw and not allow_base_url:
        credential_fields_raw = dict(credential_fields_raw)
        credential_fields_raw.pop("base_url", None)

    try:
        credential_fields = validate_credential_fields(
            provider_norm,
            credential_fields_raw,
            allow_base_url=allow_base_url,
        )
        if "base_url" in credential_fields:
            credential_fields["base_url"] = validate_base_url_override(
                credential_fields["base_url"]
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid provider credential fields") from exc

    try:
        model_used = await test_provider_credentials(
            provider=provider_norm,
            api_key=api_key,
            credential_fields=credential_fields,
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Provider credential validation failed") from exc
    except ChatAPIError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Provider test call failed") from exc

    await _touch_shared_last_used_if_match(
        repo,
        scope_type="org",
        scope_id=org_id,
        provider=provider_norm,
        api_key=api_key,
    )
    return SharedProviderKeyTestResponse(
        scope_type="org",
        scope_id=org_id,
        provider=provider_norm,
        status="valid",
        model=model_used,
    )


@router.delete(
    "/orgs/{org_id}/keys/shared/{provider}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_org_shared_key(
    org_id: int,
    provider: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> Response:
    _require_byok_enabled()
    await _require_org_manager(principal, org_id)
    repo = await _get_shared_byok_repo()
    actor_id = _principal_user_id(principal)
    deleted = await repo.delete_secret(
        "org",
        org_id,
        normalize_provider_name(provider),
        revoked_by=actor_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Key not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/teams/{team_id}/keys/shared",
    response_model=SharedProviderKeyResponse,
    status_code=status.HTTP_200_OK,
)
async def upsert_team_shared_key(
    team_id: int,
    payload: UserProviderKeyUpsertRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> SharedProviderKeyResponse:
    _require_byok_enabled()
    await _require_team_manager(principal, team_id)

    provider_norm = normalize_provider_name(payload.provider)
    if not is_provider_allowlisted(provider_norm):
        raise HTTPException(status_code=403, detail="Provider not allowed for BYOK")

    api_key = (payload.api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    allow_base_url = is_trusted_base_url_request(request, principal=principal)
    raw_fields = payload.credential_fields or {}
    if isinstance(raw_fields, dict) and "base_url" in raw_fields and not allow_base_url:
        raise HTTPException(
            status_code=400,
            detail="base_url override requires admin or service principal",
        )

    try:
        credential_fields = validate_credential_fields(
            provider_norm,
            payload.credential_fields,
            allow_base_url=allow_base_url,
        )
        if "base_url" in credential_fields:
            credential_fields["base_url"] = validate_base_url_override(
                credential_fields["base_url"]
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid provider credential fields") from exc

    try:
        await test_provider_credentials(
            provider=provider_norm,
            api_key=api_key,
            credential_fields=credential_fields,
            model=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Provider credential validation failed") from exc
    except ChatAPIError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Provider test call failed") from exc

    secret_payload = build_secret_payload(api_key, credential_fields or None)
    try:
        envelope = encrypt_byok_payload(secret_payload)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="BYOK encryption is not configured") from exc

    repo = await _get_shared_byok_repo()
    now = datetime.now(timezone.utc)
    actor_id = _principal_user_id(principal)
    row = await repo.upsert_secret(
        scope_type="team",
        scope_id=team_id,
        provider=provider_norm,
        encrypted_blob=dumps_envelope(envelope),
        key_hint=key_hint_for_api_key(api_key),
        metadata=payload.metadata,
        updated_at=now,
        created_by=actor_id,
        updated_by=actor_id,
    )
    return SharedProviderKeyResponse(
        scope_type="team",
        scope_id=team_id,
        provider=provider_norm,
        key_hint=row.get("key_hint") or key_hint_for_api_key(api_key),
        updated_at=row.get("updated_at") or now,
    )


@router.get("/teams/{team_id}/keys/shared", response_model=SharedProviderKeysResponse)
async def list_team_shared_keys(
    team_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> SharedProviderKeysResponse:
    _require_byok_enabled()
    await _require_team_manager(principal, team_id)
    repo = await _get_shared_byok_repo()
    rows = await repo.list_secrets(scope_type="team", scope_id=team_id)
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


@router.post(
    "/teams/{team_id}/keys/shared/test",
    response_model=SharedProviderKeyTestResponse,
    status_code=status.HTTP_200_OK,
)
async def test_team_shared_key(
    team_id: int,
    payload: ProviderKeyTestRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> SharedProviderKeyTestResponse:
    _require_byok_enabled()
    await _require_team_manager(principal, team_id)

    provider_norm = normalize_provider_name(payload.provider)
    if not is_provider_allowlisted(provider_norm):
        raise HTTPException(status_code=403, detail="Provider not allowed for BYOK")

    repo = await _get_shared_byok_repo()
    row = await repo.fetch_secret("team", team_id, provider_norm)
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

    allow_base_url = is_trusted_base_url_request(request, principal=principal)
    credential_fields_raw = stored_payload.get("credential_fields") or {}
    if isinstance(credential_fields_raw, dict) and "base_url" in credential_fields_raw and not allow_base_url:
        credential_fields_raw = dict(credential_fields_raw)
        credential_fields_raw.pop("base_url", None)

    try:
        credential_fields = validate_credential_fields(
            provider_norm,
            credential_fields_raw,
            allow_base_url=allow_base_url,
        )
        if "base_url" in credential_fields:
            credential_fields["base_url"] = validate_base_url_override(
                credential_fields["base_url"]
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid provider credential fields") from exc

    try:
        model_used = await test_provider_credentials(
            provider=provider_norm,
            api_key=api_key,
            credential_fields=credential_fields,
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Provider credential validation failed") from exc
    except ChatAPIError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Provider test call failed") from exc

    await _touch_shared_last_used_if_match(
        repo,
        scope_type="team",
        scope_id=team_id,
        provider=provider_norm,
        api_key=api_key,
    )
    return SharedProviderKeyTestResponse(
        scope_type="team",
        scope_id=team_id,
        provider=provider_norm,
        status="valid",
        model=model_used,
    )


@router.delete(
    "/teams/{team_id}/keys/shared/{provider}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_team_shared_key(
    team_id: int,
    provider: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> Response:
    _require_byok_enabled()
    await _require_team_manager(principal, team_id)
    repo = await _get_shared_byok_repo()
    actor_id = _principal_user_id(principal)
    deleted = await repo.delete_secret(
        "team",
        team_id,
        normalize_provider_name(provider),
        revoked_by=actor_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Key not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
