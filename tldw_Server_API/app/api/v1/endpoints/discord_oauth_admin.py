from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from fastapi import HTTPException, status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User

from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import key_hint_for_api_key


async def discord_oauth_start_impl(
    *,
    user: User,
    workspace_org_id: int | None,
    oauth_client_id: Callable[[], str | None],
    oauth_redirect_uri: Callable[[], str],
    oauth_state_ttl_seconds: Callable[[], int],
    get_oauth_state_repo: Callable[[], Awaitable[Any]],
    encrypt_discord_payload: Callable[[dict[str, Any]], str],
    oauth_auth_url: Callable[[], str],
    oauth_scope: Callable[[], str],
    oauth_permissions: Callable[[], str | None],
    urlencode_fn: Callable[[dict[str, str]], str],
) -> dict[str, Any]:
    client_id = oauth_client_id()
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DISCORD_CLIENT_ID is not configured",
        )
    redirect_uri = oauth_redirect_uri()
    state = secrets.token_urlsafe(32)
    auth_session_id = secrets.token_urlsafe(24)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=max(1, oauth_state_ttl_seconds()))

    state_repo = await get_oauth_state_repo()
    state_payload: dict[str, Any] = {"nonce": secrets.token_urlsafe(24)}
    if workspace_org_id is not None:
        state_payload["org_id"] = int(workspace_org_id)
    state_secret = encrypt_discord_payload(state_payload)
    await state_repo.create_state(
        state=state,
        user_id=int(user.id),
        provider="discord",
        auth_session_id=auth_session_id,
        redirect_uri=redirect_uri,
        pkce_verifier_encrypted=state_secret,
        expires_at=expires_at,
        created_at=now,
    )

    query: dict[str, str] = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": oauth_scope(),
        "state": state,
    }
    permissions = oauth_permissions()
    if permissions:
        query["permissions"] = permissions
    auth_url = f"{oauth_auth_url()}?{urlencode_fn(query)}"
    return {
        "ok": True,
        "status": "ready",
        "auth_url": auth_url,
        "auth_session_id": auth_session_id,
        "expires_at": expires_at.isoformat(),
    }


async def discord_oauth_callback_impl(
    *,
    code: str,
    state: str,
    guild_id: str | None,
    guild_name: str | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_oauth_state_repo: Callable[[], Awaitable[Any]],
    oauth_client_id: Callable[[], str | None],
    oauth_client_secret: Callable[[], str | None],
    oauth_token_url: Callable[[], str],
    discord_oauth_token_exchange: Callable[..., Awaitable[dict[str, Any]]],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_discord_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_discord_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    code_value = coerce_nonempty_string(code)
    state_value = coerce_nonempty_string(state)
    if not code_value or not state_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing OAuth callback parameters",
        )

    state_repo = await get_oauth_state_repo()
    state_record = await state_repo.consume_state(
        state=state_value,
        provider="discord",
    )
    if not state_record:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired OAuth state",
        )

    redirect_uri = coerce_nonempty_string(state_record.get("redirect_uri"))
    if not redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth state is missing redirect metadata",
        )

    state_payload = decrypt_discord_payload(state_record.get("pkce_verifier_encrypted"))
    state_org_id: int | None = None
    if isinstance(state_payload, dict):
        try:
            candidate = int(state_payload.get("org_id"))
            if candidate > 0:
                state_org_id = candidate
        except (TypeError, ValueError):
            state_org_id = None
    user_id_raw = state_record.get("user_id")
    try:
        user_id = int(user_id_raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth state user context is invalid",
        ) from exc

    client_id = oauth_client_id()
    client_secret = oauth_client_secret()
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Discord OAuth client credentials are not configured",
        )

    token_payload = await discord_oauth_token_exchange(
        token_url=oauth_token_url(),
        form_data={
            "grant_type": "authorization_code",
            "code": code_value,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        },
    )
    access_token = coerce_nonempty_string(token_payload.get("access_token"))
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Discord OAuth response missing access_token",
        )

    payload_guild = token_payload.get("guild")
    resolved_guild_id = (
        coerce_nonempty_string(payload_guild.get("id")) if isinstance(payload_guild, dict) else None
    ) or coerce_nonempty_string(guild_id)
    resolved_guild_name = (
        coerce_nonempty_string(payload_guild.get("name")) if isinstance(payload_guild, dict) else None
    ) or coerce_nonempty_string(guild_name)
    if not resolved_guild_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Discord OAuth callback is missing guild_id",
        )

    user_repo = await get_user_secret_repo()
    existing_row = await user_repo.fetch_secret_for_user(user_id, "discord")
    existing_payload = decrypt_discord_payload(existing_row.get("encrypted_blob")) if existing_row else None
    merged_payload = normalize_installations_payload(existing_payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict):
        installations = {}
        merged_payload["installations"] = installations

    now = datetime.now(timezone.utc)
    installations[resolved_guild_id] = {
        "guild_id": resolved_guild_id,
        "guild_name": resolved_guild_name,
        "access_token": access_token,
        "refresh_token": coerce_nonempty_string(token_payload.get("refresh_token")),
        "scope": coerce_nonempty_string(token_payload.get("scope")),
        "installed_at": now.isoformat(),
        "installed_by": user_id,
        "disabled": False,
    }

    encrypted_blob = encrypt_discord_payload(merged_payload)
    await user_repo.upsert_secret(
        user_id=user_id,
        provider="discord",
        encrypted_blob=encrypted_blob,
        key_hint=key_hint_for_api_key(access_token),
        metadata={"installation_count": len(installations)},
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )

    workspace_repo = await get_workspace_provider_installations_repo()
    org_id = state_org_id if state_org_id is not None else await resolve_workspace_org_id(user_id)
    await workspace_repo.upsert_installation(
        org_id=int(org_id),
        provider="discord",
        external_id=resolved_guild_id,
        display_name=resolved_guild_name,
        installed_by_user_id=user_id,
        disabled=False,
    )
    return {
        "ok": True,
        "status": "installed",
        "guild_id": resolved_guild_id,
        "guild_name": resolved_guild_name,
    }


def discord_admin_get_policy_impl(
    *,
    guild_id: str | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    discord_policy_for_guild: Callable[[str | None], dict[str, Any]],
) -> dict[str, Any]:
    cleaned_guild_id = coerce_nonempty_string(guild_id)
    policy = discord_policy_for_guild(cleaned_guild_id)
    return {
        "ok": True,
        "guild_id": cleaned_guild_id,
        "policy": policy,
    }


def discord_admin_set_policy_impl(
    *,
    payload: dict[str, Any] | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    set_discord_policy: Callable[[str | None, dict[str, Any]], tuple[str | None, dict[str, Any]]],
    emit_discord_counter: Callable[..., None],
) -> dict[str, Any]:
    body = dict(payload or {})
    cleaned_guild_id = coerce_nonempty_string(body.pop("guild_id", None))
    scope = "guild" if cleaned_guild_id else "default"
    guild_id, policy = set_discord_policy(cleaned_guild_id, body)
    emit_discord_counter("discord_policy_updates_total", scope=scope)
    return {
        "ok": True,
        "status": "updated",
        "guild_id": guild_id,
        "policy": policy,
    }


async def discord_admin_list_installations_impl(
    *,
    user: User,
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    decrypt_discord_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    public_installation_record: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    user_repo = await get_user_secret_repo()
    row = await user_repo.fetch_secret_for_user(int(user.id), "discord")
    payload = decrypt_discord_payload(row.get("encrypted_blob")) if row else None
    merged_payload = normalize_installations_payload(payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict):
        installations = {}
    results = []
    for guild_id_key, installation in installations.items():
        if not isinstance(installation, dict):
            continue
        record = public_installation_record(installation)
        record["guild_id"] = record.get("guild_id") or guild_id_key
        results.append(record)
    results.sort(key=lambda item: str(item.get("guild_id") or ""))
    return {"ok": True, "installations": results}


async def discord_admin_delete_installation_impl(
    *,
    guild_id: str,
    user: User,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_discord_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_discord_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    cleaned_guild_id = coerce_nonempty_string(guild_id)
    if not cleaned_guild_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="guild_id is required")
    user_id = int(user.id)
    user_repo = await get_user_secret_repo()
    row = await user_repo.fetch_secret_for_user(user_id, "discord")
    payload = decrypt_discord_payload(row.get("encrypted_blob")) if row else None
    merged_payload = normalize_installations_payload(payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict) or cleaned_guild_id not in installations:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="installation_not_found")

    installations.pop(cleaned_guild_id, None)
    now = datetime.now(timezone.utc)
    if not installations:
        await user_repo.delete_secret(
            user_id=user_id,
            provider="discord",
            revoked_by=user_id,
            revoked_at=now,
        )
    else:
        replacement_token: str | None = None
        for remaining in installations.values():
            if isinstance(remaining, dict):
                replacement_token = coerce_nonempty_string(remaining.get("access_token"))
                if replacement_token:
                    break
        await user_repo.upsert_secret(
            user_id=user_id,
            provider="discord",
            encrypted_blob=encrypt_discord_payload(merged_payload),
            key_hint=key_hint_for_api_key(replacement_token) if replacement_token else None,
            metadata={"installation_count": len(installations)},
            updated_at=now,
            created_by=user_id,
            updated_by=user_id,
        )

    workspace_repo = await get_workspace_provider_installations_repo()
    org_id = await resolve_workspace_org_id(user_id)
    await workspace_repo.delete_installation(
        org_id=int(org_id),
        provider="discord",
        external_id=cleaned_guild_id,
    )
    return {"ok": True, "status": "deleted", "guild_id": cleaned_guild_id}


async def discord_admin_set_installation_state_impl(
    *,
    guild_id: str,
    payload: dict[str, Any] | None,
    user: User,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_discord_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_discord_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    cleaned_guild_id = coerce_nonempty_string(guild_id)
    if not cleaned_guild_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="guild_id is required")

    disabled = bool((payload or {}).get("disabled"))
    user_id = int(user.id)
    user_repo = await get_user_secret_repo()
    row = await user_repo.fetch_secret_for_user(user_id, "discord")
    stored_payload = decrypt_discord_payload(row.get("encrypted_blob")) if row else None
    merged_payload = normalize_installations_payload(stored_payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict):
        installations = {}
        merged_payload["installations"] = installations
    installation = installations.get(cleaned_guild_id)
    if not isinstance(installation, dict):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="installation_not_found")

    installation["disabled"] = disabled
    now = datetime.now(timezone.utc)
    key_hint_token = coerce_nonempty_string(installation.get("access_token"))
    await user_repo.upsert_secret(
        user_id=user_id,
        provider="discord",
        encrypted_blob=encrypt_discord_payload(merged_payload),
        key_hint=key_hint_for_api_key(key_hint_token) if key_hint_token else None,
        metadata={"installation_count": len(installations)},
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )

    workspace_repo = await get_workspace_provider_installations_repo()
    org_id = await resolve_workspace_org_id(user_id)
    await workspace_repo.set_disabled(
        org_id=int(org_id),
        provider="discord",
        external_id=cleaned_guild_id,
        disabled=disabled,
    )
    return {
        "ok": True,
        "status": "updated",
        "guild_id": cleaned_guild_id,
        "disabled": disabled,
    }
