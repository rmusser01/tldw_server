from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from fastapi import HTTPException, status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User

from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import key_hint_for_api_key


async def slack_oauth_start_impl(
    *,
    user: User,
    workspace_org_id: int | None,
    oauth_client_id: Callable[[], str | None],
    oauth_redirect_uri: Callable[[], str],
    oauth_state_ttl_seconds: Callable[[], int],
    get_oauth_state_repo: Callable[[], Awaitable[Any]],
    encrypt_slack_payload: Callable[[dict[str, Any]], str],
    oauth_auth_url: Callable[[], str],
    oauth_scopes: Callable[[], str],
    urlencode_fn: Callable[[dict[str, str]], str],
) -> dict[str, Any]:
    client_id = oauth_client_id()
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="SLACK_CLIENT_ID is not configured",
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
    state_secret = encrypt_slack_payload(state_payload)
    await state_repo.create_state(
        state=state,
        user_id=int(user.id),
        provider="slack",
        auth_session_id=auth_session_id,
        redirect_uri=redirect_uri,
        pkce_verifier_encrypted=state_secret,
        expires_at=expires_at,
        created_at=now,
    )

    query = {
        "client_id": client_id,
        "scope": oauth_scopes(),
        "redirect_uri": redirect_uri,
        "state": state,
    }
    auth_url = f"{oauth_auth_url()}?{urlencode_fn(query)}"
    return {
        "ok": True,
        "status": "ready",
        "auth_url": auth_url,
        "auth_session_id": auth_session_id,
        "expires_at": expires_at.isoformat(),
    }


async def slack_oauth_callback_impl(
    *,
    code: str,
    state: str,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_oauth_state_repo: Callable[[], Awaitable[Any]],
    oauth_client_id: Callable[[], str | None],
    oauth_client_secret: Callable[[], str | None],
    oauth_token_url: Callable[[], str],
    slack_oauth_token_exchange: Callable[..., Awaitable[dict[str, Any]]],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_slack_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_slack_payload: Callable[[dict[str, Any]], str],
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
        provider="slack",
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

    state_payload = decrypt_slack_payload(state_record.get("pkce_verifier_encrypted"))
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
            detail="Slack OAuth client credentials are not configured",
        )

    token_payload = await slack_oauth_token_exchange(
        token_url=oauth_token_url(),
        form_data={
            "code": code_value,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        },
    )
    if not bool(token_payload.get("ok")):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Slack OAuth token exchange failed",
        )

    access_token = coerce_nonempty_string(token_payload.get("access_token"))
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Slack OAuth response missing access_token",
        )

    team_data = token_payload.get("team")
    team_id = coerce_nonempty_string(team_data.get("id")) if isinstance(team_data, dict) else None
    team_name = coerce_nonempty_string(team_data.get("name")) if isinstance(team_data, dict) else None
    if not team_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Slack OAuth response missing team.id",
        )

    user_repo = await get_user_secret_repo()
    existing_row = await user_repo.fetch_secret_for_user(user_id, "slack")
    existing_payload = decrypt_slack_payload(existing_row.get("encrypted_blob")) if existing_row else None
    merged_payload = normalize_installations_payload(existing_payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict):
        installations = {}
        merged_payload["installations"] = installations

    now = datetime.now(timezone.utc)
    authed_user = token_payload.get("authed_user")
    authed_user_id = coerce_nonempty_string(authed_user.get("id")) if isinstance(authed_user, dict) else None
    installations[team_id] = {
        "team_id": team_id,
        "team_name": team_name,
        "enterprise_id": coerce_nonempty_string(token_payload.get("enterprise_id")),
        "bot_user_id": coerce_nonempty_string(token_payload.get("bot_user_id")),
        "scope": coerce_nonempty_string(token_payload.get("scope")),
        "authed_user_id": authed_user_id,
        "access_token": access_token,
        "installed_at": now.isoformat(),
        "installed_by": user_id,
        "disabled": False,
    }

    encrypted_blob = encrypt_slack_payload(merged_payload)
    await user_repo.upsert_secret(
        user_id=user_id,
        provider="slack",
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
        provider="slack",
        external_id=team_id,
        display_name=team_name,
        installed_by_user_id=user_id,
        disabled=False,
    )

    return {
        "ok": True,
        "status": "installed",
        "team_id": team_id,
        "team_name": team_name,
    }


def slack_admin_get_policy_impl(
    *,
    team_id: str | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    slack_policy_for_workspace: Callable[[str | None], dict[str, Any]],
) -> dict[str, Any]:
    cleaned_team_id = coerce_nonempty_string(team_id)
    policy = slack_policy_for_workspace(cleaned_team_id)
    return {
        "ok": True,
        "team_id": cleaned_team_id,
        "policy": policy,
    }


def slack_admin_set_policy_impl(
    *,
    payload: dict[str, Any] | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    set_slack_policy: Callable[[str | None, dict[str, Any]], tuple[str | None, dict[str, Any]]],
    emit_slack_counter: Callable[..., None],
) -> dict[str, Any]:
    body = dict(payload or {})
    cleaned_team_id = coerce_nonempty_string(body.pop("team_id", None))
    scope = "workspace" if cleaned_team_id else "default"
    team_id, policy = set_slack_policy(cleaned_team_id, body)
    emit_slack_counter("slack_policy_updates_total", scope=scope)
    return {
        "ok": True,
        "status": "updated",
        "team_id": team_id,
        "policy": policy,
    }


async def slack_admin_list_installations_impl(
    *,
    user: User,
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    decrypt_slack_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    public_installation_record: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    user_repo = await get_user_secret_repo()
    row = await user_repo.fetch_secret_for_user(int(user.id), "slack")
    payload = decrypt_slack_payload(row.get("encrypted_blob")) if row else None
    merged_payload = normalize_installations_payload(payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict):
        installations = {}
    results = []
    for team_id, installation in installations.items():
        if not isinstance(installation, dict):
            continue
        record = public_installation_record(installation)
        record["team_id"] = record.get("team_id") or team_id
        results.append(record)
    results.sort(key=lambda item: str(item.get("team_id") or ""))
    return {"ok": True, "installations": results}


async def slack_admin_delete_installation_impl(
    *,
    team_id: str,
    user: User,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_slack_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_slack_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    cleaned_team_id = coerce_nonempty_string(team_id)
    if not cleaned_team_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="team_id is required")
    user_id = int(user.id)
    user_repo = await get_user_secret_repo()
    row = await user_repo.fetch_secret_for_user(user_id, "slack")
    payload = decrypt_slack_payload(row.get("encrypted_blob")) if row else None
    merged_payload = normalize_installations_payload(payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict) or cleaned_team_id not in installations:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="installation_not_found")

    installations.pop(cleaned_team_id, None)
    now = datetime.now(timezone.utc)
    if not installations:
        await user_repo.delete_secret(
            user_id=user_id,
            provider="slack",
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
            provider="slack",
            encrypted_blob=encrypt_slack_payload(merged_payload),
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
        provider="slack",
        external_id=cleaned_team_id,
    )
    return {"ok": True, "status": "deleted", "team_id": cleaned_team_id}


async def slack_admin_set_installation_state_impl(
    *,
    team_id: str,
    payload: dict[str, Any] | None,
    user: User,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_slack_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_slack_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    cleaned_team_id = coerce_nonempty_string(team_id)
    if not cleaned_team_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="team_id is required")

    disabled = bool((payload or {}).get("disabled"))
    user_id = int(user.id)
    user_repo = await get_user_secret_repo()
    row = await user_repo.fetch_secret_for_user(user_id, "slack")
    stored_payload = decrypt_slack_payload(row.get("encrypted_blob")) if row else None
    merged_payload = normalize_installations_payload(stored_payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict):
        installations = {}
        merged_payload["installations"] = installations
    installation = installations.get(cleaned_team_id)
    if not isinstance(installation, dict):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="installation_not_found")

    installation["disabled"] = disabled
    now = datetime.now(timezone.utc)
    key_hint_token = coerce_nonempty_string(installation.get("access_token")) or ""
    await user_repo.upsert_secret(
        user_id=user_id,
        provider="slack",
        encrypted_blob=encrypt_slack_payload(merged_payload),
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
        provider="slack",
        external_id=cleaned_team_id,
        disabled=disabled,
    )
    return {
        "ok": True,
        "status": "updated",
        "team_id": cleaned_team_id,
        "disabled": disabled,
    }
