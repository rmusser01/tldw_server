"""The OAuth install/admin flow shared by Discord and Slack.

``discord_oauth_admin.py`` and ``slack_oauth_admin.py`` were 90.9% identical after
normalising ``discord``/``slack`` and ``guild``/``team``. What actually differs is
small and protocol-shaped, and is carried by :class:`ChatOpsOAuthProvider`:

* the authorize-URL query (Discord sends ``response_type`` and an optional
  ``permissions``; Slack sends only ``scope``),
* the token-exchange form (Discord sends ``grant_type``; Slack does not),
* what counts as a successful token response (Slack carries an ``ok`` flag, and
  answers 502 where Discord answers 400 for a missing workspace id),
* which fields of the token response are kept on the installation record (Discord
  keeps ``refresh_token``; Slack keeps ``enterprise_id``, ``bot_user_id`` and
  ``authed_user_id``),
* the public key names (``guild_id``/``guild_name`` against ``team_id``/``team_name``)
  and the policy scope label (``guild`` against ``workspace``).

Those last two are API and metric contracts, so they are described rather than
unified -- changing them would break clients and dashboards for no gain here. Every
other line is one implementation.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import key_hint_for_api_key

__all__ = [
    "ChatOpsOAuthProvider",
    "oauth_start",
    "oauth_callback",
    "admin_get_policy",
    "admin_set_policy",
    "admin_list_installations",
    "admin_delete_installation",
    "admin_set_installation_state",
]


@dataclass(frozen=True)
class ChatOpsOAuthProvider:
    """Everything that differs between one ChatOps provider and another."""

    #: Provider key stored against secrets and installation rows.
    name: str
    #: Human-readable name used in error details.
    label: str
    #: Public key for the installed workspace id, e.g. "guild_id" / "team_id".
    entity_id_key: str
    #: Public key for its display name, e.g. "guild_name" / "team_name".
    entity_name_key: str
    #: Field of the OAuth token response holding the workspace object
    #: ("guild" / "team"). Named without "token" so bandit B106 does not read the
    #: literal as a credential.
    response_entity_field: str
    #: Metric label for a scoped policy write ("guild"/"workspace").
    policy_scope_label: str
    #: Name of the client-id setting, quoted in the 501 detail.
    client_id_setting: str
    #: Status code for a token response with no usable workspace id.
    missing_entity_status: int
    #: Detail for that failure.
    missing_entity_detail: str
    #: Extra installation-record fields, read from the token response.
    installation_extra_keys: tuple[str, ...] = ()
    #: Whether the token response carries an "ok" flag that must be true.
    requires_ok_flag: bool = False
    #: Whether the token form carries grant_type=authorization_code.
    sends_grant_type: bool = True
    #: Whether the authorize query carries response_type=code.
    sends_response_type: bool = True
    #: Whether a blank key hint is stored as "" rather than omitted.
    blank_key_hint_as_empty: bool = False
    #: Nested token-response fields to lift, as public_key -> (container, field).
    installation_nested_keys: dict[str, tuple[str, str]] = field(default_factory=dict)


async def oauth_start(
    provider: ChatOpsOAuthProvider,
    *,
    user: User,
    workspace_org_id: int | None,
    oauth_client_id: Callable[[], str | None],
    oauth_redirect_uri: Callable[[], str],
    oauth_state_ttl_seconds: Callable[[], int],
    get_oauth_state_repo: Callable[[], Awaitable[Any]],
    encrypt_payload: Callable[[dict[str, Any]], str],
    oauth_auth_url: Callable[[], str],
    oauth_scope: Callable[[], str],
    urlencode_fn: Callable[[dict[str, str]], str],
    oauth_permissions: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    """Mint OAuth state and return the provider's authorize URL."""
    client_id = oauth_client_id()
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"{provider.client_id_setting} is not configured",
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
    state_secret = encrypt_payload(state_payload)
    await state_repo.create_state(
        state=state,
        user_id=int(user.id),
        provider=provider.name,
        auth_session_id=auth_session_id,
        redirect_uri=redirect_uri,
        pkce_verifier_encrypted=state_secret,
        expires_at=expires_at,
        created_at=now,
    )

    query: dict[str, str] = {"client_id": client_id}
    if provider.sends_response_type:
        query["response_type"] = "code"
    query["redirect_uri"] = redirect_uri
    query["scope"] = oauth_scope()
    query["state"] = state
    if oauth_permissions is not None:
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


async def oauth_callback(
    provider: ChatOpsOAuthProvider,
    *,
    code: str,
    state: str,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_oauth_state_repo: Callable[[], Awaitable[Any]],
    oauth_client_id: Callable[[], str | None],
    oauth_client_secret: Callable[[], str | None],
    oauth_token_url: Callable[[], str],
    oauth_token_exchange: Callable[..., Awaitable[dict[str, Any]]],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_payload: Callable[[dict[str, Any]], str],
    fallback_entity_id: str | None = None,
    fallback_entity_name: str | None = None,
) -> dict[str, Any]:
    """Consume OAuth state, exchange the code, and record the installation."""
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
        provider=provider.name,
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

    state_payload = decrypt_payload(state_record.get("pkce_verifier_encrypted"))
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
            detail=f"{provider.label} OAuth client credentials are not configured",
        )

    form_data: dict[str, str] = {}
    if provider.sends_grant_type:
        form_data["grant_type"] = "authorization_code"
    form_data.update(
        {
            "code": code_value,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        }
    )
    token_payload = await oauth_token_exchange(
        token_url=oauth_token_url(),
        form_data=form_data,
    )
    if provider.requires_ok_flag and not bool(token_payload.get("ok")):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{provider.label} OAuth token exchange failed",
        )
    access_token = coerce_nonempty_string(token_payload.get("access_token"))
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{provider.label} OAuth response missing access_token",
        )

    entity = token_payload.get(provider.response_entity_field)
    entity_id = (
        coerce_nonempty_string(entity.get("id")) if isinstance(entity, dict) else None
    ) or coerce_nonempty_string(fallback_entity_id)
    entity_name = (
        coerce_nonempty_string(entity.get("name")) if isinstance(entity, dict) else None
    ) or coerce_nonempty_string(fallback_entity_name)
    if not entity_id:
        raise HTTPException(
            status_code=provider.missing_entity_status,
            detail=provider.missing_entity_detail,
        )

    user_repo = await get_user_secret_repo()
    existing_row = await user_repo.fetch_secret_for_user(user_id, provider.name)
    existing_payload = (
        decrypt_payload(existing_row.get("encrypted_blob")) if existing_row else None
    )
    merged_payload = normalize_installations_payload(existing_payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict):
        installations = {}
        merged_payload["installations"] = installations

    now = datetime.now(timezone.utc)
    record: dict[str, Any] = {
        provider.entity_id_key: entity_id,
        provider.entity_name_key: entity_name,
        "access_token": access_token,
    }
    for key in provider.installation_extra_keys:
        record[key] = coerce_nonempty_string(token_payload.get(key))
    for public_key, (container_key, inner_key) in provider.installation_nested_keys.items():
        container = token_payload.get(container_key)
        record[public_key] = (
            coerce_nonempty_string(container.get(inner_key))
            if isinstance(container, dict)
            else None
        )
    record.update(
        {
            "installed_at": now.isoformat(),
            "installed_by": user_id,
            "disabled": False,
        }
    )
    installations[entity_id] = record

    encrypted_blob = encrypt_payload(merged_payload)
    await user_repo.upsert_secret(
        user_id=user_id,
        provider=provider.name,
        encrypted_blob=encrypted_blob,
        key_hint=key_hint_for_api_key(access_token),
        metadata={"installation_count": len(installations)},
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )

    workspace_repo = await get_workspace_provider_installations_repo()
    org_id = (
        state_org_id if state_org_id is not None else await resolve_workspace_org_id(user_id)
    )
    await workspace_repo.upsert_installation(
        org_id=int(org_id),
        provider=provider.name,
        external_id=entity_id,
        display_name=entity_name,
        installed_by_user_id=user_id,
        disabled=False,
    )
    return {
        "ok": True,
        "status": "installed",
        provider.entity_id_key: entity_id,
        provider.entity_name_key: entity_name,
    }


def admin_get_policy(
    provider: ChatOpsOAuthProvider,
    *,
    entity_id: str | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    policy_for_entity: Callable[[str | None], dict[str, Any]],
) -> dict[str, Any]:
    cleaned = coerce_nonempty_string(entity_id)
    return {
        "ok": True,
        provider.entity_id_key: cleaned,
        "policy": policy_for_entity(cleaned),
    }


def admin_set_policy(
    provider: ChatOpsOAuthProvider,
    *,
    payload: dict[str, Any] | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    set_policy: Callable[[str | None, dict[str, Any]], tuple[str | None, dict[str, Any]]],
    emit_counter: Callable[..., None],
    counter_name: str,
) -> dict[str, Any]:
    body = dict(payload or {})
    cleaned = coerce_nonempty_string(body.pop(provider.entity_id_key, None))
    scope = provider.policy_scope_label if cleaned else "default"
    entity_id, policy = set_policy(cleaned, body)
    emit_counter(counter_name, scope=scope)
    return {
        "ok": True,
        "status": "updated",
        provider.entity_id_key: entity_id,
        "policy": policy,
    }


async def admin_list_installations(
    provider: ChatOpsOAuthProvider,
    *,
    user: User,
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    decrypt_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    public_installation_record: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    user_repo = await get_user_secret_repo()
    row = await user_repo.fetch_secret_for_user(int(user.id), provider.name)
    payload = decrypt_payload(row.get("encrypted_blob")) if row else None
    merged_payload = normalize_installations_payload(payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict):
        installations = {}
    results = []
    for entity_key, installation in installations.items():
        if not isinstance(installation, dict):
            continue
        record = public_installation_record(installation)
        record[provider.entity_id_key] = record.get(provider.entity_id_key) or entity_key
        results.append(record)
    results.sort(key=lambda item: str(item.get(provider.entity_id_key) or ""))
    return {"ok": True, "installations": results}


async def admin_delete_installation(
    provider: ChatOpsOAuthProvider,
    *,
    entity_id: str,
    user: User,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    cleaned = coerce_nonempty_string(entity_id)
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{provider.entity_id_key} is required",
        )
    user_id = int(user.id)
    user_repo = await get_user_secret_repo()
    row = await user_repo.fetch_secret_for_user(user_id, provider.name)
    payload = decrypt_payload(row.get("encrypted_blob")) if row else None
    merged_payload = normalize_installations_payload(payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict) or cleaned not in installations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="installation_not_found",
        )

    installations.pop(cleaned, None)
    now = datetime.now(timezone.utc)
    if not installations:
        await user_repo.delete_secret(
            user_id=user_id,
            provider=provider.name,
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
            provider=provider.name,
            encrypted_blob=encrypt_payload(merged_payload),
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
        provider=provider.name,
        external_id=cleaned,
    )
    return {"ok": True, "status": "deleted", provider.entity_id_key: cleaned}


async def admin_set_installation_state(
    provider: ChatOpsOAuthProvider,
    *,
    entity_id: str,
    payload: dict[str, Any] | None,
    user: User,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    cleaned = coerce_nonempty_string(entity_id)
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{provider.entity_id_key} is required",
        )

    disabled = bool((payload or {}).get("disabled"))
    user_id = int(user.id)
    user_repo = await get_user_secret_repo()
    row = await user_repo.fetch_secret_for_user(user_id, provider.name)
    stored_payload = decrypt_payload(row.get("encrypted_blob")) if row else None
    merged_payload = normalize_installations_payload(stored_payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict):
        installations = {}
        merged_payload["installations"] = installations
    installation = installations.get(cleaned)
    if not isinstance(installation, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="installation_not_found",
        )

    installation["disabled"] = disabled
    now = datetime.now(timezone.utc)
    key_hint_token = coerce_nonempty_string(installation.get("access_token"))
    if provider.blank_key_hint_as_empty:
        key_hint_token = key_hint_token or ""
    await user_repo.upsert_secret(
        user_id=user_id,
        provider=provider.name,
        encrypted_blob=encrypt_payload(merged_payload),
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
        provider=provider.name,
        external_id=cleaned,
        disabled=disabled,
    )
    return {
        "ok": True,
        "status": "updated",
        provider.entity_id_key: cleaned,
        "disabled": disabled,
    }
