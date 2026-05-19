from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_request_user, RequireRole, User

from tldw_Server_API.app.api.v1.endpoints.discord_oauth_admin import discord_oauth_start_impl
from tldw_Server_API.app.api.v1.endpoints.discord_support import (
    _decrypt_discord_payload,
    _discord_policy_for_guild,
    _encrypt_discord_payload,
    _get_oauth_state_repo as _get_discord_oauth_state_repo,
    _normalize_installations_payload as _normalize_discord_installations_payload,
    _oauth_auth_url as _discord_oauth_auth_url,
    _oauth_client_id as _discord_oauth_client_id,
    _oauth_permissions as _discord_oauth_permissions,
    _oauth_redirect_uri as _discord_oauth_redirect_uri,
    _oauth_scope as _discord_oauth_scope,
    _oauth_state_ttl_seconds as _discord_oauth_state_ttl_seconds,
    _set_discord_policy,
)
from tldw_Server_API.app.api.v1.endpoints.slack_oauth_admin import slack_oauth_start_impl
from tldw_Server_API.app.api.v1.endpoints.slack_support import (
    _coerce_nonempty_string,
    _decrypt_slack_payload,
    _encrypt_slack_payload,
    _get_oauth_state_repo as _get_slack_oauth_state_repo,
    _normalize_installations_payload as _normalize_slack_installations_payload,
    _oauth_auth_url as _slack_oauth_auth_url,
    _oauth_client_id as _slack_oauth_client_id,
    _oauth_redirect_uri as _slack_oauth_redirect_uri,
    _oauth_scopes as _slack_oauth_scopes,
    _oauth_state_ttl_seconds as _slack_oauth_state_ttl_seconds,
    _set_slack_policy,
    _slack_policy_for_workspace,
)
from tldw_Server_API.app.api.v1.endpoints.telegram_support import (
    TelegramScope,
    _resolve_shared_scope,
    telegram_admin_get_bot_impl,
    telegram_admin_list_linked_actors_impl,
    telegram_admin_put_bot_impl,
    telegram_admin_revoke_linked_actor_impl,
    telegram_admin_start_link_impl,
)
from tldw_Server_API.app.api.v1.schemas.integrations_control_plane_schemas import (
    DiscordWorkspacePolicy,
    DiscordWorkspacePolicyResponse,
    DiscordWorkspacePolicyUpdate,
    IntegrationOverviewResponse,
    IntegrationConnection,
    PersonalIntegrationConnectResponse,
    PersonalIntegrationDeleteResponse,
    PersonalIntegrationProvider,
    PersonalIntegrationUpdateRequest,
    SlackWorkspacePolicy,
    SlackWorkspacePolicyResponse,
    SlackWorkspacePolicyUpdate,
)
from tldw_Server_API.app.api.v1.schemas.telegram_schemas import (
    TelegramBotConfigResponse,
    TelegramBotConfigUpdate,
    TelegramLinkedActorListResponse,
    TelegramLinkedActorRevokeResponse,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos import get_workspace_provider_installations_repo
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import AuthnzOrgProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import key_hint_for_api_key
from tldw_Server_API.app.services.integrations_control_plane_service import IntegrationsControlPlaneService

router = APIRouter(prefix="/integrations", tags=["integrations"])


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_scope_ids(*value_lists: Any) -> list[int]:
    values: set[int] = set()
    for raw_list in value_lists:
        if not isinstance(raw_list, (list, tuple, set)):
            continue
        for raw in raw_list:
            coerced = _coerce_int(raw)
            if coerced is not None:
                values.add(coerced)
    return sorted(values)


def _normalize_installation_ids(rows: list[dict[str, Any]]) -> list[str]:
    installation_ids: list[str] = []
    for row in rows:
        external_id = str((row or {}).get("external_id") or "").strip()
        if external_id and external_id not in installation_ids:
            installation_ids.append(external_id)
    return installation_ids


def _policies_are_uniform(policies: list[dict[str, Any]]) -> bool:
    if not policies:
        return True
    baseline = dict(policies[0])
    return all(dict(candidate) == baseline for candidate in policies[1:])


def _resolve_workspace_org_id(principal: AuthPrincipal, request: Request | None = None) -> int:
    request_active_org_id = _coerce_int(getattr(request.state, "active_org_id", None)) if request else None
    principal_active_org_id = _coerce_int(principal.active_org_id)
    org_ids = _collect_scope_ids(getattr(request.state, "org_ids", None) if request else None, principal.org_ids)

    for candidate in (request_active_org_id, principal_active_org_id):
        if candidate is not None and (not org_ids or candidate in org_ids):
            return candidate

    if len(org_ids) == 1:
        return org_ids[0]

    if get_settings().auth_mode == "single_user":
        return 1

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="An active org scope is required",
    )


def _resolve_workspace_telegram_scope(principal: AuthPrincipal, request: Request | None, *, org_id: int) -> TelegramScope:
    try:
        return _resolve_shared_scope(principal=principal, request=request)
    except HTTPException:
        if get_settings().auth_mode == "single_user":
            return TelegramScope(scope_type="org", scope_id=int(org_id))
        raise


async def get_integrations_control_plane_service() -> IntegrationsControlPlaneService:
    pool = await get_db_pool()
    user_repo = AuthnzUserProviderSecretsRepo(pool)
    await user_repo.ensure_tables()
    org_repo = AuthnzOrgProviderSecretsRepo(pool)
    await org_repo.ensure_tables()
    workspace_repo = await get_workspace_provider_installations_repo()
    return IntegrationsControlPlaneService(
        user_provider_secrets_repo=user_repo,
        org_provider_secrets_repo=org_repo,
        workspace_installations_repo=workspace_repo,
    )


def _require_personal_provider(provider: str) -> PersonalIntegrationProvider:
    cleaned = str(provider or "").strip().lower()
    if cleaned in {"slack", "discord"}:
        return cleaned  # type: ignore[return-value]
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unsupported personal integration provider")


def _normalize_personal_connection_id(provider: PersonalIntegrationProvider, connection_id: str) -> str:
    cleaned = str(connection_id or "").strip()
    expected = f"personal:{provider}"
    if cleaned == provider:
        return expected
    if cleaned == expected:
        return expected
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="connection_id is required")
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"connection_id must target {expected}")


def _personal_payload_helpers(provider: PersonalIntegrationProvider):
    if provider == "slack":
        return (
            _decrypt_slack_payload,
            _encrypt_slack_payload,
            _normalize_slack_installations_payload,
            "team_id",
        )
    return (
        _decrypt_discord_payload,
        _encrypt_discord_payload,
        _normalize_discord_installations_payload,
        "guild_id",
    )


def _extract_user_id(user: User, principal: AuthPrincipal) -> int:
    for candidate in (getattr(user, "id", None), principal.user_id):
        coerced = _coerce_int(candidate)
        if coerced is not None:
            return coerced
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Authenticated user id is required")


def _build_personal_installation_metadata(installations: dict[str, Any]) -> dict[str, int]:
    installation_count = 0
    active_installation_count = 0
    for installation in installations.values():
        if not isinstance(installation, dict):
            continue
        installation_count += 1
        if not bool(installation.get("disabled")):
            active_installation_count += 1
    return {
        "installation_count": installation_count,
        "active_installation_count": active_installation_count,
    }


def _external_ids_from_installations(
    installations: dict[str, Any],
    *,
    external_id_key: str,
) -> list[str]:
    external_ids: list[str] = []
    for installation_key, installation in installations.items():
        if not isinstance(installation, dict):
            continue
        external_id = _coerce_nonempty_string(installation.get(external_id_key)) or _coerce_nonempty_string(installation_key)
        if external_id and external_id not in external_ids:
            external_ids.append(external_id)
    return external_ids


def _pick_key_hint_token(installations: dict[str, Any]) -> str | None:
    for installation in installations.values():
        if not isinstance(installation, dict):
            continue
        token = _coerce_nonempty_string(installation.get("access_token"))
        if token:
            return token
    return None


async def _start_personal_integration_connect_flow(
    *,
    provider: PersonalIntegrationProvider,
    user: User,
    workspace_org_id: int,
) -> PersonalIntegrationConnectResponse:
    if provider == "slack":
        response = await slack_oauth_start_impl(
            user=user,
            workspace_org_id=workspace_org_id,
            oauth_client_id=_slack_oauth_client_id,
            oauth_redirect_uri=_slack_oauth_redirect_uri,
            oauth_state_ttl_seconds=_slack_oauth_state_ttl_seconds,
            get_oauth_state_repo=_get_slack_oauth_state_repo,
            encrypt_slack_payload=_encrypt_slack_payload,
            oauth_auth_url=_slack_oauth_auth_url,
            oauth_scopes=_slack_oauth_scopes,
            urlencode_fn=urlencode,
        )
    else:
        response = await discord_oauth_start_impl(
            user=user,
            workspace_org_id=workspace_org_id,
            oauth_client_id=_discord_oauth_client_id,
            oauth_redirect_uri=_discord_oauth_redirect_uri,
            oauth_state_ttl_seconds=_discord_oauth_state_ttl_seconds,
            get_oauth_state_repo=_get_discord_oauth_state_repo,
            encrypt_discord_payload=_encrypt_discord_payload,
            oauth_auth_url=_discord_oauth_auth_url,
            oauth_scope=_discord_oauth_scope,
            oauth_permissions=_discord_oauth_permissions,
            urlencode_fn=urlencode,
        )

    return PersonalIntegrationConnectResponse(
        provider=provider,
        connection_id=f"personal:{provider}",
        status="ready",
        auth_url=str(response.get("auth_url") or ""),
        auth_session_id=str(response.get("auth_session_id") or ""),
        expires_at=response["expires_at"],
    )


async def _load_personal_installations(
    *,
    provider: PersonalIntegrationProvider,
    user_id: int,
    service: IntegrationsControlPlaneService,
):
    decrypt_payload, encrypt_payload, normalize_payload, external_id_key = _personal_payload_helpers(provider)
    row = await service.user_provider_secrets_repo.fetch_secret_for_user(user_id, provider)
    payload = decrypt_payload(row.get("encrypted_blob")) if row else None
    merged_payload = normalize_payload(payload)
    installations = merged_payload.get("installations")
    if not isinstance(installations, dict) or not any(isinstance(item, dict) for item in installations.values()):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="integration_not_found")
    return row, merged_payload, installations, encrypt_payload, external_id_key


async def _build_personal_provider_response(
    *,
    provider: PersonalIntegrationProvider,
    user_id: int,
    service: IntegrationsControlPlaneService,
) -> IntegrationConnection:
    overview = await service.build_personal_overview(user_id=user_id)
    for item in overview.items:
        if item.provider == provider:
            return item
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="integration_state_unavailable")


async def _list_workspace_installation_ids(
    *,
    workspace_repo: Any,
    org_id: int,
    provider: str,
) -> list[str]:
    rows = await workspace_repo.list_installations(org_id=org_id, provider=provider, include_disabled=True)
    return _normalize_installation_ids(rows)


async def _build_slack_workspace_policy_response(
    *,
    workspace_repo: Any,
    org_id: int,
) -> SlackWorkspacePolicyResponse:
    installation_ids = await _list_workspace_installation_ids(workspace_repo=workspace_repo, org_id=org_id, provider="slack")
    policies = [_slack_policy_for_workspace(installation_id) for installation_id in installation_ids]
    if not policies:
        policies = [_slack_policy_for_workspace(None)]
    return SlackWorkspacePolicyResponse(
        installation_ids=installation_ids,
        uniform=_policies_are_uniform(policies),
        policy=SlackWorkspacePolicy.model_validate(policies[0]),
    )


async def _update_slack_workspace_policy(
    *,
    workspace_repo: Any,
    org_id: int,
    payload: SlackWorkspacePolicyUpdate,
) -> SlackWorkspacePolicyResponse:
    installation_ids = await _list_workspace_installation_ids(workspace_repo=workspace_repo, org_id=org_id, provider="slack")
    update_payload = payload.model_dump(exclude_none=True)
    applied_policies: list[dict[str, Any]] = []
    if installation_ids:
        for installation_id in installation_ids:
            _, applied_policy = _set_slack_policy(installation_id, update_payload)
            applied_policies.append(applied_policy)
    else:
        _, applied_policy = _set_slack_policy(None, update_payload)
        applied_policies.append(applied_policy)
    return SlackWorkspacePolicyResponse(
        installation_ids=installation_ids,
        uniform=_policies_are_uniform(applied_policies),
        policy=SlackWorkspacePolicy.model_validate(applied_policies[0]),
    )


async def _build_discord_workspace_policy_response(
    *,
    workspace_repo: Any,
    org_id: int,
) -> DiscordWorkspacePolicyResponse:
    installation_ids = await _list_workspace_installation_ids(
        workspace_repo=workspace_repo,
        org_id=org_id,
        provider="discord",
    )
    policies = [_discord_policy_for_guild(installation_id) for installation_id in installation_ids]
    if not policies:
        policies = [_discord_policy_for_guild(None)]
    return DiscordWorkspacePolicyResponse(
        installation_ids=installation_ids,
        uniform=_policies_are_uniform(policies),
        policy=DiscordWorkspacePolicy.model_validate(policies[0]),
    )


async def _update_discord_workspace_policy(
    *,
    workspace_repo: Any,
    org_id: int,
    payload: DiscordWorkspacePolicyUpdate,
) -> DiscordWorkspacePolicyResponse:
    installation_ids = await _list_workspace_installation_ids(
        workspace_repo=workspace_repo,
        org_id=org_id,
        provider="discord",
    )
    update_payload = payload.model_dump(exclude_none=True)
    applied_policies: list[dict[str, Any]] = []
    if installation_ids:
        for installation_id in installation_ids:
            _, applied_policy = _set_discord_policy(installation_id, update_payload)
            applied_policies.append(applied_policy)
    else:
        _, applied_policy = _set_discord_policy(None, update_payload)
        applied_policies.append(applied_policy)
    return DiscordWorkspacePolicyResponse(
        installation_ids=installation_ids,
        uniform=_policies_are_uniform(applied_policies),
        policy=DiscordWorkspacePolicy.model_validate(applied_policies[0]),
    )


@router.get("/personal", response_model=IntegrationOverviewResponse)
async def list_personal_integrations(
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: IntegrationsControlPlaneService = Depends(get_integrations_control_plane_service),
) -> IntegrationOverviewResponse:
    user_id = _coerce_int(principal.user_id)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Authenticated user id is required")
    return await service.build_personal_overview(user_id=user_id)


@router.post("/personal/{provider}/connect", response_model=PersonalIntegrationConnectResponse)
async def connect_personal_integration(
    provider: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    user: User = Depends(get_request_user),
) -> PersonalIntegrationConnectResponse:
    resolved_provider = _require_personal_provider(provider)
    workspace_org_id = _resolve_workspace_org_id(principal, request)
    return await _start_personal_integration_connect_flow(
        provider=resolved_provider,
        user=user,
        workspace_org_id=workspace_org_id,
    )


@router.patch("/personal/{provider}/{connection_id}", response_model=IntegrationConnection)
async def update_personal_integration(
    provider: str,
    connection_id: str,
    payload: PersonalIntegrationUpdateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    user: User = Depends(get_request_user),
    service: IntegrationsControlPlaneService = Depends(get_integrations_control_plane_service),
) -> IntegrationConnection:
    resolved_provider = _require_personal_provider(provider)
    _normalize_personal_connection_id(resolved_provider, connection_id)
    user_id = _extract_user_id(user, principal)
    workspace_org_id = _resolve_workspace_org_id(principal, request)
    _, merged_payload, installations, encrypt_payload, external_id_key = await _load_personal_installations(
        provider=resolved_provider,
        user_id=user_id,
        service=service,
    )

    disabled = not bool(payload.enabled)
    for installation in installations.values():
        if isinstance(installation, dict):
            installation["disabled"] = disabled

    now = datetime.now(timezone.utc)
    key_hint_token = _pick_key_hint_token(installations)
    await service.user_provider_secrets_repo.upsert_secret(
        user_id=user_id,
        provider=resolved_provider,
        encrypted_blob=encrypt_payload(merged_payload),
        key_hint=key_hint_for_api_key(key_hint_token) if key_hint_token else None,
        metadata=_build_personal_installation_metadata(installations),
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )

    for external_id in _external_ids_from_installations(installations, external_id_key=external_id_key):
        await service.workspace_installations_repo.set_disabled(
            org_id=workspace_org_id,
            provider=resolved_provider,
            external_id=external_id,
            disabled=disabled,
        )

    return await _build_personal_provider_response(
        provider=resolved_provider,
        user_id=user_id,
        service=service,
    )


@router.delete("/personal/{provider}/{connection_id}", response_model=PersonalIntegrationDeleteResponse)
async def delete_personal_integration(
    provider: str,
    connection_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    user: User = Depends(get_request_user),
    service: IntegrationsControlPlaneService = Depends(get_integrations_control_plane_service),
) -> PersonalIntegrationDeleteResponse:
    resolved_provider = _require_personal_provider(provider)
    normalized_connection_id = _normalize_personal_connection_id(resolved_provider, connection_id)
    user_id = _extract_user_id(user, principal)
    workspace_org_id = _resolve_workspace_org_id(principal, request)
    _, _, installations, _, external_id_key = await _load_personal_installations(
        provider=resolved_provider,
        user_id=user_id,
        service=service,
    )

    now = datetime.now(timezone.utc)
    await service.user_provider_secrets_repo.delete_secret(
        user_id=user_id,
        provider=resolved_provider,
        revoked_by=user_id,
        revoked_at=now,
    )

    for external_id in _external_ids_from_installations(installations, external_id_key=external_id_key):
        await service.workspace_installations_repo.delete_installation(
            org_id=workspace_org_id,
            provider=resolved_provider,
            external_id=external_id,
        )

    return PersonalIntegrationDeleteResponse(
        deleted=True,
        provider=resolved_provider,
        connection_id=normalized_connection_id,
    )


@router.get("/workspace", dependencies=[Depends(RequireRole("admin"))], response_model=IntegrationOverviewResponse)
async def list_workspace_integrations(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: IntegrationsControlPlaneService = Depends(get_integrations_control_plane_service),
) -> IntegrationOverviewResponse:
    org_id = _resolve_workspace_org_id(principal, request)
    telegram_scope = _resolve_workspace_telegram_scope(principal, request, org_id=org_id)
    return await service.build_workspace_overview(
        org_id=org_id,
        scope_type=telegram_scope.scope_type,
        scope_id=telegram_scope.scope_id,
    )


@router.get(
    "/workspace/slack/policy",
    dependencies=[Depends(RequireRole("admin"))],
    response_model=SlackWorkspacePolicyResponse,
)
async def get_workspace_slack_policy(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    workspace_repo=Depends(get_workspace_provider_installations_repo),
) -> SlackWorkspacePolicyResponse:
    org_id = _resolve_workspace_org_id(principal, request)
    return await _build_slack_workspace_policy_response(workspace_repo=workspace_repo, org_id=org_id)


@router.put(
    "/workspace/slack/policy",
    dependencies=[Depends(RequireRole("admin"))],
    response_model=SlackWorkspacePolicyResponse,
)
async def put_workspace_slack_policy(
    request: Request,
    payload: SlackWorkspacePolicyUpdate,
    principal: AuthPrincipal = Depends(get_auth_principal),
    workspace_repo=Depends(get_workspace_provider_installations_repo),
) -> SlackWorkspacePolicyResponse:
    org_id = _resolve_workspace_org_id(principal, request)
    return await _update_slack_workspace_policy(workspace_repo=workspace_repo, org_id=org_id, payload=payload)


@router.get(
    "/workspace/discord/policy",
    dependencies=[Depends(RequireRole("admin"))],
    response_model=DiscordWorkspacePolicyResponse,
)
async def get_workspace_discord_policy(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    workspace_repo=Depends(get_workspace_provider_installations_repo),
) -> DiscordWorkspacePolicyResponse:
    org_id = _resolve_workspace_org_id(principal, request)
    return await _build_discord_workspace_policy_response(workspace_repo=workspace_repo, org_id=org_id)


@router.put(
    "/workspace/discord/policy",
    dependencies=[Depends(RequireRole("admin"))],
    response_model=DiscordWorkspacePolicyResponse,
)
async def put_workspace_discord_policy(
    request: Request,
    payload: DiscordWorkspacePolicyUpdate,
    principal: AuthPrincipal = Depends(get_auth_principal),
    workspace_repo=Depends(get_workspace_provider_installations_repo),
) -> DiscordWorkspacePolicyResponse:
    org_id = _resolve_workspace_org_id(principal, request)
    return await _update_discord_workspace_policy(workspace_repo=workspace_repo, org_id=org_id, payload=payload)


@router.get(
    "/workspace/telegram/bot",
    dependencies=[Depends(RequireRole("admin"))],
    response_model=TelegramBotConfigResponse,
)
async def get_workspace_telegram_bot(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TelegramBotConfigResponse:
    return await telegram_admin_get_bot_impl(principal=principal, request=request)


@router.put(
    "/workspace/telegram/bot",
    dependencies=[Depends(RequireRole("admin"))],
    response_model=TelegramBotConfigResponse,
)
async def put_workspace_telegram_bot(
    request: Request,
    payload: TelegramBotConfigUpdate,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TelegramBotConfigResponse:
    return await telegram_admin_put_bot_impl(principal=principal, payload=payload, request=request)


@router.post("/workspace/telegram/pairing-code", dependencies=[Depends(RequireRole("admin"))])
async def create_workspace_telegram_pairing_code(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    return await telegram_admin_start_link_impl(principal=principal, request=request)


@router.get(
    "/workspace/telegram/linked-actors",
    dependencies=[Depends(RequireRole("admin"))],
    response_model=TelegramLinkedActorListResponse,
)
async def list_workspace_telegram_linked_actors(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TelegramLinkedActorListResponse:
    return await telegram_admin_list_linked_actors_impl(principal=principal, request=request)


@router.delete(
    "/workspace/telegram/linked-actors/{actor_id}",
    dependencies=[Depends(RequireRole("admin"))],
    response_model=TelegramLinkedActorRevokeResponse,
)
async def revoke_workspace_telegram_linked_actor(
    actor_id: int,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TelegramLinkedActorRevokeResponse:
    return await telegram_admin_revoke_linked_actor_impl(link_id=actor_id, principal=principal, request=request)
