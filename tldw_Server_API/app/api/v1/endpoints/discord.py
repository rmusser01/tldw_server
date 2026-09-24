from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, RequireRole, User

from tldw_Server_API.app.api.v1.endpoints._chatops import ingress as _chatops_ingress
from tldw_Server_API.app.api.v1.endpoints._chatops import policy as _chatops_policy
from tldw_Server_API.app.api.v1.endpoints.discord_oauth_admin import (
    discord_admin_delete_installation_impl,
    discord_admin_get_policy_impl,
    discord_admin_list_installations_impl,
    discord_admin_set_installation_state_impl,
    discord_admin_set_policy_impl,
    discord_oauth_callback_impl,
    discord_oauth_start_impl,
)
from tldw_Server_API.app.api.v1.endpoints.discord_support import (
    _INTERACTION_RECEIPTS,
    _RATE_LIMITER,
    _coerce_nonempty_string,
    _decrypt_discord_payload,
    _dedupe_ttl_seconds,
    _discord_oauth_token_exchange,
    _discord_policy_error_response,
    _discord_policy_for_guild,
    _discord_response_mode,
    _encrypt_discord_payload,
    _error_response,
    _evaluate_discord_policy,
    _get_job_manager,
    _get_oauth_state_repo,
    _get_user_secret_repo,
    _ingress_rate_limit_per_minute,
    _interaction_dedupe_key,
    _normalize_installations_payload,
    _oauth_auth_url,
    _oauth_client_id,
    _oauth_client_secret,
    _oauth_permissions,
    _oauth_redirect_uri,
    _oauth_scope,
    _oauth_state_ttl_seconds,
    _oauth_token_url,
    _parse_discord_interaction_command,
    _public_installation_record,
    _rate_limit_key,
    _reset_discord_state_for_tests,
    _resolve_discord_actor_id,
    _safe_int,
    _set_discord_policy,
    _verify_discord_signature,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_org_memberships_for_user
from tldw_Server_API.app.core.AuthNZ.repos import (
    get_workspace_provider_installations_repo as _get_workspace_provider_installations_repo_impl,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter

router = APIRouter(prefix="/discord", tags=["discord"])


def _emit_discord_counter(metric_name: str, **labels: Any) -> None:
    try:
        log_counter(metric_name, labels=_chatops_policy.metric_labels(**labels))
    except Exception:
        logger.debug("Failed to emit Discord metric")


async def _get_workspace_provider_installations_repo():
    return await _get_workspace_provider_installations_repo_impl()


async def _resolve_workspace_org_id(request: Request | None, user_id: int) -> int:
    return await _chatops_ingress.resolve_workspace_org_id(
        request,
        user_id,
        list_memberships=list_org_memberships_for_user,
        safe_int=_safe_int,
        auth_mode=str(getattr(get_settings(), "AUTH_MODE", "")),
    )


def _enqueue_discord_job(
    *,
    payload: dict[str, Any],
    parsed_command: dict[str, Any],
    owner_user_id: str | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    owner = _coerce_nonempty_string(owner_user_id)
    if not owner and isinstance(payload.get("member"), dict):
        owner = _coerce_nonempty_string(payload.get("member", {}).get("user", {}).get("id"))
    return _chatops_ingress.submit_job(
        _get_job_manager(),
        domain="discord",
        action=str(parsed_command.get("action") or "ask"),
        request_id=_coerce_nonempty_string(payload.get("id")) or secrets.token_urlsafe(12),
        owner_user_id=owner,
        payload={
            "application_id": _coerce_nonempty_string(payload.get("application_id")),
            "guild_id": _coerce_nonempty_string(payload.get("guild_id")),
            "channel_id": _coerce_nonempty_string(payload.get("channel_id")),
            "command": parsed_command,
        },
        response_mode=_discord_response_mode(payload, policy),
        safe_int=_safe_int,
    )


@router.post("/interactions")
async def discord_interactions(request: Request) -> JSONResponse:
    raw_body = await request.body()
    ok, error = _verify_discord_signature(
        raw_body,
        request.headers.get("x-signature-timestamp"),
        request.headers.get("x-signature-ed25519"),
    )
    if not ok:
        status = 503 if error == "public_key_not_configured" else 401
        _emit_discord_counter(
            "discord_signature_failures_total",
            endpoint="interactions",
            reason=error or "unknown",
        )
        return _error_response(status, str(error or "invalid_request"), "Discord request verification failed")

    try:
        payload = await request.json()
    except Exception:
        return _error_response(400, "invalid_json", "Invalid JSON payload")

    if not isinstance(payload, dict):
        return _error_response(400, "invalid_payload", "Payload must be a JSON object")

    allowed, retry_after = _RATE_LIMITER.allow(
        _rate_limit_key(payload, request),
        _ingress_rate_limit_per_minute(),
    )
    if not allowed:
        _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="rate_limited")
        return _chatops_ingress.rate_limited_response(retry_after)

    interaction_type = payload.get("type")
    if interaction_type == 1:
        _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="accepted", action="ping")
        return JSONResponse(status_code=200, content={"type": 1})

    dedupe_key = _interaction_dedupe_key(payload, raw_body)
    is_duplicate = _INTERACTION_RECEIPTS.seen_or_store(dedupe_key, _dedupe_ttl_seconds())
    if is_duplicate:
        _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="duplicate")
        return _chatops_ingress.duplicate_response()

    if interaction_type == 2:
        parsed_command, parse_error = _parse_discord_interaction_command(payload)
        if parse_error:
            _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="invalid_command")
            return JSONResponse(status_code=400, content={"ok": False, **parse_error})
        action = str(parsed_command.get("action") or "")
        guild_id = _coerce_nonempty_string(payload.get("guild_id"))
        channel_id = _coerce_nonempty_string(payload.get("channel_id"))
        member = payload.get("member") if isinstance(payload.get("member"), dict) else {}
        request_user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
        discord_user_id = _coerce_nonempty_string(
            member.get("user", {}).get("id") if isinstance(member.get("user"), dict) else None
        ) or _coerce_nonempty_string(request_user.get("id"))
        policy = _discord_policy_for_guild(guild_id)
        actor_user_id, mapping_error = _resolve_discord_actor_id(policy, discord_user_id)
        if mapping_error:
            return _discord_policy_error_response(mapping_error, guild_id=guild_id, action=action)

        policy_error = _evaluate_discord_policy(
            policy=policy,
            guild_id=guild_id,
            channel_id=channel_id,
            actor_user_id=actor_user_id,
            action=action,
        )
        if policy_error:
            return _discord_policy_error_response(policy_error, guild_id=guild_id, action=action)

        logger.bind(
            integration="discord",
            guild_id=guild_id or "na",
            channel_id=channel_id or "na",
            command=action or "na",
            interaction_id=_coerce_nonempty_string(payload.get("id")) or "na",
            actor_user_id=actor_user_id or "na",
        ).info("Discord interaction accepted")

        if action in {"ask", "rag", "summarize"} and not bool(parsed_command.get("inferred")):
            enqueued = _enqueue_discord_job(
                payload=payload,
                parsed_command=parsed_command,
                owner_user_id=actor_user_id,
                policy=policy,
            )
            _emit_discord_counter("discord_jobs_enqueued_total", action=action, guild_id=guild_id or "na")
            _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="queued", action=action)
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "status": "queued",
                    "parsed": parsed_command,
                    **enqueued,
                },
            )

        if action == "status":
            return _chatops_ingress.status_command_response(
                _get_job_manager(),
                parsed_command=parsed_command,
                policy=policy,
                tenant_field="guild_id",
                tenant_id=guild_id,
                actor_user_id=actor_user_id,
                emit=_emit_discord_counter,
                requests_metric="discord_requests_total",
                endpoint="interactions",
                coerce=_coerce_nonempty_string,
                safe_int=_safe_int,
            )

        _emit_discord_counter(
            "discord_requests_total", endpoint="interactions", outcome="accepted", action=action or "na"
        )
        return JSONResponse(status_code=200, content={"ok": True, "status": "accepted", "parsed": parsed_command})

    _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="accepted")
    return JSONResponse(status_code=200, content={"ok": True, "status": "accepted"})


@router.get(
    "/jobs/{job_id}",
    dependencies=[Depends(RequireRole("admin"))],
)
async def discord_job_status(
    job_id: int,
    user: User = Depends(get_request_user),
):
    return await _chatops_ingress.job_status_payload(
        _get_job_manager(),
        job_id,
        domain="discord",
        tenant_field="guild_id",
        user_id=int(user.id),
        list_memberships=list_org_memberships_for_user,
        get_installations_repo=_get_workspace_provider_installations_repo,
        policy_for=_discord_policy_for_guild,
        coerce=_coerce_nonempty_string,
        auth_mode=str(getattr(get_settings(), "AUTH_MODE", "")),
    )


@router.post("/oauth/start")
async def discord_oauth_start(
    request: Request,
    user: User = Depends(get_request_user),
):
    workspace_org_id = await _resolve_workspace_org_id(request, int(user.id))
    return await discord_oauth_start_impl(
        user=user,
        workspace_org_id=workspace_org_id,
        oauth_client_id=_oauth_client_id,
        oauth_redirect_uri=_oauth_redirect_uri,
        oauth_state_ttl_seconds=_oauth_state_ttl_seconds,
        get_oauth_state_repo=_get_oauth_state_repo,
        encrypt_discord_payload=_encrypt_discord_payload,
        oauth_auth_url=_oauth_auth_url,
        oauth_scope=_oauth_scope,
        oauth_permissions=_oauth_permissions,
        urlencode_fn=urlencode,
    )


@router.get("/oauth/callback")
async def discord_oauth_callback(
    code: str,
    state: str,
    guild_id: str | None = Query(default=None),
    guild_name: str | None = Query(default=None),
):
    return await discord_oauth_callback_impl(
        code=code,
        state=state,
        guild_id=guild_id,
        guild_name=guild_name,
        coerce_nonempty_string=_coerce_nonempty_string,
        get_oauth_state_repo=_get_oauth_state_repo,
        oauth_client_id=_oauth_client_id,
        oauth_client_secret=_oauth_client_secret,
        oauth_token_url=_oauth_token_url,
        discord_oauth_token_exchange=_discord_oauth_token_exchange,
        get_user_secret_repo=_get_user_secret_repo,
        get_workspace_provider_installations_repo=_get_workspace_provider_installations_repo,
        resolve_workspace_org_id=lambda user_id: _resolve_workspace_org_id(None, user_id),
        decrypt_discord_payload=_decrypt_discord_payload,
        normalize_installations_payload=_normalize_installations_payload,
        encrypt_discord_payload=_encrypt_discord_payload,
    )


@router.get(
    "/admin/policy",
    dependencies=[Depends(RequireRole("admin"))],
)
async def discord_admin_get_policy(
    guild_id: str | None = Query(default=None),
):
    return discord_admin_get_policy_impl(
        guild_id=guild_id,
        coerce_nonempty_string=_coerce_nonempty_string,
        discord_policy_for_guild=_discord_policy_for_guild,
    )


@router.put(
    "/admin/policy",
    dependencies=[Depends(RequireRole("admin"))],
)
async def discord_admin_set_policy(
    payload: dict[str, Any] | None = None,
):
    return discord_admin_set_policy_impl(
        payload=payload,
        coerce_nonempty_string=_coerce_nonempty_string,
        set_discord_policy=_set_discord_policy,
        emit_discord_counter=_emit_discord_counter,
    )


@router.get("/admin/installations", dependencies=[Depends(RequireRole("admin"))])
async def discord_admin_list_installations(
    user: User = Depends(get_request_user),
):
    return await discord_admin_list_installations_impl(
        user=user,
        get_user_secret_repo=_get_user_secret_repo,
        decrypt_discord_payload=_decrypt_discord_payload,
        normalize_installations_payload=_normalize_installations_payload,
        public_installation_record=_public_installation_record,
    )


@router.delete("/admin/installations/{guild_id}", dependencies=[Depends(RequireRole("admin"))])
async def discord_admin_delete_installation(
    request: Request,
    guild_id: str,
    user: User = Depends(get_request_user),
):
    return await discord_admin_delete_installation_impl(
        guild_id=guild_id,
        user=user,
        coerce_nonempty_string=_coerce_nonempty_string,
        get_user_secret_repo=_get_user_secret_repo,
        get_workspace_provider_installations_repo=_get_workspace_provider_installations_repo,
        resolve_workspace_org_id=lambda resolved_user_id: _resolve_workspace_org_id(request, resolved_user_id),
        decrypt_discord_payload=_decrypt_discord_payload,
        normalize_installations_payload=_normalize_installations_payload,
        encrypt_discord_payload=_encrypt_discord_payload,
    )


@router.put("/admin/installations/{guild_id}", dependencies=[Depends(RequireRole("admin"))])
async def discord_admin_set_installation_state(
    request: Request,
    guild_id: str,
    payload: dict[str, Any] | None = None,
    user: User = Depends(get_request_user),
):
    return await discord_admin_set_installation_state_impl(
        guild_id=guild_id,
        payload=payload,
        user=user,
        coerce_nonempty_string=_coerce_nonempty_string,
        get_user_secret_repo=_get_user_secret_repo,
        get_workspace_provider_installations_repo=_get_workspace_provider_installations_repo,
        resolve_workspace_org_id=lambda resolved_user_id: _resolve_workspace_org_id(request, resolved_user_id),
        decrypt_discord_payload=_decrypt_discord_payload,
        normalize_installations_payload=_normalize_installations_payload,
        encrypt_discord_payload=_encrypt_discord_payload,
    )
