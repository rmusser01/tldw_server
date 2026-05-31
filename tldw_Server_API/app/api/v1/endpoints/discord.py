from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, RequireRole, User

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


def _metric_labels(**labels: Any) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in labels.items():
        if value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized


def _emit_discord_counter(metric_name: str, **labels: Any) -> None:
    try:
        log_counter(metric_name, labels=_metric_labels(**labels))
    except Exception:
        logger.debug("Failed to emit Discord metric")


async def _get_workspace_provider_installations_repo():
    return await _get_workspace_provider_installations_repo_impl()


async def _resolve_workspace_org_id(request: Request | None, user_id: int) -> int:
    if request is not None:
        active_org_id = _safe_int(getattr(request.state, "active_org_id", None))
        if active_org_id is not None and active_org_id > 0:
            return active_org_id
        request_org_ids = getattr(request.state, "org_ids", None)
        if isinstance(request_org_ids, (list, tuple, set)):
            for candidate in request_org_ids:
                org_id = _safe_int(candidate)
                if org_id is not None and org_id > 0:
                    return org_id

    settings = get_settings()
    if str(getattr(settings, "AUTH_MODE", "")).strip().lower() == "single_user":
        return 1

    memberships = await list_org_memberships_for_user(int(user_id))
    for membership in memberships or []:
        try:
            org_id = int((membership or {}).get("org_id"))
        except (TypeError, ValueError):
            continue
        if org_id <= 0:
            continue
        status_value = str((membership or {}).get("status") or "").strip().lower()
        if not status_value or status_value in {"active", "member", "approved"}:
            return org_id

    for membership in memberships or []:
        try:
            org_id = int((membership or {}).get("org_id"))
        except (TypeError, ValueError):
            continue
        if org_id > 0:
            return org_id

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unable to resolve workspace organization for installation",
    )


def _discord_policy_error_response(
    policy_error: dict[str, Any], *, guild_id: str | None, action: str | None
) -> JSONResponse:
    status_code = int(policy_error.get("status_code") or status.HTTP_403_FORBIDDEN)
    response_payload = {k: v for k, v in policy_error.items() if k != "status_code"}
    headers: dict[str, str] = {}
    retry_after = _safe_int(policy_error.get("retry_after_seconds"))
    if retry_after is not None and retry_after > 0:
        headers["Retry-After"] = str(retry_after)
        _emit_discord_counter(
            "discord_policy_quota_rejections_total",
            guild_id=guild_id or "na",
            action=action or "na",
            error=response_payload.get("error"),
        )
    else:
        _emit_discord_counter(
            "discord_policy_denied_total",
            guild_id=guild_id or "na",
            action=action or "na",
            error=response_payload.get("error"),
        )
    logger.warning(
        "Discord policy denied request: guild_id={} action={} error={}",
        guild_id or "na",
        action or "na",
        response_payload.get("error"),
    )
    return JSONResponse(status_code=status_code, headers=headers, content={"ok": False, **response_payload})


def _enqueue_discord_job(
    *,
    payload: dict[str, Any],
    parsed_command: dict[str, Any],
    owner_user_id: str | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    jm = _get_job_manager()
    request_id = _coerce_nonempty_string(payload.get("id")) or secrets.token_urlsafe(12)
    owner = _coerce_nonempty_string(owner_user_id)
    if not owner and isinstance(payload.get("member"), dict):
        owner = _coerce_nonempty_string(payload.get("member", {}).get("user", {}).get("id"))
    response_mode = _discord_response_mode(payload, policy)
    action = str(parsed_command.get("action") or "ask")
    job = jm.create_job(
        domain="discord",
        queue="default",
        job_type=f"discord_{action}",
        payload={
            "request_id": request_id,
            "application_id": _coerce_nonempty_string(payload.get("application_id")),
            "guild_id": _coerce_nonempty_string(payload.get("guild_id")),
            "channel_id": _coerce_nonempty_string(payload.get("channel_id")),
            "command": parsed_command,
            "response_mode": response_mode,
        },
        owner_user_id=owner,
        request_id=request_id,
    )
    job_id = _safe_int(job.get("id"))
    return {
        "job_id": job_id,
        "request_id": request_id,
        "response_mode": response_mode,
        "job_status": str(job.get("status") or "queued"),
    }


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
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={"ok": False, "error": "rate_limited", "retry_after_seconds": retry_after},
        )

    interaction_type = payload.get("type")
    if interaction_type == 1:
        _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="accepted", action="ping")
        return JSONResponse(status_code=200, content={"type": 1})

    dedupe_key = _interaction_dedupe_key(payload, raw_body)
    is_duplicate = _INTERACTION_RECEIPTS.seen_or_store(dedupe_key, _dedupe_ttl_seconds())
    if is_duplicate:
        _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="duplicate")
        return JSONResponse(status_code=200, content={"ok": True, "status": "duplicate"})

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
            jm = _get_job_manager()
            requested_job_id = _safe_int(parsed_command.get("input"))
            if requested_job_id is None:
                _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="invalid_status_query")
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "error": "invalid_status_query",
                        "message": "Status command requires a numeric job id. Example: status 42",
                    },
                )
            job = jm.get_job(requested_job_id)
            job_payload = job.get("payload") if isinstance(job, dict) and isinstance(job.get("payload"), dict) else {}
            job_guild_id = _coerce_nonempty_string(job_payload.get("guild_id"))
            owner_user_id = _coerce_nonempty_string(job.get("owner_user_id")) if isinstance(job, dict) else None
            status_scope = str(policy.get("status_scope") or "guild").strip().lower()
            wrong_guild = bool(job_guild_id and guild_id and job_guild_id != guild_id)
            wrong_user_scope = bool(
                status_scope == "guild_and_user" and actor_user_id and owner_user_id and actor_user_id != owner_user_id
            )
            if not job or wrong_guild or wrong_user_scope:
                _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="status_denied")
                return JSONResponse(
                    status_code=404,
                    content={"ok": False, "error": "job_not_found", "job_id": requested_job_id},
                )
            _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="accepted", action=action)
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "status": "accepted",
                    "parsed": parsed_command,
                    "job": {
                        "id": requested_job_id,
                        "status": job.get("status"),
                        "domain": job.get("domain"),
                        "queue": job.get("queue"),
                        "job_type": job.get("job_type"),
                    },
                },
            )

        _emit_discord_counter(
            "discord_requests_total", endpoint="interactions", outcome="accepted", action=action or "na"
        )
        return JSONResponse(status_code=200, content={"ok": True, "status": "accepted", "parsed": parsed_command})

    _emit_discord_counter("discord_requests_total", endpoint="interactions", outcome="accepted")
    return JSONResponse(status_code=200, content={"ok": True, "status": "accepted"})


@router.get("/jobs/{job_id}")
async def discord_job_status(
    job_id: int,
):
    jm = _get_job_manager()
    job = jm.get_job(int(job_id))
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
    if str(job.get("domain") or "").strip().lower() != "discord":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
    return {
        "ok": True,
        "job": {
            "id": int(job.get("id") or job_id),
            "status": job.get("status"),
            "domain": job.get("domain"),
            "queue": job.get("queue"),
            "job_type": job.get("job_type"),
        },
    }


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
