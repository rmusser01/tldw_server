from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any
from urllib.parse import parse_qs, urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, RequireRole, User

from tldw_Server_API.app.api.v1.endpoints.slack_oauth_admin import (
    slack_admin_delete_installation_impl,
    slack_admin_get_policy_impl,
    slack_admin_list_installations_impl,
    slack_admin_set_installation_state_impl,
    slack_admin_set_policy_impl,
    slack_oauth_callback_impl,
    slack_oauth_start_impl,
)
from tldw_Server_API.app.api.v1.endpoints.slack_support import (
    _COMMAND_RECEIPTS,
    _EVENT_RECEIPTS,
    _RATE_LIMITER,
    _coerce_nonempty_string,
    _command_fingerprint,
    _decrypt_slack_payload,
    _dedupe_ttl_seconds,
    _encrypt_slack_payload,
    _error_response,
    _evaluate_slack_policy,
    _get_job_manager,
    _get_oauth_state_repo,
    _get_user_secret_repo,
    _ingress_rate_limit_per_minute,
    _is_bot_event,
    _normalize_installations_payload,
    _oauth_auth_url,
    _oauth_client_id,
    _oauth_client_secret,
    _oauth_redirect_uri,
    _oauth_scopes,
    _oauth_state_ttl_seconds,
    _oauth_token_url,
    _parse_slack_command,
    _parse_slack_mention,
    _public_installation_record,
    _rate_limit_key_for_commands,
    _rate_limit_key_for_events,
    _reset_slack_state_for_tests,
    _resolve_slack_actor_id,
    _safe_int,
    _set_slack_policy,
    _slack_oauth_token_exchange,
    _slack_policy_for_workspace,
    _slack_response_mode,
    _verify_slack_signature,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_org_memberships_for_user
from tldw_Server_API.app.core.AuthNZ.repos import (
    get_workspace_provider_installations_repo as _get_workspace_provider_installations_repo_impl,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter

router = APIRouter(prefix="/slack", tags=["slack"])


def _metric_labels(**labels: Any) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in labels.items():
        if value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized


def _emit_slack_counter(metric_name: str, **labels: Any) -> None:
    try:
        log_counter(metric_name, labels=_metric_labels(**labels))
    except Exception:
        logger.debug("Failed to emit Slack metric")


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


def _slack_policy_error_response(
    policy_error: dict[str, Any], *, team_id: str | None, action: str | None
) -> JSONResponse:
    status_code = int(policy_error.get("status_code") or status.HTTP_403_FORBIDDEN)
    response_payload = {k: v for k, v in policy_error.items() if k != "status_code"}
    headers: dict[str, str] = {}
    retry_after = _safe_int(policy_error.get("retry_after_seconds"))
    if retry_after is not None and retry_after > 0:
        headers["Retry-After"] = str(retry_after)
        _emit_slack_counter(
            "slack_policy_quota_rejections_total",
            team_id=team_id or "na",
            action=action or "na",
            error=response_payload.get("error"),
        )
    else:
        _emit_slack_counter(
            "slack_policy_denied_total",
            team_id=team_id or "na",
            action=action or "na",
            error=response_payload.get("error"),
        )
    logger.warning(
        "Slack policy denied request: team_id={} action={} error={}",
        team_id or "na",
        action or "na",
        response_payload.get("error"),
    )
    return JSONResponse(status_code=status_code, headers=headers, content={"ok": False, **response_payload})


def _enqueue_slack_job(
    *,
    form_payload: dict[str, str],
    parsed_command: dict[str, Any],
    owner_user_id: str | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    jm = _get_job_manager()
    request_id = _coerce_nonempty_string(form_payload.get("trigger_id")) or secrets.token_urlsafe(12)
    owner = _coerce_nonempty_string(owner_user_id) or _coerce_nonempty_string(form_payload.get("user_id")) or None
    action = str(parsed_command.get("action") or "ask")
    response_mode = _slack_response_mode(form_payload, policy)
    job = jm.create_job(
        domain="slack",
        queue="default",
        job_type=f"slack_{action}",
        payload={
            "request_id": request_id,
            "team_id": _coerce_nonempty_string(form_payload.get("team_id")),
            "channel_id": _coerce_nonempty_string(form_payload.get("channel_id")),
            "thread_ts": _coerce_nonempty_string(form_payload.get("thread_ts")),
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


@router.post("/events")
async def slack_events(request: Request) -> JSONResponse:
    raw_body = await request.body()
    ok, error = _verify_slack_signature(
        raw_body,
        request.headers.get("x-slack-request-timestamp"),
        request.headers.get("x-slack-signature"),
    )
    if not ok:
        status = 503 if error == "signing_secret_not_configured" else 401
        _emit_slack_counter(
            "slack_signature_failures_total",
            endpoint="events",
            reason=error or "unknown",
        )
        return _error_response(status, str(error or "invalid_request"), "Slack request verification failed")

    try:
        payload = json.loads(raw_body.decode("utf-8") or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error_response(400, "invalid_json", "Invalid JSON payload")

    if not isinstance(payload, dict):
        return _error_response(400, "invalid_payload", "Payload must be a JSON object")

    allowed, retry_after = _RATE_LIMITER.allow(
        _rate_limit_key_for_events(payload, request),
        _ingress_rate_limit_per_minute(),
    )
    if not allowed:
        _emit_slack_counter("slack_requests_total", endpoint="events", outcome="rate_limited")
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={"ok": False, "error": "rate_limited", "retry_after_seconds": retry_after},
        )

    event_type = str(payload.get("type") or "").strip()
    if event_type == "url_verification":
        challenge = payload.get("challenge")
        if not isinstance(challenge, str) or not challenge:
            return _error_response(400, "missing_challenge", "Missing challenge")
        return JSONResponse(status_code=200, content={"challenge": challenge})

    if event_type == "event_callback":
        event_id = str(payload.get("event_id") or "").strip()
        dedupe_key = event_id or hashlib.sha256(raw_body).hexdigest()
        is_duplicate = _EVENT_RECEIPTS.seen_or_store(dedupe_key, _dedupe_ttl_seconds())
        if is_duplicate:
            _emit_slack_counter("slack_requests_total", endpoint="events", outcome="duplicate")
            return JSONResponse(status_code=200, content={"ok": True, "status": "duplicate"})

        if _is_bot_event(payload):
            _emit_slack_counter("slack_requests_total", endpoint="events", outcome="ignored_bot_event")
            return JSONResponse(status_code=200, content={"ok": True, "status": "ignored_bot_event"})

        mention_parsed = _parse_slack_mention(payload)
        if mention_parsed:
            event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
            team_id = _coerce_nonempty_string(payload.get("team_id"))
            channel_id = _coerce_nonempty_string(event.get("channel")) if isinstance(event, dict) else None
            slack_user_id = _coerce_nonempty_string(event.get("user")) if isinstance(event, dict) else None
            policy = _slack_policy_for_workspace(team_id)
            actor_user_id, mapping_error = _resolve_slack_actor_id(policy, slack_user_id)
            if mapping_error:
                return _slack_policy_error_response(
                    mapping_error,
                    team_id=team_id,
                    action=str(mention_parsed.get("action") or ""),
                )
            policy_error = _evaluate_slack_policy(
                policy=policy,
                team_id=team_id,
                channel_id=channel_id,
                actor_user_id=actor_user_id,
                action=str(mention_parsed.get("action") or ""),
            )
            if policy_error:
                return _slack_policy_error_response(
                    policy_error,
                    team_id=team_id,
                    action=str(mention_parsed.get("action") or ""),
                )
            _emit_slack_counter(
                "slack_requests_total",
                endpoint="events",
                outcome="accepted",
                action=str(mention_parsed.get("action") or "na"),
            )
            return JSONResponse(
                status_code=200,
                content={"ok": True, "status": "accepted", "parsed": mention_parsed},
            )
        _emit_slack_counter("slack_requests_total", endpoint="events", outcome="accepted")
        return JSONResponse(status_code=200, content={"ok": True, "status": "accepted"})

    _emit_slack_counter("slack_requests_total", endpoint="events", outcome="accepted")
    return JSONResponse(status_code=200, content={"ok": True, "status": "accepted"})


@router.post("/commands")
async def slack_commands(request: Request) -> JSONResponse:
    raw_body = await request.body()
    ok, error = _verify_slack_signature(
        raw_body,
        request.headers.get("x-slack-request-timestamp"),
        request.headers.get("x-slack-signature"),
    )
    if not ok:
        status = 503 if error == "signing_secret_not_configured" else 401
        _emit_slack_counter(
            "slack_signature_failures_total",
            endpoint="commands",
            reason=error or "unknown",
        )
        return _error_response(status, str(error or "invalid_request"), "Slack request verification failed")

    try:
        parsed = parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
    except UnicodeDecodeError:
        return _error_response(400, "invalid_form", "Unable to parse form body")

    form_payload = {k: (v[0] if v else "") for k, v in parsed.items()}
    allowed, retry_after = _RATE_LIMITER.allow(
        _rate_limit_key_for_commands(form_payload, request),
        _ingress_rate_limit_per_minute(),
    )
    if not allowed:
        _emit_slack_counter("slack_requests_total", endpoint="commands", outcome="rate_limited")
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={"ok": False, "error": "rate_limited", "retry_after_seconds": retry_after},
        )

    dedupe_key = _command_fingerprint(raw_body)
    is_duplicate = _COMMAND_RECEIPTS.seen_or_store(dedupe_key, _dedupe_ttl_seconds())
    if is_duplicate:
        _emit_slack_counter("slack_requests_total", endpoint="commands", outcome="duplicate")
        return JSONResponse(status_code=200, content={"ok": True, "status": "duplicate"})

    parsed_command, parse_error = _parse_slack_command(form_payload)
    if parse_error:
        _emit_slack_counter("slack_requests_total", endpoint="commands", outcome="invalid_command")
        return JSONResponse(
            status_code=400,
            content={"ok": False, **parse_error},
        )
    action = str(parsed_command.get("action") or "")
    team_id = _coerce_nonempty_string(form_payload.get("team_id")) or _coerce_nonempty_string(
        form_payload.get("team_domain")
    )
    channel_id = _coerce_nonempty_string(form_payload.get("channel_id"))
    slack_user_id = _coerce_nonempty_string(form_payload.get("user_id"))
    policy = _slack_policy_for_workspace(team_id)
    actor_user_id, mapping_error = _resolve_slack_actor_id(policy, slack_user_id)
    if mapping_error:
        return _slack_policy_error_response(mapping_error, team_id=team_id, action=action)

    policy_error = _evaluate_slack_policy(
        policy=policy,
        team_id=team_id,
        channel_id=channel_id,
        actor_user_id=actor_user_id,
        action=action,
    )
    if policy_error:
        return _slack_policy_error_response(policy_error, team_id=team_id, action=action)

    logger.bind(
        integration="slack",
        workspace_id=team_id or "na",
        channel_id=channel_id or "na",
        command=action or "na",
        request_id=_coerce_nonempty_string(form_payload.get("trigger_id")) or "na",
        actor_user_id=actor_user_id or "na",
    ).info("Slack command accepted")

    if action in {"ask", "rag", "summarize"}:
        enqueued = _enqueue_slack_job(
            form_payload=form_payload,
            parsed_command=parsed_command,
            owner_user_id=actor_user_id,
            policy=policy,
        )
        _emit_slack_counter("slack_jobs_enqueued_total", action=action, team_id=team_id or "na")
        _emit_slack_counter("slack_requests_total", endpoint="commands", outcome="queued", action=action)
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
            _emit_slack_counter("slack_requests_total", endpoint="commands", outcome="invalid_status_query")
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
        job_team_id = _coerce_nonempty_string(job_payload.get("team_id"))
        owner_user_id = _coerce_nonempty_string(job.get("owner_user_id")) if isinstance(job, dict) else None
        status_scope = str(policy.get("status_scope") or "workspace").strip().lower()
        wrong_workspace = bool(job_team_id and team_id and job_team_id != team_id)
        wrong_user_scope = bool(
            status_scope == "workspace_and_user" and actor_user_id and owner_user_id and actor_user_id != owner_user_id
        )
        if not job or wrong_workspace or wrong_user_scope:
            _emit_slack_counter("slack_requests_total", endpoint="commands", outcome="status_denied")
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": "job_not_found", "job_id": requested_job_id},
            )
        _emit_slack_counter("slack_requests_total", endpoint="commands", outcome="accepted", action=action)
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

    _emit_slack_counter("slack_requests_total", endpoint="commands", outcome="accepted", action=action or "na")
    return JSONResponse(
        status_code=200,
        content={"ok": True, "status": "accepted", "parsed": parsed_command},
    )


@router.get("/jobs/{job_id}")
async def slack_job_status(
    job_id: int,
):
    jm = _get_job_manager()
    job = jm.get_job(int(job_id))
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
    if str(job.get("domain") or "").strip().lower() != "slack":
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
async def slack_oauth_start(
    request: Request,
    user: User = Depends(get_request_user),
):
    workspace_org_id = await _resolve_workspace_org_id(request, int(user.id))
    return await slack_oauth_start_impl(
        user=user,
        workspace_org_id=workspace_org_id,
        oauth_client_id=_oauth_client_id,
        oauth_redirect_uri=_oauth_redirect_uri,
        oauth_state_ttl_seconds=_oauth_state_ttl_seconds,
        get_oauth_state_repo=_get_oauth_state_repo,
        encrypt_slack_payload=_encrypt_slack_payload,
        oauth_auth_url=_oauth_auth_url,
        oauth_scopes=_oauth_scopes,
        urlencode_fn=urlencode,
    )


@router.get("/oauth/callback")
async def slack_oauth_callback(
    code: str,
    state: str,
):
    return await slack_oauth_callback_impl(
        code=code,
        state=state,
        coerce_nonempty_string=_coerce_nonempty_string,
        get_oauth_state_repo=_get_oauth_state_repo,
        oauth_client_id=_oauth_client_id,
        oauth_client_secret=_oauth_client_secret,
        oauth_token_url=_oauth_token_url,
        slack_oauth_token_exchange=_slack_oauth_token_exchange,
        get_user_secret_repo=_get_user_secret_repo,
        get_workspace_provider_installations_repo=_get_workspace_provider_installations_repo,
        resolve_workspace_org_id=lambda user_id: _resolve_workspace_org_id(None, user_id),
        decrypt_slack_payload=_decrypt_slack_payload,
        normalize_installations_payload=_normalize_installations_payload,
        encrypt_slack_payload=_encrypt_slack_payload,
    )


@router.get(
    "/admin/policy",
    dependencies=[Depends(RequireRole("admin"))],
)
async def slack_admin_get_policy(
    team_id: str | None = Query(default=None),
):
    return slack_admin_get_policy_impl(
        team_id=team_id,
        coerce_nonempty_string=_coerce_nonempty_string,
        slack_policy_for_workspace=_slack_policy_for_workspace,
    )


@router.put(
    "/admin/policy",
    dependencies=[Depends(RequireRole("admin"))],
)
async def slack_admin_set_policy(
    payload: dict[str, Any] | None = None,
):
    return slack_admin_set_policy_impl(
        payload=payload,
        coerce_nonempty_string=_coerce_nonempty_string,
        set_slack_policy=_set_slack_policy,
        emit_slack_counter=_emit_slack_counter,
    )


@router.get("/admin/installations", dependencies=[Depends(RequireRole("admin"))])
async def slack_admin_list_installations(
    user: User = Depends(get_request_user),
):
    return await slack_admin_list_installations_impl(
        user=user,
        get_user_secret_repo=_get_user_secret_repo,
        decrypt_slack_payload=_decrypt_slack_payload,
        normalize_installations_payload=_normalize_installations_payload,
        public_installation_record=_public_installation_record,
    )


@router.delete("/admin/installations/{team_id}", dependencies=[Depends(RequireRole("admin"))])
async def slack_admin_delete_installation(
    request: Request,
    team_id: str,
    user: User = Depends(get_request_user),
):
    return await slack_admin_delete_installation_impl(
        team_id=team_id,
        user=user,
        coerce_nonempty_string=_coerce_nonempty_string,
        get_user_secret_repo=_get_user_secret_repo,
        get_workspace_provider_installations_repo=_get_workspace_provider_installations_repo,
        resolve_workspace_org_id=lambda resolved_user_id: _resolve_workspace_org_id(request, resolved_user_id),
        decrypt_slack_payload=_decrypt_slack_payload,
        normalize_installations_payload=_normalize_installations_payload,
        encrypt_slack_payload=_encrypt_slack_payload,
    )


@router.put("/admin/installations/{team_id}", dependencies=[Depends(RequireRole("admin"))])
async def slack_admin_set_installation_state(
    request: Request,
    team_id: str,
    payload: dict[str, Any] | None = None,
    user: User = Depends(get_request_user),
):
    return await slack_admin_set_installation_state_impl(
        team_id=team_id,
        payload=payload,
        user=user,
        coerce_nonempty_string=_coerce_nonempty_string,
        get_user_secret_repo=_get_user_secret_repo,
        get_workspace_provider_installations_repo=_get_workspace_provider_installations_repo,
        resolve_workspace_org_id=lambda resolved_user_id: _resolve_workspace_org_id(request, resolved_user_id),
        decrypt_slack_payload=_decrypt_slack_payload,
        normalize_installations_payload=_normalize_installations_payload,
        encrypt_slack_payload=_encrypt_slack_payload,
    )
