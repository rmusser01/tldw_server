from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager as _global_get_job_manager
from tldw_Server_API.app.api.v1.endpoints._in_memory_limits import SlidingWindowLimiter, TTLReceiptStore
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.byok_oauth_state_repo import AuthnzByokOAuthStateRepo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    loads_envelope,
)
from tldw_Server_API.app.core.http_client import RetryPolicy as _RetryPolicy
from tldw_Server_API.app.core.http_client import afetch as _http_afetch
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter

from ._chatops import policy as _chatops_policy
from ._chatops.settings import ChatOpsSettings, coerce_nonempty_string, env_int

_EVENT_RECEIPTS = TTLReceiptStore()
_COMMAND_RECEIPTS = TTLReceiptStore()
_RATE_LIMITER = SlidingWindowLimiter()
_POLICY_RATE_LIMITER = SlidingWindowLimiter()


def _reset_slack_state_for_tests() -> None:
    _EVENT_RECEIPTS.clear()
    _COMMAND_RECEIPTS.clear()
    _RATE_LIMITER.clear()
    _POLICY_RATE_LIMITER.clear()
    _POLICY_STORE.clear()


_SETTINGS = ChatOpsSettings(
    provider="slack",
    env_prefix="SLACK",
    default_oauth_auth_url="https://slack.com/oauth/v2/authorize",
    default_oauth_token_url="https://slack.com/api/oauth.v2.access",
    installation_fields=('team_id', 'team_name', 'enterprise_id', 'bot_user_id'),
)
# The names the slack endpoints import; one implementation in _chatops/settings.py.
_env_int = env_int
_coerce_nonempty_string = coerce_nonempty_string
_replay_window_seconds = _SETTINGS.replay_window_seconds
_dedupe_ttl_seconds = _SETTINGS.dedupe_ttl_seconds
_ingress_rate_limit_per_minute = _SETTINGS.ingress_rate_limit_per_minute
_policy_user_quota_per_minute = _SETTINGS.policy_user_quota_per_minute
_oauth_client_id = _SETTINGS.oauth_client_id
_oauth_client_secret = _SETTINGS.oauth_client_secret
_oauth_redirect_uri = _SETTINGS.oauth_redirect_uri
_oauth_auth_url = _SETTINGS.oauth_auth_url
_oauth_token_url = _SETTINGS.oauth_token_url
_oauth_state_ttl_seconds = _SETTINGS.oauth_state_ttl_seconds
_default_installations_payload = _SETTINGS.default_installations_payload
_normalize_installations_payload = _SETTINGS.normalize_installations_payload
_public_installation_record = _SETTINGS.public_installation_record


def _signing_secret() -> str:
    return (os.getenv("SLACK_SIGNING_SECRET") or "").strip()


def _policy_workspace_quota_per_minute() -> int:
    return _env_int("SLACK_POLICY_WORKSPACE_QUOTA_PER_MINUTE", 120)


def _oauth_scopes() -> str:
    raw = _coerce_nonempty_string(os.getenv("SLACK_OAUTH_SCOPES")) or "commands,chat:write"
    scopes: list[str] = []
    for part in raw.replace(" ", ",").split(","):
        cleaned = part.strip()
        if cleaned and cleaned not in scopes:
            scopes.append(cleaned)
    return ",".join(scopes)


def _get_job_manager() -> Any:
    return _global_get_job_manager()


async def _close_http_response(response: Any) -> None:
    close_async = getattr(response, "aclose", None)
    if callable(close_async):
        await close_async()
        return
    close_sync = getattr(response, "close", None)
    if callable(close_sync):
        close_sync()


async def _get_oauth_state_repo() -> AuthnzByokOAuthStateRepo:
    pool = await get_db_pool()
    repo = AuthnzByokOAuthStateRepo(pool)
    await repo.ensure_tables()
    return repo


async def _get_user_secret_repo() -> AuthnzUserProviderSecretsRepo:
    pool = await get_db_pool()
    repo = AuthnzUserProviderSecretsRepo(pool)
    await repo.ensure_tables()
    return repo


def _encrypt_slack_payload(payload: dict[str, Any]) -> str:
    return dumps_envelope(encrypt_byok_payload(payload))


def _decrypt_slack_payload(encrypted_blob: str) -> dict[str, Any] | None:
    if not encrypted_blob:
        return None
    try:
        payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
    except Exception:
        logger.warning("Failed to decrypt Slack installation payload")
        return None
    return payload if isinstance(payload, dict) else None


async def _slack_oauth_token_exchange(*, token_url: str, form_data: dict[str, Any]) -> dict[str, Any]:
    response = await _http_afetch(
        method="POST",
        url=token_url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data=form_data,
        timeout=30,
        retry=_RetryPolicy(attempts=1),
    )
    try:
        status_code = int(getattr(response, "status_code", 0))
        payload: dict[str, Any] | None = None
        try:
            maybe_payload = response.json()
            if isinstance(maybe_payload, dict):
                payload = dict(maybe_payload)
        except Exception:
            payload = None

        if status_code < 200 or status_code >= 300:
            detail = "Slack OAuth token exchange failed"
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=detail,
            )
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Slack OAuth token exchange returned invalid payload",
            )
        return payload
    finally:
        await _close_http_response(response)


_error_response = _chatops_policy.error_response
_metric_labels = _chatops_policy.metric_labels


def _emit_slack_counter(metric_name: str, **labels: Any) -> None:
    try:
        log_counter(metric_name, labels=_metric_labels(**labels))
    except Exception:
        logger.debug("Failed to emit Slack metric")


def _extract_timestamp(header_value: str | None) -> int | None:
    if not header_value:
        return None
    try:
        return int(header_value.strip())
    except (TypeError, ValueError):
        return None


def _verify_slack_signature(
    raw_body: bytes, timestamp_header: str | None, signature_header: str | None
) -> tuple[bool, str | None]:
    secret = _signing_secret()
    if not secret:
        logger.warning("Slack signing secret is not configured")
        return False, "signing_secret_not_configured"

    timestamp = _extract_timestamp(timestamp_header)
    if timestamp is None:
        return False, "invalid_timestamp"

    now = int(time.time())
    if abs(now - timestamp) > max(1, _replay_window_seconds()):
        return False, "stale_request"

    if not signature_header or not signature_header.startswith("v0="):
        return False, "invalid_signature"

    base = f"v0:{timestamp}:".encode() + raw_body
    expected = "v0=" + hmac.new(secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_header.strip()):
        return False, "invalid_signature"

    return True, None


def _rate_limit_key_for_events(payload: dict, request: Request) -> str:
    team_id = payload.get("team_id")
    if isinstance(team_id, dict):
        team_id = team_id.get("id")
    app_id = payload.get("api_app_id")
    fallback = request.client.host if request.client else "unknown"
    return f"slack:events:{app_id or 'na'}:{team_id or fallback}"


def _rate_limit_key_for_commands(form_payload: dict[str, str], request: Request) -> str:
    team_id = form_payload.get("team_id") or form_payload.get("team_domain")
    app_id = form_payload.get("api_app_id") or "na"
    fallback = request.client.host if request.client else "unknown"
    return f"slack:commands:{app_id}:{team_id or fallback}"


def _is_bot_event(payload: dict) -> bool:
    event = payload.get("event")
    if not isinstance(event, dict):
        return False
    subtype = str(event.get("subtype") or "").strip().lower()
    return bool(event.get("bot_id") or subtype == "bot_message")


def _command_fingerprint(raw_body: bytes) -> str:
    return hashlib.sha256(raw_body).hexdigest()


_SUPPORTED_SLACK_ACTIONS = ("help", "ask", "rag", "summarize", "status")


def _slack_usage_text() -> str:
    return "Supported commands: help | ask <query> | rag <query> | summarize <text> | status"


def _normalize_string_list(raw: Any) -> list[str]:
    return _chatops_policy.normalize_string_list(raw, coerce=_coerce_nonempty_string)


_SLACK_POLICY_SPEC = _chatops_policy.ChatOpsPolicySpec(
    supported_actions=_SUPPORTED_SLACK_ACTIONS,
    scope_quota_field="workspace_quota_per_minute",
    scope_quota_default=_policy_workspace_quota_per_minute,
    user_quota_default=_policy_user_quota_per_minute,
    status_scope_values=("workspace", "workspace_and_user"),
    response_modes=("ephemeral", "thread", "channel"),
)


def _default_slack_policy() -> dict[str, Any]:
    return _chatops_policy.default_policy(_SLACK_POLICY_SPEC)


def _normalize_slack_policy_payload(
    payload: dict[str, Any] | None,
    *,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _chatops_policy.normalize_policy_payload(
        _SLACK_POLICY_SPEC,
        payload,
        base=base,
        coerce=_coerce_nonempty_string,
        safe_int=_safe_int,
    )


_POLICY_STORE = _chatops_policy.PolicyStore(
    normalize=_normalize_slack_policy_payload,
    default_policy=_default_slack_policy,
    coerce=_coerce_nonempty_string,
)
_slack_policy_for_workspace = _POLICY_STORE.get
_set_slack_policy = _POLICY_STORE.set


def _resolve_slack_actor_id(
    policy: dict[str, Any], slack_user_id: str | None
) -> tuple[str | None, dict[str, Any] | None]:
    return _chatops_policy.resolve_actor_id(
        policy, slack_user_id, provider_label="Slack", coerce=_coerce_nonempty_string, http_status=status
    )


_SLACK_POLICY_RUNTIME = _chatops_policy.ChatOpsPolicyRuntime(
    name="slack",
    scope_word="workspace",
    scope_quota_field="workspace_quota_per_minute",
    scope_quota_default=_policy_workspace_quota_per_minute,
    user_quota_default=_policy_user_quota_per_minute,
    scope_label_field="team_id",
    quota_rejection_counter="slack_policy_quota_rejections_total",
    denial_counter="slack_policy_denied_total",
)


def _evaluate_slack_policy(
    *,
    policy: dict[str, Any],
    team_id: str | None,
    channel_id: str | None,
    actor_user_id: str | None,
    action: str,
) -> dict[str, Any] | None:
    return _chatops_policy.evaluate_policy(
        _SLACK_POLICY_RUNTIME,
        policy=policy,
        scope_id=team_id,
        channel_id=channel_id,
        actor_user_id=actor_user_id,
        action=action,
        rate_limiter=_POLICY_RATE_LIMITER,
        coerce=_coerce_nonempty_string,
        safe_int=_safe_int,
        http_status=status,
    )


def _slack_policy_error_response(
    policy_error: dict[str, Any], *, team_id: str | None, action: str | None
) -> JSONResponse:
    return _chatops_policy.policy_error_response(
        _SLACK_POLICY_RUNTIME,
        policy_error,
        scope_id=team_id,
        action=action,
        emit_counter=_emit_slack_counter,
        log=logger,
        safe_int=_safe_int,
        http_status=status,
    )


def _slack_action_route(action: str) -> str:
    return _chatops_policy.action_route(_SLACK_POLICY_RUNTIME, action)


def _parse_slack_text_command(text: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw = (text or "").strip()
    if not raw:
        action = "help"
        return {
            "action": action,
            "route": _slack_action_route(action),
            "input": "",
        }, None

    parts = raw.split(maxsplit=1)
    command = parts[0].strip().lower()
    remainder = parts[1].strip() if len(parts) > 1 else ""
    if command not in _SUPPORTED_SLACK_ACTIONS:
        return None, {
            "error": "unknown_command",
            "message": f"Unknown command '{command}'. {_slack_usage_text()}",
            "usage": _slack_usage_text(),
        }
    return {
        "action": command,
        "route": _slack_action_route(command),
        "input": remainder,
    }, None


def _parse_slack_command(form_payload: dict[str, str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    text = form_payload.get("text")
    return _parse_slack_text_command(text)


def _parse_slack_mention(payload: dict[str, Any]) -> dict[str, Any] | None:
    event = payload.get("event")
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("type") or "").strip().lower()
    if event_type != "app_mention":
        return None

    text = str(event.get("text") or "").strip()
    if not text:
        action = "help"
        return {"action": action, "route": _slack_action_route(action), "input": ""}

    normalized = " ".join(part for part in text.split() if not part.startswith("<@")).strip()
    parsed, _error = _parse_slack_text_command(normalized)
    if parsed:
        return parsed

    # Mention defaults to ask when input does not start with a supported command.
    return {
        "action": "ask",
        "route": _slack_action_route("ask"),
        "input": normalized,
    }


def _slack_response_mode(form_payload: dict[str, str], policy: dict[str, Any] | None = None) -> str:
    raw = str(form_payload.get("response_mode") or "").strip().lower()
    if not raw and isinstance(policy, dict):
        raw = str(policy.get("default_response_mode") or "").strip().lower()
    if raw in {"ephemeral", "thread", "channel"}:
        return raw
    return "ephemeral"


def _safe_int(raw_value: Any) -> int | None:
    try:
        return int(str(raw_value).strip())
    except (TypeError, ValueError):
        return None


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
