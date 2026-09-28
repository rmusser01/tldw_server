from __future__ import annotations

import os
import secrets
import time
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
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

_INTERACTION_RECEIPTS = TTLReceiptStore()
_RATE_LIMITER = SlidingWindowLimiter()
_POLICY_RATE_LIMITER = SlidingWindowLimiter()


def _reset_discord_state_for_tests() -> None:
    _INTERACTION_RECEIPTS.clear()
    _RATE_LIMITER.clear()
    _POLICY_RATE_LIMITER.clear()
    _POLICY_STORE.clear()


_SETTINGS = ChatOpsSettings(
    provider="discord",
    env_prefix="DISCORD",
    default_oauth_auth_url="https://discord.com/oauth2/authorize",
    default_oauth_token_url="https://discord.com/api/oauth2/token",
    installation_fields=('guild_id', 'guild_name'),
)
# The names the discord endpoints import; one implementation in _chatops/settings.py.
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


def _discord_public_key_hex() -> str:
    return (os.getenv("DISCORD_PUBLIC_KEY") or "").strip()


def _policy_guild_quota_per_minute() -> int:
    return _env_int("DISCORD_POLICY_GUILD_QUOTA_PER_MINUTE", 120)


def _oauth_scope() -> str:
    raw_scope = _coerce_nonempty_string(os.getenv("DISCORD_OAUTH_SCOPE")) or "bot applications.commands"
    parts = [part.strip() for part in raw_scope.split() if part.strip()]
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return " ".join(deduped)


def _oauth_permissions() -> str | None:
    return _coerce_nonempty_string(os.getenv("DISCORD_OAUTH_PERMISSIONS"))


def _get_job_manager():
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


def _encrypt_discord_payload(payload: dict[str, Any]) -> str:
    return dumps_envelope(encrypt_byok_payload(payload))


def _decrypt_discord_payload(encrypted_blob: str) -> dict[str, Any] | None:
    if not encrypted_blob:
        return None
    try:
        payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
    except Exception:
        logger.warning("Failed to decrypt Discord installation payload")
        return None
    return payload if isinstance(payload, dict) else None


async def _discord_oauth_token_exchange(*, token_url: str, form_data: dict[str, Any]) -> dict[str, Any]:
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
            detail = "Discord OAuth token exchange failed"
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=detail,
            )
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Discord OAuth token exchange returned invalid payload",
            )
        return payload
    finally:
        await _close_http_response(response)


_error_response = _chatops_policy.error_response
_metric_labels = _chatops_policy.metric_labels


def _emit_discord_counter(metric_name: str, **labels: Any) -> None:
    try:
        log_counter(metric_name, labels=_metric_labels(**labels))
    except Exception:
        logger.debug("Failed to emit Discord metric")


def _extract_timestamp(header_value: str | None) -> int | None:
    if not header_value:
        return None
    try:
        return int(header_value.strip())
    except (TypeError, ValueError):
        return None


def _load_discord_public_key() -> Ed25519PublicKey | None:
    key_hex = _discord_public_key_hex()
    if not key_hex:
        return None
    try:
        key_bytes = bytes.fromhex(key_hex)
        return Ed25519PublicKey.from_public_bytes(key_bytes)
    except ValueError:
        logger.warning("DISCORD_PUBLIC_KEY is malformed")
        return None


def _verify_discord_signature(
    raw_body: bytes, timestamp_header: str | None, signature_header: str | None
) -> tuple[bool, str | None]:
    public_key = _load_discord_public_key()
    if public_key is None:
        return False, "public_key_not_configured"

    timestamp = _extract_timestamp(timestamp_header)
    if timestamp is None:
        return False, "invalid_timestamp"

    now = int(time.time())
    if abs(now - timestamp) > max(1, _replay_window_seconds()):
        return False, "stale_request"

    if not signature_header:
        return False, "invalid_signature"
    try:
        signature = bytes.fromhex(signature_header.strip())
    except ValueError:
        return False, "invalid_signature"

    message = str(timestamp).encode("utf-8") + raw_body
    try:
        public_key.verify(signature, message)
    except InvalidSignature:
        return False, "invalid_signature"
    return True, None


def _interaction_dedupe_key(payload: dict, raw_body: bytes) -> str:
    interaction_id = str(payload.get("id") or "").strip()
    application_id = str(payload.get("application_id") or "").strip()
    if interaction_id and application_id:
        return f"{application_id}:{interaction_id}"
    if interaction_id:
        return interaction_id
    return raw_body.hex()


def _rate_limit_key(payload: dict, request: Request) -> str:
    application_id = str(payload.get("application_id") or "na")
    guild_id = str(payload.get("guild_id") or "")
    fallback = request.client.host if request.client else "unknown"
    return f"discord:interactions:{application_id}:{guild_id or fallback}"


_SUPPORTED_DISCORD_ACTIONS = ("help", "ask", "rag", "summarize", "status")


def _discord_usage_text() -> str:
    return "Supported commands: help | ask <query> | rag <query> | summarize <text> | status"


def _normalize_string_list(raw: Any) -> list[str]:
    return _chatops_policy.normalize_string_list(raw, coerce=_coerce_nonempty_string)


_DISCORD_POLICY_SPEC = _chatops_policy.ChatOpsPolicySpec(
    supported_actions=_SUPPORTED_DISCORD_ACTIONS,
    scope_quota_field="guild_quota_per_minute",
    scope_quota_default=_policy_guild_quota_per_minute,
    user_quota_default=_policy_user_quota_per_minute,
    status_scope_values=("guild", "guild_and_user"),
    response_modes=("ephemeral", "channel"),
)


def _default_discord_policy() -> dict[str, Any]:
    return _chatops_policy.default_policy(_DISCORD_POLICY_SPEC)


def _normalize_discord_policy_payload(
    payload: dict[str, Any] | None,
    *,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _chatops_policy.normalize_policy_payload(
        _DISCORD_POLICY_SPEC,
        payload,
        base=base,
        coerce=_coerce_nonempty_string,
        safe_int=_safe_int,
    )


_POLICY_STORE = _chatops_policy.PolicyStore(
    normalize=_normalize_discord_policy_payload,
    default_policy=_default_discord_policy,
    coerce=_coerce_nonempty_string,
)
_discord_policy_for_guild = _POLICY_STORE.get
_set_discord_policy = _POLICY_STORE.set


def _resolve_discord_actor_id(
    policy: dict[str, Any], discord_user_id: str | None
) -> tuple[str | None, dict[str, Any] | None]:
    return _chatops_policy.resolve_actor_id(
        policy, discord_user_id, provider_label="Discord", coerce=_coerce_nonempty_string, http_status=status
    )


_DISCORD_POLICY_RUNTIME = _chatops_policy.ChatOpsPolicyRuntime(
    name="discord",
    scope_word="guild",
    scope_quota_field="guild_quota_per_minute",
    scope_quota_default=_policy_guild_quota_per_minute,
    user_quota_default=_policy_user_quota_per_minute,
    scope_label_field="guild_id",
    quota_rejection_counter="discord_policy_quota_rejections_total",
    denial_counter="discord_policy_denied_total",
)


def _evaluate_discord_policy(
    *,
    policy: dict[str, Any],
    guild_id: str | None,
    channel_id: str | None,
    actor_user_id: str | None,
    action: str,
) -> dict[str, Any] | None:
    return _chatops_policy.evaluate_policy(
        _DISCORD_POLICY_RUNTIME,
        policy=policy,
        scope_id=guild_id,
        channel_id=channel_id,
        actor_user_id=actor_user_id,
        action=action,
        rate_limiter=_POLICY_RATE_LIMITER,
        coerce=_coerce_nonempty_string,
        safe_int=_safe_int,
        http_status=status,
    )


def _discord_policy_error_response(
    policy_error: dict[str, Any], *, guild_id: str | None, action: str | None
) -> JSONResponse:
    return _chatops_policy.policy_error_response(
        _DISCORD_POLICY_RUNTIME,
        policy_error,
        scope_id=guild_id,
        action=action,
        emit_counter=_emit_discord_counter,
        log=logger,
        safe_int=_safe_int,
        http_status=status,
    )


def _discord_action_route(action: str) -> str:
    return _chatops_policy.action_route(_DISCORD_POLICY_RUNTIME, action)


def _parse_discord_interaction_command(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    data = payload.get("data")
    if not isinstance(data, dict):
        # Default to ask when command metadata is unavailable.
        return {"action": "ask", "route": _discord_action_route("ask"), "input": "", "inferred": True}, None

    candidate = _coerce_nonempty_string(data.get("name"))
    input_text = ""
    options = data.get("options")
    if isinstance(options, list) and options:
        first = options[0]
        if isinstance(first, dict):
            option_name = _coerce_nonempty_string(first.get("name"))
            option_value = first.get("value")
            if isinstance(option_value, str):
                input_text = option_value.strip()
            if option_name:
                candidate = option_name
            nested = first.get("options")
            if isinstance(nested, list):
                for opt in nested:
                    if isinstance(opt, dict) and isinstance(opt.get("value"), str):
                        input_text = str(opt.get("value")).strip()
                        break

    if not candidate:
        candidate = "ask"
    action = candidate.lower()
    if action == "tldw":
        action = "ask"
    if action not in _SUPPORTED_DISCORD_ACTIONS:
        return None, {
            "error": "unknown_command",
            "message": f"Unknown command '{action}'. {_discord_usage_text()}",
            "usage": _discord_usage_text(),
        }
    return {
        "action": action,
        "route": _discord_action_route(action),
        "input": input_text,
    }, None


def _safe_int(raw_value: Any) -> int | None:
    try:
        return int(str(raw_value).strip())
    except (TypeError, ValueError):
        return None


def _discord_response_mode(payload: dict[str, Any], policy: dict[str, Any] | None = None) -> str:
    data = payload.get("data")
    if isinstance(data, dict):
        # Optional extension field for bot workflow policy.
        mode = _coerce_nonempty_string(data.get("response_mode"))
        if mode:
            mode = mode.lower()
            if mode in {"ephemeral", "channel"}:
                return mode
    if isinstance(policy, dict):
        default_mode = _coerce_nonempty_string(policy.get("default_response_mode"))
        if default_mode and default_mode.lower() in {"ephemeral", "channel"}:
            return default_mode.lower()
    return "ephemeral"


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
