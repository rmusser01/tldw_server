from __future__ import annotations

import os
import secrets
import threading
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

_INTERACTION_RECEIPTS = TTLReceiptStore()
_RATE_LIMITER = SlidingWindowLimiter()
_POLICY_RATE_LIMITER = SlidingWindowLimiter()
_DISCORD_POLICY_LOCK = threading.Lock()
_DISCORD_POLICIES: dict[str, dict[str, Any]] = {}
_POLICY_DEFAULT_KEY = "__default__"


def _reset_discord_state_for_tests() -> None:
    _INTERACTION_RECEIPTS.clear()
    _RATE_LIMITER.clear()
    _POLICY_RATE_LIMITER.clear()
    with _DISCORD_POLICY_LOCK:
        _DISCORD_POLICIES.clear()


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _discord_public_key_hex() -> str:
    return (os.getenv("DISCORD_PUBLIC_KEY") or "").strip()


def _replay_window_seconds() -> int:
    return _env_int("DISCORD_REPLAY_WINDOW_SECONDS", 300)


def _dedupe_ttl_seconds() -> int:
    return _env_int("DISCORD_DEDUPE_TTL_SECONDS", 3600)


def _ingress_rate_limit_per_minute() -> int:
    return _env_int("DISCORD_INGRESS_RATE_LIMIT_PER_MINUTE", 120)


def _policy_guild_quota_per_minute() -> int:
    return _env_int("DISCORD_POLICY_GUILD_QUOTA_PER_MINUTE", 120)


def _policy_user_quota_per_minute() -> int:
    return _env_int("DISCORD_POLICY_USER_QUOTA_PER_MINUTE", 60)


def _coerce_nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _oauth_client_id() -> str:
    return (os.getenv("DISCORD_CLIENT_ID") or "").strip()


def _oauth_client_secret() -> str:
    return (os.getenv("DISCORD_CLIENT_SECRET") or "").strip()


def _oauth_redirect_uri() -> str:
    redirect_uri = _coerce_nonempty_string(os.getenv("DISCORD_OAUTH_REDIRECT_URI"))
    if not redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="DISCORD_OAUTH_REDIRECT_URI is not configured",
        )
    return redirect_uri


def _oauth_auth_url() -> str:
    return _coerce_nonempty_string(os.getenv("DISCORD_OAUTH_AUTH_URL")) or "https://discord.com/oauth2/authorize"


def _oauth_token_url() -> str:
    return _coerce_nonempty_string(os.getenv("DISCORD_OAUTH_TOKEN_URL")) or "https://discord.com/api/oauth2/token"


def _oauth_state_ttl_seconds() -> int:
    return _env_int("DISCORD_OAUTH_STATE_TTL_SECONDS", 600)


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


def _default_installations_payload() -> dict[str, Any]:
    return {
        "provider": "discord",
        "credential_version": 1,
        "installations": {},
    }


def _normalize_installations_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    merged = _default_installations_payload()
    if isinstance(payload, dict):
        merged.update(payload)
    installations = merged.get("installations")
    if not isinstance(installations, dict):
        merged["installations"] = {}
    return merged


def _public_installation_record(installation: dict[str, Any]) -> dict[str, Any]:
    return {
        "guild_id": installation.get("guild_id"),
        "guild_name": installation.get("guild_name"),
        "scope": installation.get("scope"),
        "installed_at": installation.get("installed_at"),
        "installed_by": installation.get("installed_by"),
        "disabled": bool(installation.get("disabled")),
    }


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


def _error_response(status_code: int, error: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "error": error, "message": message},
    )


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
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    for item in raw:
        cleaned = _coerce_nonempty_string(item)
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values


def _default_discord_policy() -> dict[str, Any]:
    return {
        "allowed_commands": list(_SUPPORTED_DISCORD_ACTIONS),
        "channel_allowlist": [],
        "channel_denylist": [],
        "default_response_mode": "ephemeral",
        "strict_user_mapping": False,
        "service_user_id": None,
        "user_mappings": {},
        "guild_quota_per_minute": _policy_guild_quota_per_minute(),
        "user_quota_per_minute": _policy_user_quota_per_minute(),
        "status_scope": "guild",
    }


def _normalize_discord_policy_payload(
    payload: dict[str, Any] | None,
    *,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(base or _default_discord_policy())
    data = payload if isinstance(payload, dict) else {}

    if "allowed_commands" in data:
        allowed: list[str] = []
        for candidate in _normalize_string_list(data.get("allowed_commands")):
            lowered = candidate.lower()
            if lowered in _SUPPORTED_DISCORD_ACTIONS and lowered not in allowed:
                allowed.append(lowered)
        merged["allowed_commands"] = allowed or list(_SUPPORTED_DISCORD_ACTIONS)

    if "channel_allowlist" in data:
        merged["channel_allowlist"] = _normalize_string_list(data.get("channel_allowlist"))
    if "channel_denylist" in data:
        merged["channel_denylist"] = _normalize_string_list(data.get("channel_denylist"))

    if "default_response_mode" in data:
        mode = _coerce_nonempty_string(data.get("default_response_mode"))
        if mode and mode.lower() in {"ephemeral", "channel"}:
            merged["default_response_mode"] = mode.lower()

    if "strict_user_mapping" in data:
        merged["strict_user_mapping"] = bool(data.get("strict_user_mapping"))
    if "service_user_id" in data:
        merged["service_user_id"] = _coerce_nonempty_string(data.get("service_user_id"))

    if "user_mappings" in data and isinstance(data.get("user_mappings"), dict):
        normalized_mappings: dict[str, str] = {}
        for raw_key, raw_value in dict(data.get("user_mappings") or {}).items():
            key = _coerce_nonempty_string(raw_key)
            value = _coerce_nonempty_string(raw_value)
            if key and value:
                normalized_mappings[key] = value
        merged["user_mappings"] = normalized_mappings

    if "guild_quota_per_minute" in data:
        value = _safe_int(data.get("guild_quota_per_minute"))
        if value is not None and value > 0:
            merged["guild_quota_per_minute"] = value
    if "user_quota_per_minute" in data:
        value = _safe_int(data.get("user_quota_per_minute"))
        if value is not None and value > 0:
            merged["user_quota_per_minute"] = value

    if "status_scope" in data:
        scope = _coerce_nonempty_string(data.get("status_scope"))
        if scope and scope.lower() in {"guild", "guild_and_user"}:
            merged["status_scope"] = scope.lower()

    return merged


def _policy_key(guild_id: str | None) -> str:
    return _coerce_nonempty_string(guild_id) or _POLICY_DEFAULT_KEY


def _discord_policy_for_guild(guild_id: str | None) -> dict[str, Any]:
    key = _policy_key(guild_id)
    with _DISCORD_POLICY_LOCK:
        default_raw = _DISCORD_POLICIES.get(_POLICY_DEFAULT_KEY)
        default_policy = _normalize_discord_policy_payload(default_raw if isinstance(default_raw, dict) else None)
        if key == _POLICY_DEFAULT_KEY:
            return default_policy
        selected = _DISCORD_POLICIES.get(key)
        if isinstance(selected, dict):
            return _normalize_discord_policy_payload(selected, base=default_policy)
        return default_policy


def _set_discord_policy(guild_id: str | None, payload: dict[str, Any] | None) -> tuple[str | None, dict[str, Any]]:
    key = _policy_key(guild_id)
    cleaned_guild = _coerce_nonempty_string(guild_id)
    with _DISCORD_POLICY_LOCK:
        default_raw = _DISCORD_POLICIES.get(_POLICY_DEFAULT_KEY)
        default_policy = _normalize_discord_policy_payload(default_raw if isinstance(default_raw, dict) else None)
        base = default_policy if key != _POLICY_DEFAULT_KEY else _default_discord_policy()
        normalized = _normalize_discord_policy_payload(payload, base=base)
        _DISCORD_POLICIES[key] = dict(normalized)
    return cleaned_guild, normalized


def _resolve_discord_actor_id(
    policy: dict[str, Any], discord_user_id: str | None
) -> tuple[str | None, dict[str, Any] | None]:
    requested_user_id = _coerce_nonempty_string(discord_user_id)
    user_mappings = policy.get("user_mappings") if isinstance(policy.get("user_mappings"), dict) else {}
    mapped_user = user_mappings.get(requested_user_id) if requested_user_id else None
    if mapped_user:
        return _coerce_nonempty_string(mapped_user), None

    service_user_id = _coerce_nonempty_string(policy.get("service_user_id"))
    strict_mapping = bool(policy.get("strict_user_mapping"))
    if strict_mapping and not service_user_id:
        return None, {
            "status_code": status.HTTP_403_FORBIDDEN,
            "error": "unknown_user_mapping",
            "message": "Discord user is not mapped to a local user and strict mapping is enabled",
        }
    return requested_user_id or service_user_id, None


def _evaluate_discord_policy(
    *,
    policy: dict[str, Any],
    guild_id: str | None,
    channel_id: str | None,
    actor_user_id: str | None,
    action: str,
) -> dict[str, Any] | None:
    allowed_commands = policy.get("allowed_commands")
    if isinstance(allowed_commands, list) and allowed_commands:
        if action not in {str(item).lower() for item in allowed_commands}:
            return {
                "status_code": status.HTTP_403_FORBIDDEN,
                "error": "command_blocked_by_policy",
                "message": f"Command '{action}' is not allowed for this guild",
            }

    deny_channels = set(_normalize_string_list(policy.get("channel_denylist")))
    allow_channels = set(_normalize_string_list(policy.get("channel_allowlist")))
    if channel_id and channel_id in deny_channels:
        return {
            "status_code": status.HTTP_403_FORBIDDEN,
            "error": "channel_blocked_by_policy",
            "message": f"Channel '{channel_id}' is blocked by policy",
        }
    if allow_channels and channel_id and channel_id not in allow_channels:
        return {
            "status_code": status.HTTP_403_FORBIDDEN,
            "error": "channel_not_allowed_by_policy",
            "message": f"Channel '{channel_id}' is not in the allowlist",
        }

    guild_limit = _safe_int(policy.get("guild_quota_per_minute")) or _policy_guild_quota_per_minute()
    guild_key = _coerce_nonempty_string(guild_id) or "unknown"
    allowed_guild, retry_after_guild = _POLICY_RATE_LIMITER.allow(
        f"discord:guild:{guild_key}",
        max(1, guild_limit),
    )
    if not allowed_guild:
        return {
            "status_code": status.HTTP_429_TOO_MANY_REQUESTS,
            "error": "guild_quota_exceeded",
            "message": "Guild command quota exceeded",
            "retry_after_seconds": retry_after_guild,
        }

    if actor_user_id:
        user_limit = _safe_int(policy.get("user_quota_per_minute")) or _policy_user_quota_per_minute()
        allowed_user, retry_after_user = _POLICY_RATE_LIMITER.allow(
            f"discord:user:{guild_key}:{actor_user_id}",
            max(1, user_limit),
        )
        if not allowed_user:
            return {
                "status_code": status.HTTP_429_TOO_MANY_REQUESTS,
                "error": "user_quota_exceeded",
                "message": "User command quota exceeded",
                "retry_after_seconds": retry_after_user,
            }

    return None


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


def _discord_action_route(action: str) -> str:
    routes = {
        "help": "discord.help",
        "ask": "chat.ask",
        "rag": "rag.search",
        "summarize": "summarize.run",
        "status": "jobs.status",
    }
    return routes.get(action, "chat.ask")


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
