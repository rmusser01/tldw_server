from __future__ import annotations

import asyncio
import copy
import contextlib
import json
import os
import sqlite3
import threading
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable
from weakref import WeakKeyDictionary

import aiosqlite
import asyncpg
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_config import (
    PROVIDER_APP_CONFIG_KEYS,
    merge_app_config_overrides,
)
from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
    is_byok_enabled,
    is_provider_allowlisted,
    is_trusted_base_url_request,
    resolve_byok_base_url_allowlist,
    resolve_server_default_key,
    validate_base_url_override,
    validate_credential_fields,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError as AuthNZDatabaseError
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_memberships_for_user
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
    normalize_provider_name,
)
from tldw_Server_API.app.core.config import loaded_config_data
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_provider_number,
    custom_openai_section_name,
)
from tldw_Server_API.app.core.http_client import RetryPolicy as _RetryPolicy
from tldw_Server_API.app.core.http_client import afetch as _http_afetch
from tldw_Server_API.app.core.Metrics import increment_counter
from tldw_Server_API.app.core.exceptions import (
    EgressPolicyError,
    NetworkError,
    RetryExhaustedError,
)

DEFAULT_LAST_USED_THROTTLE_SECONDS = 300
DEFAULT_OPENAI_OAUTH_REFRESH_SKEW_SECONDS = 120
DEFAULT_OPENAI_OAUTH_REFRESH_LOCK_BACKEND = "memory"
_OPENAI_PROVIDER = "openai"
_OPENAI_SOURCE_API_KEY = "api_key"
_OPENAI_SOURCE_OAUTH = "oauth"
_OPENAI_CREDENTIAL_VERSION = 2

_BYOK_RESOLUTION_ERROR_CODES = frozenset(
    {
        "invalid_provider_credentials",
        "credential_store_unavailable",
        "credential_scope_revoked",
    }
)
_SHARED_APP_CONFIG_KEYS = {
    "HTTP": frozenset(
        {
            "connect_timeout",
            "read_timeout",
            "write_timeout",
            "pool_timeout",
            "proxy_allowlist",
            "enforce_tls_min_version",
            "tls_min_version",
            "allow_redirects",
            "max_redirects",
            "allow_cross_host_redirects",
        }
    ),
    "Egress": frozenset(
        {
            "egress_allowlist",
            "egress_denylist",
            "workflows_allowlist",
            "workflows_denylist",
            "allowed_ports",
            "profile",
            "block_private",
        }
    ),
}
_LOCAL_PROVIDER_APP_CONFIG_KEYS = {
    "llama.cpp": "llama_api",
    "kobold": "kobold_api",
    "ooba": "ooba_api",
    "tabbyapi": "tabby_api",
    "vllm": "vllm_api",
    "local-llm": "local_llm",
    "ollama": "ollama_api",
    "aphrodite": "aphrodite_api",
    "mlx": "mlx",
}
_PROVIDER_CONFIG_KEYS = frozenset(
    {
        "model",
        "model_id",
        "model_path",
        "mlx_model_path",
        "timeout",
        "connect_timeout",
        "read_timeout",
        "write_timeout",
        "pool_timeout",
        "api_timeout",
        "retry",
        "retries",
        "retry_attempts",
        "retry_delay",
        "api_retries",
        "api_retry_delay",
        "backoff_base_ms",
        "backoff_cap_s",
        "base_url",
        "api_base_url",
        "api_url",
        "api_ip",
        "endpoint",
        "runtime_endpoint",
        "organization",
        "organization_id",
        "org_id",
        "project",
        "project_id",
    }
)

_openai_oauth_refresh_lock_guard = threading.Lock()
_openai_oauth_refresh_locks: "WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]]" = (
    WeakKeyDictionary()
)
_openai_oauth_refresh_backend_warned: set[str] = set()

_BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    EOFError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)
_BYOK_CREDENTIAL_STORE_EXCEPTIONS = (
    sqlite3.Error,
    aiosqlite.Error,
    asyncpg.PostgresError,
    AuthNZDatabaseError,
    OSError,
    TimeoutError,
)
_BYOK_OAUTH_TRANSPORT_EXCEPTIONS = (
    EgressPolicyError,
    NetworkError,
    RetryExhaustedError,
    OSError,
    TimeoutError,
)
_CREDENTIAL_FIELDS_MISSING = object()


def _last_used_throttle_seconds() -> int:
    raw = os.getenv("BYOK_LAST_USED_THROTTLE_SECONDS")
    if not raw:
        return DEFAULT_LAST_USED_THROTTLE_SECONDS
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_LAST_USED_THROTTLE_SECONDS


def _parse_last_used(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
            return None
    return None


def _should_touch(last_used_at: datetime | None) -> bool:
    if last_used_at is None:
        return True
    throttle = _last_used_throttle_seconds()
    if throttle <= 0:
        return True
    delta = datetime.now(timezone.utc) - last_used_at
    return delta.total_seconds() >= throttle


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def _can_use_base_url_override(
    provider: str,
    request: Any | None,
    trusted_base_url_override: bool | None = None,
) -> bool:
    trusted = (
        is_trusted_base_url_request(request) if trusted_base_url_override is None else trusted_base_url_override is True
    )
    if not trusted:
        return False
    provider_norm = normalize_provider_name(provider)
    return provider_norm in resolve_byok_base_url_allowlist()


def _sanitize_credential_fields(
    provider: str,
    credential_fields_raw: Any,
    *,
    allow_base_url: bool,
) -> dict[str, Any]:
    if credential_fields_raw is None:
        return {}
    if not isinstance(credential_fields_raw, dict):
        raise ValueError("credential_fields must be an object")

    cleaned = dict(credential_fields_raw)
    if not allow_base_url and "base_url" in cleaned:
        cleaned.pop("base_url", None)
        logger.debug("BYOK base_url override ignored for provider={}", provider)

    validated = validate_credential_fields(provider, cleaned, allow_base_url=allow_base_url)
    if "base_url" in validated:
        validated["base_url"] = validate_base_url_override(validated["base_url"])
    return validated


def _apply_active_scope(ids: list[int], active_id: Any, *, provider: str) -> list[int]:
    if active_id is None:
        return ids if len(ids) == 1 else []
    try:
        active = int(active_id)
    except (TypeError, ValueError):
        raise ByokResolutionError("credential_scope_revoked", provider) from None
    if active not in ids:
        raise ByokResolutionError("credential_scope_revoked", provider)
    return [active]


class ByokResolutionStatus(str, Enum):
    """Non-secret terminal state for credential resolution."""

    ABSENT = "ABSENT"
    RESOLVED = "RESOLVED"


class ByokResolutionError(Exception):
    """Sanitized failure raised when credential resolution cannot continue."""

    def __init__(self, code: str, provider: str) -> None:
        if code not in _BYOK_RESOLUTION_ERROR_CODES:
            raise ValueError("Unsupported BYOK resolution error code")
        self.code = code
        self.provider = normalize_provider_name(provider)
        super().__init__(f"{self.code}: {self.provider}")


def _credential_fields_from_payload(
    payload: dict[str, Any],
    provider: str,
) -> dict[str, Any]:
    raw_fields = payload.get("credential_fields", _CREDENTIAL_FIELDS_MISSING)
    if raw_fields is _CREDENTIAL_FIELDS_MISSING or raw_fields is None:
        return {}
    if not isinstance(raw_fields, dict):
        raise ByokResolutionError("invalid_provider_credentials", provider)
    return dict(raw_fields)


@dataclass
class ResolvedByokCredentials:
    provider: str
    api_key: str | None
    app_config: dict[str, Any] | None
    credential_fields: dict[str, Any]
    source: str
    allowlisted: bool
    status: ByokResolutionStatus = ByokResolutionStatus.RESOLVED
    auth_source: str | None = None
    _touch_cb: Callable[[], Awaitable[None]] | None = None

    @property
    def uses_byok(self) -> bool:
        return self.source in {"user", "team", "org"}

    def __repr__(self) -> str:
        return (
            "ResolvedByokCredentials("
            f"provider={self.provider!r}, source={self.source!r}, "
            f"allowlisted={self.allowlisted!r}, status={self.status.value!r})"
        )

    async def touch_last_used(self) -> None:
        if not self._touch_cb:
            return
        try:
            await self._touch_cb()
        except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"BYOK last_used_at update failed for {self.provider}: {exc}")


@dataclass
class _OpenAIUserResolution:
    payload: dict[str, Any]
    api_key: str | None
    auth_source: str | None
    fail_closed: bool


def _record_byok_resolution(resolved: ResolvedByokCredentials, *, byok_enabled: bool) -> None:
    """Emit a counter entry for BYOK credential resolution."""
    try:
        increment_counter(
            "byok_resolution_total",
            labels={
                "provider": resolved.provider,
                "source": resolved.source,
                "allowlisted": _bool_label(resolved.allowlisted),
                "byok_enabled": _bool_label(byok_enabled),
            },
        )
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"BYOK resolution metrics failed for {resolved.provider}: {exc}")


def record_byok_missing_credentials(provider: str, *, operation: str) -> None:
    """Emit a counter entry for missing provider credentials."""
    provider_norm = normalize_provider_name(provider)
    try:
        allowlisted = is_provider_allowlisted(provider_norm)
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
        allowlisted = False
    try:
        byok_enabled = is_byok_enabled()
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
        byok_enabled = False
    try:
        increment_counter(
            "byok_missing_credentials_total",
            labels={
                "provider": provider_norm,
                "operation": operation,
                "allowlisted": _bool_label(allowlisted),
                "byok_enabled": _bool_label(byok_enabled),
            },
        )
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"BYOK missing-credentials metrics failed for {provider_norm}: {exc}")


def _finalize_resolution(
    resolved: ResolvedByokCredentials,
    *,
    byok_enabled: bool,
) -> ResolvedByokCredentials:
    _record_byok_resolution(resolved, byok_enabled=byok_enabled)
    return resolved


async def _get_user_repo() -> AuthnzUserProviderSecretsRepo:
    pool = await get_db_pool()
    return AuthnzUserProviderSecretsRepo(pool)


async def _get_org_repo() -> AuthnzOrgProviderSecretsRepo:
    pool = await get_db_pool()
    return AuthnzOrgProviderSecretsRepo(pool)


def _fallback_result(
    provider: str,
    *,
    allowlisted: bool,
    fallback_resolver: Callable[[str], str | None] | None,
) -> ResolvedByokCredentials:
    api_key = None
    if fallback_resolver is not None:
        try:
            api_key = fallback_resolver(provider)
        except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"BYOK fallback resolver failed for {provider}: {exc}")
            api_key = None
    if api_key is None:
        api_key = resolve_server_default_key(provider)
    source = "server_default" if api_key else "none"
    return ResolvedByokCredentials(
        provider=provider,
        api_key=api_key,
        app_config=_build_app_config(provider, {}),
        credential_fields={},
        source=source,
        allowlisted=allowlisted,
        status=(ByokResolutionStatus.RESOLVED if api_key else ByokResolutionStatus.ABSENT),
        auth_source=None,
        _touch_cb=None,
    )


def _build_app_config(provider: str, credential_fields: dict[str, Any]) -> dict[str, Any] | None:
    try:
        base_cfg = loaded_config_data
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
        base_cfg = None
    provider_norm = normalize_provider_name(provider)
    section = PROVIDER_APP_CONFIG_KEYS.get(provider_norm) or _LOCAL_PROVIDER_APP_CONFIG_KEYS.get(provider_norm)
    custom_number = custom_openai_provider_number(provider_norm)
    if section is None and custom_number is not None:
        section = custom_openai_section_name(custom_number)
    if section is None:
        section = f"{provider_norm.replace('.', '_').replace('-', '_')}_api"

    scrubbed_cfg: dict[str, Any] = {}
    if base_cfg:
        try:
            if section and isinstance(base_cfg.get(section), dict):
                cleaned_provider = {
                    k: copy.deepcopy(v) for k, v in base_cfg.get(section, {}).items() if k in _PROVIDER_CONFIG_KEYS
                }
                if cleaned_provider:
                    scrubbed_cfg[section] = cleaned_provider

            for canonical_section, allowed_keys in _SHARED_APP_CONFIG_KEYS.items():
                for candidate in (canonical_section, canonical_section.lower()):
                    shared = base_cfg.get(candidate)
                    if isinstance(shared, dict):
                        cleaned_shared = {k: copy.deepcopy(v) for k, v in shared.items() if k in allowed_keys}
                        if cleaned_shared:
                            scrubbed_cfg[candidate] = cleaned_shared
                        break
        except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
            scrubbed_cfg = {}
    merged = merge_app_config_overrides(scrubbed_cfg or None, provider, credential_fields)
    return merged or None


def _extract_payload(row: dict[str, Any], provider: str) -> dict[str, Any]:
    encrypted_blob = row.get("encrypted_blob")
    if not encrypted_blob:
        raise ByokResolutionError("invalid_provider_credentials", provider)
    try:
        payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
        logger.warning("BYOK decrypt failed for provider={}", provider)
        raise ByokResolutionError("invalid_provider_credentials", provider) from None
    if not isinstance(payload, dict) or not payload:
        raise ByokResolutionError("invalid_provider_credentials", provider)
    return payload


def _coerce_nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _extract_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_metadata_value(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return dict(parsed) if isinstance(parsed, dict) else None


async def _close_http_response(response: Any) -> None:
    close_async = getattr(response, "aclose", None)
    if callable(close_async):
        await close_async()
        return
    close_sync = getattr(response, "close", None)
    if callable(close_sync):
        close_sync()


def _is_openai_v2_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("credential_version") != _OPENAI_CREDENTIAL_VERSION:
        return False
    credentials = payload.get("credentials")
    return isinstance(credentials, dict)


def _openai_credentials_map(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not _is_openai_v2_payload(payload):
        return {}
    credentials = payload.get("credentials")
    return dict(credentials) if isinstance(credentials, dict) else {}


def _openai_source_payload(payload: dict[str, Any] | None, source: str) -> dict[str, Any]:
    credentials = _openai_credentials_map(payload)
    blob = credentials.get(source)
    return dict(blob) if isinstance(blob, dict) else {}


def _legacy_payload_api_key(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    return _coerce_nonempty_string(payload.get("api_key"))


def _v2_payload_api_key(payload: dict[str, Any] | None) -> str | None:
    source_blob = _openai_source_payload(payload, _OPENAI_SOURCE_API_KEY)
    return _coerce_nonempty_string(source_blob.get("api_key"))


def _v2_payload_oauth_access_token(payload: dict[str, Any] | None) -> str | None:
    source_blob = _openai_source_payload(payload, _OPENAI_SOURCE_OAUTH)
    return _coerce_nonempty_string(source_blob.get("access_token"))


def _v2_payload_oauth_refresh_token(payload: dict[str, Any] | None) -> str | None:
    source_blob = _openai_source_payload(payload, _OPENAI_SOURCE_OAUTH)
    return _coerce_nonempty_string(source_blob.get("refresh_token"))


def _extract_api_key_from_v2_source(credentials: dict[str, Any], source: str) -> str | None:
    source_blob = credentials.get(source)
    if not isinstance(source_blob, dict):
        return None
    if source == _OPENAI_SOURCE_OAUTH:
        return _coerce_nonempty_string(source_blob.get("access_token"))
    if source == _OPENAI_SOURCE_API_KEY:
        return _coerce_nonempty_string(source_blob.get("api_key"))
    return None


def _v2_source_available(
    payload: dict[str, Any] | None,
    source: str,
    *,
    require_access_for_oauth: bool = False,
) -> bool:
    if source == _OPENAI_SOURCE_API_KEY:
        return bool(_v2_payload_api_key(payload))
    if source == _OPENAI_SOURCE_OAUTH:
        access_token = _v2_payload_oauth_access_token(payload)
        refresh_token = _v2_payload_oauth_refresh_token(payload)
        if require_access_for_oauth:
            return bool(access_token)
        return bool(access_token or refresh_token)
    return False


def _extract_runtime_auth_source(
    payload: dict[str, Any] | None,
    *,
    require_access_for_oauth: bool = True,
) -> str | None:
    legacy = _legacy_payload_api_key(payload)
    if legacy:
        return _OPENAI_SOURCE_API_KEY

    if not _is_openai_v2_payload(payload):
        return None

    active_source_raw = payload.get("active_auth_source")
    active_source = active_source_raw.strip().lower() if isinstance(active_source_raw, str) else ""
    if active_source in {_OPENAI_SOURCE_API_KEY, _OPENAI_SOURCE_OAUTH} and _v2_source_available(
        payload,
        active_source,
        require_access_for_oauth=require_access_for_oauth,
    ):
        return active_source

    if _v2_source_available(
        payload,
        _OPENAI_SOURCE_API_KEY,
        require_access_for_oauth=require_access_for_oauth,
    ):
        return _OPENAI_SOURCE_API_KEY
    if _v2_source_available(
        payload,
        _OPENAI_SOURCE_OAUTH,
        require_access_for_oauth=require_access_for_oauth,
    ):
        return _OPENAI_SOURCE_OAUTH
    return None


def _extract_runtime_api_key(payload: dict[str, Any]) -> str | None:
    runtime_source = _extract_runtime_auth_source(payload, require_access_for_oauth=True)
    if runtime_source == _OPENAI_SOURCE_API_KEY:
        return _legacy_payload_api_key(payload) or _v2_payload_api_key(payload)
    if runtime_source == _OPENAI_SOURCE_OAUTH:
        return _v2_payload_oauth_access_token(payload)
    return None


def _openai_has_any_credentials(payload: dict[str, Any] | None) -> bool:
    if not _is_openai_v2_payload(payload):
        return False
    return _v2_source_available(payload, _OPENAI_SOURCE_API_KEY) or _v2_source_available(
        payload,
        _OPENAI_SOURCE_OAUTH,
        require_access_for_oauth=False,
    )


def _payload_key_hint(payload: dict[str, Any]) -> str:
    auth_source = _extract_runtime_auth_source(payload, require_access_for_oauth=False)
    if auth_source == _OPENAI_SOURCE_OAUTH:
        return _OPENAI_SOURCE_OAUTH
    key = _legacy_payload_api_key(payload) or _v2_payload_api_key(payload)
    return key_hint_for_api_key(key) if key else ""


def _openai_oauth_refresh_skew_seconds() -> int:
    raw = os.getenv("OPENAI_OAUTH_REFRESH_SKEW_SECONDS")
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    try:
        settings = get_settings()
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
        return DEFAULT_OPENAI_OAUTH_REFRESH_SKEW_SECONDS
    raw_setting = getattr(
        settings,
        "OPENAI_OAUTH_REFRESH_SKEW_SECONDS",
        DEFAULT_OPENAI_OAUTH_REFRESH_SKEW_SECONDS,
    )
    try:
        parsed = int(raw_setting)
    except (TypeError, ValueError):
        return DEFAULT_OPENAI_OAUTH_REFRESH_SKEW_SECONDS
    return max(0, parsed)


def _normalize_openai_oauth_refresh_lock_backend(raw_value: Any) -> str:
    text = _coerce_nonempty_string(raw_value)
    if text is None:
        return DEFAULT_OPENAI_OAUTH_REFRESH_LOCK_BACKEND
    normalized = text.lower()
    if normalized in {"memory", "redis", "db"}:
        return normalized
    return DEFAULT_OPENAI_OAUTH_REFRESH_LOCK_BACKEND


def _openai_oauth_refresh_lock_backend() -> str:
    env_override = os.getenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND")
    if env_override is not None:
        return _normalize_openai_oauth_refresh_lock_backend(env_override)
    try:
        settings = get_settings()
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
        return DEFAULT_OPENAI_OAUTH_REFRESH_LOCK_BACKEND
    setting_value = getattr(
        settings,
        "OPENAI_OAUTH_REFRESH_LOCK_BACKEND",
        DEFAULT_OPENAI_OAUTH_REFRESH_LOCK_BACKEND,
    )
    return _normalize_openai_oauth_refresh_lock_backend(setting_value)


def _openai_refresh_lock_key(*, user_id: int, provider: str) -> str:
    return f"{int(user_id)}:{provider}"


def _get_openai_refresh_lock(lock_key: str) -> asyncio.Lock:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
    if loop is None:
        return asyncio.Lock()
    with _openai_oauth_refresh_lock_guard:
        bucket = _openai_oauth_refresh_locks.get(loop)
        if bucket is None:
            bucket = {}
            _openai_oauth_refresh_locks[loop] = bucket
        lock = bucket.get(lock_key)
        if lock is None:
            lock = asyncio.Lock()
            bucket[lock_key] = lock
        return lock


@contextlib.asynccontextmanager
async def _openai_oauth_refresh_lock(*, user_id: int, provider: str):
    backend = _openai_oauth_refresh_lock_backend()
    if backend != "memory":
        with _openai_oauth_refresh_lock_guard:
            should_log = backend not in _openai_oauth_refresh_backend_warned
            if should_log:
                _openai_oauth_refresh_backend_warned.add(backend)
        if should_log:
            logger.warning(
                "OPENAI_OAUTH_REFRESH_LOCK_BACKEND={} requested; falling back to in-process memory lock.",
                backend,
            )
    lock = _get_openai_refresh_lock(_openai_refresh_lock_key(user_id=user_id, provider=provider))
    async with lock:
        yield


def _openai_payload_needs_refresh(
    payload: dict[str, Any],
    *,
    force_oauth_refresh: bool,
    now: datetime,
    skew_seconds: int,
) -> bool:
    active_source = _extract_runtime_auth_source(payload, require_access_for_oauth=False)
    if active_source != _OPENAI_SOURCE_OAUTH:
        return False
    if force_oauth_refresh:
        return True

    access_token = _v2_payload_oauth_access_token(payload)
    if not access_token:
        return True

    expires_at = _parse_iso_datetime(_openai_source_payload(payload, _OPENAI_SOURCE_OAUTH).get("expires_at"))
    if expires_at is None:
        return False
    return expires_at <= (now + timedelta(seconds=max(0, skew_seconds)))


def _coerce_openai_payload_v2(payload: dict[str, Any]) -> dict[str, Any]:
    credential_fields = _credential_fields_from_payload(payload, _OPENAI_PROVIDER)
    result: dict[str, Any] = {
        "credential_version": _OPENAI_CREDENTIAL_VERSION,
        "credentials": {},
    }
    credentials: dict[str, Any] = {}

    existing_credentials = _openai_credentials_map(payload)
    api_blob = existing_credentials.get(_OPENAI_SOURCE_API_KEY)
    if isinstance(api_blob, dict):
        api_key = _coerce_nonempty_string(api_blob.get("api_key"))
        if api_key:
            copied_api_blob = dict(api_blob)
            copied_api_blob["api_key"] = api_key
            credentials[_OPENAI_SOURCE_API_KEY] = copied_api_blob

    oauth_blob = existing_credentials.get(_OPENAI_SOURCE_OAUTH)
    if isinstance(oauth_blob, dict):
        copied_oauth_blob: dict[str, Any] = {}
        for key in (
            "access_token",
            "refresh_token",
            "token_type",
            "scope",
            "subject",
            "issued_at",
            "expires_at",
        ):
            value = oauth_blob.get(key)
            if isinstance(value, datetime):
                copied_oauth_blob[key] = value.astimezone(timezone.utc).isoformat()
                continue
            text = _coerce_nonempty_string(value)
            if text:
                copied_oauth_blob[key] = text
        if copied_oauth_blob:
            credentials[_OPENAI_SOURCE_OAUTH] = copied_oauth_blob

    legacy_api_key = _legacy_payload_api_key(payload)
    if legacy_api_key and _OPENAI_SOURCE_API_KEY not in credentials:
        credentials[_OPENAI_SOURCE_API_KEY] = {"api_key": legacy_api_key}

    result["credentials"] = credentials

    if credential_fields:
        result["credential_fields"] = credential_fields

    active_source = _extract_runtime_auth_source(payload, require_access_for_oauth=False)
    if active_source in {_OPENAI_SOURCE_API_KEY, _OPENAI_SOURCE_OAUTH} and _v2_source_available(
        result,
        active_source,
        require_access_for_oauth=False,
    ):
        result["active_auth_source"] = active_source
    elif _v2_source_available(result, _OPENAI_SOURCE_API_KEY):
        result["active_auth_source"] = _OPENAI_SOURCE_API_KEY
    elif _v2_source_available(result, _OPENAI_SOURCE_OAUTH, require_access_for_oauth=False):
        result["active_auth_source"] = _OPENAI_SOURCE_OAUTH

    return result


async def _openai_oauth_token_refresh(
    *,
    token_url: str,
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> dict[str, Any] | None:
    try:
        response = await _http_afetch(
            method="POST",
            url=token_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=30,
            retry=_RetryPolicy(attempts=1),
        )
    except _BYOK_OAUTH_TRANSPORT_EXCEPTIONS:
        logger.debug("OpenAI OAuth refresh request failed")
        return None

    try:
        status_code = int(getattr(response, "status_code", 0))
        payload: dict[str, Any] | None = None
        try:
            maybe_payload = response.json()
            if isinstance(maybe_payload, dict):
                payload = dict(maybe_payload)
        except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
            payload = None

        if status_code < 200 or status_code >= 300:
            logger.debug("OpenAI OAuth refresh rejected with status={}", status_code)
            return None
        return payload if payload is not None else None
    finally:
        await _close_http_response(response)


async def _persist_user_payload_update(
    *,
    repo: AuthnzUserProviderSecretsRepo,
    provider: str,
    user_id: int,
    row: dict[str, Any],
    payload: dict[str, Any],
    updated_at: datetime,
) -> None:
    key_hint = _payload_key_hint(payload)
    if not key_hint:
        key_hint = row.get("key_hint") or ""
    try:
        envelope = encrypt_byok_payload(payload)
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
        logger.warning("BYOK encrypt failed while persisting provider={}", provider)
        raise ByokResolutionError("invalid_provider_credentials", provider) from None

    metadata_to_store = _parse_metadata_value(row.get("metadata"))
    try:
        await repo.upsert_secret(
            user_id=user_id,
            provider=provider,
            encrypted_blob=dumps_envelope(envelope),
            key_hint=key_hint or None,
            metadata=metadata_to_store,
            updated_at=updated_at,
            created_by=user_id,
            updated_by=user_id,
        )
    except _BYOK_CREDENTIAL_STORE_EXCEPTIONS:
        logger.warning(
            "BYOK payload persist failed for user_id={} provider={}",
            user_id,
            provider,
        )
        raise ByokResolutionError("credential_store_unavailable", provider) from None


async def _resolve_openai_user_payload(
    *,
    user_repo: AuthnzUserProviderSecretsRepo,
    user_id: int,
    row: dict[str, Any],
    payload: dict[str, Any],
    force_oauth_refresh: bool,
) -> _OpenAIUserResolution:
    merged_payload = _coerce_openai_payload_v2(payload)
    now = datetime.now(timezone.utc)
    runtime_api_key = _extract_runtime_api_key(merged_payload)
    runtime_auth_source = _extract_runtime_auth_source(
        merged_payload,
        require_access_for_oauth=True,
    )
    needs_refresh = _openai_payload_needs_refresh(
        merged_payload,
        force_oauth_refresh=force_oauth_refresh,
        now=now,
        skew_seconds=_openai_oauth_refresh_skew_seconds(),
    )

    if needs_refresh:
        async with _openai_oauth_refresh_lock(user_id=user_id, provider=_OPENAI_PROVIDER):
            try:
                latest_row = await user_repo.fetch_secret_for_user(int(user_id), _OPENAI_PROVIDER)
            except _BYOK_CREDENTIAL_STORE_EXCEPTIONS:
                logger.debug(
                    "BYOK user reload before OAuth refresh failed for user_id={} provider={}",
                    user_id,
                    _OPENAI_PROVIDER,
                )
                raise ByokResolutionError(
                    "credential_store_unavailable",
                    _OPENAI_PROVIDER,
                ) from None

            if latest_row:
                latest_payload = _extract_payload(latest_row, _OPENAI_PROVIDER)
                if latest_payload:
                    row = latest_row
                    merged_payload = _coerce_openai_payload_v2(latest_payload)

            now = datetime.now(timezone.utc)
            runtime_api_key = _extract_runtime_api_key(merged_payload)
            runtime_auth_source = _extract_runtime_auth_source(
                merged_payload,
                require_access_for_oauth=True,
            )
            needs_refresh = _openai_payload_needs_refresh(
                merged_payload,
                force_oauth_refresh=force_oauth_refresh,
                now=now,
                skew_seconds=_openai_oauth_refresh_skew_seconds(),
            )

            if needs_refresh:
                settings = get_settings()
                token_url = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_TOKEN_URL", None))
                client_id = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_CLIENT_ID", None))
                client_secret = _coerce_nonempty_string(getattr(settings, "OPENAI_OAUTH_CLIENT_SECRET", None))
                oauth_enabled = bool(getattr(settings, "OPENAI_OAUTH_ENABLED", False))
                refresh_token = _v2_payload_oauth_refresh_token(merged_payload)
                token_payload: dict[str, Any] | None = None
                refresh_succeeded = False
                if oauth_enabled and token_url and client_id and client_secret and refresh_token:
                    token_payload = await _openai_oauth_token_refresh(
                        token_url=token_url,
                        client_id=client_id,
                        client_secret=client_secret,
                        refresh_token=refresh_token,
                    )

                if token_payload:
                    access_token = _coerce_nonempty_string(token_payload.get("access_token"))
                    if access_token:
                        oauth_payload = _openai_source_payload(merged_payload, _OPENAI_SOURCE_OAUTH)
                        next_refresh_token = (
                            _coerce_nonempty_string(token_payload.get("refresh_token")) or refresh_token
                        )
                        token_type = (
                            _coerce_nonempty_string(token_payload.get("token_type"))
                            or _coerce_nonempty_string(oauth_payload.get("token_type"))
                            or "Bearer"
                        )
                        scope = _coerce_nonempty_string(token_payload.get("scope")) or _coerce_nonempty_string(
                            oauth_payload.get("scope")
                        )
                        expires_in = _extract_positive_int(token_payload.get("expires_in"))
                        refreshed_at = datetime.now(timezone.utc)
                        refreshed_oauth_payload = dict(oauth_payload)
                        refreshed_oauth_payload["access_token"] = access_token
                        if next_refresh_token:
                            refreshed_oauth_payload["refresh_token"] = next_refresh_token
                        refreshed_oauth_payload["token_type"] = token_type
                        refreshed_oauth_payload["issued_at"] = refreshed_at.isoformat()
                        if scope:
                            refreshed_oauth_payload["scope"] = scope
                        if expires_in:
                            refreshed_oauth_payload["expires_at"] = (
                                refreshed_at + timedelta(seconds=expires_in)
                            ).isoformat()
                        else:
                            refreshed_oauth_payload.pop("expires_at", None)

                        credentials = _openai_credentials_map(merged_payload)
                        credentials[_OPENAI_SOURCE_OAUTH] = refreshed_oauth_payload
                        merged_payload["credentials"] = credentials
                        merged_payload["active_auth_source"] = _OPENAI_SOURCE_OAUTH
                        runtime_auth_source = _OPENAI_SOURCE_OAUTH
                        runtime_api_key = access_token
                        refresh_succeeded = True
                        await _persist_user_payload_update(
                            repo=user_repo,
                            provider=_OPENAI_PROVIDER,
                            user_id=user_id,
                            row=row,
                            payload=merged_payload,
                            updated_at=refreshed_at,
                        )
                    else:
                        logger.debug("OpenAI OAuth refresh response missing access_token")

                if not refresh_succeeded:
                    runtime_api_key = None
                    runtime_auth_source = None

                if not runtime_api_key:
                    fallback_api_key = _v2_payload_api_key(merged_payload)
                    if fallback_api_key:
                        runtime_api_key = fallback_api_key
                        runtime_auth_source = _OPENAI_SOURCE_API_KEY
                        merged_payload["active_auth_source"] = _OPENAI_SOURCE_API_KEY
                        await _persist_user_payload_update(
                            repo=user_repo,
                            provider=_OPENAI_PROVIDER,
                            user_id=user_id,
                            row=row,
                            payload=merged_payload,
                            updated_at=datetime.now(timezone.utc),
                        )
                        return _OpenAIUserResolution(
                            payload=merged_payload,
                            api_key=runtime_api_key,
                            auth_source=runtime_auth_source,
                            fail_closed=False,
                        )

                    return _OpenAIUserResolution(
                        payload=merged_payload,
                        api_key=None,
                        auth_source=None,
                        fail_closed=True,
                    )

    if runtime_api_key:
        return _OpenAIUserResolution(
            payload=merged_payload,
            api_key=runtime_api_key,
            auth_source=runtime_auth_source,
            fail_closed=False,
        )

    if _openai_has_any_credentials(merged_payload):
        return _OpenAIUserResolution(
            payload=merged_payload,
            api_key=None,
            auth_source=None,
            fail_closed=True,
        )

    return _OpenAIUserResolution(
        payload=merged_payload,
        api_key=None,
        auth_source=None,
        fail_closed=False,
    )


def _build_touch_cb(
    *,
    provider: str,
    last_used_at: datetime | None,
    repo: AuthnzUserProviderSecretsRepo | AuthnzOrgProviderSecretsRepo,
    user_id: int | None = None,
    scope_type: str | None = None,
    scope_id: int | None = None,
) -> Callable[[], Awaitable[None]]:
    async def _touch() -> None:
        if not _should_touch(last_used_at):
            return
        now = datetime.now(timezone.utc)
        if isinstance(repo, AuthnzUserProviderSecretsRepo):
            if user_id is None:
                return
            await repo.touch_last_used(int(user_id), provider, now)
        else:
            if not scope_type or scope_id is None:
                return
            await repo.touch_last_used(scope_type, int(scope_id), provider, now)

    return _touch


async def resolve_byok_credentials(
    provider: str,
    *,
    user_id: int | None,
    request: Any | None = None,
    team_ids: list[int] | None = None,
    org_ids: list[int] | None = None,
    fallback_resolver: Callable[[str], str | None] | None = None,
    force_oauth_refresh: bool = False,
    trusted_base_url_override: bool | None = None,
) -> ResolvedByokCredentials:
    provider_norm = normalize_provider_name(provider)
    byok_enabled = is_byok_enabled()
    allowlisted = is_provider_allowlisted(provider_norm)
    allow_base_url = _can_use_base_url_override(
        provider_norm,
        request,
        trusted_base_url_override,
    )

    if not byok_enabled or user_id is None or not allowlisted:
        return _finalize_resolution(
            _fallback_result(
                provider_norm,
                allowlisted=allowlisted,
                fallback_resolver=fallback_resolver,
            ),
            byok_enabled=byok_enabled,
        )

    # Resolve user key
    try:
        user_repo = await _get_user_repo()
        user_row = await user_repo.fetch_secret_for_user(int(user_id), provider_norm)
    except _BYOK_CREDENTIAL_STORE_EXCEPTIONS:
        logger.debug(
            "BYOK user lookup failed for user_id={} provider={}",
            user_id,
            provider_norm,
        )
        raise ByokResolutionError("credential_store_unavailable", provider_norm) from None

    if user_row is not None:
        payload = _extract_payload(user_row, provider_norm)
        if payload:
            runtime_payload = payload
            runtime_auth_source = _extract_runtime_auth_source(
                runtime_payload,
                require_access_for_oauth=True,
            )
            if provider_norm == _OPENAI_PROVIDER and _is_openai_v2_payload(payload):
                openai_resolution = await _resolve_openai_user_payload(
                    user_repo=user_repo,
                    user_id=int(user_id),
                    row=user_row,
                    payload=payload,
                    force_oauth_refresh=force_oauth_refresh,
                )
                runtime_payload = openai_resolution.payload
                runtime_auth_source = openai_resolution.auth_source
                if openai_resolution.fail_closed:
                    raise ByokResolutionError("invalid_provider_credentials", provider_norm)

            api_key = _extract_runtime_api_key(runtime_payload)
            if not api_key:
                raise ByokResolutionError("invalid_provider_credentials", provider_norm)
            if api_key:
                credential_fields_raw = _credential_fields_from_payload(
                    runtime_payload,
                    provider_norm,
                )
                try:
                    credential_fields = _sanitize_credential_fields(
                        provider_norm,
                        credential_fields_raw,
                        allow_base_url=allow_base_url,
                    )
                except ValueError:
                    logger.warning(
                        "BYOK credential_fields invalid for user_id={} provider={}",
                        user_id,
                        provider_norm,
                    )
                    raise ByokResolutionError(
                        "invalid_provider_credentials",
                        provider_norm,
                    ) from None
                last_used_at = _parse_last_used(user_row.get("last_used_at"))
                return _finalize_resolution(
                    ResolvedByokCredentials(
                        provider=provider_norm,
                        api_key=api_key,
                        app_config=_build_app_config(provider_norm, credential_fields),
                        credential_fields=credential_fields,
                        source="user",
                        allowlisted=True,
                        auth_source=runtime_auth_source,
                        _touch_cb=_build_touch_cb(
                            provider=provider_norm,
                            last_used_at=last_used_at,
                            repo=user_repo,
                            user_id=int(user_id),
                        ),
                    ),
                    byok_enabled=byok_enabled,
                )

    # Determine org/team scopes if not supplied
    active_team_id = None
    active_org_id = None
    if request is not None and hasattr(request, "state"):
        active_team_id = getattr(request.state, "active_team_id", None)
        active_org_id = getattr(request.state, "active_org_id", None)

    if team_ids is None or org_ids is None:
        if request is not None and hasattr(request, "state"):
            if team_ids is None:
                team_ids = list(getattr(request.state, "team_ids", None) or [])
            if org_ids is None:
                org_ids = list(getattr(request.state, "org_ids", None) or [])

    if team_ids is None or org_ids is None:
        try:
            memberships = await list_memberships_for_user(int(user_id))
            if team_ids is None:
                team_ids = [m.get("team_id") for m in memberships if m.get("team_id") is not None]
            if org_ids is None:
                org_ids = sorted({m.get("org_id") for m in memberships if m.get("org_id") is not None})
        except _BYOK_CREDENTIAL_STORE_EXCEPTIONS:
            logger.debug("BYOK membership lookup failed for user_id={}", user_id)
            raise ByokResolutionError("credential_store_unavailable", provider_norm) from None

    team_ids = team_ids or []
    org_ids = org_ids or []
    team_ids = _apply_active_scope(team_ids, active_team_id, provider=provider_norm)
    org_ids = _apply_active_scope(org_ids, active_org_id, provider=provider_norm)

    shared_repo = None
    if team_ids or org_ids:
        try:
            shared_repo = await _get_org_repo()
        except _BYOK_CREDENTIAL_STORE_EXCEPTIONS:
            logger.debug("BYOK shared repo init failed for provider={}", provider_norm)
            raise ByokResolutionError("credential_store_unavailable", provider_norm) from None

    if shared_repo:
        # Prefer team scope over org scope, mirroring list_user_provider_keys()
        for team_id in sorted({int(tid) for tid in team_ids if tid is not None}):
            try:
                row = await shared_repo.fetch_secret("team", int(team_id), provider_norm)
            except _BYOK_CREDENTIAL_STORE_EXCEPTIONS:
                logger.debug(
                    "BYOK team lookup failed for team_id={} provider={}",
                    team_id,
                    provider_norm,
                )
                raise ByokResolutionError(
                    "credential_store_unavailable",
                    provider_norm,
                ) from None
            if row is None:
                continue
            payload = _extract_payload(row, provider_norm)
            api_key = _extract_runtime_api_key(payload)
            if not api_key:
                raise ByokResolutionError("invalid_provider_credentials", provider_norm)
            credential_fields_raw = _credential_fields_from_payload(payload, provider_norm)
            try:
                credential_fields = _sanitize_credential_fields(
                    provider_norm,
                    credential_fields_raw,
                    allow_base_url=allow_base_url,
                )
            except ValueError:
                logger.warning(
                    "BYOK credential_fields invalid for team_id={} provider={}",
                    team_id,
                    provider_norm,
                )
                raise ByokResolutionError(
                    "invalid_provider_credentials",
                    provider_norm,
                ) from None
            last_used_at = _parse_last_used(row.get("last_used_at"))
            return _finalize_resolution(
                ResolvedByokCredentials(
                    provider=provider_norm,
                    api_key=api_key,
                    app_config=_build_app_config(provider_norm, credential_fields),
                    credential_fields=credential_fields,
                    source="team",
                    allowlisted=True,
                    auth_source=_extract_runtime_auth_source(payload, require_access_for_oauth=True),
                    _touch_cb=_build_touch_cb(
                        provider=provider_norm,
                        last_used_at=last_used_at,
                        repo=shared_repo,
                        scope_type="team",
                        scope_id=int(team_id),
                    ),
                ),
                byok_enabled=byok_enabled,
            )

        for org_id in sorted({int(oid) for oid in org_ids if oid is not None}):
            try:
                row = await shared_repo.fetch_secret("org", int(org_id), provider_norm)
            except _BYOK_CREDENTIAL_STORE_EXCEPTIONS:
                logger.debug(
                    "BYOK org lookup failed for org_id={} provider={}",
                    org_id,
                    provider_norm,
                )
                raise ByokResolutionError(
                    "credential_store_unavailable",
                    provider_norm,
                ) from None
            if row is None:
                continue
            payload = _extract_payload(row, provider_norm)
            api_key = _extract_runtime_api_key(payload)
            if not api_key:
                raise ByokResolutionError("invalid_provider_credentials", provider_norm)
            credential_fields_raw = _credential_fields_from_payload(payload, provider_norm)
            try:
                credential_fields = _sanitize_credential_fields(
                    provider_norm,
                    credential_fields_raw,
                    allow_base_url=allow_base_url,
                )
            except ValueError:
                logger.warning(
                    "BYOK credential_fields invalid for org_id={} provider={}",
                    org_id,
                    provider_norm,
                )
                raise ByokResolutionError(
                    "invalid_provider_credentials",
                    provider_norm,
                ) from None
            last_used_at = _parse_last_used(row.get("last_used_at"))
            return _finalize_resolution(
                ResolvedByokCredentials(
                    provider=provider_norm,
                    api_key=api_key,
                    app_config=_build_app_config(provider_norm, credential_fields),
                    credential_fields=credential_fields,
                    source="org",
                    allowlisted=True,
                    auth_source=_extract_runtime_auth_source(payload, require_access_for_oauth=True),
                    _touch_cb=_build_touch_cb(
                        provider=provider_norm,
                        last_used_at=last_used_at,
                        repo=shared_repo,
                        scope_type="org",
                        scope_id=int(org_id),
                    ),
                ),
                byok_enabled=byok_enabled,
            )

    return _finalize_resolution(
        _fallback_result(
            provider_norm,
            allowlisted=allowlisted,
            fallback_resolver=fallback_resolver,
        ),
        byok_enabled=byok_enabled,
    )
