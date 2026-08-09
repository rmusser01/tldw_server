from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import inspect
import json
import os
import secrets
import sqlite3
import threading
from collections.abc import AsyncIterator, Awaitable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from weakref import WeakKeyDictionary

import asyncpg
from loguru import logger
from redis import asyncio as redis_async

from tldw_Server_API.app.core.AuthNZ.byok_config import (
    PROVIDER_APP_CONFIG_KEYS,
    merge_app_config_overrides,
    runtime_base_url_override_provenance,
)
from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
    get_byok_gateway_spec,
    is_byok_enabled,
    is_provider_allowlisted,
    is_trusted_base_url_request,
    load_server_config_snapshot,
    resolve_byok_base_url_allowlist,
    resolve_server_default_key_from_snapshot,
    validate_base_url_override,
    validate_credential_fields,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseLockError,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    DatabaseError as AuthNZDatabaseError,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_memberships_for_user
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    ProviderCredentialAliasConflictError,
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
)
from tldw_Server_API.app.core.config import loaded_config_data
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_provider_number,
    custom_openai_section_name,
)
from tldw_Server_API.app.core.exceptions import (
    EgressPolicyError,
    NetworkError,
    RetryExhaustedError,
    raise_detached_error,
)
from tldw_Server_API.app.core.http_client import RetryPolicy as _RetryPolicy
from tldw_Server_API.app.core.http_client import afetch as _http_afetch
from tldw_Server_API.app.core.Infrastructure.distributed_lock import FileLock
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.Metrics import increment_counter

DEFAULT_LAST_USED_THROTTLE_SECONDS = 300
DEFAULT_OPENAI_OAUTH_REFRESH_SKEW_SECONDS = 120
DEFAULT_OPENAI_OAUTH_REFRESH_LOCK_BACKEND = "db"
OPENAI_OAUTH_REFRESH_LOCK_TIMEOUT_SECONDS = 45.0
OPENAI_OAUTH_REFRESH_LOCK_TTL_SECONDS = 120
OPENAI_OAUTH_REFRESH_LOCK_RENEW_INTERVAL_SECONDS = (
    OPENAI_OAUTH_REFRESH_LOCK_TTL_SECONDS / 3
)
_OPENAI_PROVIDER = "openai"
_OPENAI_SOURCE_API_KEY = "api_key"
_OPENAI_SOURCE_OAUTH = "oauth"
_OPENAI_CREDENTIAL_VERSION = 2
_OPENAI_OAUTH_GENERATION_FIELDS = (
    "access_token",
)

_BYOK_RESOLUTION_ERROR_CODES = frozenset(
    {
        "invalid_provider_credentials",
        "credential_store_unavailable",
        "credential_scope_revoked",
    }
)
_BYOK_REQUIRED_SOURCES = frozenset(
    {"user", "team", "org", "server_default", "none"}
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
        "api_base",
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
_HUGGINGFACE_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "use_router_url_format",
        "huggingface_use_router_url_format",
        "router_base_url",
        "huggingface_router_base_url",
        "api_chat_path",
        "huggingface_api_chat_path",
    }
)
_REGIONAL_RUNTIME_CONFIG_KEYS = frozenset({"region"})
_OPENROUTER_RUNTIME_CONFIG_KEYS = frozenset({"site_url", "site_name"})
# Only provider-section defaults that adapters consume as bounded generation
# behavior belong here. Request-authority and arbitrary payload fields such as
# headers, tools, user identifiers, logit_bias, and extra_body stay excluded.
_PROVIDER_BEHAVIOR_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "openai": frozenset(
        {
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
            "stop",
        }
    ),
    "cohere": frozenset(
        {
            "temperature",
            "top_p",
            "p",
            "top_k",
            "k",
            "max_tokens",
            "stop_sequences",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "num_generations",
        }
    ),
    "deepseek": frozenset(
        {
            "temperature",
            "top_p",
            "max_tokens",
            "seed",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
            "response_format",
            "n",
        }
    ),
    "google": frozenset(
        {
            "temperature",
            "top_p",
            "topP",
            "top_k",
            "topK",
            "max_output_tokens",
            "max_tokens",
            "stop_sequences",
            "candidate_count",
            "n",
            "response_format",
        }
    ),
    "mistral": frozenset(
        {
            "temperature",
            "top_p",
            "max_tokens",
            "random_seed",
            "top_k",
            "safe_prompt",
            "response_format",
        }
    ),
    "qwen": frozenset(
        {
            "temperature",
            "top_p",
            "max_tokens",
            "seed",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
            "response_format",
            "n",
        }
    ),
    "moonshot": frozenset(
        {"temperature", "top_p", "max_tokens"}
    ),
    "zai": frozenset(
        {"temperature", "top_p", "max_tokens"}
    ),
    "local-llm": frozenset(
        {
            "temperature",
            "streaming",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "seed",
            "stop",
            "response_format",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
            "strict_openai_compat",
        }
    ),
    "llama.cpp": frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "n_predict",
            "seed",
            "stop",
            "response_format",
            "n",
            "n_probs",
            "presence_penalty",
            "frequency_penalty",
        }
    ),
    "kobold": frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "max_length",
            "stop_sequence",
            "num_responses",
            "seed",
            "max_context_length",
            "rep_pen",
        }
    ),
    "ooba": frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "seed",
            "stop",
            "response_format",
            "n",
            "presence_penalty",
            "frequency_penalty",
        }
    ),
    "tabbyapi": frozenset(
        {
            "temperature",
            "temp",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "seed",
            "stop",
            "response_format",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
        }
    ),
    "vllm": frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "seed",
            "stop",
            "response_format",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
        }
    ),
    "aphrodite": frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "seed",
            "stop",
            "response_format",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
        }
    ),
    "ollama": frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "num_predict",
            "seed",
            "stop",
            "format",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
        }
    ),
}

_openai_oauth_refresh_lock_guard = threading.Lock()
_openai_oauth_refresh_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = (
    WeakKeyDictionary()
)

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
_DIRECT_CREDENTIAL_STORE_UNAVAILABLE_EXCEPTIONS = (
    sqlite3.OperationalError,
    sqlite3.InterfaceError,
    asyncpg.InterfaceError,
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    asyncpg.TooManyConnectionsError,
    ConnectionPoolExhaustedError,
    DatabaseLockError,
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
_OPENAI_CREDENTIAL_METADATA_FIELDS = frozenset(
    {
        "organization",
        "organization_id",
        "org_id",
        "project",
        "project_id",
    }
)


def _is_credential_store_unavailable(exc: Exception) -> bool:
    """Return whether an exception represents an operational store outage."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _DIRECT_CREDENTIAL_STORE_UNAVAILABLE_EXCEPTIONS):
            return True
        if not isinstance(current, AuthNZDatabaseError):
            return False
        current = current.__cause__
    return False


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
    provider_norm = canonical_provider_name(provider)
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
        self.provider = canonical_provider_name(provider)
        super().__init__(f"{self.code}: {self.provider}")


@dataclass(frozen=True)
class ServerFallbackCredentials:
    """Atomic server-side fallback credentials for one provider."""

    api_key: str | None
    credential_fields: Mapping[str, Any]
    auth_source: str | None = None
    app_config: Mapping[str, Any] | None = None

    def __repr__(self) -> str:
        """Return a bounded representation without credential-derived data."""
        return "ServerFallbackCredentials(credentials=[REDACTED])"


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
    credential_scope_token: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _touch_cb: Callable[[], Awaitable[None]] | None = None
    _credential_generation: str | None = None

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
            logger.debug(
                "BYOK last_used_at update failed provider={} error_type={}",
                (self.provider or "").strip().lower() or "unknown",
                type(exc).__name__,
            )


@dataclass
class _OpenAIUserResolution:
    payload: dict[str, Any]
    api_key: str | None
    auth_source: str | None
    fail_closed: bool
    credential_generation: str | None

    def __repr__(self) -> str:
        """Return a bounded representation without decrypted OAuth material."""
        return "_OpenAIUserResolution(credentials=[REDACTED])"


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
    provider_norm = canonical_provider_name(provider)
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


async def _fetch_authorized_shared_secret(
    repo: AuthnzOrgProviderSecretsRepo,
    scope_type: str,
    scope_id: int,
    user_id: int,
    provider: str,
) -> dict[str, Any] | None:
    """Atomically bind active membership/entity state to the fetched credential."""
    row = await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id,
        provider,
    )
    if row is not None and row.get("revoked_at") is not None:
        raise ByokResolutionError("invalid_provider_credentials", provider)
    return row


def _fallback_result(
    provider: str,
    *,
    allowlisted: bool,
    fallback_resolver: Callable[[str], str | ServerFallbackCredentials | None] | None,
    fallback_override: str | ServerFallbackCredentials | None,
    server_config_snapshot: Mapping[str, Any],
) -> ResolvedByokCredentials:
    fallback_value = fallback_override
    if fallback_value is None and fallback_resolver is not None:
        try:
            fallback_value = fallback_resolver(provider)
        except ByokResolutionError:
            raise
        except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "BYOK fallback resolver failed provider={} error_type={}",
                provider,
                type(exc).__name__,
            )
            fallback_value = None
    if fallback_value is None:
        fallback_value = resolve_static_server_fallback_from_snapshot(
            provider,
            server_config_snapshot,
        )

    credential_fields: dict[str, Any] = {}
    auth_source: str | None = None
    captured_app_config: dict[str, Any] | None = None
    has_captured_app_config = False
    if isinstance(fallback_value, ServerFallbackCredentials):
        try:
            api_key = _coerce_nonempty_string(fallback_value.api_key)
            auth_source = _coerce_nonempty_string(fallback_value.auth_source)
            if fallback_value.api_key is not None and api_key is None:
                raise ValueError("api_key must be a non-empty string")
            if fallback_value.auth_source is not None and auth_source is None:
                raise ValueError("auth_source must be a non-empty string")
            credential_fields = _sanitize_credential_fields(
                provider,
                dict(fallback_value.credential_fields),
                allow_base_url=True,
            )
            if fallback_value.app_config is not None:
                if not isinstance(fallback_value.app_config, Mapping):
                    raise ValueError("app_config must be an object")
                has_captured_app_config = True
                captured_app_config = _build_app_config(
                    provider,
                    credential_fields,
                    auth_source=auth_source,
                    base_config=fallback_value.app_config,
                )
        except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
            raise_detached_error(
                ByokResolutionError("invalid_provider_credentials", provider)
            )

        valid_api_key_auth = bool(api_key) and auth_source in {None, "api_key"}
        valid_default_chain_auth = (
            provider == "bedrock"
            and api_key is None
            and auth_source == "aws_default_chain"
        )
        valid_config_only_absence = (
            api_key is None
            and auth_source is None
            and not credential_fields
            and has_captured_app_config
        )
        if not (
            valid_api_key_auth
            or valid_default_chain_auth
            or valid_config_only_absence
        ):
            raise ByokResolutionError("invalid_provider_credentials", provider)
    elif isinstance(fallback_value, str):
        api_key = _coerce_nonempty_string(fallback_value)
    else:
        raise ByokResolutionError("invalid_provider_credentials", provider)

    is_resolved = bool(api_key) or auth_source == "aws_default_chain"
    source = "server_default" if is_resolved else "none"
    return ResolvedByokCredentials(
        provider=provider,
        api_key=api_key,
        app_config=(
            captured_app_config
            if has_captured_app_config
            else _build_app_config(
                provider,
                credential_fields,
                auth_source=auth_source,
                base_config=server_config_snapshot,
            )
        ),
        credential_fields=credential_fields,
        source=source,
        allowlisted=allowlisted,
        status=(ByokResolutionStatus.RESOLVED if is_resolved else ByokResolutionStatus.ABSENT),
        auth_source=auth_source,
        _touch_cb=None,
    )


def _gateway_scope_token(*parts: Any) -> str | None:
    """Derive an opaque cache scope from secret-free identity and revision data."""
    if any(part is None or str(part).strip() == "" for part in parts):
        return None
    material = "\x1f".join(("tts-gateway-scope-v1", *(str(part) for part in parts)))
    return hashlib.sha256(material.encode("utf-8"), usedforsecurity=True).hexdigest()


def _gateway_user_credential_revision(row: dict[str, Any]) -> str | None:
    """Return a usage-stable, secret-safe revision for one encrypted credential.

    Explicit credential revisions take precedence. Otherwise the stored
    ciphertext is immediately reduced to a one-way fingerprint; neither the
    ciphertext nor its fingerprint is logged or emitted.
    """
    for field_name in ("revision", "version"):
        value = row.get(field_name)
        if value is not None and str(value).strip():
            return f"{field_name}:{value}"

    ciphertext = row.get("encrypted_blob")
    if not isinstance(ciphertext, str) or not ciphertext:
        return None
    return hashlib.sha256(
        ciphertext.encode("utf-8"),
        usedforsecurity=True,
    ).hexdigest()


def _gateway_admin_credential_revision(api_key: str) -> str:
    """Return a one-way revision for an admin-configured gateway key."""
    return hashlib.sha256(
        api_key.encode("utf-8"),
        usedforsecurity=True,
    ).hexdigest()


def _unavailable_gateway_result(provider: str, *, allowlisted: bool) -> ResolvedByokCredentials:
    return ResolvedByokCredentials(
        provider=provider,
        api_key=None,
        app_config=None,
        credential_fields={},
        source="none",
        allowlisted=allowlisted,
        auth_source=None,
        credential_scope_token=None,
        _touch_cb=None,
    )


def _build_app_config(
    provider: str,
    credential_fields: dict[str, Any],
    *,
    auth_source: str | None = None,
    base_config: Mapping[str, Any] | None = None,
    replace_credential_metadata: bool = False,
) -> dict[str, Any] | None:
    has_authoritative_base_config = base_config is not None
    if base_config is not None:
        base_cfg = base_config
    else:
        try:
            base_cfg = loaded_config_data
        except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
            base_cfg = None
    provider_norm = canonical_provider_name(provider)
    section = PROVIDER_APP_CONFIG_KEYS.get(provider_norm)
    custom_number = custom_openai_provider_number(provider_norm)
    if section is None and custom_number is not None:
        section = custom_openai_section_name(custom_number)
    if section is None:
        section = f"{provider_norm.replace('.', '_').replace('-', '_')}_api"

    scrubbed_cfg: dict[str, Any] = {}
    if base_cfg:
        try:
            if section and isinstance(base_cfg.get(section), dict):
                provider_section = base_cfg.get(section, {})
                behavior_keys = (
                    _PROVIDER_BEHAVIOR_CONFIG_KEYS.get(provider_norm, frozenset())
                    if has_authoritative_base_config
                    else frozenset()
                )
                allowed_provider_keys = _PROVIDER_CONFIG_KEYS | behavior_keys
                if provider_norm == "huggingface":
                    allowed_provider_keys = allowed_provider_keys | _HUGGINGFACE_RUNTIME_CONFIG_KEYS
                elif provider_norm in {"bedrock", "qwen"}:
                    allowed_provider_keys = allowed_provider_keys | _REGIONAL_RUNTIME_CONFIG_KEYS
                elif provider_norm == "openrouter":
                    allowed_provider_keys = allowed_provider_keys | _OPENROUTER_RUNTIME_CONFIG_KEYS
                cleaned_provider = {
                    k: copy.deepcopy(v)
                    for k, v in provider_section.items()
                    if k in allowed_provider_keys
                }
                if (
                    has_authoritative_base_config
                    and provider_norm == "cohere"
                    and provider_section.get("top_p") is None
                    and provider_section.get("p") is None
                    and provider_section.get("max_p") is not None
                ):
                    # The config loader emits ``max_p`` while Cohere consumes
                    # ``top_p``/``p``. Normalize at the snapshot boundary so
                    # adapters receive one live spelling instead of a dead alias.
                    cleaned_provider["top_p"] = copy.deepcopy(
                        provider_section["max_p"]
                    )
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
    if replace_credential_metadata and provider_norm == _OPENAI_PROVIDER and section:
        selected = scrubbed_cfg.get(section)
        if isinstance(selected, dict):
            selected_config = {
                key: value
                for key, value in selected.items()
                if key not in _OPENAI_CREDENTIAL_METADATA_FIELDS
            }
            if selected_config:
                scrubbed_cfg[section] = selected_config
            else:
                scrubbed_cfg.pop(section, None)
    merged = merge_app_config_overrides(scrubbed_cfg or None, provider, credential_fields)
    credential_base_url = credential_fields.get("base_url")
    if (
        provider_norm == "huggingface"
        and isinstance(credential_base_url, str)
        and credential_base_url.strip()
        and section
    ):
        selected = merged.get(section)
        selected_config = dict(selected) if isinstance(selected, dict) else {}
        selected_config["_runtime_base_url_override"] = runtime_base_url_override_provenance()
        merged[section] = selected_config
    if auth_source is not None and section:
        selected = merged.get(section)
        selected_config = dict(selected) if isinstance(selected, dict) else {}
        selected_config["_runtime_auth_source"] = auth_source
        merged[section] = selected_config
    return merged or None


def merge_server_fallback_snapshot(
    provider: str,
    base_fallback: ServerFallbackCredentials,
    *,
    api_key: str | None,
    credential_fields: Mapping[str, Any],
    auth_source: str | None,
    provider_config: Mapping[str, Any],
    replace_credential_metadata: bool = False,
) -> ServerFallbackCredentials:
    """Overlay one flat provider override onto an already frozen fallback."""
    provider_norm = canonical_provider_name(provider)
    if not isinstance(base_fallback, ServerFallbackCredentials):
        raise ByokResolutionError("invalid_provider_credentials", provider_norm)
    if base_fallback.app_config is not None and not isinstance(
        base_fallback.app_config,
        Mapping,
    ):
        raise ByokResolutionError("invalid_provider_credentials", provider_norm)
    if not isinstance(credential_fields, Mapping) or not isinstance(
        provider_config,
        Mapping,
    ):
        raise ByokResolutionError("invalid_provider_credentials", provider_norm)

    try:
        base_config = copy.deepcopy(dict(base_fallback.app_config or {}))
        flat_override = copy.deepcopy(dict(provider_config))
        section = PROVIDER_APP_CONFIG_KEYS.get(provider_norm)
        custom_number = custom_openai_provider_number(provider_norm)
        if section is None and custom_number is not None:
            section = custom_openai_section_name(custom_number)
        if section is None:
            section = f"{provider_norm.replace('.', '_').replace('-', '_')}_api"

        selected = base_config.get(section)
        selected_config = dict(selected) if isinstance(selected, dict) else {}
        default_model = flat_override.pop("default_model", None)
        flat_override.pop("auth_source", None)
        selected_config.update(flat_override)
        if isinstance(default_model, str) and default_model.strip():
            selected_config["model"] = default_model.strip()
        base_config[section] = selected_config
        selected_fields = copy.deepcopy(dict(credential_fields))
        app_config = _build_app_config(
            provider_norm,
            selected_fields,
            auth_source=auth_source,
            base_config=base_config,
            replace_credential_metadata=replace_credential_metadata,
        )
    except ByokResolutionError:
        raise
    except Exception:  # noqa: BLE001 - malformed stored configuration fails closed
        raise_detached_error(
            ByokResolutionError(
                "invalid_provider_credentials",
                provider_norm,
            )
        )

    return ServerFallbackCredentials(
        api_key=api_key,
        credential_fields=selected_fields,
        auth_source=auth_source,
        app_config=app_config or {},
    )


def resolve_static_server_fallback_from_snapshot(
    provider: str,
    config_snapshot: Mapping[str, Any],
) -> ServerFallbackCredentials:
    """Build an atomic server fallback from a caller-owned config snapshot."""
    provider_norm = canonical_provider_name(provider)
    frozen_snapshot = copy.deepcopy(dict(config_snapshot))
    api_key = resolve_server_default_key_from_snapshot(
        provider_norm,
        frozen_snapshot,
    )
    auth_source = (
        "aws_default_chain"
        if provider_norm == "bedrock" and api_key is None
        else None
    )
    app_config = _build_app_config(
        provider_norm,
        {},
        auth_source=auth_source,
        base_config=frozen_snapshot,
    )
    return ServerFallbackCredentials(
        api_key=api_key,
        credential_fields={},
        auth_source=auth_source,
        # An empty mapping is an authoritative snapshot too: it prevents a
        # second lazy config read after the static key decision.
        app_config=app_config or {},
    )


def resolve_static_server_fallback(provider: str) -> ServerFallbackCredentials:
    """Capture a static server key and adapter config from one config load."""
    return resolve_static_server_fallback_from_snapshot(
        provider,
        load_server_config_snapshot(),
    )


def _extract_payload(row: dict[str, Any], provider: str) -> dict[str, Any]:
    encrypted_blob = row.get("encrypted_blob")
    if not encrypted_blob:
        raise ByokResolutionError("invalid_provider_credentials", provider)
    try:
        payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
        logger.warning("BYOK decrypt failed for provider={}", provider)
        raise_detached_error(
            ByokResolutionError("invalid_provider_credentials", provider)
        )
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


def _openai_refresh_advisory_lock_id(lock_key: str) -> int:
    """Return a stable signed PostgreSQL advisory-lock identifier."""
    digest = hashlib.blake2b(lock_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


class _PostgresConnectionBoundPool:
    """Expose one advisory-lock connection through the repository's pool API."""

    pool = object()

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        """Run repository mutations on the advisory-lock-owning connection."""
        async with self._connection.transaction():
            yield self._connection

    async def fetchone(self, query: str, *args: Any) -> dict[str, Any] | None:
        row = await self._connection.fetchrow(query, *args)
        return dict(row) if row else None

    async def fetchall(self, query: str, *args: Any) -> list[dict[str, Any]]:
        """Read a repository row set on the advisory-lock-owning connection."""
        rows = await self._connection.fetch(query, *args)
        return [dict(row) for row in rows]

    async def execute(self, query: str, *args: Any) -> Any:
        """Execute one mutation on the advisory-lock-owning connection."""
        return await self._connection.execute(query, *args)


async def _await_lock_cleanup(awaitable: Awaitable[Any], *, operation: str) -> Any:
    """Finish lock cleanup before propagating cancellation to the caller."""
    cleanup_task = asyncio.create_task(awaitable)
    try:
        return await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
        try:
            cleanup_task.result()
        except asyncio.CancelledError:
            logger.warning("OpenAI OAuth lock cleanup was cancelled operation={}", operation)
        except Exception as exc:  # noqa: BLE001 - cleanup errors are logged by type only
            logger.warning(
                "OpenAI OAuth lock cleanup failed operation={} error_type={}",
                operation,
                type(exc).__name__,
            )
        raise


async def _wait_for_sync_lock(lock: FileLock, *, provider: str) -> None:
    """Acquire a non-blocking native lock without blocking the event loop."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + OPENAI_OAUTH_REFRESH_LOCK_TIMEOUT_SECONDS
    while True:
        # timeout=0 makes this one native non-blocking lock attempt. Calling it
        # inline avoids a cancellation race where an abandoned worker thread
        # could acquire the lock after cleanup already ran.
        if lock.acquire():
            return
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise ByokResolutionError("credential_store_unavailable", provider)
        await asyncio.sleep(min(0.05, remaining))


@contextlib.asynccontextmanager
async def _openai_db_refresh_lock(*, lock_key: str, provider: str):
    """Use PostgreSQL advisory locks or a native SQLite-process file lock."""
    try:
        pool = await get_db_pool()
        backend = pool.backend_type
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - backend failures must be sanitized
        raise_detached_error(
            ByokResolutionError("credential_store_unavailable", provider)
        )

    if backend == "postgres":
        lock_id = _openai_refresh_advisory_lock_id(lock_key)
        protected_body_failed = False
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + OPENAI_OAUTH_REFRESH_LOCK_TIMEOUT_SECONDS
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise ByokResolutionError(
                        "credential_store_unavailable",
                        provider,
                    )
                acquired = False
                lock_attempted = False
                lock_attempt_finished = False
                try:
                    async with pool.acquire_openai_credential_lock_connection(
                        timeout=remaining
                    ) as conn:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise ByokResolutionError(
                                "credential_store_unavailable",
                                provider,
                            )
                        lock_attempted = True
                        try:
                            acquired = bool(
                                await asyncio.wait_for(
                                    conn.fetchval(
                                        "SELECT pg_try_advisory_lock($1)",
                                        lock_id,
                                    ),
                                    timeout=remaining,
                                )
                            )
                            lock_attempt_finished = True
                            if acquired:
                                # The winner must reload and CAS through this
                                # connection. Otherwise a pool-sized refresh
                                # burst can deadlock the winner's second
                                # acquisition.
                                try:
                                    yield AuthnzUserProviderSecretsRepo(
                                        _PostgresConnectionBoundPool(conn)
                                    )
                                except BaseException:
                                    protected_body_failed = True
                                    raise
                                else:
                                    return
                        finally:
                            if acquired or (
                                lock_attempted and not lock_attempt_finished
                            ):
                                try:
                                    unlock_confirmed = bool(
                                        await _await_lock_cleanup(
                                            conn.fetchval(
                                                "SELECT pg_advisory_unlock($1)",
                                                lock_id,
                                            ),
                                            operation="postgres_unlock",
                                        )
                                    )
                                except asyncio.CancelledError:
                                    raise
                                except Exception as exc:  # noqa: BLE001 - preserve body error
                                    if not protected_body_failed:
                                        raise_detached_error(
                                            ByokResolutionError(
                                                "credential_store_unavailable",
                                                provider,
                                            )
                                        )
                                    logger.warning(
                                        "OpenAI PostgreSQL advisory unlock failed while "
                                        "propagating a protected-body error error_type={}",
                                        type(exc).__name__,
                                    )
                                else:
                                    if not unlock_confirmed:
                                        if not protected_body_failed:
                                            raise ByokResolutionError(
                                                "credential_store_unavailable",
                                                provider,
                                            )
                                        logger.warning(
                                            "OpenAI PostgreSQL advisory unlock was not "
                                            "confirmed while propagating a protected-body error"
                                        )
                except asyncio.TimeoutError:
                    raise_detached_error(
                        ByokResolutionError(
                            "credential_store_unavailable",
                            provider,
                        )
                    )
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise ByokResolutionError(
                        "credential_store_unavailable",
                        provider,
                    )
                await asyncio.sleep(min(0.05, remaining))
        except asyncio.CancelledError:
            raise
        except ByokResolutionError:
            raise
        except Exception:
            if protected_body_failed:
                raise
            raise_detached_error(
                ByokResolutionError("credential_store_unavailable", provider)
            )
        return

    lock_dir = Path(
        os.getenv("OPENAI_OAUTH_REFRESH_LOCK_DIR")
        or (Path.home() / ".tldw" / "locks")
    )
    try:
        lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        raise_detached_error(
            ByokResolutionError("credential_store_unavailable", provider)
        )
    lock_name = hashlib.sha256(lock_key.encode("utf-8")).hexdigest()
    file_lock = FileLock(
        lock_dir / f"openai-oauth-refresh-{lock_name}.lock",
        timeout=0,
    )
    try:
        await _wait_for_sync_lock(file_lock, provider=provider)
    except asyncio.CancelledError:
        raise
    except ByokResolutionError:
        raise
    except Exception:  # noqa: BLE001 - native lock failures must be sanitized
        raise_detached_error(
            ByokResolutionError("credential_store_unavailable", provider)
        )

    try:
        # Exceptions from the protected mutation belong to the caller. Only
        # acquisition failures are credential-store failures at this boundary.
        yield None
    finally:
        file_lock.release()


def _openai_oauth_redis_client():
    """Build the configured async Redis client or fail closed."""
    try:
        redis_url = _coerce_nonempty_string(getattr(get_settings(), "REDIS_URL", None))
    except _BYOK_RUNTIME_NONCRITICAL_EXCEPTIONS:
        redis_url = None
    if redis_url is None:
        raise ByokResolutionError("credential_store_unavailable", _OPENAI_PROVIDER)
    return redis_async.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


async def _close_openai_oauth_redis_client(client: Any) -> None:
    """Close async Redis clients across redis-py 4.x and 5.x APIs."""
    close = getattr(client, "aclose", None)
    if not callable(close):
        close = getattr(client, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


@contextlib.asynccontextmanager
async def _openai_redis_refresh_lock(*, lock_key: str, provider: str):
    """Serialize OAuth refreshes through an expiring Redis ownership token."""
    redis_key = f"tldw:openai-oauth-refresh:{hashlib.sha256(lock_key.encode('utf-8')).hexdigest()}"
    token = secrets.token_hex(32)
    client = None
    acquired = False
    renewal_task: asyncio.Task[None] | None = None
    lease_lost = False
    release_confirmed = False
    release_script = (
        "if redis.call('get', KEYS[1]) == ARGV[1] then "
        "return redis.call('del', KEYS[1]) else return 0 end"
    )
    renew_script = (
        "if redis.call('get', KEYS[1]) == ARGV[1] then "
        "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"
    )
    try:
        try:
            client = _openai_oauth_redis_client()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + OPENAI_OAUTH_REFRESH_LOCK_TIMEOUT_SECONDS
            while True:
                acquired = bool(
                    await client.set(
                        redis_key,
                        token,
                        ex=OPENAI_OAUTH_REFRESH_LOCK_TTL_SECONDS,
                        nx=True,
                    )
                )
                if acquired:
                    break
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise ByokResolutionError(
                        "credential_store_unavailable",
                        provider,
                    )
                await asyncio.sleep(min(0.05, remaining))

            owner_task = asyncio.current_task()

            async def _renew_redis_lease() -> None:
                nonlocal lease_lost
                try:
                    while True:
                        await asyncio.sleep(
                            OPENAI_OAUTH_REFRESH_LOCK_RENEW_INTERVAL_SECONDS
                        )
                        renewed = bool(
                            await client.eval(
                                renew_script,
                                1,
                                redis_key,
                                token,
                                OPENAI_OAUTH_REFRESH_LOCK_TTL_SECONDS,
                            )
                        )
                        if renewed:
                            continue
                        break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - lease failures are fail-closed
                    logger.warning(
                        "OpenAI OAuth Redis lock renewal failed error_type={}",
                        type(exc).__name__,
                    )
                lease_lost = True
                if owner_task is not None and not owner_task.done():
                    owner_task.cancel()

            renewal_task = asyncio.create_task(_renew_redis_lease())
        except asyncio.CancelledError:
            raise
        except ByokResolutionError:
            raise
        except Exception:  # noqa: BLE001 - Redis failures must be sanitized
            raise_detached_error(
                ByokResolutionError("credential_store_unavailable", provider)
            )

        try:
            yield
        except asyncio.CancelledError:
            if lease_lost:
                raise_detached_error(
                    ByokResolutionError(
                        "credential_store_unavailable",
                        provider,
                    )
                )
            raise
    finally:
        if client is not None:
            async def _cleanup_redis_lock() -> bool:
                if renewal_task is not None:
                    renewal_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await renewal_task
                # SET may have succeeded server-side even if cancellation or
                # transport failure prevented its result from reaching us.
                # The Lua token check makes an unconditional release attempt
                # safe for both certain and uncertain acquisition outcomes.
                try:
                    released = bool(
                        await client.eval(release_script, 1, redis_key, token)
                    )
                except Exception as exc:  # noqa: BLE001 - final release is best effort
                    logger.warning(
                        "OpenAI OAuth Redis lock release failed error_type={}",
                        type(exc).__name__,
                    )
                    released = False
                try:
                    await _close_openai_oauth_redis_client(client)
                except Exception as exc:  # noqa: BLE001 - close must not skip release result
                    logger.debug(
                        "OpenAI OAuth Redis client close failed error_type={}",
                        type(exc).__name__,
                    )
                return released

            try:
                release_confirmed = bool(
                    await _await_lock_cleanup(
                        _cleanup_redis_lock(),
                        operation="redis_release",
                    )
                )
            except asyncio.CancelledError:
                if lease_lost:
                    raise_detached_error(
                        ByokResolutionError(
                            "credential_store_unavailable",
                            provider,
                        )
                    )
                raise

    if lease_lost or not release_confirmed:
        raise ByokResolutionError(
            "credential_store_unavailable",
            provider,
        )


@contextlib.asynccontextmanager
async def openai_credential_mutation_lock(
    *,
    user_id: int,
    provider: str = _OPENAI_PROVIDER,
) -> AsyncIterator[AuthnzUserProviderSecretsRepo | None]:
    """Serialize one whole-row OpenAI credential mutation across workers."""
    provider_norm = canonical_provider_name(provider)
    if provider_norm != _OPENAI_PROVIDER:
        raise ByokResolutionError("invalid_provider_credentials", provider_norm)
    backend = _openai_oauth_refresh_lock_backend()
    lock_key = _openai_refresh_lock_key(user_id=user_id, provider=provider_norm)
    if backend == "db":
        async with _openai_db_refresh_lock(
            lock_key=lock_key,
            provider=provider_norm,
        ) as locked_user_repo:
            yield locked_user_repo
        return
    if backend == "redis":
        async with _openai_redis_refresh_lock(lock_key=lock_key, provider=provider_norm):
            yield None
        return
    lock = _get_openai_refresh_lock(lock_key)
    async with lock:
        yield None


@contextlib.asynccontextmanager
async def _openai_oauth_refresh_lock(*, user_id: int, provider: str):
    """Compatibility wrapper for the shared credential-mutation lock."""
    async with openai_credential_mutation_lock(
        user_id=user_id,
        provider=provider,
    ) as locked_user_repo:
        yield locked_user_repo


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


def openai_oauth_credential_generation(payload: dict[str, Any]) -> str | None:
    """Return a non-secret access-token generation digest for refresh coalescing."""
    oauth_payload = _openai_source_payload(payload, _OPENAI_SOURCE_OAUTH)
    values = tuple(
        _coerce_nonempty_string(oauth_payload.get(field))
        for field in _OPENAI_OAUTH_GENERATION_FIELDS
    )
    if not any(values):
        return None
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def openai_oauth_refresh_state_generation(
    payload: dict[str, Any] | None,
) -> str | None:
    """Return an opaque digest proving publication of a refreshed access token."""
    oauth_payload = _openai_source_payload(payload, _OPENAI_SOURCE_OAUTH)
    if not oauth_payload:
        return None
    values = (
        _coerce_nonempty_string(oauth_payload.get("access_token")),
        _coerce_nonempty_string(oauth_payload.get("issued_at")),
    )
    if not any(values):
        return None
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _openai_oauth_generation(payload: dict[str, Any]) -> str | None:
    """Compatibility wrapper for the public OAuth credential generation seam."""
    return openai_oauth_credential_generation(payload)


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
        raise_detached_error(
            ByokResolutionError("invalid_provider_credentials", provider)
        )

    metadata_to_store = _parse_metadata_value(row.get("metadata"))
    try:
        updated = await repo.update_secret_if_active_and_unchanged(
            user_id=user_id,
            provider=provider,
            encrypted_blob=dumps_envelope(envelope),
            expected_encrypted_blob=str(row.get("encrypted_blob") or ""),
            key_hint=key_hint or None,
            metadata=metadata_to_store,
            updated_at=updated_at,
            updated_by=user_id,
        )
        if not updated:
            raise ByokResolutionError("invalid_provider_credentials", provider)
    except ProviderCredentialAliasConflictError:
        raise_detached_error(
            ByokResolutionError("invalid_provider_credentials", provider)
        )
    except ByokResolutionError:
        raise
    except Exception as exc:
        if not _is_credential_store_unavailable(exc):
            raise
        logger.warning(
            "BYOK payload persist failed for user_id={} provider={}",
            user_id,
            provider,
        )
        raise_detached_error(
            ByokResolutionError("credential_store_unavailable", provider)
        )


async def _resolve_openai_user_payload(
    *,
    user_repo: AuthnzUserProviderSecretsRepo,
    user_id: int,
    row: dict[str, Any],
    payload: dict[str, Any],
    force_oauth_refresh: bool,
    rejected_credential_generation: str | None,
) -> _OpenAIUserResolution:
    merged_payload = _coerce_openai_payload_v2(payload)
    rejected_generation = (
        _coerce_nonempty_string(rejected_credential_generation)
        or openai_oauth_credential_generation(merged_payload)
    )
    rejected_refresh_state_generation = openai_oauth_refresh_state_generation(
        merged_payload
    )
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
        async with _openai_oauth_refresh_lock(
            user_id=user_id,
            provider=_OPENAI_PROVIDER,
        ) as locked_user_repo:
            refresh_repo = locked_user_repo or user_repo
            try:
                latest_row = await refresh_repo.fetch_secret_for_active_user(
                    int(user_id),
                    _OPENAI_PROVIDER,
                )
            except ProviderCredentialAliasConflictError:
                raise_detached_error(
                    ByokResolutionError(
                        "invalid_provider_credentials",
                        _OPENAI_PROVIDER,
                    )
                )
            except Exception as exc:
                if not _is_credential_store_unavailable(exc):
                    raise
                logger.debug(
                    "BYOK user reload before OAuth refresh failed for user_id={} provider={}",
                    user_id,
                    _OPENAI_PROVIDER,
                )
                raise_detached_error(
                    ByokResolutionError(
                        "credential_store_unavailable",
                        _OPENAI_PROVIDER,
                    )
                )

            if latest_row is None:
                raise ByokResolutionError(
                    "invalid_provider_credentials",
                    _OPENAI_PROVIDER,
                )
            latest_payload = _extract_payload(latest_row, _OPENAI_PROVIDER)
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
                # Only an OAuth-generation change proves another resolver
                # published a winning rotation while this request waited.
                # Unrelated row edits must not suppress a requested refresh.
                force_oauth_refresh=(
                    force_oauth_refresh
                    and openai_oauth_credential_generation(merged_payload)
                    == rejected_generation
                    and openai_oauth_refresh_state_generation(merged_payload)
                    == rejected_refresh_state_generation
                ),
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
                            repo=refresh_repo,
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
                            repo=refresh_repo,
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
                            credential_generation=openai_oauth_credential_generation(merged_payload),
                        )

                    return _OpenAIUserResolution(
                        payload=merged_payload,
                        api_key=None,
                        auth_source=None,
                        fail_closed=True,
                        credential_generation=openai_oauth_credential_generation(merged_payload),
                    )

    if runtime_api_key:
        return _OpenAIUserResolution(
            payload=merged_payload,
            api_key=runtime_api_key,
            auth_source=runtime_auth_source,
            fail_closed=False,
            credential_generation=openai_oauth_credential_generation(merged_payload),
        )

    if _openai_has_any_credentials(merged_payload):
        return _OpenAIUserResolution(
            payload=merged_payload,
            api_key=None,
            auth_source=None,
            fail_closed=True,
            credential_generation=openai_oauth_credential_generation(merged_payload),
        )

    return _OpenAIUserResolution(
        payload=merged_payload,
        api_key=None,
        auth_source=None,
        fail_closed=False,
        credential_generation=openai_oauth_credential_generation(merged_payload),
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


async def resolve_gateway_byok_credentials(
    backend: str,
    *,
    user_id: int | None,
    gateway_spec: Any | None = None,
) -> ResolvedByokCredentials:
    """Resolve one gateway attempt without accepting user-controlled authority.

    A present user record is authoritative, including when its payload is
    malformed or missing key material. Callers resolve each fallback backend by
    invoking this function again with that backend's current spec.
    """
    provider_norm = canonical_provider_name(backend)
    spec = gateway_spec if gateway_spec is not None else get_byok_gateway_spec(provider_norm)
    if (
        spec is None
        or not bool(getattr(spec, "enabled", False))
        or getattr(spec, "backend_id", None) != provider_norm
    ):
        return _unavailable_gateway_result(provider_norm, allowlisted=False)

    allow_user_key = bool(getattr(spec, "allow_user_api_key", False))
    if allow_user_key and user_id is not None and is_byok_enabled():
        try:
            user_repo = await _get_user_repo()
            user_row = await user_repo.fetch_secret_for_user(int(user_id), provider_norm)
            payload = None
            if user_row is not None:
                encrypted_blob = user_row.get("encrypted_blob")
                if encrypted_blob:
                    payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
        except Exception as exc:  # noqa: BLE001 - fail closed on repository/crypto drivers
            logger.debug(
                "Gateway BYOK user lookup failed for provider={}: {}",
                provider_norm,
                type(exc).__name__,
            )
            return _unavailable_gateway_result(provider_norm, allowlisted=True)

        if user_row is not None:
            api_key = None
            if payload is not None:
                api_key = _legacy_payload_api_key(payload) or _v2_payload_api_key(payload)
            scope_token = None
            if api_key:
                revision = _gateway_user_credential_revision(user_row)
                scope_token = _gateway_scope_token(
                    provider_norm,
                    user_row.get("id"),
                    revision,
                )
            return ResolvedByokCredentials(
                provider=provider_norm,
                api_key=api_key,
                app_config=None,
                credential_fields={},
                source="user",
                allowlisted=True,
                auth_source="api_key" if api_key else None,
                credential_scope_token=scope_token,
                _touch_cb=(
                    _build_touch_cb(
                        provider=provider_norm,
                        last_used_at=_parse_last_used(user_row.get("last_used_at")),
                        repo=user_repo,
                        user_id=int(user_id),
                    )
                    if api_key
                    else None
                ),
            )

    admin_key = _coerce_nonempty_string(getattr(spec, "api_key", None))
    if not admin_key:
        return _unavailable_gateway_result(
            provider_norm,
            allowlisted=allow_user_key,
        )
    return ResolvedByokCredentials(
        provider=provider_norm,
        api_key=admin_key,
        app_config=None,
        credential_fields={},
        source="server_default",
        allowlisted=allow_user_key,
        auth_source="api_key",
        credential_scope_token=_gateway_scope_token(
            provider_norm,
            getattr(spec, "config_generation", None),
            _gateway_admin_credential_revision(admin_key),
        ),
        _touch_cb=None,
    )


async def resolve_byok_credentials(
    provider: str,
    *,
    user_id: int | None,
    request: Any | None = None,
    team_ids: list[int] | None = None,
    org_ids: list[int] | None = None,
    fallback_resolver: Callable[[str], str | ServerFallbackCredentials | None] | None = None,
    fallback_override: str | ServerFallbackCredentials | None = None,
    server_config_snapshot: Mapping[str, Any] | None = None,
    force_oauth_refresh: bool = False,
    rejected_credential_generation: str | None = None,
    trusted_base_url_override: bool | None = None,
    required_source: str | None = None,
) -> ResolvedByokCredentials:
    provider_norm = canonical_provider_name(provider)
    if required_source is not None:
        if type(required_source) is not str or required_source not in _BYOK_REQUIRED_SOURCES:
            raise ByokResolutionError(
                "invalid_provider_credentials",
                provider_norm,
            ) from None
        valid_scope_shape = (
            isinstance(team_ids, list)
            and isinstance(org_ids, list)
            and (
                (
                    required_source == "team"
                    and len(team_ids) == 1
                    and not org_ids
                    and type(team_ids[0]) is int
                    and team_ids[0] > 0
                )
                or (
                    required_source == "org"
                    and len(org_ids) == 1
                    and not team_ids
                    and type(org_ids[0]) is int
                    and org_ids[0] > 0
                )
                or (
                    required_source in {"user", "server_default", "none"}
                    and not team_ids
                    and not org_ids
                )
            )
        )
        if not valid_scope_shape:
            raise ByokResolutionError(
                "invalid_provider_credentials",
                provider_norm,
            ) from None
    if server_config_snapshot is not None and fallback_resolver is not None:
        raise ByokResolutionError("invalid_provider_credentials", provider_norm)
    if fallback_override is not None and fallback_resolver is not None:
        raise ByokResolutionError("invalid_provider_credentials", provider_norm)
    try:
        frozen_server_config = copy.deepcopy(
            dict(
                server_config_snapshot
                if server_config_snapshot is not None
                else load_server_config_snapshot()
            )
        )
    except Exception:  # noqa: BLE001 - malformed server snapshots fail closed
        raise_detached_error(
            ByokResolutionError("invalid_provider_credentials", provider_norm)
        )
    byok_enabled = is_byok_enabled()
    allowlisted = is_provider_allowlisted(provider_norm)
    allow_base_url = _can_use_base_url_override(
        provider_norm,
        request,
        trusted_base_url_override,
    )

    if (
        not byok_enabled
        or user_id is None
        or not allowlisted
        or required_source in {"server_default", "none"}
    ):
        return _finalize_resolution(
            _fallback_result(
                provider_norm,
                allowlisted=allowlisted,
                fallback_resolver=fallback_resolver,
                fallback_override=fallback_override,
                server_config_snapshot=frozen_server_config,
            ),
            byok_enabled=byok_enabled,
        )

    # Resolve user key only when the caller did not bind another source.
    user_repo = None
    user_row = None
    if required_source in {None, "user"}:
        try:
            user_repo = await _get_user_repo()
            user_row = await user_repo.fetch_secret_for_active_user(
                int(user_id),
                provider_norm,
                include_revoked=True,
            )
            if user_row is not None and user_row.get("revoked_at") is not None:
                raise ByokResolutionError("invalid_provider_credentials", provider_norm)
            if user_row is None:
                unrestricted_user_row = await user_repo.fetch_secret_for_user(
                    int(user_id),
                    provider_norm,
                    include_revoked=True,
                )
                if unrestricted_user_row is not None:
                    raise_detached_error(
                        ByokResolutionError(
                            "invalid_provider_credentials",
                            provider_norm,
                        )
                    )
        except ByokResolutionError:
            raise
        except ProviderCredentialAliasConflictError:
            raise_detached_error(
                ByokResolutionError("invalid_provider_credentials", provider_norm)
            )
        except Exception as exc:
            if not _is_credential_store_unavailable(exc):
                raise
            logger.debug(
                "BYOK user lookup failed for user_id={} provider={}",
                user_id,
                provider_norm,
            )
            raise_detached_error(
                ByokResolutionError("credential_store_unavailable", provider_norm)
            )

    if user_row is not None:
        payload = _extract_payload(user_row, provider_norm)
        if payload:
            runtime_payload = payload
            credential_generation = None
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
                    rejected_credential_generation=rejected_credential_generation,
                )
                runtime_payload = openai_resolution.payload
                runtime_auth_source = openai_resolution.auth_source
                credential_generation = openai_resolution.credential_generation
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
                    raise_detached_error(
                        ByokResolutionError(
                            "invalid_provider_credentials",
                            provider_norm,
                        )
                    )
                last_used_at = _parse_last_used(user_row.get("last_used_at"))
                return _finalize_resolution(
                    ResolvedByokCredentials(
                        provider=provider_norm,
                        api_key=api_key,
                        app_config=_build_app_config(
                            provider_norm,
                            credential_fields,
                            base_config=frozen_server_config,
                            replace_credential_metadata=True,
                        ),
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
                        _credential_generation=credential_generation,
                    ),
                    byok_enabled=byok_enabled,
                )

    if required_source == "user":
        team_ids = []
        org_ids = []
    elif required_source == "team":
        org_ids = []
    elif required_source == "org":
        team_ids = []

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
        except Exception as exc:
            if not _is_credential_store_unavailable(exc):
                raise
            logger.debug("BYOK membership lookup failed for user_id={}", user_id)
            raise_detached_error(
                ByokResolutionError("credential_store_unavailable", provider_norm)
            )

    team_ids = team_ids or []
    org_ids = org_ids or []
    team_ids = _apply_active_scope(team_ids, active_team_id, provider=provider_norm)
    org_ids = _apply_active_scope(org_ids, active_org_id, provider=provider_norm)

    shared_repo = None
    if team_ids or org_ids:
        try:
            shared_repo = await _get_org_repo()
        except Exception as exc:
            if not _is_credential_store_unavailable(exc):
                raise
            logger.debug("BYOK shared repo init failed for provider={}", provider_norm)
            raise_detached_error(
                ByokResolutionError("credential_store_unavailable", provider_norm)
            )

    if shared_repo:
        # Prefer team scope over org scope, mirroring list_user_provider_keys()
        for team_id in sorted({int(tid) for tid in team_ids if tid is not None}):
            try:
                row = await _fetch_authorized_shared_secret(
                    shared_repo,
                    "team",
                    int(team_id),
                    int(user_id),
                    provider_norm,
                )
            except ByokResolutionError:
                raise
            except ProviderCredentialAliasConflictError:
                raise_detached_error(
                    ByokResolutionError("invalid_provider_credentials", provider_norm)
                )
            except Exception as exc:
                if not _is_credential_store_unavailable(exc):
                    raise
                logger.debug(
                    "BYOK team lookup failed for team_id={} provider={}",
                    team_id,
                    provider_norm,
                )
                raise_detached_error(
                    ByokResolutionError(
                        "credential_store_unavailable",
                        provider_norm,
                    )
                )
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
                raise_detached_error(
                    ByokResolutionError(
                        "invalid_provider_credentials",
                        provider_norm,
                    )
                )
            last_used_at = _parse_last_used(row.get("last_used_at"))
            return _finalize_resolution(
                ResolvedByokCredentials(
                    provider=provider_norm,
                    api_key=api_key,
                    app_config=_build_app_config(
                        provider_norm,
                        credential_fields,
                        base_config=frozen_server_config,
                        replace_credential_metadata=True,
                    ),
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
                row = await _fetch_authorized_shared_secret(
                    shared_repo,
                    "org",
                    int(org_id),
                    int(user_id),
                    provider_norm,
                )
            except ByokResolutionError:
                raise
            except ProviderCredentialAliasConflictError:
                raise_detached_error(
                    ByokResolutionError("invalid_provider_credentials", provider_norm)
                )
            except Exception as exc:
                if not _is_credential_store_unavailable(exc):
                    raise
                logger.debug(
                    "BYOK org lookup failed for org_id={} provider={}",
                    org_id,
                    provider_norm,
                )
                raise_detached_error(
                    ByokResolutionError(
                        "credential_store_unavailable",
                        provider_norm,
                    )
                )
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
                raise_detached_error(
                    ByokResolutionError(
                        "invalid_provider_credentials",
                        provider_norm,
                    )
                )
            last_used_at = _parse_last_used(row.get("last_used_at"))
            return _finalize_resolution(
                ResolvedByokCredentials(
                    provider=provider_norm,
                    api_key=api_key,
                    app_config=_build_app_config(
                        provider_norm,
                        credential_fields,
                        base_config=frozen_server_config,
                        replace_credential_metadata=True,
                    ),
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
            fallback_override=fallback_override,
            server_config_snapshot=frozen_server_config,
        ),
        byok_enabled=byok_enabled,
    )
