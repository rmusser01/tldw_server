from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from .modules.base import BaseModule


class InvalidParamsException(Exception):
    """Raised when tool parameters fail validation or validators are missing for write tools."""
    pass


class GovernanceDeniedError(PermissionError):
    """Permission error carrying structured governance decision details."""

    def __init__(self, message: str, governance: dict[str, Any] | None = None):
        super().__init__(message)
        self.governance = governance or {}


class ApprovalRequiredError(PermissionError):
    """Permission error carrying structured MCP Hub approval request details."""

    def __init__(self, message: str, approval: dict[str, Any] | None = None):
        super().__init__(message)
        self.approval = approval or {}


class RequestContext:
    """Context for request processing.

    Request contexts store caller metadata and explicit database path mappings
    only. Host-specific database path resolution is owned by MCPProtocol or
    MCPServer dependencies so standalone callers do not import tldw_server
    adapters through this neutral context object.
    """
    def __init__(
        self,
        request_id: str,
        user_id: str | None = None,
        client_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        db_paths: dict[str, str] | None = None,
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.client_id = client_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.start_time = datetime.now(timezone.utc)
        self.db_paths = dict(db_paths or {})
        # Build a bound logger for this request
        self.logger = logger.bind(
            request_id=request_id,
            user_id=user_id,
            client_id=client_id,
            session_id=session_id,
        )


class _TrustedCompatClaimsSentinel:
    """Object-identity marker for server-created mounted auth compatibility claims."""

    def __repr__(self) -> str:
        return "<trusted_mcp_compat_auth>"


_TRUSTED_COMPAT_CLAIMS_SENTINEL = _TrustedCompatClaimsSentinel()
_TRUSTED_COMPAT_CLAIMS_SENTINEL_KEY = "_server_auth_compat_sentinel"
_TRUSTED_COMPAT_AUTH_VIA = frozenset({"single_user_api_key", "single_user_test_api_key"})
_TRUSTED_COMPAT_CLAIMS_SOURCES = frozenset({"mounted_http", "mounted_ws"})


def _metadata_claim_values(value: Any) -> tuple[Any, ...]:
    """Return metadata claim values without iterating strings character-by-character."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return ()


def _trusted_compat_claims_metadata(*, auth_via: str, compat_claims_source: str) -> dict[str, Any]:
    """Return server-only metadata for mounted single-user compatibility claims."""
    if auth_via not in _TRUSTED_COMPAT_AUTH_VIA:
        raise ValueError("Unsupported compatibility auth source")
    if compat_claims_source not in _TRUSTED_COMPAT_CLAIMS_SOURCES:
        raise ValueError("Unsupported compatibility claims source")
    return {
        "auth_via": auth_via,
        "trusted_auth_claims": True,
        "compat_claims_source": compat_claims_source,
        _TRUSTED_COMPAT_CLAIMS_SENTINEL_KEY: _TRUSTED_COMPAT_CLAIMS_SENTINEL,
    }


def _metadata_has_admin_claims(metadata: dict[str, Any]) -> bool:
    """Return True when trusted metadata carries wildcard or admin claims."""
    roles = {
        str(role).strip().lower()
        for role in _metadata_claim_values(metadata.get("roles"))
        if str(role).strip()
    }
    permissions = {
        str(permission).strip().lower()
        for permission in _metadata_claim_values(metadata.get("permissions"))
        if str(permission).strip()
    }
    return "admin" in roles or "*" in permissions


def _has_trusted_compat_claims(context: RequestContext) -> bool:
    """Return True only for server-created mounted compatibility auth claims."""
    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    server_auth_keys = {
        key
        for key in metadata
        if isinstance(key, str) and key.startswith("_server_auth_")
    }
    if server_auth_keys != {_TRUSTED_COMPAT_CLAIMS_SENTINEL_KEY}:
        return False
    if metadata.get(_TRUSTED_COMPAT_CLAIMS_SENTINEL_KEY) is not _TRUSTED_COMPAT_CLAIMS_SENTINEL:
        return False
    if metadata.get("trusted_auth_claims") is not True:
        return False
    if metadata.get("auth_via") not in _TRUSTED_COMPAT_AUTH_VIA:
        return False
    if metadata.get("compat_claims_source") not in _TRUSTED_COMPAT_CLAIMS_SOURCES:
        return False
    return _metadata_has_admin_claims(metadata)


@dataclass(frozen=True, slots=True)
class PreparedToolCall:
    """Prepared tool execution context reused by nested tool orchestration."""

    tool_name: str
    tool_args: Any
    module: BaseModule
    module_id: str | None
    tool_def: dict[str, Any] | None
    is_write: bool | None
    normalized_idempotency_key: str | None
    idempotency_cache_key: str | None
    arguments_hash: str | None
    context_fingerprint: str
    integrity_tag: str
    context: RequestContext
    scope_payload: dict[str, Any] | None = None
