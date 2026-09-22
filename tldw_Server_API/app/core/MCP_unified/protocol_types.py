from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

# Source of truth for which roles are platform administrators. Imported rather than
# restated so MCP cannot drift from AuthNZ. (_PLATFORM_ADMIN_ROLES is itself spelled
# out in four places under app/core -- see TASK-13345.)
from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import _PLATFORM_ADMIN_ROLES

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


@dataclass(frozen=True, slots=True)
class AuthenticatedExecutionScope:
    """Explicit server-authenticated active organization/team replay scope."""

    active_org_id: int | None = None
    active_team_id: int | None = None

    def __post_init__(self) -> None:
        if self.active_org_id is None and self.active_team_id is None:
            raise ValueError("Authenticated execution scope requires at least one active ID")
        for value in (self.active_org_id, self.active_team_id):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("Active scope IDs must be positive non-boolean integers")

    def canonical_object(self) -> dict[str, JsonValue]:
        """Return the canonical scope object, omitting absent dimensions."""

        payload: dict[str, JsonValue] = {}
        if self.active_org_id is not None:
            payload["active_org_id"] = self.active_org_id
        if self.active_team_id is not None:
            payload["active_team_id"] = self.active_team_id
        return payload


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
        server_auth_scope: AuthenticatedExecutionScope | None = None,
    ):
        if server_auth_scope is not None and not isinstance(
            server_auth_scope,
            AuthenticatedExecutionScope,
        ):
            raise TypeError("server_auth_scope must be an AuthenticatedExecutionScope or None")
        self.request_id = request_id
        self.user_id = user_id
        self.client_id = client_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.start_time = datetime.now(timezone.utc)
        self.db_paths = dict(db_paths or {})
        self.server_auth_scope = server_auth_scope
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


def metadata_has_admin_claims(metadata: Any) -> bool:
    """Return True when metadata claims make the caller an MCP administrator.

    The one admin predicate for MCP. Six had grown independently -- media, notes,
    kanban, sandbox, mcp_discovery and this one -- with six different claim sets, and
    they disagreed in both directions:

    * Under-grant: AuthNZ treats "owner" and "super_admin" as platform admins, and
      server.py writes those straight into metadata["roles"], but every MCP predicate
      tested the literal "admin". A platform owner was refused permanent media delete,
      permanent note delete and all kanban policy operations while being an
      administrator everywhere else in the product.
    * Over-grant: sandbox_module alone accepted the "system.configure" permission, and
      used that to pass its CROSS-USER session gate -- so an API key with no admin role
      could reach another user's sandbox session, while this predicate, MCP's own
      trusted-claims gate, said the same caller was not an admin.

    Roles are taken from AuthNZ's _PLATFORM_ADMIN_ROLES so the two cannot drift again.
    Permissions deliberately do NOT follow AuthNZ's _ADMIN_CLAIM_PERMISSIONS: only the
    "*" wildcard grants admin here, not "system.configure" or the "admin" permission.
    A configuration permission should not authorise permanently deleting another
    user's media and notes, which is what full parity would have granted. That
    narrowing is the only intentional divergence from AuthNZ; see
    Docs/ADR/048-mcp-admin-claims.md.
    """
    if not isinstance(metadata, dict):
        return False
    roles = {
        str(role).strip().lower()
        for role in _metadata_claim_values(metadata.get("roles"))
        if str(role).strip()
    }
    if roles & _PLATFORM_ADMIN_ROLES:
        return True
    permissions = {
        str(permission).strip().lower()
        for permission in _metadata_claim_values(metadata.get("permissions"))
        if str(permission).strip()
    }
    return "*" in permissions


# Back-compat alias: this was the private canonical predicate before it was shared.
_metadata_has_admin_claims = metadata_has_admin_claims


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
    """HMAC-bound prepared execution state reused by nested orchestration.

    ``tool_def`` and ``scope_payload`` are detached observer compatibility
    views. Security decisions must use ``policy`` and the signed snapshots,
    never these freshly decoded dictionaries.
    """

    tool_name: str
    tool_args: Any
    module: BaseModule
    module_id: str | None
    policy: PreparedExecutionPolicy
    arguments_snapshot: CanonicalJsonSnapshot
    tool_definition_snapshot: CanonicalJsonSnapshot
    scope_reporting_snapshot: CanonicalJsonSnapshot
    normalized_idempotency_key: str | None
    normalized_idempotency_key_digest: str
    idempotency_cache_key: str | None
    arguments_hash: str | None
    context_fingerprint: str
    idempotency_scope_fingerprint: str
    integrity_tag: str
    context: RequestContext

    @property
    def tool_def(self) -> dict[str, Any] | None:
        """Return a fresh observer-only tool-definition copy."""

        from .tool_execution.canonical import (
            TOOL_DEFINITION_MAX_BYTES,
            decode_canonical_json_object_or_none,
        )

        return decode_canonical_json_object_or_none(
            self.tool_definition_snapshot.encoded,
            max_bytes=TOOL_DEFINITION_MAX_BYTES,
        )

    @property
    def scope_payload(self) -> dict[str, Any] | None:
        """Return a fresh observer-only scope-reporting copy."""

        from .tool_execution.canonical import (
            SCOPE_REPORTING_MAX_BYTES,
            decode_canonical_json_object_or_none,
        )

        return decode_canonical_json_object_or_none(
            self.scope_reporting_snapshot.encoded,
            max_bytes=SCOPE_REPORTING_MAX_BYTES,
        )

    @property
    def is_write(self) -> bool:
        """Return the immutable prepared effect for compatibility observers."""

        return self.policy.effect == "write"


# Resolve public postponed annotations only after the shared protocol types are
# initialized; tool_execution package setup imports RequestContext back here.
from .tool_execution.canonical import JsonValue as JsonValue  # noqa: E402
from .tool_execution.models import (  # noqa: E402
    CanonicalJsonSnapshot as CanonicalJsonSnapshot,
)
from .tool_execution.models import (  # noqa: E402
    PreparedExecutionPolicy as PreparedExecutionPolicy,
)
