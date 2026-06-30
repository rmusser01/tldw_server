"""Gateway management surface for TTL-bound policy grants (approval leases)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from loguru import logger

from mcp_unified.interfaces.storage import AuditStore
from mcp_unified.policy_grants import PolicyGrant, PolicyGrantStore
from mcp_unified.storage.models import AuditEvent

APPROVAL_GRANT_MIN_TTL_SECONDS = 60
APPROVAL_GRANT_DEFAULT_TTL_SECONDS = 900
APPROVAL_GRANT_MAX_TTL_SECONDS = 86_400


class GatewayPolicyGrantManagementError(RuntimeError):
    """Domain error for expected gateway policy-grant failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        grant_id: str | None = None,
        profile_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.grant_id = grant_id
        self.profile_id = profile_id

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe error payload."""

        payload: dict[str, Any] = {
            "ok": False,
            "error": str(self),
            "reason_code": self.reason_code,
        }
        if self.grant_id is not None:
            payload["grant_id"] = self.grant_id
        if self.profile_id is not None:
            payload["profile_id"] = self.profile_id
        return payload


class GatewayPolicyGrantManager:
    """Manage TTL-bound policy grants for the standalone gateway."""

    def __init__(
        self,
        *,
        policy_grant_store: PolicyGrantStore,
        audit_store: AuditStore | None = None,
    ) -> None:
        self.policy_grant_store = policy_grant_store
        self.audit_store = audit_store

    async def grant_approval(
        self,
        *,
        profile_id: str,
        subject_type: str,
        value: str,
        ttl_seconds: int = APPROVAL_GRANT_DEFAULT_TTL_SECONDS,
        session_id: str | None = None,
        granted_by: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Create one approval lease with clamped TTL and audit provenance."""

        ttl = min(
            APPROVAL_GRANT_MAX_TTL_SECONDS,
            max(APPROVAL_GRANT_MIN_TTL_SECONDS, int(ttl_seconds)),
        )
        try:
            grant = self.policy_grant_store.create_grant(
                profile_id=profile_id,
                grant_type="approval",
                subject_type=subject_type,
                value=value,
                ttl_seconds=ttl,
                session_id=session_id,
                granted_by=granted_by,
                reason=reason,
            )
        except ValueError as exc:
            raise GatewayPolicyGrantManagementError(
                f"Invalid policy grant request: {exc}",
                reason_code="invalid_policy_grant",
                profile_id=profile_id if isinstance(profile_id, str) else None,
            ) from exc
        await self._append_audit_event(
            "policy_grant.approval.created",
            grant=grant,
        )
        return {"grant": grant.safe_payload()}

    async def grant_path(
        self,
        *,
        profile_id: str,
        prefix: str,
        actions: tuple[str, ...],
        ttl_seconds: int = APPROVAL_GRANT_DEFAULT_TTL_SECONDS,
        session_id: str | None = None,
        granted_by: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Create one TTL-bound path grant with clamped TTL and audit provenance."""

        ttl = min(
            APPROVAL_GRANT_MAX_TTL_SECONDS,
            max(APPROVAL_GRANT_MIN_TTL_SECONDS, int(ttl_seconds)),
        )
        try:
            grant = self.policy_grant_store.create_grant(
                profile_id=profile_id,
                grant_type="path",
                subject_type="path",
                value=prefix,
                ttl_seconds=ttl,
                actions=actions,
                session_id=session_id,
                granted_by=granted_by,
                reason=reason,
            )
        except ValueError as exc:
            raise GatewayPolicyGrantManagementError(
                f"Invalid policy grant request: {exc}",
                reason_code="invalid_policy_grant",
                profile_id=profile_id if isinstance(profile_id, str) else None,
            ) from exc
        await self._append_audit_event("policy_grant.path.created", grant=grant)
        return {"grant": grant.safe_payload()}

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        grant_type: str | None = None,
    ) -> dict[str, Any]:
        """Return active grants, optionally filtered by profile and type."""

        grants = self.policy_grant_store.list_active_grants(
            profile_id=profile_id,
            grant_type=grant_type,
        )
        return {"grants": [grant.safe_payload() for grant in grants]}

    async def revoke_grant(self, grant_id: str) -> dict[str, Any]:
        """Revoke one active grant by id."""

        grant = self.policy_grant_store.revoke_grant(grant_id)
        if grant is None:
            raise GatewayPolicyGrantManagementError(
                "Policy grant not found or already expired",
                reason_code="policy_grant_not_found",
                grant_id=grant_id,
            )
        await self._append_audit_event("policy_grant.revoked", grant=grant)
        return {"grant": grant.safe_payload()}

    async def _append_audit_event(
        self,
        event_type: str,
        *,
        grant: PolicyGrant,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Append an audit event when an audit store is configured."""

        if self.audit_store is None:
            return
        event = AuditEvent(
            id=str(uuid4()),
            event_type=event_type,
            profile_id=grant.profile_id,
            target_type="policy_grant",
            target_id=grant.grant_id,
            payload=dict(payload or grant.safe_payload()),
            provenance={"source": "gateway_policy_grant_manager"},
            created_at=datetime.now(timezone.utc),
        )
        try:
            await self.audit_store.append_event(event)
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).warning(
                "Failed to append policy grant audit event {event_type}",
                event_type=event_type,
            )


__all__ = [
    "APPROVAL_GRANT_DEFAULT_TTL_SECONDS",
    "APPROVAL_GRANT_MAX_TTL_SECONDS",
    "APPROVAL_GRANT_MIN_TTL_SECONDS",
    "GatewayPolicyGrantManagementError",
    "GatewayPolicyGrantManager",
]
