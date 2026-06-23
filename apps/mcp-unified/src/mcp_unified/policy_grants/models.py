"""Shared TTL-bound policy grant models for the standalone MCP gateway."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from mcp_unified.profiles.path_grants import PATH_GRANT_ACTIONS, normalize_path_grant_prefix
from mcp_unified.profiles.permission_rules import normalize_permission_subject_value

POLICY_GRANT_TYPES = frozenset({"approval", "path"})
APPROVAL_SUBJECT_TYPES = frozenset({"tool", "path", "domain", "command", "mcp"})


@dataclass(frozen=True, slots=True)
class PolicyGrant:
    """One TTL-bound operator-issued policy grant (approval lease or path grant)."""

    grant_id: str
    profile_id: str
    grant_type: str
    subject_type: str
    value: str
    expires_at: float
    ttl_seconds: int
    actions: tuple[str, ...] = ()
    effect: str = "allow"
    session_id: str | None = None
    user_id: str | None = None
    granted_by: str | None = None
    reason: str | None = None

    def expires_at_iso(self) -> str:
        """Return the grant expiry timestamp in UTC ISO-8601 form."""

        return datetime.fromtimestamp(self.expires_at, tz=timezone.utc).isoformat()

    def is_active(self, now: float) -> bool:
        """Return whether the grant is still active at the supplied time."""

        return self.expires_at > now

    def matches_session(self, session_id: str | None) -> bool:
        """Return whether the grant applies to the supplied session scope."""

        return self.session_id is None or self.session_id == session_id

    def safe_payload(self) -> dict[str, Any]:
        """Return grant metadata safe for operator-facing responses."""

        return {
            "grant_id": self.grant_id,
            "profile_id": self.profile_id,
            "grant_type": self.grant_type,
            "subject_type": self.subject_type,
            "value": self.value,
            "actions": list(self.actions),
            "effect": self.effect,
            "session_id": self.session_id,
            "granted_by": self.granted_by,
            "reason": self.reason,
            "expires_at": self.expires_at_iso(),
            "ttl_seconds": self.ttl_seconds,
        }


class PolicyGrantStore(Protocol):
    """Backend contract for TTL-bound policy grants."""

    def create_grant(
        self,
        *,
        profile_id: str,
        grant_type: str,
        subject_type: str,
        value: str,
        ttl_seconds: int,
        actions: tuple[str, ...] = (),
        effect: str = "allow",
        session_id: str | None = None,
        user_id: str | None = None,
        granted_by: str | None = None,
        reason: str | None = None,
    ) -> PolicyGrant:
        """Create one validated grant and return it."""
        ...

    def list_active_grants(
        self,
        *,
        profile_id: str | None = None,
        grant_type: str | None = None,
    ) -> list[PolicyGrant]:
        """Return active grants, optionally filtered by profile and type."""
        ...

    def revoke_grant(self, grant_id: str) -> PolicyGrant | None:
        """Revoke one grant by id, returning it when it was active."""
        ...

    def find_active_grant(
        self,
        *,
        profile_id: str,
        grant_type: str,
        subject_type: str,
        value: str,
        session_id: str | None = None,
    ) -> PolicyGrant | None:
        """Return one active grant matching the normalized subject, if any."""
        ...


def validate_grant_request(
    *,
    profile_id: str,
    grant_type: str,
    subject_type: str,
    value: str,
) -> tuple[str, str, str, str]:
    """Validate and normalize one grant request's identity dimensions."""

    normalized_profile_id = profile_id.strip() if isinstance(profile_id, str) else ""
    if not normalized_profile_id:
        raise ValueError("policy grant requires a non-empty profile_id")
    if grant_type not in POLICY_GRANT_TYPES:
        raise ValueError(f"unsupported policy grant_type: {grant_type!r}")
    if grant_type == "approval":
        if subject_type not in APPROVAL_SUBJECT_TYPES:
            raise ValueError(f"unsupported approval subject_type: {subject_type!r}")
        normalized_value = normalize_permission_subject_value(subject_type, value)  # type: ignore[arg-type]
    else:
        if subject_type != "path":
            raise ValueError("path policy grants require subject_type 'path'")
        normalized_prefix = normalize_path_grant_prefix(value)
        if normalized_prefix is None:
            raise ValueError(
                "path policy grants require a workspace-relative prefix "
                "without '..' or absolute segments"
            )
        normalized_value = normalized_prefix
    return normalized_profile_id, grant_type, subject_type, normalized_value


def validate_grant_actions(grant_type: str, actions: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and normalize one grant request's action list."""

    normalized = tuple(
        str(action or "").strip().lower()
        for action in actions or ()
        if str(action or "").strip()
    )
    if grant_type != "path":
        return normalized
    if not normalized:
        raise ValueError("path policy grants require at least one action")
    invalid = sorted(action for action in normalized if action not in PATH_GRANT_ACTIONS)
    if invalid:
        raise ValueError(f"unsupported path grant actions: {', '.join(invalid)}")
    return normalized


def validate_grant_effect(effect: str) -> str:
    """Validate and normalize one grant request's effect.

    TTL-bound grants can only widen policy temporarily; deny effects belong
    in the profile document, so anything except "allow" is rejected.
    """

    normalized = str(effect or "").strip().lower()
    if normalized != "allow":
        raise ValueError("policy grants only support effect 'allow'")
    return normalized


__all__ = [
    "APPROVAL_SUBJECT_TYPES",
    "POLICY_GRANT_TYPES",
    "PolicyGrant",
    "PolicyGrantStore",
    "validate_grant_actions",
    "validate_grant_effect",
    "validate_grant_request",
]
