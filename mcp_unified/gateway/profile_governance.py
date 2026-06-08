"""Permission-change governance primitives for gateway profile mutations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

PermissionChangeOutcome = Literal["deny", "ask", "allow"]
_VALID_PERMISSION_CHANGE_OUTCOMES = frozenset({"deny", "ask", "allow"})


@dataclass(frozen=True, slots=True)
class PermissionChangeRequest:
    """Redacted permission-change summary passed to governance hooks."""

    action: str
    profile_id: str | None
    target_type: str
    target_id: str
    changed_fields: tuple[str, ...] = ()
    policy_fields: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PermissionChangeDecision:
    """Governance result for one permission-changing profile mutation."""

    outcome: PermissionChangeOutcome
    reason_code: str = "allowed"
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the decision shape at the trust boundary."""
        if self.outcome not in _VALID_PERMISSION_CHANGE_OUTCOMES:
            raise ValueError("permission change outcome must be deny, ask, or allow")
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            object.__setattr__(self, "reason_code", self.outcome)
        if not isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", {})


class PermissionChangeGovernor(Protocol):
    """Async governance hook for profile permission-surface mutations."""

    async def evaluate_permission_change(
        self,
        request: PermissionChangeRequest,
    ) -> PermissionChangeDecision:
        """Return the governance decision for a profile permission change."""


class AllowPermissionChangeGovernor:
    """Default governor that preserves existing profile-management behavior."""

    async def evaluate_permission_change(
        self,
        request: PermissionChangeRequest,
    ) -> PermissionChangeDecision:
        """Allow the permission change without requiring approval."""
        return PermissionChangeDecision(outcome="allow", reason_code="allowed")
