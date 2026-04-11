"""Policy conditions for RBAC enrichment.

Conditions are pre-resolved into the policy snapshot at build time.
GovernanceFilter evaluates them synchronously (no DB lookups at call time).

Condition types:
- Time windows: valid_from / valid_until
- Labels: required_labels (AND semantics)
- Delegation: principal_type + principal_id for ancestry-based access
- Source IPs: CIDR ranges (evaluated at endpoint layer, not in GovernanceFilter)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class DelegationCondition:
    """Delegation condition for ancestry-based transitive access."""
    principal_type: str = "user"   # "user" | "agent"
    principal_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> DelegationCondition | None:
        if not d:
            return None
        return cls(
            principal_type=d.get("principal_type", "user"),
            principal_id=d.get("principal_id", ""),
        )


@dataclass
class PolicyConditions:
    """Conditions that must be met for a policy to apply."""
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    source_ips: list[str] | None = None
    required_labels: dict[str, str] | None = None
    delegation: DelegationCondition | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.valid_from:
            d["valid_from"] = self.valid_from.isoformat()
        if self.valid_until:
            d["valid_until"] = self.valid_until.isoformat()
        if self.source_ips:
            d["source_ips"] = self.source_ips
        if self.required_labels:
            d["required_labels"] = self.required_labels
        if self.delegation:
            d["delegation"] = self.delegation.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PolicyConditions:
        if not d:
            return cls()
        vf = d.get("valid_from")
        vu = d.get("valid_until")
        return cls(
            valid_from=datetime.fromisoformat(vf) if vf else None,
            valid_until=datetime.fromisoformat(vu) if vu else None,
            source_ips=d.get("source_ips"),
            required_labels=d.get("required_labels"),
            delegation=DelegationCondition.from_dict(d.get("delegation")),
        )

    @classmethod
    def from_json(cls, s: str | None) -> PolicyConditions:
        if not s:
            return cls()
        return cls.from_dict(json.loads(s))

    def is_empty(self) -> bool:
        return (
            self.valid_from is None
            and self.valid_until is None
            and self.source_ips is None
            and self.required_labels is None
            and self.delegation is None
        )


def evaluate_conditions(
    conditions: PolicyConditions,
    *,
    resource_labels: dict[str, str] | None = None,
    ancestry_chain: list[str] | None = None,
    now: datetime | None = None,
) -> bool:
    """Evaluate policy conditions. Returns True if all conditions pass.

    This is designed to be called synchronously from GovernanceFilter.
    No DB lookups -- all data is pre-resolved into the arguments.

    Parameters
    ----------
    conditions:
        The conditions to evaluate.
    resource_labels:
        Labels on the resource being accessed.
    ancestry_chain:
        The session's ancestry chain [root_user, ..., parent_agent].
    now:
        Current time (for testing; defaults to utcnow).
    """
    if conditions.is_empty():
        return True

    if now is None:
        now = datetime.now(timezone.utc)

    # Time window check
    if conditions.valid_from and now < conditions.valid_from:
        return False
    if conditions.valid_until and now > conditions.valid_until:
        return False

    # Label matching (AND semantics)
    if conditions.required_labels:
        if resource_labels is None:
            return False
        for key, value in conditions.required_labels.items():
            if resource_labels.get(key) != value:
                return False

    # Delegation check (ancestry-based)
    if conditions.delegation and conditions.delegation.principal_id:
        if not ancestry_chain:
            return False
        if conditions.delegation.principal_id not in ancestry_chain:
            return False

    # source_ips are NOT evaluated here -- they're checked at the endpoint layer
    return True
