"""Policy conditions for RBAC enrichment.

Conditions are pre-resolved into the policy snapshot at build time.
GovernanceFilter evaluates them synchronously (no DB lookups at call time).
"""
from __future__ import annotations

from ipaddress import ip_address, ip_network
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class DelegationCondition:
    """Identifies the delegating principal whose ancestry must be present."""

    principal_type: str  # "user" | "agent"
    principal_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "principal_type": self.principal_type,
            "principal_id": self.principal_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DelegationCondition | None:
        if not data or not isinstance(data, dict):
            return None
        principal_type = str(data.get("principal_type") or "").strip()
        principal_id = str(data.get("principal_id") or "").strip()
        if not principal_type or not principal_id:
            return None
        return cls(principal_type=principal_type, principal_id=principal_id)


def _parse_datetime(value: Any) -> datetime | None:
    """Parse an ISO-format datetime string, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        dt = datetime.fromisoformat(str(value))
        # Ensure timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _format_datetime(dt: datetime | None) -> str | None:
    """Format a datetime as ISO string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _normalize_source_ips(value: Any) -> list[str] | None:
    """Normalize source IP conditions to a list of non-empty IP/CIDR strings."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else None
    if not isinstance(value, (list, tuple, set)):
        return None
    out = [str(entry).strip() for entry in value if str(entry).strip()]
    return out or None


def _normalize_required_labels(value: Any) -> dict[str, str] | None:
    """Normalize required_labels to a mapping, or None when the payload is invalid."""
    return dict(value) if isinstance(value, dict) else None


@dataclass
class PolicyConditions:
    """Conditions attached to a policy that must be satisfied for it to apply.

    All fields are optional.  An empty ``PolicyConditions`` (all None) means
    the policy applies unconditionally.
    """

    valid_from: datetime | None = None
    valid_until: datetime | None = None
    source_ips: list[str] | None = None
    required_labels: dict[str, str] | None = None
    delegation: DelegationCondition | None = None

    def is_empty(self) -> bool:
        """Return True when no conditions are set (unconditional policy)."""
        return (
            self.valid_from is None
            and self.valid_until is None
            and self.source_ips is None
            and self.required_labels is None
            and self.delegation is None
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.valid_from is not None:
            d["valid_from"] = _format_datetime(self.valid_from)
        if self.valid_until is not None:
            d["valid_until"] = _format_datetime(self.valid_until)
        if self.source_ips is not None:
            d["source_ips"] = list(self.source_ips)
        if self.required_labels is not None:
            d["required_labels"] = dict(self.required_labels)
        if self.delegation is not None:
            d["delegation"] = self.delegation.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PolicyConditions:
        """Deserialize from a plain dict.  ``None`` / empty dict yields empty conditions."""
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            valid_from=_parse_datetime(data.get("valid_from")),
            valid_until=_parse_datetime(data.get("valid_until")),
            source_ips=_normalize_source_ips(data.get("source_ips")),
            required_labels=_normalize_required_labels(data.get("required_labels")),
            delegation=DelegationCondition.from_dict(data.get("delegation")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str | None) -> PolicyConditions:
        if not raw:
            return cls()
        try:
            return cls.from_dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return cls()


def evaluate_conditions(
    conditions: PolicyConditions,
    *,
    resource_labels: dict[str, str] | None = None,
    ancestry_chain: list[str] | None = None,
    source_ip: str | None = None,
    now: datetime | None = None,
) -> bool:
    """Evaluate policy conditions synchronously.

    Returns ``True`` if all conditions are satisfied (or no conditions are set).

    Notes:
        - ``source_ips`` supports both individual IP addresses and CIDR ranges.
        - Missing or invalid request IP context causes IP-scoped policies to fail closed.
        - Empty conditions always pass.
    """
    if conditions.is_empty():
        return True

    if now is None:
        now = datetime.now(timezone.utc)

    # --- Time window ---
    if conditions.valid_from is not None and now < conditions.valid_from:
        return False
    if conditions.valid_until is not None and now > conditions.valid_until:
        return False

    # --- Source IP allowlist ---
    if conditions.source_ips is not None:
        if not source_ip:
            return False
        try:
            client_ip = ip_address(str(source_ip).strip())
        except ValueError:
            return False

        ip_match = False
        for allowed_entry in conditions.source_ips:
            try:
                if client_ip in ip_network(str(allowed_entry).strip(), strict=False):
                    ip_match = True
                    break
            except ValueError:
                continue
        if not ip_match:
            return False

    # --- Required labels (AND semantics) ---
    if conditions.required_labels is not None:
        labels = resource_labels or {}
        for key, expected_value in conditions.required_labels.items():
            if labels.get(key) != expected_value:
                return False

    # --- Delegation ancestry ---
    if conditions.delegation is not None:
        chain = ancestry_chain or []
        if conditions.delegation.principal_id not in chain:
            return False

    return True
