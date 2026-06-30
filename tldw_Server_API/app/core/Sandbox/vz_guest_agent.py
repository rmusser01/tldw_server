"""VZ Linux guest-agent compatibility helpers for diagnostics and reuse gates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

VZ_LINUX_EXPECTED_GUEST_WORKSPACE_ROOT = "/workspace"
VZ_LINUX_REQUIRED_GUEST_CAPABILITIES = ("exec",)
VZ_LINUX_GUEST_AGENT_WORKSPACE_MISMATCH = "vz_linux_guest_agent_workspace_mismatch"
VZ_LINUX_GUEST_AGENT_REQUIRED_CAPABILITY_MISSING = (
    "vz_linux_guest_agent_required_capability_missing"
)

GuestAgentCompatibility = Literal["compatible", "unknown", "mismatch"]


def _detail_text(details: Mapping[str, object], key: str) -> str | None:
    """Extract a non-empty text detail from helper metadata."""

    value = details.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _detail_bool(details: Mapping[str, object], key: str) -> bool | None:
    """Coerce helper detail booleans without trusting arbitrary truthy strings."""

    value = details.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def _detail_csv(details: Mapping[str, object], key: str) -> list[str]:
    """Extract a list-like helper detail from a list or comma-separated string."""

    value = details.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def classify_vz_linux_guest_agent(details: Mapping[str, Any] | None) -> dict[str, object]:
    """Classify guest-agent metadata without rejecting older metadata-light guests."""

    payload: Mapping[str, object] = details if isinstance(details, Mapping) else {}
    version = _detail_text(payload, "guest_version")
    workspace_root = _detail_text(payload, "guest_workspace_root")
    capabilities_known = _detail_bool(payload, "guest_capabilities_known")
    capabilities = _detail_csv(payload, "guest_capabilities")
    capability_set = set(capabilities)
    missing_required = [
        capability
        for capability in VZ_LINUX_REQUIRED_GUEST_CAPABILITIES
        if capability not in capability_set
    ]

    reasons: list[str] = []
    if workspace_root and workspace_root != VZ_LINUX_EXPECTED_GUEST_WORKSPACE_ROOT:
        reasons.append(VZ_LINUX_GUEST_AGENT_WORKSPACE_MISMATCH)
    if capabilities_known is True and missing_required:
        reasons.append(VZ_LINUX_GUEST_AGENT_REQUIRED_CAPABILITY_MISSING)

    if reasons:
        compatibility: GuestAgentCompatibility = "mismatch"
    elif capabilities_known is True and not missing_required:
        compatibility = "compatible"
    else:
        compatibility = "unknown"

    return {
        "version": version,
        "workspace_root": workspace_root,
        "capabilities_known": capabilities_known,
        "capabilities": capabilities,
        "compatibility": compatibility,
        "reasons": reasons,
        "expected_workspace_root": VZ_LINUX_EXPECTED_GUEST_WORKSPACE_ROOT,
        "required_capabilities": list(VZ_LINUX_REQUIRED_GUEST_CAPABILITIES),
        "missing_required_capabilities": missing_required if capabilities_known is True else [],
    }


def vz_linux_guest_agent_mismatched(details: Mapping[str, Any] | None) -> bool:
    """Return true only for explicit guest-agent contract mismatches."""

    return classify_vz_linux_guest_agent(details).get("compatibility") == "mismatch"
