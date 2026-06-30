"""Workspace membership request metadata helpers."""
from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_ADMIN


def normalize_claim_values(values: Any) -> list[str]:
    """Normalize role/permission claims to lowercase strings."""
    raw_values = values if isinstance(values, (list, tuple, set)) else ([values] if values is not None else [])
    normalized: list[str] = []
    for value in raw_values:
        text = str(value).strip().lower()
        if text:
            normalized.append(text)
    return normalized


def is_workflows_admin_user(current_user: Any) -> bool:
    """Return whether explicit role/permission claims grant workflow-admin access."""
    try:
        if "admin" in normalize_claim_values(getattr(current_user, "roles", [])):
            return True
        permission_values = normalize_claim_values(getattr(current_user, "permissions", []))
        return (
            WORKFLOWS_ADMIN.lower() in permission_values
            or "*" in permission_values
            or "system.configure" in permission_values
        )
    except (AttributeError, TypeError, ValueError):
        return False


def build_workspace_membership_request_metadata(current_user: Any) -> dict[str, Any]:
    """Build request metadata consumed by domain-owned membership adapters."""
    raw_tenant_id = getattr(current_user, "tenant_id", None)
    tenant_id = str(raw_tenant_id).strip() if raw_tenant_id is not None else ""
    return {
        "tenant_id": tenant_id,
        "is_workflows_admin": is_workflows_admin_user(current_user),
    }
