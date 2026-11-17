"""Projection of server-authenticated identity data into MCP execution scope."""

from __future__ import annotations

from typing import Any

from .protocol_types import AuthenticatedExecutionScope


def _positive_user_id(value: Any) -> int:
    if type(value) is int:
        normalized = value
    elif type(value) is str and value.isascii() and value.isdecimal() and len(value) <= 20:
        normalized = int(value)
    else:
        raise ValueError("Authenticated user IDs must be positive integers")
    if normalized < 1:
        raise ValueError("Authenticated user IDs must be positive integers")
    return normalized


def _positive_scope_id(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("Active scope IDs must be positive non-boolean integers")
    return value


def project_authenticated_execution_scope(
    *,
    authenticated_user_id: Any | None,
    principal_user_id: Any | None = None,
    principal_active_org_id: Any | None = None,
    principal_active_team_id: Any | None = None,
    api_key_info: dict[str, Any] | None = None,
) -> AuthenticatedExecutionScope | None:
    """Build scope from verified principals and API-key records, bound to one owner."""

    if api_key_info is not None and not isinstance(api_key_info, dict):
        raise ValueError("API-key identity must be a mapping")
    api_key_snapshot = dict(api_key_info) if api_key_info is not None else None

    principal_org_id = _positive_scope_id(principal_active_org_id)
    principal_team_id = _positive_scope_id(principal_active_team_id)
    api_key_org_id = _positive_scope_id(api_key_snapshot.get("org_id")) if api_key_snapshot is not None else None
    api_key_team_id = _positive_scope_id(api_key_snapshot.get("team_id")) if api_key_snapshot is not None else None

    if principal_org_id is not None and api_key_org_id is not None and principal_org_id != api_key_org_id:
        raise ValueError("Conflicting authenticated organization IDs")
    if principal_team_id is not None and api_key_team_id is not None and principal_team_id != api_key_team_id:
        raise ValueError("Conflicting authenticated team IDs")

    has_scope = any(
        value is not None
        for value in (
            principal_org_id,
            principal_team_id,
            api_key_org_id,
            api_key_team_id,
        )
    )
    if not has_scope and api_key_snapshot is None:
        return None

    owner_ids: list[int] = []
    if authenticated_user_id is not None:
        owner_ids.append(_positive_user_id(authenticated_user_id))
    if principal_user_id is not None:
        owner_ids.append(_positive_user_id(principal_user_id))
    if api_key_snapshot is not None:
        if "user_id" not in api_key_snapshot:
            raise ValueError("Authenticated API-key owner is required")
        owner_ids.append(_positive_user_id(api_key_snapshot.get("user_id")))
    if not owner_ids or any(owner_id != owner_ids[0] for owner_id in owner_ids[1:]):
        raise ValueError("Authenticated scope owners do not match")

    active_org_id = principal_org_id if principal_org_id is not None else api_key_org_id
    active_team_id = principal_team_id if principal_team_id is not None else api_key_team_id
    if active_org_id is None and active_team_id is None:
        return None
    return AuthenticatedExecutionScope(
        active_org_id=active_org_id,
        active_team_id=active_team_id,
    )
