"""Authorization helpers for Chat slash commands."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CommandAuthorizationContext:
    auth_user_id: int | None
    user_id: str
    permissions: frozenset[str]
    roles: frozenset[str]
    is_admin: bool
    auth_mode: str | None
    is_single_user_owner: bool


@dataclass(frozen=True)
class CommandAuthorizationDecision:
    allowed: bool
    metadata: dict[str, Any]


def _normalized_strings(values: Any) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, str):
        values = [values]
    try:
        return frozenset(str(value).strip() for value in values if str(value).strip())
    except TypeError:
        return frozenset()


def build_command_authorization_context(ctx: Any) -> CommandAuthorizationContext:
    request_meta = getattr(ctx, "request_meta", None) or {}
    if not isinstance(request_meta, dict):
        request_meta = {}
    permissions = _normalized_strings(request_meta.get("permissions"))
    roles = _normalized_strings(request_meta.get("roles"))
    auth_mode_raw = request_meta.get("auth_mode")
    auth_mode = str(auth_mode_raw).strip().lower() if auth_mode_raw else None
    is_admin = bool(
        request_meta.get("is_admin", False)
        or "admin" in {role.lower() for role in roles}
        or "*" in permissions
        or "system.configure" in permissions
    )
    return CommandAuthorizationContext(
        auth_user_id=getattr(ctx, "auth_user_id", None),
        user_id=str(getattr(ctx, "user_id", "anonymous")),
        permissions=permissions,
        roles=roles,
        is_admin=is_admin,
        auth_mode=auth_mode,
        is_single_user_owner=bool(request_meta.get("is_single_user_owner", False)),
    )


def _permission_in_claims(permission: str, permissions: frozenset[str]) -> bool:
    if permission in permissions or "*" in permissions:
        return True
    parts = permission.split(".")
    for end in range(len(parts), 0, -1):
        wildcard = ".".join(parts[:end]) + ".*"
        if wildcard in permissions:
            return True
    return False


def authorize_command(
    *,
    spec: Any,
    context: CommandAuthorizationContext,
    permission_checker: Callable[[int, str], bool],
) -> CommandAuthorizationDecision:
    required_permission = getattr(spec, "required_permission", None)
    rbac_required = bool(getattr(spec, "rbac_required", bool(required_permission)))
    if not required_permission or not rbac_required:
        return CommandAuthorizationDecision(True, {"checked": False})

    metadata = {"checked": True, "required_permission": required_permission}
    if context.is_admin:
        return CommandAuthorizationDecision(True, {**metadata, "source": "admin"})
    if context.auth_mode == "single_user" and context.is_single_user_owner:
        return CommandAuthorizationDecision(True, {**metadata, "source": "single_user_owner"})
    if _permission_in_claims(required_permission, context.permissions):
        return CommandAuthorizationDecision(True, {**metadata, "source": "claims"})
    if context.auth_user_id is None:
        return CommandAuthorizationDecision(False, {**metadata, "permitted": False})

    try:
        permitted = bool(permission_checker(int(context.auth_user_id), required_permission))
    except Exception:
        permitted = False
    if permitted:
        return CommandAuthorizationDecision(True, {**metadata, "source": "db"})
    return CommandAuthorizationDecision(False, {**metadata, "permitted": False})
