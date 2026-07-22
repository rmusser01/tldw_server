"""Per-request content scope context for Media/Content databases."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeContext:
    """Represents the active authorization scope for content operations."""

    user_id: int | None
    org_ids: list[int]
    team_ids: list[int]
    active_org_id: int | None
    active_team_id: int | None
    is_admin: bool = False
    session_role: str | None = None

    @property
    def effective_org_id(self) -> int | None:
        if self.active_org_id is not None:
            return self.active_org_id
        return self.org_ids[0] if self.org_ids else None

    @property
    def effective_team_id(self) -> int | None:
        if self.active_team_id is not None:
            return self.active_team_id
        return self.team_ids[0] if self.team_ids else None


_CACHE_SESSION_ROLE_MAX_CHARS = 128


def _cache_scope_int(value: object, *, field: str) -> int | None:
    """Return a strictly typed optional integer for cache authorization identity."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer or None")
    return value


def _cache_scope_ids(values: object, *, field: str) -> tuple[int, ...]:
    """Return a sorted immutable copy of a scope membership collection."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{field} must be an iterable of integers")
    items = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in items):
        raise TypeError(f"{field} must contain only integers")
    return tuple(sorted(set(items)))


def content_authorization_cache_scope(scope: ScopeContext) -> dict[str, object]:
    """Snapshot the content authorization fields that can change retrieval results."""
    if not isinstance(scope, ScopeContext):
        raise TypeError("scope must be a ScopeContext")
    if not isinstance(scope.is_admin, bool):
        raise TypeError("is_admin must be a boolean")
    if scope.session_role is not None and not isinstance(scope.session_role, str):
        raise TypeError("session_role must be a string or None")

    role = scope.session_role
    if role is not None and len(role) > _CACHE_SESSION_ROLE_MAX_CHARS:
        role = f"sha256:{hashlib.sha256(role.encode('utf-8')).hexdigest()}"

    return {
        "user_id": _cache_scope_int(scope.user_id, field="user_id"),
        "org_ids": _cache_scope_ids(scope.org_ids, field="org_ids"),
        "team_ids": _cache_scope_ids(scope.team_ids, field="team_ids"),
        "active_org_id": _cache_scope_int(scope.active_org_id, field="active_org_id"),
        "active_team_id": _cache_scope_int(scope.active_team_id, field="active_team_id"),
        "is_admin": scope.is_admin,
        "session_role": role,
    }


def _ordered_unique_ints(values: Iterable[int]) -> list[int]:
    """Return integers in first-seen order without duplicates."""
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value is None:
            continue
        try:
            as_int = int(value)
        except (TypeError, ValueError):
            continue
        if as_int in seen:
            continue
        seen.add(as_int)
        ordered.append(as_int)
    return ordered


_SCOPE_CTX: ContextVar[ScopeContext | None] = ContextVar("content_scope_ctx", default=None)


def set_scope(
    *,
    user_id: int | None,
    org_ids: Iterable[int] = (),
    team_ids: Iterable[int] = (),
    active_org_id: int | None = None,
    active_team_id: int | None = None,
    is_admin: bool = False,
    session_role: str | None = None,
) -> Token:
    """Set the current scope context and return a token for later reset."""
    org_list = _ordered_unique_ints(org_ids)
    team_list = _ordered_unique_ints(team_ids)

    scope = ScopeContext(
        user_id=user_id,
        org_ids=org_list,
        team_ids=team_list,
        active_org_id=int(active_org_id) if active_org_id is not None else None,
        active_team_id=int(active_team_id) if active_team_id is not None else None,
        is_admin=is_admin,
        session_role=str(session_role) if session_role else None,
    )
    return _SCOPE_CTX.set(scope)


def reset_scope(token: Token) -> None:
    """Restore the scope context from a prior set_scope call."""
    _SCOPE_CTX.reset(token)


def get_scope() -> ScopeContext | None:
    """Return the currently active scope (if any)."""
    return _SCOPE_CTX.get()


@contextmanager
def scoped_context(**kwargs) -> ScopeContext:
    """Context manager helper for temporarily setting a scope."""
    token = set_scope(**kwargs)
    try:
        scope = get_scope()
        yield scope
    finally:
        reset_scope(token)
