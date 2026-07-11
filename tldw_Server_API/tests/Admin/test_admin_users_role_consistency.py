from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.admin_schemas import UserUpdateRequest
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_users_service


@dataclass
class _RoleState:
    legacy_role: str
    memberships: set[int]
    roles: dict[str, tuple[int, bool]] = field(
        default_factory=lambda: {
            "admin": (1, True),
            "user": (2, True),
            "service": (3, True),
            "custom-reviewer": (9, False),
        }
    )


class _Cursor:
    def __init__(self, row=None, *, rowcount: int = 1) -> None:
        self._row = row
        self.rowcount = rowcount

    async def fetchone(self):
        return self._row


class _SqliteRoleDb:
    def __init__(self, state: _RoleState) -> None:
        self.state = state
        self.commit_count = 0

    async def execute(self, query: str, params=()):
        normalized = " ".join(query.split()).lower()
        if normalized.startswith("select id") and "from roles" in normalized:
            role = self.state.roles.get(str(params[0]))
            return _Cursor(role)
        if normalized.startswith("update users"):
            self.state.legacy_role = str(params[0])
            return _Cursor(rowcount=1)
        if normalized.startswith("delete from user_roles"):
            selected_role_id = int(params[-1])
            system_role_ids = {role_id for role_id, is_system in self.state.roles.values() if is_system}
            self.state.memberships -= system_role_ids - {selected_role_id}
            return _Cursor()
        if normalized.startswith("insert or ignore into user_roles"):
            self.state.memberships.add(int(params[1]))
            return _Cursor()
        raise AssertionError(f"Unexpected SQLite query: {query}")

    async def commit(self) -> None:
        self.commit_count += 1


class _PostgresRoleDb:
    def __init__(self, state: _RoleState) -> None:
        self.state = state

    async def fetchrow(self, query: str, *params):
        normalized = " ".join(query.split()).lower()
        if normalized.startswith("select id") and "from roles" in normalized:
            role = self.state.roles.get(str(params[0]))
            if role is None:
                return None
            return {"id": role[0], "is_system": role[1]}
        if normalized.startswith("update users"):
            self.state.legacy_role = str(params[0])
            return {"id": int(params[-1])}
        raise AssertionError(f"Unexpected PostgreSQL fetchrow: {query}")

    async def execute(self, query: str, *params):
        normalized = " ".join(query.split()).lower()
        if normalized.startswith("delete from user_roles"):
            selected_role_id = int(params[-1])
            system_role_ids = {role_id for role_id, is_system in self.state.roles.values() if is_system}
            self.state.memberships -= system_role_ids - {selected_role_id}
            return "DELETE"
        if normalized.startswith("insert into user_roles"):
            self.state.memberships.add(int(params[1]))
            return "INSERT"
        raise AssertionError(f"Unexpected PostgreSQL execute: {query}")


def _principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, roles=["admin"], is_admin=True)


async def _allow(*_args, **_kwargs) -> None:
    return None


async def _reauth(*_args, **_kwargs) -> str:
    return "Support case 123"


async def _no_audit(**_kwargs) -> None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "target_role", "initial_role", "initial_memberships", "expected_memberships"),
    [
        ("sqlite", "admin", "user", {2, 9}, {1, 9}),
        ("sqlite", "user", "admin", {1, 2, 9}, {2, 9}),
        ("postgres", "admin", "user", {2, 9}, {1, 9}),
        ("postgres", "user", "admin", {1, 2, 9}, {2, 9}),
    ],
)
async def test_update_user_keeps_one_system_role_and_preserves_custom_grants(
    monkeypatch,
    backend: str,
    target_role: str,
    initial_role: str,
    initial_memberships: set[int],
    expected_memberships: set[int],
) -> None:
    state = _RoleState(initial_role, set(initial_memberships))
    db = _PostgresRoleDb(state) if backend == "postgres" else _SqliteRoleDb(state)
    monkeypatch.setattr(admin_users_service.admin_scope_service, "enforce_admin_user_scope", _allow)
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _reauth)
    monkeypatch.setattr(admin_users_service, "_emit_admin_account_audit_event", _no_audit)

    await admin_users_service.update_user(
        _principal(),
        42,
        UserUpdateRequest(role=target_role, reason="Support case 123"),
        db,
        password_service=object(),
        is_pg_fn=lambda: _is_postgres(backend),
    )

    assert state.legacy_role == target_role
    assert state.memberships == expected_memberships
    if backend == "sqlite":
        assert db.commit_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_update_user_unknown_role_fails_without_mutating_role_state(monkeypatch, backend: str) -> None:
    state = _RoleState("user", {2, 9})
    db = _PostgresRoleDb(state) if backend == "postgres" else _SqliteRoleDb(state)
    monkeypatch.setattr(admin_users_service.admin_scope_service, "enforce_admin_user_scope", _allow)
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _reauth)

    request = UserUpdateRequest.model_construct(role="missing-role", reason="Support case 123")
    with pytest.raises(HTTPException) as exc_info:
        await admin_users_service.update_user(
            _principal(),
            42,
            request,
            db,
            password_service=object(),
            is_pg_fn=lambda: _is_postgres(backend),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Unknown role 'missing-role'"
    assert state.legacy_role == "user"
    assert state.memberships == {2, 9}
    if backend == "sqlite":
        assert db.commit_count == 0


async def _is_postgres(backend: str) -> bool:
    return backend == "postgres"
