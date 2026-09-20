from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_rbac
from tldw_Server_API.app.api.v1.schemas.admin_schemas import UserUpdateRequest
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql
from tldw_Server_API.app.services import admin_users_service


@dataclass
class _RoleState:
    legacy_role: str
    memberships: set[int]
    is_superuser: bool = True
    user_exists: bool = True
    expired_memberships: set[int] = field(default_factory=set)
    roles: dict[str, tuple[int, bool]] = field(
        default_factory=lambda: {
            "admin": (1, True),
            "user": (2, True),
            "service": (3, True),
            "custom-reviewer": (9, False),
        }
    )


class _Cursor:
    def __init__(self, row=None, *, rows=None, rowcount: int = 1) -> None:
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount

    async def fetchone(self):
        return self._row

    async def fetchall(self):
        return self._rows


class _SqliteRoleDb:
    def __init__(self, state: _RoleState) -> None:
        self.state = state
        self.commit_count = 0
        self.events: list[str] = []
        self._authnz_profile_user_backend = "sqlite"
        self._authnz_profile_user_guard_identity = self

    async def execute(self, query: object, params=()):
        concrete = _guard_sql(
            query,
            backend="sqlite",
            connection_identity=self,
            operation="execute",
        )
        normalized = " ".join(concrete.split()).lower()
        if "source_tag" in normalized:
            if "select users.id, users.profile_version" in normalized and "where users.id = ?" in normalized:
                self.events.append("lock-user")
            rows = (
                [("user", int(params[0]), "2026-07-26T12:00:00.000000Z")]
                if self.state.user_exists
                else []
            )
            return _Cursor(rows=rows)
        if normalized.startswith("select id") and "from roles" in normalized:
            role = self.state.roles.get(str(params[0]))
            return _Cursor(role)
        if normalized.startswith("select id") and "from users" in normalized:
            self.events.append("lock-user")
            return _Cursor((42,) if self.state.user_exists else None)
        if normalized.startswith("select 1 from users"):
            return _Cursor((1,) if self.state.user_exists else None)
        if normalized.startswith("update main.users"):
            if "set profile_version" in normalized:
                self.events.append("touch-user")
                return _Cursor(rowcount=1 if self.state.user_exists else 0)
            self.events.append("update-user")
            if not self.state.user_exists:
                return _Cursor(rowcount=0)
            self.state.legacy_role = str(params[0])
            if "is_superuser = 0" in normalized:
                self.state.is_superuser = False
            return _Cursor(rowcount=1)
        if normalized.startswith("delete from user_roles"):
            self.events.append("delete-memberships")
            selected_role_id = int(params[-1])
            system_role_ids = {role_id for role_id, is_system in self.state.roles.values() if is_system}
            self.state.memberships -= system_role_ids - {selected_role_id}
            return _Cursor()
        if normalized.startswith("insert into user_roles"):
            self.events.append("upsert-membership")
            role_id = int(params[1])
            self.state.memberships.add(role_id)
            if "do update set expires_at = null" in normalized:
                self.state.expired_memberships.discard(role_id)
            return _Cursor()
        raise AssertionError(f"Unexpected SQLite query: {query}")

    async def commit(self) -> None:
        self.commit_count += 1


class _PostgresRoleDb:
    def __init__(self, state: _RoleState) -> None:
        self.state = state
        self.events: list[str] = []
        self._authnz_profile_user_backend = "postgres"
        self._authnz_profile_user_guard_identity = self

    async def fetchrow(self, query: object, *params):
        concrete = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self,
            operation="fetchrow",
        )
        normalized = " ".join(concrete.split()).lower()
        if normalized.startswith("select id") and "from roles" in normalized:
            role = self.state.roles.get(str(params[0]))
            if role is None:
                return None
            return {"id": role[0], "is_system": role[1]}
        if normalized.startswith("select id") and "from users" in normalized:
            self.events.append("lock-user")
            return {"id": 42} if self.state.user_exists else None
        if normalized.startswith("update public.users"):
            self.events.append("update-user")
            if not self.state.user_exists:
                return None
            self.state.legacy_role = str(params[0])
            if "is_superuser = false" in normalized:
                self.state.is_superuser = False
            return {"id": int(params[-1])}
        raise AssertionError(f"Unexpected PostgreSQL fetchrow: {query}")

    async def fetch(self, query: object, *params):
        concrete = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self,
            operation="fetch",
        )
        normalized = " ".join(concrete.split()).lower()
        if "source_tag" not in normalized:
            raise AssertionError(f"Unexpected PostgreSQL fetch: {query}")
        if "for update" in normalized:
            self.events.append("lock-user")
        if not self.state.user_exists:
            return []
        return [
            {
                "source_tag": "user",
                "source_id": int(params[0]),
                "candidate_value": datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc),
            }
        ]

    async def execute(self, query: object, *params):
        concrete = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self,
            operation="execute",
        )
        normalized = " ".join(concrete.split()).lower()
        if normalized.startswith("update public.users"):
            if "set profile_version" in normalized:
                self.events.append("touch-user")
                return "UPDATE 1" if self.state.user_exists else "UPDATE 0"
            self.events.append("update-user")
            if not self.state.user_exists:
                return "UPDATE 0"
            self.state.legacy_role = str(params[0])
            if "is_superuser = false" in normalized:
                self.state.is_superuser = False
            return "UPDATE 1"
        if normalized.startswith("delete from user_roles"):
            self.events.append("delete-memberships")
            selected_role_id = int(params[-1])
            system_role_ids = {role_id for role_id, is_system in self.state.roles.values() if is_system}
            self.state.memberships -= system_role_ids - {selected_role_id}
            return "DELETE"
        if normalized.startswith("insert into user_roles"):
            self.events.append("upsert-membership")
            role_id = int(params[1])
            self.state.memberships.add(role_id)
            if "do update set expires_at = null" in normalized:
                self.state.expired_memberships.discard(role_id)
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
        assert db.commit_count == 0


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


def test_request_scoped_admin_operations_do_not_commit_their_connection() -> None:
    assert "await db.commit()" not in inspect.getsource(admin_users_service)
    assert "await db.commit()" not in inspect.getsource(admin_rbac)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_update_user_reactivates_expired_selected_system_role(monkeypatch, backend: str) -> None:
    state = _RoleState("user", {1, 2, 9}, expired_memberships={1})
    db = _PostgresRoleDb(state) if backend == "postgres" else _SqliteRoleDb(state)
    monkeypatch.setattr(admin_users_service.admin_scope_service, "enforce_admin_user_scope", _allow)
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _reauth)
    monkeypatch.setattr(admin_users_service, "_emit_admin_account_audit_event", _no_audit)

    await admin_users_service.update_user(
        _principal(),
        42,
        UserUpdateRequest(role="admin", reason="Support case 123"),
        db,
        password_service=object(),
        is_pg_fn=lambda: _is_postgres(backend),
    )

    assert state.memberships == {1, 9}
    assert state.expired_memberships == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_update_user_demotion_clears_superuser_bypass(monkeypatch, backend: str) -> None:
    state = _RoleState("admin", {1, 9}, is_superuser=True)
    db = _PostgresRoleDb(state) if backend == "postgres" else _SqliteRoleDb(state)
    monkeypatch.setattr(admin_users_service.admin_scope_service, "enforce_admin_user_scope", _allow)
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _reauth)
    monkeypatch.setattr(admin_users_service, "_emit_admin_account_audit_event", _no_audit)

    await admin_users_service.update_user(
        _principal(),
        42,
        UserUpdateRequest(role="user", reason="Support case 123"),
        db,
        password_service=object(),
        is_pg_fn=lambda: _is_postgres(backend),
    )

    assert state.legacy_role == "user"
    assert state.memberships == {2, 9}
    assert state.is_superuser is False


@pytest.mark.asyncio
async def test_postgres_role_change_locks_user_before_replacing_memberships(monkeypatch) -> None:
    state = _RoleState("user", {2, 9}, is_superuser=False)
    db = _PostgresRoleDb(state)
    monkeypatch.setattr(admin_users_service.admin_scope_service, "enforce_admin_user_scope", _allow)
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _reauth)
    monkeypatch.setattr(admin_users_service, "_emit_admin_account_audit_event", _no_audit)

    await admin_users_service.update_user(
        _principal(),
        42,
        UserUpdateRequest(role="admin", reason="Support case 123"),
        db,
        password_service=object(),
        is_pg_fn=lambda: _is_postgres("postgres"),
    )

    assert db.events[:3] == ["lock-user", "delete-memberships", "upsert-membership"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_role_change_for_missing_user_returns_404_without_membership_writes(
    monkeypatch,
    backend: str,
) -> None:
    state = _RoleState("user", {2, 9}, is_superuser=False, user_exists=False)
    db = _PostgresRoleDb(state) if backend == "postgres" else _SqliteRoleDb(state)
    monkeypatch.setattr(admin_users_service.admin_scope_service, "enforce_admin_user_scope", _allow)
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _reauth)

    with pytest.raises(HTTPException) as exc_info:
        await admin_users_service.update_user(
            _principal(),
            999,
            UserUpdateRequest(role="admin", reason="Support case 123"),
            db,
            password_service=object(),
            is_pg_fn=lambda: _is_postgres(backend),
        )

    assert exc_info.value.status_code == 404
    assert state.memberships == {2, 9}
    assert "delete-memberships" not in db.events
    assert "upsert-membership" not in db.events


async def _is_postgres(backend: str) -> bool:
    return backend == "postgres"
