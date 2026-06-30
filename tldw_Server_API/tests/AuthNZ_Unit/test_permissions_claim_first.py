from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ import permissions as perms_mod
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import require_permissions, require_roles


def test_permissions_module_no_legacy_require_role_helper():
    assert not hasattr(perms_mod, "require_role")


class _FakeUserDB:
    def __init__(self):
        self.calls: list[tuple[str, int, str]] = []
        self._has_perm_result = False
        self._has_role_result = False

    def set_permission_result(self, value: bool) -> None:
        self._has_perm_result = bool(value)

    def set_role_result(self, value: bool) -> None:
        self._has_role_result = bool(value)

    def has_permission(self, user_id, permission: str) -> bool:
        self.calls.append(("perm", int(user_id), permission))
        return self._has_perm_result

    def has_role(self, user_id, role: str) -> bool:
        self.calls.append(("role", int(user_id), role))
        return self._has_role_result


class _FailingUserDB:
    def has_permission(self, user_id, permission: str) -> bool:
        raise RuntimeError("permission database failed at /private/permissions.db")

    def has_role(self, user_id, role: str) -> bool:
        raise RuntimeError("role database failed at /private/roles.db")


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str, *args, **kwargs) -> None:
        self.errors.append(message)

    def debug(self, message: str, *args, **kwargs) -> None:
        pass


def _make_user(roles=None, permissions=None, is_admin: bool = False) -> User:
    return User(
        id=1,
        username="testuser",
        email=None,
        is_active=True,
        roles=list(roles or []),
        permissions=list(permissions or []),
        is_admin=is_admin,
    )


def _make_principal(
    *,
    user_id: int = 7,
    api_key_id: int | None = None,
    roles=None,
    permissions=None,
    is_admin: bool = False,
    subject: str = "test-user",
    kind: str = "user",
) -> AuthPrincipal:
    return AuthPrincipal(
        kind=kind,
        user_id=user_id,
        api_key_id=api_key_id,
        subject=subject,
        token_type="access",  # nosec B106
        jti=None,
        roles=list(roles or []),
        permissions=list(permissions or []),
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


def test_check_permission_uses_claims_true(monkeypatch):


    # Not in single-user mode
    monkeypatch.setattr(perms_mod, "is_single_user_mode", lambda: False)

    fake_db = _FakeUserDB()

    def _fake_get_user_database():

        return fake_db

    monkeypatch.setattr(perms_mod, "get_user_database", _fake_get_user_database)

    user = _make_user(permissions=["media.read", "media.create"])

    assert perms_mod.check_permission(user, "media.read") is True
    # When claims list is present, DB should not be consulted
    assert fake_db.calls == []


def test_check_permission_uses_claims_false_without_db(monkeypatch):


    monkeypatch.setattr(perms_mod, "is_single_user_mode", lambda: False)

    def _broken_get_user_database():

        pytest.fail("get_user_database should not be called when permissions claims are present")

    monkeypatch.setattr(perms_mod, "get_user_database", _broken_get_user_database)

    user = _make_user(permissions=["media.read"])

    # Claims present but missing; DB must not be consulted even if available
    assert perms_mod.check_permission(user, "media.create") is False


def test_check_permission_uses_claims_even_if_db_unavailable(monkeypatch):


    monkeypatch.setattr(perms_mod, "is_single_user_mode", lambda: False)

    def _broken_get_user_database():

        pytest.fail("get_user_database should not be called when permissions claims are present")

    monkeypatch.setattr(perms_mod, "get_user_database", _broken_get_user_database)

    user = _make_user(permissions=["media.read", "media.create"])

    assert perms_mod.check_permission(user, "media.read") is True


def test_check_permission_falls_back_to_db_when_permissions_absent(monkeypatch):


    monkeypatch.setattr(perms_mod, "is_single_user_mode", lambda: False)

    fake_db = _FakeUserDB()
    fake_db.set_permission_result(True)

    def _fake_get_user_database():

        return fake_db

    monkeypatch.setattr(perms_mod, "get_user_database", _fake_get_user_database)

    # Use a minimal object without a permissions attribute
    user = SimpleNamespace(id=5, username="legacy-user", is_active=True)

    assert perms_mod.check_permission(user, "media.read") is True
    assert fake_db.calls == [("perm", 5, "media.read")]


def test_check_role_uses_claims_true(monkeypatch):


    monkeypatch.setattr(perms_mod, "is_single_user_mode", lambda: False)

    fake_db = _FakeUserDB()

    def _fake_get_user_database():

        return fake_db

    monkeypatch.setattr(perms_mod, "get_user_database", _fake_get_user_database)

    user = _make_user(roles=["user", "editor"])

    assert perms_mod.check_role(user, "editor") is True
    assert fake_db.calls == []


def test_check_role_uses_claims_false_without_db(monkeypatch):


    monkeypatch.setattr(perms_mod, "is_single_user_mode", lambda: False)

    def _broken_get_user_database():

        pytest.fail("get_user_database should not be called when role claims are present")

    monkeypatch.setattr(perms_mod, "get_user_database", _broken_get_user_database)

    user = _make_user(roles=["user"])

    # Claims present but missing; DB must not be consulted even if available
    assert perms_mod.check_role(user, "admin") is False


def test_check_role_uses_claims_even_if_db_unavailable(monkeypatch):


    monkeypatch.setattr(perms_mod, "is_single_user_mode", lambda: False)

    def _broken_get_user_database():

        pytest.fail("get_user_database should not be called when role claims are present")

    monkeypatch.setattr(perms_mod, "get_user_database", _broken_get_user_database)

    user = _make_user(roles=["user", "editor"])

    assert perms_mod.check_role(user, "editor") is True


def test_check_role_falls_back_to_db_when_roles_absent(monkeypatch):


    monkeypatch.setattr(perms_mod, "is_single_user_mode", lambda: False)

    fake_db = _FakeUserDB()
    fake_db.set_role_result(True)

    def _fake_get_user_database():

        return fake_db

    monkeypatch.setattr(perms_mod, "get_user_database", _fake_get_user_database)

    user = SimpleNamespace(id=8, username="legacy-user", is_active=True)

    assert perms_mod.check_role(user, "admin") is True
    assert fake_db.calls == [("role", 8, "admin")]


@pytest.mark.asyncio
async def test_check_permission_single_user_mode_prefers_claims(monkeypatch):
    monkeypatch.setattr(perms_mod, "is_single_user_mode", lambda: True)

    def _fake_get_user_database():

        pytest.fail("DB should not be used when permissions claims are present in single-user mode")

    monkeypatch.setattr(perms_mod, "get_user_database", _fake_get_user_database)

    user = _make_user(permissions=["media.read"])

    # Claims list governs access even when single-user mode is enabled
    assert perms_mod.check_permission(user, "media.read") is True
    assert perms_mod.check_permission(user, "media.create") is False


def test_check_permission_single_user_mode_without_claims_falls_back_to_db(monkeypatch):


    monkeypatch.setattr(perms_mod, "is_single_user_mode", lambda: True)

    fake_db = _FakeUserDB()
    fake_db.set_permission_result(True)

    def _fake_get_user_database():

        return fake_db

    monkeypatch.setattr(perms_mod, "get_user_database", _fake_get_user_database)

    # Minimal legacy-style user without a permissions attribute
    user = SimpleNamespace(id=42, username="single-user-legacy", is_active=True)

    assert perms_mod.check_permission(user, "media.read") is True
    assert fake_db.calls == [("perm", 42, "media.read")]


@pytest.mark.asyncio
async def test_check_role_admin_implies_admin_and_user(monkeypatch):
    def _fake_get_user_database():
        pytest.fail("DB should not be used when role claims are present in single-user mode")

    monkeypatch.setattr(perms_mod, "get_user_database", _fake_get_user_database)

    # Single-user admin claim implies both 'admin' and 'user' level access
    user = _make_user(roles=["admin"])

    assert perms_mod.check_role(user, "admin") is True
    assert perms_mod.check_role(user, "user") is True
    # Non-admin/user roles still rejected
    assert perms_mod.check_role(user, "viewer") is False


def test_check_role_without_roles_falls_back_to_db(monkeypatch):


    fake_db = _FakeUserDB()
    fake_db.set_role_result(True)

    def _fake_get_user_database():

        return fake_db

    monkeypatch.setattr(perms_mod, "get_user_database", _fake_get_user_database)

    user = SimpleNamespace(id=99, username="single-user-legacy", is_active=True)

    assert perms_mod.check_role(user, "admin") is True
    assert fake_db.calls == [("role", 99, "admin")]


@pytest.mark.parametrize("redact", [False, True])
def test_check_permission_db_failure_log_is_sanitized(monkeypatch, redact):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(perms_mod, "get_user_database", lambda: _FailingUserDB())
    monkeypatch.setattr(perms_mod, "get_settings", lambda: SimpleNamespace(PII_REDACT_LOGS=redact))
    monkeypatch.setattr(perms_mod, "logger", logger_stub)
    monkeypatch.setattr(perms_mod, "_PERMISSION_DB_FALLBACK_LOGGED", True)

    user = SimpleNamespace(id=123, username="legacy-user", is_active=True)

    assert perms_mod.check_permission(user, "media.secret") is False
    assert logger_stub.errors == ["Error checking permission"]
    assert "permission database failed" not in str(logger_stub.errors)
    assert "/private/permissions.db" not in str(logger_stub.errors)
    assert "media.secret" not in str(logger_stub.errors)
    assert "123" not in str(logger_stub.errors)


@pytest.mark.parametrize("redact", [False, True])
def test_check_role_db_failure_log_is_sanitized(monkeypatch, redact):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(perms_mod, "get_user_database", lambda: _FailingUserDB())
    monkeypatch.setattr(perms_mod, "get_settings", lambda: SimpleNamespace(PII_REDACT_LOGS=redact))
    monkeypatch.setattr(perms_mod, "logger", logger_stub)
    monkeypatch.setattr(perms_mod, "_ROLE_DB_FALLBACK_LOGGED", True)

    user = SimpleNamespace(id=456, username="legacy-user", is_active=True)

    assert perms_mod.check_role(user, "admin.secret") is False
    assert logger_stub.errors == ["Error checking role"]
    assert "role database failed" not in str(logger_stub.errors)
    assert "/private/roles.db" not in str(logger_stub.errors)
    assert "admin.secret" not in str(logger_stub.errors)
    assert "456" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_check_any_permission_uses_claims_even_if_db_unavailable(monkeypatch):
    def _broken_get_user_database():
        pytest.fail("get_user_database should not be called when permissions claims are present (any)")

    monkeypatch.setattr(perms_mod, "get_user_database", _broken_get_user_database)

    user = _make_user(permissions=["media.read"])

    # any-permission helper should rely purely on claims when present
    assert perms_mod.check_any_permission(user, ["media.read", "media.create"]) is True
    assert perms_mod.check_any_permission(user, ["media.create", "media.update"]) is False


@pytest.mark.asyncio
async def test_check_all_permissions_uses_claims_even_if_db_unavailable(monkeypatch):
    def _broken_get_user_database():
        pytest.fail("get_user_database should not be called when permissions claims are present (all)")

    monkeypatch.setattr(perms_mod, "get_user_database", _broken_get_user_database)

    user = _make_user(permissions=["media.read", "media.create"])

    # all-permissions helper should rely purely on claims when present
    assert perms_mod.check_all_permissions(user, ["media.read", "media.create"]) is True
    assert perms_mod.check_all_permissions(user, ["media.read", "media.update"]) is False


@pytest.mark.asyncio
async def test_require_permissions_respects_claims_and_admin_bypass():
    checker = require_permissions("media.create")
    admin_principal = _make_principal(roles=["admin"], permissions=[], is_admin=False)
    assert await checker(admin_principal) is admin_principal

    limited_principal = _make_principal(roles=["user"], permissions=["media.read"], is_admin=False)
    with pytest.raises(HTTPException) as exc:
        await checker(limited_principal)
    assert exc.value.status_code == 403

    permitted_principal = _make_principal(permissions=["media.create"], roles=["user"], is_admin=False)
    assert await checker(permitted_principal) is permitted_principal


@pytest.mark.asyncio
async def test_require_permissions_denies_boolean_admin_without_admin_claims():
    checker = require_permissions("media.create")
    principal = _make_principal(roles=["user"], permissions=[], is_admin=True)
    with pytest.raises(HTTPException) as exc:
        await checker(principal)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_require_roles_checks_claims_and_allows_admin():
    checker = require_roles("admin")

    user_principal = _make_principal(roles=["user"], permissions=[])
    with pytest.raises(HTTPException) as exc:
        await checker(user_principal)
    assert exc.value.status_code == 403

    editor_principal = _make_principal(roles=["editor", "admin"], permissions=[])
    assert await checker(editor_principal) is editor_principal

    admin_override = _make_principal(roles=["viewer"], permissions=["*"], is_admin=False)
    assert await checker(admin_override) is admin_override


@pytest.mark.asyncio
async def test_require_roles_denies_boolean_admin_without_admin_claims():
    checker = require_roles("admin")
    principal = _make_principal(roles=["viewer"], permissions=[], is_admin=True)
    with pytest.raises(HTTPException) as exc:
        await checker(principal)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_require_permissions_denies_single_user_principal_without_claim():
    checker = require_permissions("media.create")
    principal = _make_principal(
        kind="user",
        subject="single_user",
        roles=["user"],
        permissions=["media.read"],
        is_admin=False,
    )
    with pytest.raises(HTTPException) as exc:
        await checker(principal)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_require_roles_denies_single_user_principal_without_admin_role():
    checker = require_roles("admin")
    principal = _make_principal(
        kind="user",
        subject="single_user",
        roles=["user"],
        permissions=["media.read"],
        is_admin=False,
    )
    with pytest.raises(HTTPException) as exc:
        await checker(principal)
    assert exc.value.status_code == 403
