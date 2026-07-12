"""Verify admin bootstrap grants and enforces canonical RBAC membership."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ import api_key_manager as api_key_manager_module
from tldw_Server_API.app.core.AuthNZ import create_admin, initialize


class _UsersDb:
    def __init__(self, existing: dict | None) -> None:
        self.existing = existing
        self.created = False
        self.created_kwargs: list[dict] = []
        self.updated: list[tuple[int, dict]] = []
        self.list_calls: list[dict] = []
        self.paged_users: list[dict] | None = None

    async def get_user_by_username(self, _username: str):
        return self.existing

    async def create_user(self, **kwargs):
        self.created = True
        self.created_kwargs.append(dict(kwargs))
        return {"id": 81, "username": "bootstrap-admin"}

    async def update_user(self, user_id: int, **kwargs):
        self.updated.append((user_id, dict(kwargs)))
        if self.existing and int(self.existing.get("id", -1)) == user_id:
            self.existing.update(kwargs)
            return dict(self.existing)
        return {"id": user_id, **kwargs}

    async def list_users(self, **kwargs):
        self.list_calls.append(dict(kwargs))
        if self.paged_users is not None:
            offset = int(kwargs.get("offset", 0))
            limit = int(kwargs.get("limit", 100))
            return self.paged_users[offset : offset + limit]
        return [self.existing] if self.existing else []


class _Repo:
    def __init__(self, *, membership_available: bool = True) -> None:
        self.membership_available = membership_available
        self.assigned: list[tuple[int, str]] = []

    async def assign_role_if_missing(self, *, user_id: int, role_name: str) -> None:
        self.assigned.append((user_id, role_name))

    async def has_role_assignment(self, *, user_id: int, role_name: str) -> bool:
        return self.membership_available and (user_id, role_name) in self.assigned


class _RepoFactory:
    repo: _Repo

    @classmethod
    async def from_pool(cls):
        return cls.repo


class _PasswordService:
    min_length = 10

    def hash_password(self, _password: str) -> str:
        return "hashed"


class _ApiManager:
    async def create_api_key(self, **_kwargs):
        return {"key": "not-logged"}


async def _set_common_create_admin_patches(monkeypatch, module, users_db: _UsersDb, repo: _Repo) -> None:
    _RepoFactory.repo = repo
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(AUTH_MODE="multi_user"))
    monkeypatch.setattr(module, "get_users_db", _async_result(users_db))
    monkeypatch.setattr(module, "PasswordService", _PasswordService)
    monkeypatch.setattr(module, "AuthnzUsersRepo", _RepoFactory, raising=False)
    monkeypatch.setattr(module, "ensure_user_directories", _async_result(None))
    monkeypatch.setattr(api_key_manager_module, "get_api_key_manager", _async_result(_ApiManager()))


def _async_result(value):
    async def _result(*_args, **_kwargs):
        return value

    return _result


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [None, {"id": 80, "username": "bootstrap-admin", "role": "admin"}])
async def test_noninteractive_admin_bootstrap_creates_or_repairs_canonical_membership(
    monkeypatch,
    existing,
) -> None:
    users_db = _UsersDb(existing)
    repo = _Repo()
    await _set_common_create_admin_patches(monkeypatch, create_admin, users_db, repo)
    monkeypatch.setitem(create_admin.create_admin_user_non_interactive.__globals__, "AuthnzUsersRepo", _RepoFactory)

    result = await create_admin.create_admin_user_non_interactive(
        "bootstrap-admin",
        "StrongPass123!",
        "bootstrap@example.com",
    )

    assert result is True
    assert repo.assigned == [((existing or {"id": 81})["id"], "admin")]


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [None, {"id": 80, "username": "bootstrap-admin", "role": "admin"}])
async def test_interactive_admin_bootstrap_creates_or_repairs_canonical_membership(
    monkeypatch,
    existing,
) -> None:
    users_db = _UsersDb(existing)
    repo = _Repo()
    await _set_common_create_admin_patches(monkeypatch, initialize, users_db, repo)
    monkeypatch.setattr("builtins.input", lambda _prompt: "bootstrap-admin" if "username" in _prompt else "bootstrap@example.com")
    monkeypatch.setattr(initialize, "getpass", lambda _prompt: "StrongPass123!")
    monkeypatch.setattr(initialize, "get_api_key_manager", _async_result(_ApiManager()))

    result = await initialize.create_admin_user()

    assert result is True
    assert repo.assigned == [((existing or {"id": 81})["id"], "admin")]


@pytest.mark.asyncio
@pytest.mark.parametrize("module_name", ["noninteractive", "interactive"])
async def test_admin_bootstrap_fails_when_admin_role_cannot_be_verified(monkeypatch, module_name: str) -> None:
    module = create_admin if module_name == "noninteractive" else initialize
    users_db = _UsersDb({"id": 80, "username": "bootstrap-admin", "role": "admin"})
    repo = _Repo(membership_available=False)
    await _set_common_create_admin_patches(monkeypatch, module, users_db, repo)
    if module_name == "noninteractive":
        result = await create_admin.create_admin_user_non_interactive(
            "bootstrap-admin",
            "StrongPass123!",
            "bootstrap@example.com",
        )
    else:
        monkeypatch.setattr("builtins.input", lambda _prompt: "bootstrap-admin" if "username" in _prompt else "bootstrap@example.com")
        monkeypatch.setattr(initialize, "getpass", lambda _prompt: "StrongPass123!")
        monkeypatch.setattr(initialize, "get_api_key_manager", _async_result(_ApiManager()))
        result = await initialize.create_admin_user()

    assert result is False
    assert users_db.created is False


@pytest.mark.asyncio
@pytest.mark.parametrize("module_name", ["noninteractive", "interactive"])
async def test_new_admin_is_activated_only_after_canonical_membership(monkeypatch, module_name: str) -> None:
    module = create_admin if module_name == "noninteractive" else initialize
    users_db = _UsersDb(None)
    repo = _Repo()
    await _set_common_create_admin_patches(monkeypatch, module, users_db, repo)

    if module_name == "noninteractive":
        result = await create_admin.create_admin_user_non_interactive(
            "bootstrap-admin",
            "StrongPass123!",
            "bootstrap@example.com",
        )
    else:
        monkeypatch.setattr("builtins.input", lambda prompt: "bootstrap-admin" if "username" in prompt else "bootstrap@example.com")
        monkeypatch.setattr(initialize, "getpass", lambda _prompt: "StrongPass123!")
        monkeypatch.setattr(initialize, "get_api_key_manager", _async_result(_ApiManager()))
        result = await initialize.create_admin_user()

    assert result is True
    assert users_db.created_kwargs[0]["is_active"] is False
    assert users_db.created_kwargs[0]["is_superuser"] is False
    assert users_db.updated == [(81, {"is_active": True, "is_superuser": True})]


@pytest.mark.asyncio
@pytest.mark.parametrize("module_name", ["noninteractive", "interactive"])
async def test_new_admin_membership_failure_leaves_account_disabled(monkeypatch, module_name: str) -> None:
    module = create_admin if module_name == "noninteractive" else initialize
    users_db = _UsersDb(None)
    repo = _Repo(membership_available=False)
    await _set_common_create_admin_patches(monkeypatch, module, users_db, repo)

    if module_name == "noninteractive":
        result = await create_admin.create_admin_user_non_interactive(
            "bootstrap-admin",
            "StrongPass123!",
            "bootstrap@example.com",
        )
    else:
        monkeypatch.setattr("builtins.input", lambda prompt: "bootstrap-admin" if "username" in prompt else "bootstrap@example.com")
        monkeypatch.setattr(initialize, "getpass", lambda _prompt: "StrongPass123!")
        monkeypatch.setattr(initialize, "get_api_key_manager", _async_result(_ApiManager()))
        result = await initialize.create_admin_user()

    assert result is False
    assert users_db.created_kwargs[0]["is_active"] is False
    assert users_db.created_kwargs[0]["is_superuser"] is False
    assert all(update[1] == {"is_active": False, "is_superuser": False} for update in users_db.updated)


@pytest.mark.asyncio
@pytest.mark.parametrize("module_name", ["noninteractive", "interactive"])
async def test_existing_admin_membership_failure_disables_superuser(monkeypatch, module_name: str) -> None:
    module = create_admin if module_name == "noninteractive" else initialize
    existing = {
        "id": 80,
        "username": "bootstrap-admin",
        "role": "admin",
        "is_active": True,
        "is_superuser": True,
    }
    users_db = _UsersDb(existing)
    repo = _Repo(membership_available=False)
    await _set_common_create_admin_patches(monkeypatch, module, users_db, repo)

    if module_name == "noninteractive":
        result = await create_admin.create_admin_user_non_interactive(
            "bootstrap-admin",
            "StrongPass123!",
            "bootstrap@example.com",
        )
    else:
        monkeypatch.setattr("builtins.input", lambda prompt: "bootstrap-admin" if "username" in prompt else "bootstrap@example.com")
        monkeypatch.setattr(initialize, "getpass", lambda _prompt: "StrongPass123!")
        monkeypatch.setattr(initialize, "get_api_key_manager", _async_result(_ApiManager()))
        result = await initialize.create_admin_user()

    assert result is False
    assert users_db.updated == [(80, {"is_active": False, "is_superuser": False})]


@pytest.mark.asyncio
async def test_initialize_existing_admin_scan_repairs_canonical_membership(monkeypatch) -> None:
    users_db = _UsersDb({"id": 80, "username": "bootstrap-admin", "role": "admin"})
    repo = _Repo()
    await _set_common_create_admin_patches(monkeypatch, initialize, users_db, repo)

    await initialize._repair_existing_admin_memberships(users_db)

    assert repo.assigned == [(80, "admin")]


@pytest.mark.asyncio
async def test_initialize_existing_admin_scan_fails_when_admin_role_is_unavailable(monkeypatch) -> None:
    users_db = _UsersDb({"id": 80, "username": "bootstrap-admin", "role": "admin"})
    repo = _Repo(membership_available=False)
    await _set_common_create_admin_patches(monkeypatch, initialize, users_db, repo)

    with pytest.raises(initialize.AuthNZDatabaseError, match="Canonical admin role membership"):
        await initialize._repair_existing_admin_memberships(users_db)

    assert users_db.updated == [(80, {"is_active": False, "is_superuser": False})]


@pytest.mark.asyncio
async def test_initialize_existing_admin_scan_repairs_every_page(monkeypatch) -> None:
    users_db = _UsersDb(None)
    users_db.paged_users = [
        {"id": user_id, "username": f"admin-{user_id}", "role": "admin"}
        for user_id in range(1, 102)
    ]
    repo = _Repo()
    await _set_common_create_admin_patches(monkeypatch, initialize, users_db, repo)

    await initialize._repair_existing_admin_memberships(users_db)

    assert repo.assigned == [(user_id, "admin") for user_id in range(1, 102)]
    assert users_db.list_calls == [
        {"role": "admin", "offset": 0, "limit": 100},
        {"role": "admin", "offset": 100, "limit": 100},
    ]


@pytest.mark.asyncio
async def test_initialize_existing_admin_scan_disables_all_failures_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    users_db = _UsersDb(None)
    users_db.paged_users = [
        {"id": user_id, "username": f"admin-{user_id}", "role": "admin"}
        for user_id in range(1, 102)
    ]
    repo = _Repo(membership_available=False)
    await _set_common_create_admin_patches(monkeypatch, initialize, users_db, repo)

    with pytest.raises(initialize.AuthNZDatabaseError, match="Canonical admin role membership"):
        await initialize._repair_existing_admin_memberships(users_db)

    assert users_db.updated == [
        (user_id, {"is_active": False, "is_superuser": False})
        for user_id in range(1, 102)
    ]
    assert users_db.list_calls == [
        {"role": "admin", "offset": 0, "limit": 100},
        {"role": "admin", "offset": 100, "limit": 100},
    ]
