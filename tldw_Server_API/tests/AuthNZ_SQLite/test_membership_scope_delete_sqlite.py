from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    MembershipAuthority,
    MembershipAuthorizationError,
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.UserProfiles.version_gateway import ProfileVersionGateway

_BOOTSTRAP = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)
_FUTURE_FLOOR = datetime(2099, 1, 1, tzinfo=timezone.utc)


async def _create_sqlite_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, dict[str, int]]:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
        AuthnzOrgsTeamsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    user_ids: dict[str, int] = {}
    async with pool.transaction() as conn:
        gateway = VersionedUserWriteGateway(
            "sqlite",
            clock=lambda: datetime(2026, 8, 8, tzinfo=timezone.utc),
        )
        for label in ("owner", "first", "second"):
            result = await gateway.insert_user(
                conn,
                values={
                    "username": label,
                    "email": f"{label}@example.com",
                    "password_hash": "x",
                    "is_active": True,
                },
            )
            user_ids[label] = result.affected_user_ids[0]

    repo = AuthnzOrgsTeamsRepo(pool)
    organization = await repo.create_organization_with_owner_membership(
        name="Scope deletion organization",
        owner_user_id=user_ids["owner"],
        context=_BOOTSTRAP,
    )
    org_id = int(organization["id"])
    for label, role in (("first", "member"), ("second", "member")):
        await repo.add_org_member(
            org_id=org_id,
            user_id=user_ids[label],
            role=role,
            context=_BOOTSTRAP,
        )
    team = await repo.create_team(org_id=org_id, name="Affected team")
    team_id = int(team["id"])
    for label in ("first", "second"):
        await repo.add_team_member(
            team_id=team_id,
            user_id=user_ids[label],
            role="member",
            context=_BOOTSTRAP,
        )
    empty_team = await repo.create_team(org_id=org_id, name="Empty child team")
    return pool, repo, {
        **user_ids,
        "org": org_id,
        "team": team_id,
        "empty_team": int(empty_team["id"]),
    }


def _owner_context(user_id: int) -> ActorMembershipWriteContext:
    return ActorMembershipWriteContext(
        actor_user_id=user_id,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )


class _ManagedSQLiteConnectionProxy:
    _authnz_profile_user_backend = "sqlite"

    @property
    def _authnz_profile_user_guard_identity(self) -> object:
        return self._conn._authnz_profile_user_guard_identity


class _SQLiteSecretDeleteGateConnection(_ManagedSQLiteConnectionProxy):
    def __init__(
        self,
        conn: Any,
        *,
        scope_type: str,
        ready: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        self._conn = conn
        self._scope_type = scope_type
        self._ready = ready
        self._release = release
        self._paused = False

    async def execute(self, query: Any, *args: Any) -> Any:
        query_text = query if type(query) is str else getattr(query, "text", "")
        if type(query_text) is str:
            normalized = " ".join(query_text.split())
            parent_table = "organizations" if self._scope_type == "org" else "teams"
            if (
                not self._paused
                and normalized.startswith(f"DELETE FROM main.{parent_table}")
            ):
                self._paused = True
                self._ready.set()
                await self._release.wait()
        return await self._conn.execute(query, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _SQLiteSecretDeleteGatePool:
    def __init__(
        self,
        delegate: Any,
        *,
        scope_type: str,
        ready: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        self._delegate = delegate
        self.pool = delegate.pool
        self._scope_type = scope_type
        self._ready = ready
        self._release = release

    @asynccontextmanager
    async def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ):
        async with self._delegate.transaction(
            acquire_timeout_seconds=acquire_timeout_seconds,
        ) as conn:
            yield _SQLiteSecretDeleteGateConnection(
                conn,
                scope_type=self._scope_type,
                ready=self._ready,
                release=self._release,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _SQLiteTransactionAttemptPool:
    def __init__(self, delegate: Any, attempted: asyncio.Event) -> None:
        self._delegate = delegate
        self.pool = delegate.pool
        self._attempted = attempted

    @asynccontextmanager
    async def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ):
        self._attempted.set()
        async with self._delegate.transaction(
            acquire_timeout_seconds=acquire_timeout_seconds,
        ) as conn:
            yield conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _SQLiteUpsertInsertGateConnection(_ManagedSQLiteConnectionProxy):
    def __init__(
        self,
        conn: Any,
        *,
        ready: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        self._conn = conn
        self._ready = ready
        self._release = release

    async def execute(self, query: Any, *args: Any) -> Any:
        if type(query) is str and " ".join(query.split()).startswith(
            "INSERT INTO org_provider_secrets"
        ):
            self._ready.set()
            await self._release.wait()
        return await self._conn.execute(query, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _SQLiteUpsertInsertGatePool:
    def __init__(
        self,
        delegate: Any,
        *,
        ready: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        self._delegate = delegate
        self.pool = delegate.pool
        self._ready = ready
        self._release = release

    @asynccontextmanager
    async def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ):
        async with self._delegate.transaction(
            acquire_timeout_seconds=acquire_timeout_seconds,
        ) as conn:
            yield _SQLiteUpsertInsertGateConnection(
                conn,
                ready=self._ready,
                release=self._release,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


async def _record_touches(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    touched: list[int] = []
    original = VersionedUserWriteGateway.final_touch

    async def _record(self, conn, *, user_id, version_floor):
        touched.append(int(user_id))
        return await original(
            self,
            conn,
            user_id=user_id,
            version_floor=version_floor,
        )

    monkeypatch.setattr(VersionedUserWriteGateway, "final_touch", _record)
    return touched


@pytest.mark.integration
@pytest.mark.asyncio
async def test_team_delete_uses_writer_and_advances_once_beyond_inherited_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, repo, ids = await _create_sqlite_fixture(tmp_path, monkeypatch)
    async with pool.transaction() as conn:
        await conn.execute(
            "UPDATE teams SET is_active = 0 WHERE id = ?",
            (ids["team"],),
        )
        await conn.execute(
            "INSERT INTO team_config_overrides "
            "(team_id, key, value_json, updated_at) VALUES (?, ?, ?, ?)",
            (ids["team"], "future", "true", _FUTURE_FLOOR.isoformat()),
        )
        await conn.execute(
            "INSERT INTO org_provider_secrets "
            "(scope_type, scope_id, provider, encrypted_blob) VALUES (?, ?, ?, ?)",
            ("team", ids["team"], "test", "ciphertext"),
        )
    touched = await _record_touches(monkeypatch)

    await repo.delete_team_with_provider_secrets(
        team_id=ids["team"],
        context=_owner_context(ids["owner"]),
    )

    assert touched == sorted((ids["first"], ids["second"]))
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM teams WHERE id = ?", (ids["team"],)
    ) == 0
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM org_provider_secrets "
        "WHERE scope_type = 'team' AND scope_id = ?",
        (ids["team"],),
    ) == 0
    for user_id in (ids["first"], ids["second"]):
        assert await ProfileVersionGateway(pool).read(user_id) > _FUTURE_FLOOR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_org_delete_covers_empty_teams_and_touches_each_user_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, repo, ids = await _create_sqlite_fixture(tmp_path, monkeypatch)
    async with pool.transaction() as conn:
        await conn.execute(
            "UPDATE teams SET is_active = 0 WHERE id = ?",
            (ids["empty_team"],),
        )
        await conn.execute(
            "INSERT INTO org_config_overrides "
            "(org_id, key, value_json, updated_at) VALUES (?, ?, ?, ?)",
            (ids["org"], "future", "true", _FUTURE_FLOOR.isoformat()),
        )
        for scope_type, scope_id in (
            ("org", ids["org"]),
            ("team", ids["empty_team"]),
        ):
            await conn.execute(
                "INSERT INTO org_provider_secrets "
                "(scope_type, scope_id, provider, encrypted_blob) VALUES (?, ?, ?, ?)",
                (scope_type, scope_id, "test", "ciphertext"),
            )
    touched = await _record_touches(monkeypatch)

    await repo.delete_organization_with_provider_secrets(
        org_id=ids["org"],
        context=_owner_context(ids["owner"]),
    )

    assert touched == sorted((ids["owner"], ids["first"], ids["second"]))
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM organizations WHERE id = ?", (ids["org"],)
    ) == 0
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM teams WHERE org_id = ?", (ids["org"],)
    ) == 0
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM org_provider_secrets WHERE "
        "(scope_type = 'org' AND scope_id = ?) OR "
        "(scope_type = 'team' AND scope_id = ?)",
        (ids["org"], ids["empty_team"]),
    ) == 0
    for user_id in (ids["owner"], ids["first"], ids["second"]):
        assert await ProfileVersionGateway(pool).read(user_id) > _FUTURE_FLOOR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ownership_transfer_updates_pointer_roles_and_touches_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, repo, ids = await _create_sqlite_fixture(tmp_path, monkeypatch)
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO org_config_overrides "
            "(org_id, key, value_json, updated_at) VALUES (?, ?, ?, ?)",
            (ids["org"], "future", "true", _FUTURE_FLOOR.isoformat()),
        )
    touched = await _record_touches(monkeypatch)

    organization = await repo.transfer_organization_ownership(
        org_id=ids["org"],
        new_owner_user_id=ids["first"],
        current_owner_user_id=ids["owner"],
        context=_owner_context(ids["owner"]),
    )

    assert organization is not None
    assert organization["owner_user_id"] == ids["first"]
    assert touched == sorted((ids["owner"], ids["first"]))
    assert await pool.fetchval(
        "SELECT role FROM org_members WHERE org_id = ? AND user_id = ?",
        (ids["org"], ids["first"]),
    ) == "owner"
    assert await pool.fetchval(
        "SELECT role FROM org_members WHERE org_id = ? AND user_id = ?",
        (ids["org"], ids["owner"]),
    ) == "admin"
    for user_id in (ids["owner"], ids["first"]):
        assert await ProfileVersionGateway(pool).read(user_id) > _FUTURE_FLOOR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ownership_transfer_reauthorizes_locked_current_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, repo, ids = await _create_sqlite_fixture(tmp_path, monkeypatch)
    await repo.update_org_member_role(
        org_id=ids["org"],
        user_id=ids["first"],
        role="admin",
        context=_BOOTSTRAP,
    )

    with pytest.raises(MembershipAuthorizationError):
        await repo.transfer_organization_ownership(
            org_id=ids["org"],
            new_owner_user_id=ids["second"],
            current_owner_user_id=ids["owner"],
            context=_owner_context(ids["first"]),
        )

    assert await pool.fetchval(
        "SELECT owner_user_id FROM organizations WHERE id = ?",
        (ids["org"],),
    ) == ids["owner"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ownership_transfer_missing_organization_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pool, repo, ids = await _create_sqlite_fixture(tmp_path, monkeypatch)

    result = await repo.transfer_organization_ownership(
        org_id=ids["org"] + 10_000,
        new_owner_user_id=ids["first"],
        current_owner_user_id=ids["owner"],
        context=_owner_context(ids["owner"]),
    )

    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ("org", "team"))
@pytest.mark.parametrize("parent_state", ("missing", "inactive"))
async def test_provider_secret_upsert_preserves_exact_scope_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope_type: str,
    parent_state: str,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.membership_writer import (
        MembershipScopeNotFound,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )

    pool, _repo, ids = await _create_sqlite_fixture(tmp_path, monkeypatch)
    scope_id = ids[scope_type]
    if parent_state == "missing":
        scope_id += 10_000
    else:
        parent_table = "organizations" if scope_type == "org" else "teams"
        async with pool.transaction() as conn:
            await conn.execute(
                f"UPDATE main.{parent_table} SET is_active = 0 WHERE id = ?",  # nosec B608
                (scope_id,),
            )

    with pytest.raises(MembershipScopeNotFound) as exc_info:
        await AuthnzOrgProviderSecretsRepo(pool).upsert_secret(
            scope_type=scope_type,
            scope_id=scope_id,
            provider="openai",
            encrypted_blob="must-not-survive",
            key_hint="missing",
            metadata=None,
            updated_at=datetime.now(timezone.utc),
            created_by=ids["owner"],
            updated_by=ids["owner"],
        )

    assert type(exc_info.value) is MembershipScopeNotFound
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM org_provider_secrets "
        "WHERE scope_type = ? AND scope_id = ?",
        (scope_type, scope_id),
    ) == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ("org", "team"))
async def test_scope_delete_serializes_against_provider_secret_upsert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope_type: str,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.membership_writer import (
        MembershipScopeNotFound,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
        AuthnzOrgsTeamsRepo,
    )

    pool, _repo, ids = await _create_sqlite_fixture(tmp_path, monkeypatch)
    scope_id = ids[scope_type]
    delete_ready = asyncio.Event()
    release_delete = asyncio.Event()
    upsert_attempted = asyncio.Event()
    delete_repo = AuthnzOrgsTeamsRepo(
        _SQLiteSecretDeleteGatePool(
            pool,
            scope_type=scope_type,
            ready=delete_ready,
            release=release_delete,
        )
    )
    secret_repo = AuthnzOrgProviderSecretsRepo(
        _SQLiteTransactionAttemptPool(pool, upsert_attempted)
    )
    delete_call = (
        delete_repo.delete_organization_with_provider_secrets(
            org_id=scope_id,
            context=_owner_context(ids["owner"]),
        )
        if scope_type == "org"
        else delete_repo.delete_team_with_provider_secrets(
            team_id=scope_id,
            context=_owner_context(ids["owner"]),
        )
    )
    delete_task = asyncio.create_task(delete_call)
    upsert_task: asyncio.Task[Any] | None = None
    try:
        await asyncio.wait_for(delete_ready.wait(), timeout=5)
        upsert_task = asyncio.create_task(
            secret_repo.upsert_secret(
                scope_type=scope_type,
                scope_id=scope_id,
                provider="openai",
                encrypted_blob="must-not-survive",
                key_hint="race",
                metadata=None,
                updated_at=datetime.now(timezone.utc),
                created_by=ids["owner"],
                updated_by=ids["owner"],
            )
        )
        await asyncio.wait_for(upsert_attempted.wait(), timeout=5)
        release_delete.set()
        deleted, upserted = await asyncio.wait_for(
            asyncio.gather(delete_task, upsert_task, return_exceptions=True),
            timeout=10,
        )
    finally:
        release_delete.set()
        pending = [
            task
            for task in (delete_task, upsert_task)
            if task is not None and not task.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert deleted is None
    assert isinstance(upserted, MembershipScopeNotFound)
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM org_provider_secrets "
        "WHERE scope_type = ? AND scope_id = ?",
        (scope_type, scope_id),
    ) == 0

    async with pool.transaction() as conn:
        if scope_type == "org":
            await conn.execute(
                "INSERT INTO organizations "
                "(id, name, slug, owner_user_id, is_active) "
                "VALUES (?, ?, ?, ?, 1)",
                (
                    scope_id,
                    "Reused organization",
                    f"reused-org-{scope_id}",
                    ids["owner"],
                ),
            )
        else:
            await conn.execute(
                "INSERT INTO teams (id, org_id, name, slug, is_active) "
                "VALUES (?, ?, ?, ?, 1)",
                (
                    scope_id,
                    ids["org"],
                    "Reused team",
                    f"reused-team-{scope_id}",
                ),
            )

    assert (
        await AuthnzOrgProviderSecretsRepo(pool).fetch_secret(
            scope_type,
            scope_id,
            "openai",
        )
        is None
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ("org", "team"))
async def test_provider_secret_upsert_commits_before_scope_delete_removes_both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope_type: str,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
        AuthnzOrgsTeamsRepo,
    )

    pool, _repo, ids = await _create_sqlite_fixture(tmp_path, monkeypatch)
    scope_id = ids[scope_type]
    insert_ready = asyncio.Event()
    release_insert = asyncio.Event()
    delete_attempted = asyncio.Event()
    upsert_repo = AuthnzOrgProviderSecretsRepo(
        _SQLiteUpsertInsertGatePool(
            pool,
            ready=insert_ready,
            release=release_insert,
        )
    )
    delete_repo = AuthnzOrgsTeamsRepo(
        _SQLiteTransactionAttemptPool(pool, delete_attempted)
    )
    upsert_task = asyncio.create_task(
        upsert_repo.upsert_secret(
            scope_type=scope_type,
            scope_id=scope_id,
            provider="openai",
            encrypted_blob="committed-before-delete",
            key_hint="race",
            metadata=None,
            updated_at=datetime.now(timezone.utc),
            created_by=ids["owner"],
            updated_by=ids["owner"],
        )
    )
    delete_task: asyncio.Task[Any] | None = None
    try:
        await asyncio.wait_for(insert_ready.wait(), timeout=5)
        delete_call = (
            delete_repo.delete_organization_with_provider_secrets(
                org_id=scope_id,
                context=_owner_context(ids["owner"]),
            )
            if scope_type == "org"
            else delete_repo.delete_team_with_provider_secrets(
                team_id=scope_id,
                context=_owner_context(ids["owner"]),
            )
        )
        delete_task = asyncio.create_task(delete_call)
        await asyncio.wait_for(delete_attempted.wait(), timeout=5)
        release_insert.set()
        upserted, deleted = await asyncio.wait_for(
            asyncio.gather(upsert_task, delete_task),
            timeout=10,
        )
    finally:
        release_insert.set()
        pending = [
            task
            for task in (upsert_task, delete_task)
            if task is not None and not task.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert upserted["provider"] == "openai"
    assert deleted is None
    parent_table = "organizations" if scope_type == "org" else "teams"
    assert await pool.fetchval(
        f"SELECT COUNT(*) FROM {parent_table} WHERE id = ?",  # nosec B608
        (scope_id,),
    ) == 0
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM org_provider_secrets "
        "WHERE scope_type = ? AND scope_id = ?",
        (scope_type, scope_id),
    ) == 0
