from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    MembershipLockBackend,
    MembershipMutationResult,
    MembershipUserVersionFloor,
    MembershipWriter,
    MembershipWriteResult,
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo

_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


class _Tx:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        return False


class _PoolStub:
    def __init__(self, conn: Any, *, postgres: bool) -> None:
        self._conn = conn
        self.pool = object() if postgres else None
        self.transaction_acquire_timeouts: list[float | None] = []

    def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ) -> _Tx:
        self.transaction_acquire_timeouts.append(acquire_timeout_seconds)
        return _Tx(self._conn)


class _Cursor:
    def __init__(self, row: Any = None) -> None:
        self._row = row

    async def fetchone(self) -> Any:
        return self._row


class _SqliteConnWithPgTrap:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []
        self._default_team_select_calls = 0

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchrow")

    async def execute(self, query: str, params: Any) -> _Cursor:
        self.execute_calls.append((str(query), params))
        lower_q = str(query).lower()
        if "select tm.team_id, tm.user_id, tm.role, t.org_id" in lower_q:
            return _Cursor((2, 7, "member", 11))
        if "select id from main.teams where org_id = ? and name = ?" in lower_q:
            self._default_team_select_calls += 1
            if self._default_team_select_calls == 1:
                return _Cursor(None)
            return _Cursor((55,))
        return _Cursor(None)


class _PostgresConnWithSqliteTrap:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *params: Any) -> str:
        lower_q = str(query).lower()
        if "?" in lower_q:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.execute_calls.append((str(query), tuple(params)))
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any]:
        lower_q = str(query).lower()
        if "?" in lower_q:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.fetchrow_calls.append((str(query), tuple(params)))
        return {"team_id": 2, "user_id": 7, "role": "member", "org_id": 11}


class _Acquire:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        return False


class _ListCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _SqliteMembershipListConn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []

    async def execute(self, query: str, params: Any) -> _ListCursor:
        lowered = str(query).lower()
        self.execute_calls.append((str(query), params))
        if "$1" in lowered:
            raise AssertionError("SQLite backend path should not use Postgres placeholders")
        return _ListCursor([(2, 7, "member", 11)])


class _SqliteMembershipListPool:
    def __init__(self, conn: Any) -> None:
        self.pool = None
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)

    async def fetchall(self, *_args: Any, **_kwargs: Any):  # noqa: ANN002
        raise AssertionError("SQLite backend path should not call pool.fetchall")


class _PostgresMembershipListPool:
    def __init__(self) -> None:
        self.pool = object()
        self.fetchall_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchall(self, query: str, *params: Any) -> list[dict[str, Any]]:
        lowered = str(query).lower()
        if "?" in lowered:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.fetchall_calls.append((str(query), tuple(params)))
        return [{"team_id": 2, "user_id": 7, "role": "member", "org_id": 11}]


class _SqliteUpdateTeamConn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchrow")

    async def execute(self, query: str, params: Any) -> _Cursor:
        lowered = str(query).lower()
        if "$1" in lowered:
            raise AssertionError("SQLite backend path should not use Postgres placeholders")
        self.execute_calls.append((str(query), params))
        if "select id, org_id, name, slug, description, is_active" in lowered:
            return _Cursor((9, 3, "renamed", "renamed-team", "updated", 1, "c", "u"))
        return _Cursor(None)


class _PostgresUpdateTeamConn:
    def __init__(self) -> None:
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *params: Any):  # noqa: ANN001, ANN002
        if "?" in str(query):
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        raise AssertionError("Postgres update_team path should not call conn.execute")

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any]:
        lowered = str(query).lower()
        if "?" in lowered:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.fetchrow_calls.append((str(query), tuple(params)))
        return {
            "id": 9,
            "org_id": 3,
            "name": "renamed",
            "slug": "renamed-team",
            "description": "updated",
            "is_active": True,
            "created_at": "c",
            "updated_at": "u",
        }


class _SqliteTransferOwnershipConn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchrow")

    async def execute(self, query: str, params: Any) -> _Cursor:
        lowered = str(query).lower()
        if "$1" in lowered:
            raise AssertionError("SQLite backend path should not use Postgres placeholders")
        self.execute_calls.append((str(query), params))
        if "select id, name, slug, owner_user_id, is_active" in lowered:
            return _Cursor((3, "org", "org-slug", 22, 1, "c", "u"))
        return _Cursor(None)


class _PostgresTransferOwnershipConn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *params: Any) -> str:
        lowered = str(query).lower()
        if "?" in lowered:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.execute_calls.append((str(query), tuple(params)))
        return "UPDATE 1"

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any]:
        lowered = str(query).lower()
        if "?" in lowered:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.fetchrow_calls.append((str(query), tuple(params)))
        return {
            "id": 3,
            "name": "org",
            "slug": "org-slug",
            "owner_user_id": 22,
            "is_active": True,
            "created_at": "c",
            "updated_at": "u",
        }


class _SqliteDeleteOrgConn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchrow")

    async def execute(self, query: str, params: Any) -> _Cursor:
        lowered = str(query).lower()
        if "$1" in lowered:
            raise AssertionError("SQLite backend path should not use Postgres placeholders")
        self.execute_calls.append((str(query), params))
        return _Cursor(None)


class _PostgresDeleteTeamConn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *params: Any) -> str:
        lowered = str(query).lower()
        if "?" in lowered:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.execute_calls.append((str(query), tuple(params)))
        return "DELETE 1"

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("Postgres delete-team path should not call conn.fetchrow")


@pytest.mark.asyncio
async def test_add_team_member_sqlite_delegates_with_sqlite_writer(
    monkeypatch: pytest.MonkeyPatch,
):
    conn = _SqliteConnWithPgTrap()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))
    observed: list[tuple[Any, MembershipLockBackend]] = []

    async def _apply(writer, **kwargs):
        observed.append((kwargs["conn"], writer._backend))  # noqa: SLF001
        mutation = kwargs["mutations"][0]
        floor = datetime.now(timezone.utc)
        return MembershipWriteResult(
            mutation_results=(MembershipMutationResult(mutation, True, True, mutation.role, 11),),
            affected_user_ids=(7,),
            version_floors=(MembershipUserVersionFloor(7, floor, floor),),
        )

    monkeypatch.setattr(MembershipWriter, "apply_membership_mutations", _apply)

    row = await repo.add_team_member(
        team_id=2,
        user_id=7,
        role="member",
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )

    assert row["team_id"] == 2
    assert row["org_id"] == 11
    assert observed == [(conn, MembershipLockBackend.SQLITE)]


@pytest.mark.asyncio
async def test_add_team_member_postgres_delegates_with_postgres_writer(
    monkeypatch: pytest.MonkeyPatch,
):
    conn = _PostgresConnWithSqliteTrap()
    pool = _PoolStub(conn, postgres=True)
    repo = AuthnzOrgsTeamsRepo(db_pool=pool)
    observed: list[tuple[Any, MembershipLockBackend]] = []

    async def _apply(writer, **kwargs):
        observed.append((kwargs["conn"], writer._backend))  # noqa: SLF001
        mutation = kwargs["mutations"][0]
        floor = datetime.now(timezone.utc)
        return MembershipWriteResult(
            mutation_results=(MembershipMutationResult(mutation, True, True, mutation.role, 11),),
            affected_user_ids=(7,),
            version_floors=(MembershipUserVersionFloor(7, floor, floor),),
        )

    monkeypatch.setattr(MembershipWriter, "apply_membership_mutations", _apply)

    row = await repo.add_team_member(
        team_id=2,
        user_id=7,
        role="member",
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )

    assert row["team_id"] == 2
    assert row["org_id"] == 11
    assert observed == [(conn, MembershipLockBackend.POSTGRESQL)]
    assert pool.transaction_acquire_timeouts == [5.0]


@pytest.mark.asyncio
async def test_ensure_user_in_default_team_sqlite_backend_selection_uses_sqlite_upsert():
    conn = _SqliteConnWithPgTrap()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))

    await repo._ensure_user_in_default_team(conn, org_id=11, user_id=7)

    assert conn.execute_calls
    insert_calls = [
        q for q, _ in conn.execute_calls if "insert or ignore into team_members" in q.lower()
    ]
    assert insert_calls


@pytest.mark.asyncio
async def test_list_active_team_memberships_for_user_sqlite_backend_selection_uses_execute():
    conn = _SqliteMembershipListConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_SqliteMembershipListPool(conn))

    rows = await repo.list_active_team_memberships_for_user(user_id=7)

    assert rows == [{"team_id": 2, "user_id": 7, "role": "member", "org_id": 11}]
    assert conn.execute_calls
    assert "coalesce(tm.status, 'active') = 'active'" in conn.execute_calls[0][0].lower()


@pytest.mark.asyncio
async def test_list_active_team_memberships_for_user_postgres_backend_selection_uses_fetchall():
    pool = _PostgresMembershipListPool()
    repo = AuthnzOrgsTeamsRepo(db_pool=pool)

    rows = await repo.list_active_team_memberships_for_user(user_id=7)

    assert rows == [{"team_id": 2, "user_id": 7, "role": "member", "org_id": 11}]
    assert pool.fetchall_calls
    assert "coalesce(tm.status, 'active') = 'active'" in pool.fetchall_calls[0][0].lower()


@pytest.mark.asyncio
async def test_update_team_sqlite_backend_selection_uses_execute():
    conn = _SqliteUpdateTeamConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))

    row = await repo.update_team(team_id=9, name="renamed")

    assert row and row["id"] == 9
    assert row["name"] == "renamed"
    assert conn.execute_calls
    assert "update teams set name = ?" in conn.execute_calls[0][0].lower()


@pytest.mark.asyncio
async def test_update_team_postgres_backend_selection_uses_fetchrow():
    conn = _PostgresUpdateTeamConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=True))

    row = await repo.update_team(team_id=9, name="renamed")

    assert row and row["id"] == 9
    assert conn.fetchrow_calls
    assert "where id = $1" in conn.fetchrow_calls[0][0].lower()


@pytest.mark.asyncio
async def test_transfer_organization_ownership_sqlite_backend_selection_uses_execute():
    conn = _SqliteTransferOwnershipConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))

    row = await repo.transfer_organization_ownership(
        org_id=3,
        new_owner_user_id=22,
        current_owner_user_id=11,
    )

    assert row and row["owner_user_id"] == 22
    assert conn.execute_calls
    assert "update organizations set owner_user_id = ?" in conn.execute_calls[0][0].lower()


@pytest.mark.asyncio
async def test_transfer_organization_ownership_postgres_backend_selection_uses_fetchrow():
    conn = _PostgresTransferOwnershipConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=True))

    row = await repo.transfer_organization_ownership(
        org_id=3,
        new_owner_user_id=22,
        current_owner_user_id=11,
    )

    assert row and row["owner_user_id"] == 22
    assert conn.execute_calls
    assert conn.fetchrow_calls
    assert "where id = $1" in conn.fetchrow_calls[0][0].lower()


@pytest.mark.asyncio
async def test_delete_organization_with_provider_secrets_sqlite_backend_selection_uses_execute():
    conn = _SqliteDeleteOrgConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))

    await repo.delete_organization_with_provider_secrets(org_id=5)

    assert conn.execute_calls
    assert "scope_type = 'org' and scope_id = ?" in conn.execute_calls[0][0].lower()
    assert "delete from organizations where id = ?" in conn.execute_calls[-1][0].lower()


@pytest.mark.asyncio
async def test_delete_team_with_provider_secrets_postgres_backend_selection_uses_execute():
    conn = _PostgresDeleteTeamConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=True))

    await repo.delete_team_with_provider_secrets(team_id=9)

    assert conn.execute_calls
    assert "scope_type = 'team' and scope_id = $1" in conn.execute_calls[0][0].lower()
    assert "delete from teams where id = $1" in conn.execute_calls[-1][0].lower()
