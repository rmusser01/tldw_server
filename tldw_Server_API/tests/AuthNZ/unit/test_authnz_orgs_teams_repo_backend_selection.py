from __future__ import annotations

import ast
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_type_hints

import pytest

from tldw_Server_API.app.core.AuthNZ import orgs_teams as orgs_teams_facade
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    RollbackSignal,
    TransactionError,
    UserRegistrationException,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    AnchorOwnership,
    MembershipAuthority,
    MembershipLockBackend,
    MembershipMutationResult,
    MembershipParentRequired,
    MembershipPreflightChanged,
    MembershipScopeDeletionSnapshot,
    MembershipScopeType,
    MembershipTargetNotFound,
    MembershipUserVersionFloor,
    MembershipWriter,
    MembershipWriterContractError,
    MembershipWriteResult,
    TeamParentOrganization,
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.repos import orgs_teams_repo as repo_module
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo

_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)
_ACTOR_MEMBERSHIP_CONTEXT = ActorMembershipWriteContext(
    actor_user_id=11,
    required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
)


def test_postgres_authnz_relation_literals_are_public_qualified() -> None:
    module_path = Path(repo_module.__file__ or "")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    relation = re.compile(
        r"(?<!public\.)(?<!main\.)(?<![a-z_])"
        r"(organizations|teams|org_members|team_members)(?![a-z_])",
        re.IGNORECASE,
    )
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            value = "".join(
                item.value
                for item in node.values
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
        else:
            continue
        if "$" not in value:
            continue
        violations.extend(
            (node.lineno, match.group(0)) for match in relation.finditer(value)
        )
    assert violations == []


def test_sqlite_authnz_relation_literals_are_not_public_qualified() -> None:
    module_path = Path(repo_module.__file__ or "")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    violations = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "?" in node.value
        and "public." in node.value.lower()
    ]
    assert violations == []


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
        self.acquire_timeouts: list[float | None] = []
        self.transaction_acquire_timeouts: list[float | None] = []

    def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ) -> _Tx:
        self.transaction_acquire_timeouts.append(acquire_timeout_seconds)
        return _Tx(self._conn)

    def acquire(self, *, timeout: float | None = None) -> _Tx:
        self.acquire_timeouts.append(timeout)
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
        self.savepoints_created: list[str] = []
        self.savepoints_rolled_back: list[str] = []
        self.savepoints_released: list[str] = []

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchrow")

    async def execute(self, query: str, params: Any) -> _Cursor:
        self.execute_calls.append((str(query), params))
        lower_q = str(query).lower()
        if "select tm.team_id, tm.user_id, tm.role, t.org_id" in lower_q:
            return _Cursor((2, 7, "member", 11))
        if "select id, owner_user_id, is_active from main.organizations" in lower_q:
            return _Cursor((11, 1, 1))
        if "select id, is_active, is_superuser, role from main.users" in lower_q:
            return _Cursor((7, 1, 0, "user"))
        if "select org_id from main.teams where id = ?" in lower_q:
            return _Cursor((11,))
        if "select id from main.teams where org_id = ? and name = ?" in lower_q:
            self._default_team_select_calls += 1
            if self._default_team_select_calls == 1:
                return _Cursor(None)
            return _Cursor((55,))
        return _Cursor(None)

    async def create_savepoint(self, name: str) -> None:
        self.savepoints_created.append(name)

    async def rollback_savepoint(self, name: str) -> None:
        self.savepoints_rolled_back.append(name)

    async def release_savepoint(self, name: str) -> None:
        self.savepoints_released.append(name)


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


class _NoSecondaryAcquirePool:
    def __init__(self, *, postgres: bool) -> None:
        self.pool = object() if postgres else None

    def acquire(self):
        raise AssertionError("supplied connection must avoid pool acquisition")

    async def fetchall(self, *_args: Any, **_kwargs: Any):  # noqa: ANN002
        raise AssertionError("supplied connection must avoid pool fetches")


class _PostgresMembershipReadConn:
    def __init__(self) -> None:
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((str(query), tuple(params)))
        if "org_members" in query:
            return [{"org_id": 11, "role": "admin", "status": "active"}]
        return [
            {
                "team_id": 2,
                "user_id": 7,
                "role": "member",
                "org_id": 11,
                "team_name": "team",
                "org_name": "org",
            }
        ]


class _SqliteMembershipReadConn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []

    async def execute(self, query: str, params: Any) -> _ListCursor:
        self.execute_calls.append((str(query), params))
        if "org_members" in query:
            return _ListCursor([(11, "admin", "active")])
        return _ListCursor([(2, 7, "member", 11, "team", "org")])


class _FailingMembershipReadPool:
    def __init__(self, secret: str) -> None:
        self.pool = object()
        self._secret = secret

    async def fetchall(self, *_args: Any, **_kwargs: Any):  # noqa: ANN002
        raise RuntimeError(self._secret)


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


class _PostgresDeleteOrgConn(_PostgresDeleteTeamConn):
    pass


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
async def test_default_team_provisioning_uses_explicit_connection_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    conn = _SqliteConnWithPgTrap()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))
    observed: list[dict[str, Any]] = []

    async def _apply(_writer, **kwargs):
        observed.append(kwargs)
        floor = datetime.now(timezone.utc)
        mutation_results = tuple(
            MembershipMutationResult(
                mutation=mutation,
                changed=(index == len(kwargs["mutations"]) - 1),
                found=True,
                role=mutation.role,
                organization_id=(
                    11 if mutation.scope_type is MembershipScopeType.TEAM else None
                ),
            )
            for index, mutation in enumerate(kwargs["mutations"])
        )
        return MembershipWriteResult(
            mutation_results=mutation_results,
            affected_user_ids=(7,),
            version_floors=(MembershipUserVersionFloor(7, floor, floor),),
        )

    monkeypatch.setattr(MembershipWriter, "apply_membership_mutations", _apply)

    result = await repo.provision_org_membership_on_connection(
        conn=conn,
        org_id=11,
        user_id=7,
        org_role="member",
        team_id=None,
        team_role=None,
        team_failure_is_best_effort=False,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
        anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
        operation_time=datetime.now(timezone.utc),
    )

    assert result.org_membership.found is True
    assert len(observed) == 1
    assert all(call["conn"] is conn for call in observed)
    assert all(call["context"] is _BOOTSTRAP_MEMBERSHIP_CONTEXT for call in observed)
    assert all(
        call["anchor_ownership"] is AnchorOwnership.CALLER_OWNS_ANCHOR
        for call in observed
    )
    assert observed[0]["mutations"][1].scope_id == 55
    assert not hasattr(repo, "_ensure_user_in_default_team")


@pytest.mark.asyncio
async def test_compound_provisioning_supplies_complete_mutation_set_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _SqliteConnWithPgTrap()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))
    observed: list[tuple[Any, ...]] = []

    async def _apply(_writer, **kwargs):
        mutations = kwargs["mutations"]
        observed.append(mutations)
        floor = datetime.now(timezone.utc)
        return MembershipWriteResult(
            mutation_results=tuple(
                MembershipMutationResult(
                    mutation=mutation,
                    changed=True,
                    found=True,
                    role=mutation.role,
                    organization_id=(
                        11
                        if mutation.scope_type is MembershipScopeType.TEAM
                        else None
                    ),
                )
                for mutation in mutations
            ),
            affected_user_ids=(7,),
            version_floors=(MembershipUserVersionFloor(7, floor, floor),),
        )

    monkeypatch.setattr(MembershipWriter, "apply_membership_mutations", _apply)

    await repo.provision_org_membership_on_connection(
        conn=conn,
        org_id=11,
        user_id=7,
        org_role="member",
        team_id=77,
        team_role="member",
        team_failure_is_best_effort=False,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
        anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
        operation_time=datetime.now(timezone.utc),
    )

    assert len(observed) == 1
    assert tuple(
        (mutation.scope_type, mutation.scope_id) for mutation in observed[0]
    ) == (
        (MembershipScopeType.ORGANIZATION, 11),
        (MembershipScopeType.TEAM, 55),
        (MembershipScopeType.TEAM, 77),
    )


@pytest.mark.asyncio
async def test_explicit_team_best_effort_rolls_back_full_plan_before_base_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _SqliteConnWithPgTrap()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))
    observed: list[tuple[Any, ...]] = []

    async def _apply(_writer, **kwargs):
        mutations = kwargs["mutations"]
        observed.append(mutations)
        if len(mutations) == 3:
            raise RuntimeError("explicit team write failed")
        floor = datetime.now(timezone.utc)
        return MembershipWriteResult(
            mutation_results=tuple(
                MembershipMutationResult(
                    mutation=mutation,
                    changed=True,
                    found=True,
                    role=mutation.role,
                    organization_id=(
                        11
                        if mutation.scope_type is MembershipScopeType.TEAM
                        else None
                    ),
                )
                for mutation in mutations
            ),
            affected_user_ids=(7,),
            version_floors=(MembershipUserVersionFloor(7, floor, floor),),
        )

    monkeypatch.setattr(MembershipWriter, "apply_membership_mutations", _apply)

    result = await repo.provision_org_membership_on_connection(
        conn=conn,
        org_id=11,
        user_id=7,
        org_role="member",
        team_id=77,
        team_role="member",
        team_failure_is_best_effort=True,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
        anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
        operation_time=datetime.now(timezone.utc),
    )

    assert [len(mutations) for mutations in observed] == [3, 2]
    assert conn.savepoints_rolled_back == ["explicit_team_companion"]
    assert result.team_membership_failed is True
    assert result.team_membership is None
    assert len(result.write_results) == 1


@pytest.mark.asyncio
async def test_explicit_team_from_another_org_is_rejected_before_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CrossOrgConnection(_SqliteConnWithPgTrap):
        async def execute(self, query: str, params: Any) -> _Cursor:
            if "select org_id from main.teams where id = ?" in str(query).lower():
                return _Cursor((99,))
            return await super().execute(query, params)

    conn = _CrossOrgConnection()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))

    async def _unexpected_apply(*_args, **_kwargs):
        raise AssertionError("cross-organization team must fail before mutation")

    monkeypatch.setattr(
        MembershipWriter,
        "apply_membership_mutations",
        _unexpected_apply,
    )

    with pytest.raises(MembershipParentRequired):
        await repo.provision_org_membership_on_connection(
            conn=conn,
            org_id=11,
            user_id=7,
            org_role="member",
            team_id=77,
            team_role="member",
            team_failure_is_best_effort=False,
            context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
            anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
            operation_time=datetime.now(timezone.utc),
        )


@pytest.mark.asyncio
async def test_row_only_organization_creation_rejects_owner_pointer() -> None:
    conn = _SqliteConnWithPgTrap()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))

    with pytest.raises(MembershipWriterContractError):
        await repo.create_organization(name="Example", owner_user_id=7)


@pytest.mark.asyncio
async def test_legacy_organization_facade_routes_owner_through_bootstrap_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    class _Repo:
        async def create_organization(self, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("owner-bearing creation bypassed the writer")

        async def create_organization_with_owner_membership(
            self,
            **kwargs: Any,
        ) -> dict[str, Any]:
            observed.update(kwargs)
            return {"id": 17, "owner_user_id": kwargs["owner_user_id"]}

    async def _repo() -> _Repo:
        return _Repo()

    monkeypatch.setattr(orgs_teams_facade, "_get_orgs_teams_repo", _repo)

    organization = await orgs_teams_facade.create_organization(
        name="Legacy owner",
        owner_user_id=7,
        slug="legacy-owner",
        metadata={"source": "compatibility"},
    )

    assert organization == {"id": 17, "owner_user_id": 7}
    assert observed["name"] == "Legacy owner"
    assert observed["owner_user_id"] == 7
    assert observed["slug"] == "legacy-owner"
    assert observed["metadata"] == {"source": "compatibility"}
    assert observed["context"] == TrustedMembershipWriteContext(
        trusted_reason=TrustedMembershipReason.BOOTSTRAP,
    )


@pytest.mark.asyncio
async def test_legacy_organization_facade_preserves_missing_owner_transaction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Repo:
        async def create_organization_with_owner_membership(
            self,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            raise MembershipTargetNotFound()

    async def _repo() -> _Repo:
        return _Repo()

    monkeypatch.setattr(orgs_teams_facade, "_get_orgs_teams_repo", _repo)

    with pytest.raises(TransactionError) as exc_info:
        await orgs_teams_facade.create_organization(
            name="Missing owner",
            owner_user_id=999,
        )

    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_create_organization_with_owner_membership_reuses_one_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _SqliteConnWithPgTrap()
    pool = _PoolStub(conn, postgres=False)
    repo = AuthnzOrgsTeamsRepo(db_pool=pool)
    observed: dict[str, Any] = {}
    events: list[str] = []

    async def _authorize(
        self,
        *,
        conn: Any,
        context: Any,
        owner_user_id: int | None,
    ) -> None:
        events.append("authorize")
        observed["authorize_args"] = (self, conn, context, owner_user_id)

    async def _create(active_conn, **kwargs):
        events.append("create")
        observed["create_conn"] = active_conn
        observed["create_kwargs"] = kwargs
        return {"id": 11, "name": kwargs["name"], "owner_user_id": 7}

    async def _provision(**kwargs):
        events.append("provision")
        observed["provision_kwargs"] = kwargs

    monkeypatch.setattr(
        MembershipWriter,
        "authorize_organization_creation",
        _authorize,
        raising=False,
    )
    monkeypatch.setattr(repo, "_create_organization_on_connection", _create)
    monkeypatch.setattr(repo, "provision_org_membership_on_connection", _provision)

    organization = await repo.create_organization_with_owner_membership(
        name="Example",
        owner_user_id=7,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )

    assert organization["id"] == 11
    assert events == ["authorize", "create", "provision"]
    assert observed["authorize_args"][1] is conn
    assert observed["authorize_args"][2] is _BOOTSTRAP_MEMBERSHIP_CONTEXT
    assert observed["authorize_args"][3] == 7
    assert observed["create_conn"] is conn
    assert observed["provision_kwargs"]["conn"] is conn
    assert observed["provision_kwargs"]["org_role"] == "owner"
    assert pool.transaction_acquire_timeouts == [5.0]


@pytest.mark.asyncio
async def test_create_ownerless_organization_reauthorizes_actor_in_same_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _SqliteConnWithPgTrap()
    pool = _PoolStub(conn, postgres=False)
    repo = AuthnzOrgsTeamsRepo(db_pool=pool)
    context = ActorMembershipWriteContext(
        actor_user_id=11,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    observed: list[tuple[str, Any]] = []

    async def _authorize(
        self,
        *,
        conn: Any,
        context: Any,
        owner_user_id: int | None,
    ) -> None:
        observed.append(("authorize", (self, conn, context, owner_user_id)))

    async def _create(active_conn: Any, **kwargs: Any) -> dict[str, Any]:
        observed.append(("create", (active_conn, kwargs)))
        return {"id": 13, "name": kwargs["name"], "owner_user_id": None}

    monkeypatch.setattr(
        MembershipWriter,
        "authorize_organization_creation",
        _authorize,
        raising=False,
    )
    monkeypatch.setattr(repo, "_create_organization_on_connection", _create)

    organization = await repo.create_organization_as_actor(
        name="Ownerless",
        context=context,
    )

    assert organization["id"] == 13
    assert [event for event, _payload in observed] == ["authorize", "create"]
    assert observed[0][1][1] is conn
    assert observed[0][1][2] is context
    assert observed[0][1][3] is None
    assert observed[1][1][0] is conn
    assert pool.transaction_acquire_timeouts == [5.0]


@pytest.mark.asyncio
async def test_default_team_best_effort_log_does_not_render_backend_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Logger:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def warning(self, message: str, *args: Any) -> None:
            self.messages.append(message.format(*args))

    async def _fail_default_team(*_args, **_kwargs):
        raise RuntimeError("secret backend detail at /private/authnz.db")

    conn = _SqliteConnWithPgTrap()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))
    logger_stub = _Logger()
    monkeypatch.setattr(repo, "_get_or_create_default_team_id", _fail_default_team)
    monkeypatch.setattr(repo_module, "logger", logger_stub)

    assert await repo._create_default_team_best_effort(conn, 11) is None
    rendered = " ".join(logger_stub.messages)
    assert "Default team auto-enroll failed" in rendered
    assert "secret backend detail" not in rendered
    assert "/private/" not in rendered


@pytest.mark.asyncio
async def test_remove_team_member_log_does_not_render_backend_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Logger:
        def __init__(self) -> None:
            self.bindings: list[dict[str, Any]] = []
            self.messages: list[str] = []

        def bind(self, **kwargs: Any) -> _Logger:
            self.bindings.append(kwargs)
            return self

        def error(self, message: str, *args: Any) -> None:
            self.messages.append(message.format(*args))

    async def _fail_remove(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("secret SQL detail at /private/authnz.db")

    conn = _SqliteConnWithPgTrap()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))
    logger_stub = _Logger()
    monkeypatch.setattr(repo, "remove_team_member_on_connection", _fail_remove)
    monkeypatch.setattr(repo_module, "logger", logger_stub)

    with pytest.raises(RuntimeError):
        await repo.remove_team_member(
            team_id=11,
            user_id=7,
            context=_ACTOR_MEMBERSHIP_CONTEXT,
        )

    rendered = " ".join(logger_stub.messages)
    assert "AuthnzOrgsTeamsRepo.remove_team_member failed" in rendered
    assert logger_stub.bindings == [{"error_type": "RuntimeError"}]
    assert "secret SQL detail" not in rendered
    assert "/private/" not in rendered


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
@pytest.mark.parametrize(
    "postgres",
    (False, True),
)
async def test_membership_reads_use_supplied_connection_without_secondary_acquisition(
    postgres: bool,
):
    conn = _PostgresMembershipReadConn() if postgres else _SqliteMembershipReadConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_NoSecondaryAcquirePool(postgres=postgres))

    team_rows = await repo.list_memberships_for_user(user_id=7, conn=conn)
    org_rows = await repo.list_org_memberships_for_user(user_id=7, conn=conn)

    assert team_rows[0]["org_id"] == 11
    assert org_rows == [{"org_id": 11, "role": "admin", "status": "active"}]
    if postgres:
        queries = "\n".join(query for query, _params in conn.fetch_calls)
        assert "FROM public.team_members" in queries
        assert "JOIN public.teams" in queries
        assert "JOIN public.organizations" in queries
        assert "FROM public.org_members" in queries
    else:
        queries = "\n".join(query for query, _params in conn.execute_calls)
        assert "FROM main.team_members" in queries
        assert "JOIN main.teams" in queries
        assert "JOIN main.organizations" in queries
        assert "FROM main.org_members" in queries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ("list_memberships_for_user", "list_org_memberships_for_user"),
)
async def test_membership_read_failures_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
):
    secret = "password=private database=/private/authnz.db"
    log_events: list[tuple[dict[str, object], str]] = []

    class _BoundLogger:
        def __init__(self, fields: dict[str, object] | None = None) -> None:
            self._fields = fields or {}

        def bind(self, **fields: object):
            return _BoundLogger(fields)

        def error(self, message: str) -> None:
            log_events.append((self._fields, message))

    monkeypatch.setattr(repo_module, "logger", _BoundLogger())
    repo = AuthnzOrgsTeamsRepo(db_pool=_FailingMembershipReadPool(secret))

    with pytest.raises(UserRegistrationException) as exc_info:
        await getattr(repo, operation)(user_id=7)

    assert type(exc_info.value).__name__ == "MembershipReadError"
    assert str(exc_info.value) == "Membership state could not be read."
    assert exc_info.value.__suppress_context__ is True
    assert log_events == [
        (
            {"operation": operation, "exception_type": "RuntimeError"},
            "AuthNZ membership read failed",
        )
    ]
    assert secret not in repr(log_events)


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
async def test_transfer_organization_ownership_sqlite_backend_selection_uses_execute(
    monkeypatch: pytest.MonkeyPatch,
):
    conn = _SqliteTransferOwnershipConn()
    pool = _PoolStub(conn, postgres=False)
    repo = AuthnzOrgsTeamsRepo(db_pool=pool)
    observed: list[tuple[Any, MembershipLockBackend]] = []

    async def _transfer(writer, **kwargs):
        observed.append((kwargs["conn"], writer._backend))  # noqa: SLF001

    monkeypatch.setattr(MembershipWriter, "transfer_organization_ownership", _transfer)

    row = await repo.transfer_organization_ownership(
        org_id=3,
        new_owner_user_id=22,
        current_owner_user_id=11,
        context=_ACTOR_MEMBERSHIP_CONTEXT,
    )

    assert row and row["owner_user_id"] == 22
    assert observed == [(conn, MembershipLockBackend.SQLITE)]
    assert pool.transaction_acquire_timeouts == [5.0]


@pytest.mark.asyncio
async def test_transfer_organization_ownership_postgres_backend_selection_uses_fetchrow(
    monkeypatch: pytest.MonkeyPatch,
):
    conn = _PostgresTransferOwnershipConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=True))
    observed: list[tuple[Any, MembershipLockBackend]] = []

    async def _transfer(writer, **kwargs):
        observed.append((kwargs["conn"], writer._backend))  # noqa: SLF001

    monkeypatch.setattr(MembershipWriter, "transfer_organization_ownership", _transfer)

    row = await repo.transfer_organization_ownership(
        org_id=3,
        new_owner_user_id=22,
        current_owner_user_id=11,
        context=_ACTOR_MEMBERSHIP_CONTEXT,
    )

    assert row and row["owner_user_id"] == 22
    assert conn.fetchrow_calls
    assert "where id = $1" in conn.fetchrow_calls[0][0].lower()
    assert observed == [(conn, MembershipLockBackend.POSTGRESQL)]


@pytest.mark.parametrize(
    "reason",
    (TrustedMembershipReason.REGISTRATION, TrustedMembershipReason.BOOTSTRAP),
)
@pytest.mark.asyncio
async def test_transfer_organization_ownership_repo_rejects_trusted_context_before_transaction(
    reason: TrustedMembershipReason,
) -> None:
    class _NoTransactionPool:
        pool = None

        def transaction(self):
            raise AssertionError("trusted ownership transfer opened a transaction")

    assert (
        get_type_hints(AuthnzOrgsTeamsRepo.transfer_organization_ownership)["context"]
        is ActorMembershipWriteContext
    )
    with pytest.raises(MembershipWriterContractError):
        await AuthnzOrgsTeamsRepo(_NoTransactionPool()).transfer_organization_ownership(
            org_id=3,
            new_owner_user_id=22,
            current_owner_user_id=11,
            context=TrustedMembershipWriteContext(trusted_reason=reason),
        )


@pytest.mark.asyncio
async def test_delete_organization_with_provider_secrets_sqlite_backend_selection_uses_execute(
    monkeypatch: pytest.MonkeyPatch,
):
    conn = _SqliteDeleteOrgConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))
    snapshot = MembershipScopeDeletionSnapshot(
        scope_type=MembershipScopeType.ORGANIZATION,
        scope_id=5,
        organization_ids=(5,),
        team_parents=(),
        membership_rows=(),
    )
    observed: list[dict[str, Any]] = []

    async def _discover(_writer, **kwargs):
        assert kwargs["conn"] is conn
        return snapshot

    async def _apply(_writer, **kwargs):
        observed.append(kwargs)

    monkeypatch.setattr(MembershipWriter, "discover_scope_deletion", _discover)
    monkeypatch.setattr(MembershipWriter, "apply_scope_deletion", _apply)

    await repo.delete_organization_with_provider_secrets(
        org_id=5,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )

    assert conn.execute_calls
    statements = [" ".join(query.lower().split()) for query, _ in conn.execute_calls]
    assert "delete from main.org_provider_secrets" in statements[0]
    assert "scope_type = 'org' and scope_id = ?" in statements[0]
    assert "delete from main.org_provider_secrets" in statements[1]
    assert "select id from main.teams where org_id = ?" in statements[1]
    assert statements[-1].startswith("delete from main.organizations where id = ?")
    assert observed[0]["conn"] is conn
    assert observed[0]["context"] is _BOOTSTRAP_MEMBERSHIP_CONTEXT
    assert observed[0]["anchor_ownership"] is AnchorOwnership.WRITER_OWNS_ANCHOR


@pytest.mark.asyncio
async def test_delete_organization_with_provider_secrets_postgres_uses_public_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _PostgresDeleteOrgConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=True))
    snapshot = MembershipScopeDeletionSnapshot(
        scope_type=MembershipScopeType.ORGANIZATION,
        scope_id=5,
        organization_ids=(5,),
        team_parents=(TeamParentOrganization(team_id=9, organization_id=5),),
        membership_rows=(),
    )

    async def _discover(_writer, **_kwargs):
        return snapshot

    async def _apply(_writer, **_kwargs):
        return None

    monkeypatch.setattr(MembershipWriter, "discover_scope_deletion", _discover)
    monkeypatch.setattr(MembershipWriter, "apply_scope_deletion", _apply)

    await repo.delete_organization_with_provider_secrets(
        org_id=5,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )

    statements = [" ".join(query.lower().split()) for query, _ in conn.execute_calls]
    assert "delete from public.org_provider_secrets" in statements[0]
    assert "scope_type = 'org' and scope_id = $1" in statements[0]
    assert "delete from public.org_provider_secrets" in statements[1]
    assert "select id from public.teams where org_id = $1" in statements[1]
    assert statements[-1].startswith(
        "delete from public.organizations where id = $1"
    )


@pytest.mark.asyncio
async def test_delete_team_with_provider_secrets_postgres_backend_selection_uses_execute(
    monkeypatch: pytest.MonkeyPatch,
):
    conn = _PostgresDeleteTeamConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=True))
    snapshot = MembershipScopeDeletionSnapshot(
        scope_type=MembershipScopeType.TEAM,
        scope_id=9,
        organization_ids=(3,),
        team_parents=(TeamParentOrganization(team_id=9, organization_id=3),),
        membership_rows=(),
    )
    observed: list[dict[str, Any]] = []

    async def _discover(_writer, **kwargs):
        assert kwargs["conn"] is conn
        return snapshot

    async def _apply(_writer, **kwargs):
        observed.append(kwargs)

    monkeypatch.setattr(MembershipWriter, "discover_scope_deletion", _discover)
    monkeypatch.setattr(MembershipWriter, "apply_scope_deletion", _apply)

    await repo.delete_team_with_provider_secrets(
        team_id=9,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )

    assert conn.execute_calls
    statements = [" ".join(query.lower().split()) for query, _ in conn.execute_calls]
    assert statements[0].startswith("delete from public.org_provider_secrets")
    assert "scope_type = 'team' and scope_id = $1" in statements[0]
    assert statements[-1].startswith("delete from public.teams where id = $1")
    assert observed[0]["conn"] is conn
    assert observed[0]["context"] is _BOOTSTRAP_MEMBERSHIP_CONTEXT
    assert observed[0]["anchor_ownership"] is AnchorOwnership.WRITER_OWNS_ANCHOR


@pytest.mark.parametrize(
    ("method_name", "scope_argument"),
    (
        ("delete_organization_with_provider_secrets", "org_id"),
        ("delete_team_with_provider_secrets", "team_id"),
    ),
)
@pytest.mark.asyncio
async def test_missing_scope_delete_is_idempotent_without_parent_delete_race(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    scope_argument: str,
) -> None:
    conn = _SqliteDeleteOrgConn()
    repo = AuthnzOrgsTeamsRepo(db_pool=_PoolStub(conn, postgres=False))

    async def _discover(_writer, **_kwargs):
        return None

    monkeypatch.setattr(MembershipWriter, "discover_scope_deletion", _discover)

    await getattr(repo, method_name)(
        **{
            scope_argument: 5,
            "context": _BOOTSTRAP_MEMBERSHIP_CONTEXT,
        }
    )

    assert conn.execute_calls == []


@pytest.mark.parametrize(
    ("method_name", "scope_argument"),
    (
        ("delete_organization_with_provider_secrets", "org_id"),
        ("delete_team_with_provider_secrets", "team_id"),
    ),
)
@pytest.mark.asyncio
async def test_scope_delete_retries_bound_every_pool_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    scope_argument: str,
) -> None:
    conn = _SqliteDeleteOrgConn()
    pool = _PoolStub(conn, postgres=False)
    repo = AuthnzOrgsTeamsRepo(db_pool=pool)
    snapshot = MembershipScopeDeletionSnapshot(
        scope_type=(
            MembershipScopeType.ORGANIZATION
            if scope_argument == "org_id"
            else MembershipScopeType.TEAM
        ),
        scope_id=5,
        organization_ids=(5,),
        team_parents=(
            ()
            if scope_argument == "org_id"
            else (TeamParentOrganization(team_id=5, organization_id=5),)
        ),
        membership_rows=(),
    )

    async def _discover(_writer, **_kwargs):
        return snapshot

    async def _retry(_writer, **_kwargs):
        raise RollbackSignal()

    monkeypatch.setattr(MembershipWriter, "discover_scope_deletion", _discover)
    monkeypatch.setattr(MembershipWriter, "apply_scope_deletion", _retry)
    monkeypatch.setattr(
        MembershipWriter,
        "is_scope_deletion_retry",
        lambda _writer, _exc: True,
    )

    with pytest.raises(MembershipPreflightChanged):
        await getattr(repo, method_name)(
            **{
                scope_argument: 5,
                "context": _BOOTSTRAP_MEMBERSHIP_CONTEXT,
            }
        )

    assert pool.acquire_timeouts == [5.0, 5.0, 5.0]
    assert pool.transaction_acquire_timeouts == [5.0, 5.0, 5.0]
