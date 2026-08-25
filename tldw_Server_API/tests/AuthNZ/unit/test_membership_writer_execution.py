from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import get_type_hints

import pytest

import tldw_Server_API.app.core.AuthNZ.membership_writer as membership_writer_module
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    AnchorOwnership,
    MembershipAuthority,
    MembershipAuthorizationError,
    MembershipMutation,
    MembershipMutationKind,
    MembershipMutationResult,
    MembershipRowSnapshot,
    MembershipScopeDeletionSnapshot,
    MembershipScopeType,
    MembershipUserVersionFloor,
    MembershipWriter,
    MembershipWriterContractError,
    MembershipWriteResult,
    OfflineMigrationContextRejected,
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    add_org_member,
    add_team_member,
    remove_org_member,
    remove_team_member,
    update_org_member_role,
    update_team_member_role,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _mutation(
    kind: MembershipMutationKind = MembershipMutationKind.ADD,
) -> MembershipMutation:
    return MembershipMutation(
        scope_type=MembershipScopeType.ORGANIZATION,
        scope_id=3,
        user_id=7,
        kind=kind,
        role="member" if kind is not MembershipMutationKind.REMOVE else None,
    )


def test_execution_result_contracts_are_immutable_and_validated() -> None:
    mutation = _mutation()
    mutation_result = MembershipMutationResult(
        mutation=mutation,
        changed=True,
        found=True,
        role="member",
    )
    floor = MembershipUserVersionFloor(
        user_id=7,
        pre_mutation_floor=BASE,
        post_mutation_floor=BASE + timedelta(seconds=1),
    )
    result = MembershipWriteResult(
        mutation_results=(mutation_result,),
        affected_user_ids=(7,),
        version_floors=(floor,),
    )

    assert result.floor_for(7) == BASE + timedelta(seconds=1)
    assert mutation_result.to_legacy_result() == {
        "org_id": 3,
        "user_id": 7,
        "role": "member",
    }
    with pytest.raises(FrozenInstanceError):
        floor.user_id = 8
    with pytest.raises(ValueError, match="Invalid membership writer contract"):
        MembershipWriteResult(
            mutation_results=(mutation_result,),
            affected_user_ids=(7, 7),
            version_floors=(floor,),
        )


@pytest.mark.parametrize(
    (
        "scope_type",
        "kind",
        "changed",
        "found",
        "result_role",
        "organization_id",
        "error",
        "expected",
    ),
    (
        (
            MembershipScopeType.ORGANIZATION,
            MembershipMutationKind.ADD,
            True,
            True,
            "member",
            None,
            None,
            {"org_id": 3, "user_id": 7, "role": "member"},
        ),
        (
            MembershipScopeType.ORGANIZATION,
            MembershipMutationKind.ADD,
            False,
            True,
            "member",
            None,
            None,
            {"org_id": 3, "user_id": 7, "role": "member"},
        ),
        (
            MembershipScopeType.ORGANIZATION,
            MembershipMutationKind.UPDATE_ROLE,
            True,
            True,
            "admin",
            None,
            None,
            {"org_id": 3, "user_id": 7, "role": "admin"},
        ),
        (
            MembershipScopeType.ORGANIZATION,
            MembershipMutationKind.UPDATE_ROLE,
            False,
            False,
            None,
            None,
            None,
            None,
        ),
        (
            MembershipScopeType.ORGANIZATION,
            MembershipMutationKind.UPDATE_ROLE,
            False,
            True,
            "owner",
            None,
            "owner_required",
            {
                "org_id": 3,
                "user_id": 7,
                "role": "owner",
                "error": "owner_required",
            },
        ),
        (
            MembershipScopeType.ORGANIZATION,
            MembershipMutationKind.REMOVE,
            True,
            True,
            None,
            None,
            None,
            {"org_id": 3, "user_id": 7, "removed": True},
        ),
        (
            MembershipScopeType.ORGANIZATION,
            MembershipMutationKind.REMOVE,
            False,
            False,
            None,
            None,
            None,
            {"org_id": 3, "user_id": 7, "removed": False},
        ),
        (
            MembershipScopeType.ORGANIZATION,
            MembershipMutationKind.REMOVE,
            False,
            True,
            "owner",
            None,
            "owner_required",
            {
                "org_id": 3,
                "user_id": 7,
                "removed": False,
                "error": "owner_required",
            },
        ),
        (
            MembershipScopeType.TEAM,
            MembershipMutationKind.ADD,
            True,
            True,
            "member",
            5,
            None,
            {"team_id": 3, "user_id": 7, "role": "member", "org_id": 5},
        ),
        (
            MembershipScopeType.TEAM,
            MembershipMutationKind.ADD,
            False,
            True,
            "member",
            5,
            None,
            {"team_id": 3, "user_id": 7, "role": "member", "org_id": 5},
        ),
        (
            MembershipScopeType.TEAM,
            MembershipMutationKind.UPDATE_ROLE,
            True,
            True,
            "admin",
            5,
            None,
            {"team_id": 3, "user_id": 7, "role": "admin"},
        ),
        (
            MembershipScopeType.TEAM,
            MembershipMutationKind.UPDATE_ROLE,
            False,
            False,
            None,
            5,
            None,
            None,
        ),
        (
            MembershipScopeType.TEAM,
            MembershipMutationKind.REMOVE,
            True,
            True,
            None,
            5,
            None,
            {"team_id": 3, "user_id": 7, "removed": True},
        ),
        (
            MembershipScopeType.TEAM,
            MembershipMutationKind.REMOVE,
            False,
            False,
            None,
            5,
            None,
            {"team_id": 3, "user_id": 7, "removed": False},
        ),
    ),
)
def test_mutation_results_preserve_exact_legacy_operation_shapes(
    scope_type: MembershipScopeType,
    kind: MembershipMutationKind,
    changed: bool,
    found: bool,
    result_role: str | None,
    organization_id: int | None,
    error: str | None,
    expected: dict[str, object] | None,
) -> None:
    mutation = MembershipMutation(
        scope_type=scope_type,
        scope_id=3,
        user_id=7,
        kind=kind,
        role="member" if kind is MembershipMutationKind.ADD else (
            "admin" if kind is MembershipMutationKind.UPDATE_ROLE else None
        ),
    )
    result = MembershipMutationResult(
        mutation=mutation,
        changed=changed,
        found=found,
        role=result_role,
        organization_id=organization_id,
        error=error,
    )

    assert result.to_legacy_result() == expected


@pytest.mark.parametrize(
    ("mutation", "changed", "found", "role", "organization_id", "error"),
    (
        (_mutation(), True, True, "member", 3, None),
        (
            MembershipMutation(
                scope_type=MembershipScopeType.TEAM,
                scope_id=4,
                user_id=7,
                kind=MembershipMutationKind.ADD,
                role="member",
            ),
            True,
            True,
            "member",
            None,
            None,
        ),
        (_mutation(), False, False, "member", None, None),
        (_mutation(), False, True, "member", None, "owner_required"),
        (
            MembershipMutation(
                scope_type=MembershipScopeType.TEAM,
                scope_id=4,
                user_id=7,
                kind=MembershipMutationKind.REMOVE,
            ),
            False,
            True,
            None,
            3,
            "owner_required",
        ),
    ),
)
def test_mutation_result_rejects_impossible_cross_field_states(
    mutation: MembershipMutation,
    changed: bool,
    found: bool,
    role: str | None,
    organization_id: int | None,
    error: str | None,
) -> None:
    with pytest.raises(ValueError, match="Invalid membership writer contract"):
        MembershipMutationResult(
            mutation=mutation,
            changed=changed,
            found=found,
            role=role,
            organization_id=organization_id,
            error=error,
        )


@pytest.mark.parametrize(
    ("changed", "affected_user_ids", "version_floors"),
    (
        (True, (), ()),
        (
            False,
            (7,),
            (
                MembershipUserVersionFloor(
                    user_id=7,
                    pre_mutation_floor=BASE,
                    post_mutation_floor=BASE,
                ),
            ),
        ),
    ),
)
def test_write_result_affected_users_match_changed_mutation_results(
    changed: bool,
    affected_user_ids: tuple[int, ...],
    version_floors: tuple[MembershipUserVersionFloor, ...],
) -> None:
    mutation_result = MembershipMutationResult(
        mutation=_mutation(),
        changed=changed,
        found=True,
        role="member",
    )

    with pytest.raises(ValueError, match="Invalid membership writer contract"):
        MembershipWriteResult(
            mutation_results=(mutation_result,),
            affected_user_ids=affected_user_ids,
            version_floors=version_floors,
        )


def test_direct_membership_wrappers_require_explicit_context() -> None:
    wrappers = (
        add_org_member,
        remove_org_member,
        update_org_member_role,
        add_team_member,
        remove_team_member,
        update_team_member_role,
    )

    for wrapper in wrappers:
        parameter = inspect.signature(wrapper).parameters["context"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


@pytest.mark.asyncio
async def test_writer_rejects_offline_context_before_connection_access() -> None:
    class _Connection:
        def __getattr__(self, name: str):
            raise AssertionError(f"unexpected connection access: {name}")

    class _Pool:
        pool = None

    writer = MembershipWriter(_Pool())
    with pytest.raises(OfflineMigrationContextRejected):
        await writer.apply_membership_mutations(
            conn=_Connection(),
            context=TrustedMembershipWriteContext(
                trusted_reason=TrustedMembershipReason.OFFLINE_MIGRATION,
            ),
            mutations=(_mutation(),),
            anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
            operation_time=BASE,
        )


@pytest.mark.parametrize(
    "reason",
    (TrustedMembershipReason.REGISTRATION, TrustedMembershipReason.BOOTSTRAP),
)
@pytest.mark.asyncio
async def test_ownership_transfer_rejects_trusted_context_before_connection_access(
    reason: TrustedMembershipReason,
) -> None:
    class _Connection:
        def __getattr__(self, name: str):
            raise AssertionError(f"unexpected connection access: {name}")

    class _Pool:
        pool = None

    assert (
        get_type_hints(MembershipWriter.transfer_organization_ownership)["context"]
        is ActorMembershipWriteContext
    )
    with pytest.raises(MembershipWriterContractError):
        await MembershipWriter(_Pool()).transfer_organization_ownership(
            conn=_Connection(),
            context=TrustedMembershipWriteContext(trusted_reason=reason),
            organization_id=3,
            current_owner_user_id=11,
            new_owner_user_id=22,
            anchor_ownership=AnchorOwnership.WRITER_OWNS_ANCHOR,
            operation_time=BASE,
        )


@pytest.mark.asyncio
async def test_scoped_actor_authority_is_reread_after_lock_planning() -> None:
    class _Pool:
        pool = object()

    class _Connection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        async def fetch(self, sql, *args):
            self.statements.append(str(sql))
            if "organization_owners" in str(sql):
                return []
            raise AssertionError(str(sql))

        async def fetchrow(self, sql, *args):
            statement = str(sql)
            self.statements.append(statement)
            if "FROM public.organizations" in statement and "FOR UPDATE" not in statement:
                return {"id": 3, "is_active": True, "owner_user_id": 99}
            if "FROM public.users" in statement and "FOR UPDATE" not in statement:
                return {
                    "id": 11,
                    "is_active": True,
                    "is_superuser": False,
                    "role": "user",
                }
            if "FROM public.org_members" in statement and "FOR UPDATE" not in statement:
                return None
            if "FOR UPDATE" in statement:
                return None
            raise AssertionError(statement)

        async def fetchval(self, sql, *args):
            raise AssertionError(str(sql))

        async def execute(self, sql, *args):
            self.statements.append(str(sql))
            return "SELECT 0"

    conn = _Connection()
    writer = MembershipWriter(_Pool())
    with pytest.raises(MembershipAuthorizationError):
        await writer.apply_membership_mutations(
            conn=conn,
            context=ActorMembershipWriteContext(
                actor_user_id=11,
                required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
            ),
            mutations=(_mutation(),),
            anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
            operation_time=BASE,
        )

    lock_index = next(
        index
        for index, statement in enumerate(conn.statements)
        if "public.organizations" in statement and "FOR UPDATE" in statement
    )
    authority_index = max(
        index
        for index, statement in enumerate(conn.statements)
        if "public.org_members" in statement and "FOR UPDATE" not in statement
    )
    assert authority_index > lock_index


class _RowsCursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows


def _postgres_rbac_rows(
    statement: str,
    *,
    role_rows,
    role_permission_rows,
    direct_rows,
):
    role_names = [str(row["name"]) for row in role_rows]
    if role_permission_rows and not role_names:
        role_names.append("member")
    permission_names = list(
        dict.fromkeys(
            str(row["name"])
            for row in (*role_permission_rows, *direct_rows)
        )
    )
    permission_ids = {
        permission_name: index
        for index, permission_name in enumerate(permission_names, start=1)
    }
    normalized = " ".join(statement.split())
    if normalized.startswith("SELECT ur.role_id"):
        return tuple(
            {"role_id": index, "active": True}
            for index, _ in enumerate(role_names, start=1)
        )
    if normalized.startswith("SELECT r.id"):
        return tuple(
            {"id": index, "name": role_name}
            for index, role_name in enumerate(role_names, start=1)
        )
    if normalized.startswith("SELECT rp.role_id"):
        return tuple(
            {
                "role_id": 1,
                "permission_id": permission_ids[str(row["name"])],
            }
            for row in role_permission_rows
        )
    if normalized.startswith("SELECT p.id"):
        return tuple(
            {"id": permission_id, "name": permission_name}
            for permission_name, permission_id in permission_ids.items()
        )
    if normalized.startswith("SELECT up.permission_id"):
        return tuple(
            {
                "permission_id": permission_ids[str(row["name"])],
                "granted": bool(row["granted"]),
                "active": True,
            }
            for row in direct_rows
        )
    return None


class _RbacConnection:
    def __init__(self, *, role_rows=(), role_permission_rows=(), direct_rows=()):
        self.role_rows = role_rows
        self.role_permission_rows = role_permission_rows
        self.direct_rows = direct_rows
        self.statements: list[str] = []

    def _rows(self, sql: str):
        self.statements.append(sql)
        if "public." in sql:
            rows = _postgres_rbac_rows(
                sql,
                role_rows=self.role_rows,
                role_permission_rows=self.role_permission_rows,
                direct_rows=self.direct_rows,
            )
            if rows is not None:
                return rows
        if "JOIN public.roles" in sql or "JOIN main.roles" in sql:
            return self.role_rows
        if "JOIN public.role_permissions" in sql or "JOIN main.role_permissions" in sql:
            return self.role_permission_rows
        if "JOIN public.user_permissions" in sql or "JOIN main.user_permissions" in sql:
            return self.direct_rows
        raise AssertionError(sql)

    async def fetch(self, sql, *args):
        return self._rows(str(sql))

    async def execute(self, sql, parameters):
        return _RowsCursor(self._rows(str(sql)))


class _WriterPathCursor:
    def __init__(self, *, row=None, rows=()):
        self._row = row
        self._rows = rows

    async def fetchone(self):
        return self._row

    async def fetchall(self):
        return self._rows


class _PlatformAdminWriterConnection:
    def __init__(
        self,
        *,
        postgres: bool,
        role_rows=(),
        role_permission_rows=(),
        direct_rows=(),
    ) -> None:
        self.postgres = postgres
        self.role_rows = role_rows
        self.role_permission_rows = role_permission_rows
        self.direct_rows = direct_rows
        self.statements: list[str] = []

    def _rbac_rows(self, statement: str):
        if "public." in statement:
            rows = _postgres_rbac_rows(
                statement,
                role_rows=self.role_rows,
                role_permission_rows=self.role_permission_rows,
                direct_rows=self.direct_rows,
            )
            if rows is not None:
                return rows
        if "JOIN public.roles" in statement or "JOIN main.roles" in statement:
            return self.role_rows
        if (
            "JOIN public.role_permissions" in statement
            or "JOIN main.role_permissions" in statement
        ):
            return self.role_permission_rows
        if (
            "JOIN public.user_permissions" in statement
            or "JOIN main.user_permissions" in statement
        ):
            return self.direct_rows
        return None

    def _select_row(self, statement: str, parameters: tuple[object, ...]):
        if "FROM public.organizations" in statement or "FROM main.organizations" in statement:
            return {"id": 3, "owner_user_id": 99, "is_active": True}
        if "FROM public.users" in statement or "FROM main.users" in statement:
            return {
                "id": int(parameters[0]),
                "is_active": True,
                "is_superuser": False,
                "role": "user",
            }
        if "FROM public.org_members" in statement or "FROM main.org_members" in statement:
            return {"role": "member", "status": "active"}
        raise AssertionError(statement)

    async def fetchrow(self, sql, *parameters):
        statement = str(sql)
        self.statements.append(statement)
        if "FOR UPDATE" in statement:
            return None
        return self._select_row(statement, parameters)

    async def fetch(self, sql, *parameters):
        statement = str(sql)
        self.statements.append(statement)
        rows = self._rbac_rows(statement)
        if rows is None:
            raise AssertionError(statement)
        return rows

    async def execute(self, sql, parameters):
        statement = str(sql)
        self.statements.append(statement)
        rows = self._rbac_rows(statement)
        if rows is not None:
            return _WriterPathCursor(rows=rows)
        return _WriterPathCursor(row=self._select_row(statement, tuple(parameters)))


class _StableFloorGateway:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    async def capture_floor(self, conn, *, user_id, lock_user):
        del conn, user_id, lock_user
        return BASE


class _TeamAuthorityConnection:
    def __init__(
        self,
        *,
        team_role: str,
        team_status: str = "active",
        target_org_statuses: dict[int, str | None] | None = None,
    ) -> None:
        self.team_role = team_role
        self.team_status = team_status
        self.target_org_statuses = target_org_statuses or {9: "active"}
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    def _row(self, statement: str, parameters: tuple[object, ...]):
        if "FROM main.teams" in statement:
            return (20, 5, 1)
        if "FROM main.organizations" in statement:
            return (5, 99, 1)
        if "FROM main.users" in statement:
            return (int(parameters[0]), 1, 0, "user")
        if "FROM main.org_members" in statement:
            if tuple(parameters) == (5, 4):
                return ("member", "active")
            if int(parameters[0]) == 5:
                status = self.target_org_statuses.get(int(parameters[1]))
                return None if status is None else ("member", status)
            return None
        if "FROM main.team_members" in statement:
            if tuple(parameters) == (20, 4):
                return (self.team_role, self.team_status)
            return None
        raise AssertionError(statement)

    async def execute(self, sql, parameters=()):
        statement = str(sql)
        normalized = tuple(parameters)
        self.statements.append((statement, normalized))
        if statement.lstrip().upper().startswith("SELECT"):
            return _WriterPathCursor(row=self._row(statement, normalized))
        return _WriterPathCursor()


@pytest.mark.parametrize("team_role", ["owner", "admin", "lead"])
@pytest.mark.asyncio
async def test_writer_path_accepts_active_team_authority(
    monkeypatch,
    team_role,
) -> None:
    monkeypatch.setattr(
        membership_writer_module,
        "VersionedUserWriteGateway",
        _StableFloorGateway,
    )
    conn = _TeamAuthorityConnection(team_role=team_role)

    result = await MembershipWriter(type("Pool", (), {"pool": None})()).apply_membership_mutations(
        conn=conn,
        context=ActorMembershipWriteContext(
            actor_user_id=4,
            required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
        ),
        mutations=(
            MembershipMutation(
                scope_type=MembershipScopeType.TEAM,
                scope_id=20,
                user_id=9,
                kind=MembershipMutationKind.ADD,
                role="member",
            ),
        ),
        anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
        operation_time=BASE,
    )

    assert result.mutation_results[0].changed is True


@pytest.mark.asyncio
async def test_team_lead_does_not_authorize_organization_mutation(monkeypatch) -> None:
    monkeypatch.setattr(
        membership_writer_module,
        "VersionedUserWriteGateway",
        _StableFloorGateway,
    )
    conn = _TeamAuthorityConnection(team_role="lead")

    with pytest.raises(MembershipAuthorizationError):
        await MembershipWriter(type("Pool", (), {"pool": None})()).apply_membership_mutations(
            conn=conn,
            context=ActorMembershipWriteContext(
                actor_user_id=4,
                required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
            ),
            mutations=(_mutation(),),
            anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
            operation_time=BASE,
        )


@pytest.mark.asyncio
async def test_team_add_parent_precondition_is_per_mutation_and_never_inserts_orphan(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        membership_writer_module,
        "VersionedUserWriteGateway",
        _StableFloorGateway,
    )
    conn = _TeamAuthorityConnection(
        team_role="lead",
        target_org_statuses={9: None, 10: "active"},
    )
    mutations = tuple(
        MembershipMutation(
            scope_type=MembershipScopeType.TEAM,
            scope_id=20,
            user_id=user_id,
            kind=MembershipMutationKind.ADD,
            role="member",
        )
        for user_id in (9, 10)
    )

    result = await MembershipWriter(type("Pool", (), {"pool": None})()).apply_membership_mutations(
        conn=conn,
        context=ActorMembershipWriteContext(
            actor_user_id=4,
            required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
        ),
        mutations=mutations,
        anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
        operation_time=BASE,
    )

    assert tuple(item.mutation.user_id for item in result.mutation_results) == (9, 10)
    assert result.mutation_results[0].error == "org_membership_required"
    assert result.mutation_results[0].changed is False
    assert result.mutation_results[0].found is False
    assert result.mutation_results[1].changed is True
    assert result.affected_user_ids == (10,)
    inserted_user_ids = [
        int(parameters[1])
        for statement, parameters in conn.statements
        if "INSERT INTO main.team_members" in statement
    ]
    assert inserted_user_ids == [10]


@pytest.mark.asyncio
async def test_blocked_org_removal_does_not_skip_unmarked_team_removal(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        membership_writer_module,
        "VersionedUserWriteGateway",
        _StableFloorGateway,
    )

    class _OrderedRemovalConnection(_TeamAuthorityConnection):
        def _row(self, statement: str, parameters: tuple[object, ...]):
            if "FROM main.org_members" in statement:
                if parameters == (5, 4):
                    return ("admin", "active")
                if parameters == (5, 9):
                    return ("owner", "active")
            if "FROM main.team_members" in statement:
                if parameters == (20, 4):
                    return ("lead", "active")
                if parameters == (20, 9):
                    return ("member", "active")
            return super()._row(statement, parameters)

        async def execute(self, sql, parameters=()):
            statement = str(sql)
            normalized = tuple(parameters)
            self.statements.append((statement, normalized))
            if "SELECT user_id FROM main.org_members" in statement:
                return _WriterPathCursor(rows=((9,),))
            if statement.lstrip().upper().startswith("SELECT"):
                return _WriterPathCursor(row=self._row(statement, normalized))
            return _WriterPathCursor()

    conn = _OrderedRemovalConnection(team_role="lead")
    mutations = (
        MembershipMutation(
            scope_type=MembershipScopeType.ORGANIZATION,
            scope_id=5,
            user_id=9,
            kind=MembershipMutationKind.REMOVE,
        ),
        MembershipMutation(
            scope_type=MembershipScopeType.TEAM,
            scope_id=20,
            user_id=9,
            kind=MembershipMutationKind.REMOVE,
        ),
    )

    result = await MembershipWriter(
        type("Pool", (), {"pool": None})()
    ).apply_membership_mutations(
        conn=conn,
        context=ActorMembershipWriteContext(
            actor_user_id=4,
            required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
        ),
        mutations=mutations,
        anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
        operation_time=BASE,
    )

    assert result.mutation_results[0].error == "owner_required"
    assert result.mutation_results[0].changed is False
    assert result.mutation_results[1].changed is True
    assert [
        parameters
        for statement, parameters in conn.statements
        if "DELETE FROM main.team_members" in statement
    ] == [(20, 9)]


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_owner_queries_treat_only_explicit_active_status_as_active(postgres) -> None:
    class _OwnerConnection:
        def __init__(self) -> None:
            self.statement = ""

        async def fetch(self, sql, *parameters):
            del parameters
            self.statement = str(sql)
            return ({"user_id": 7},)

        async def execute(self, sql, parameters):
            del parameters
            self.statement = str(sql)
            return _WriterPathCursor(rows=((7,),))

    conn = _OwnerConnection()
    pool = type("Pool", (), {"pool": object() if postgres else None})()

    assert await MembershipWriter(pool)._read_owner_user_ids(conn, 3) == (7,)
    assert "COALESCE(status, 'active')" not in conn.statement
    assert "LOWER(COALESCE(status, '')) = 'active'" in conn.statement


@pytest.mark.asyncio
async def test_postgres_persisted_authority_read_does_not_acquire_unplanned_locks() -> None:
    class _LockedRbacConnection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        async def fetch(self, sql, *parameters):
            del parameters
            statement = " ".join(str(sql).split())
            self.statements.append(statement)
            if "FROM public.roles r" in statement:
                return ({"id": 2, "name": "member"},)
            if "FROM public.permissions p" in statement:
                return ({"id": 9, "name": "system.configure"},)
            if "FROM public.user_permissions up" in statement:
                return ()
            if "FROM public.role_permissions rp" in statement:
                return ({"role_id": 2, "permission_id": 9},)
            if "FROM public.user_roles ur" in statement:
                return ({"role_id": 2, "active": True},)
            raise AssertionError(statement)

    conn = _LockedRbacConnection()

    assert await MembershipWriter(type("Pool", (), {"pool": object()})())._has_persisted_platform_admin(
        conn,
        11,
    )
    def _primary_table(statement: str) -> str:
        query_prefixes = {
            "SELECT ur.role_id": "user_roles",
            "SELECT r.id": "roles",
            "SELECT rp.role_id": "role_permissions",
            "SELECT p.id": "permissions",
            "SELECT up.permission_id": "user_permissions",
        }
        return next(
            table
            for prefix, table in query_prefixes.items()
            if statement.startswith(prefix)
        )

    assert [_primary_table(statement) for statement in conn.statements] == [
        "user_roles",
        "roles",
        "role_permissions",
        "permissions",
        "user_permissions",
    ]
    assert all("FOR UPDATE" not in statement for statement in conn.statements)


async def _apply_as_persisted_platform_admin(
    monkeypatch,
    *,
    postgres: bool,
    role_rows=(),
    role_permission_rows=(),
    direct_rows=(),
):
    monkeypatch.setattr(
        membership_writer_module,
        "VersionedUserWriteGateway",
        _StableFloorGateway,
    )
    pool = type("Pool", (), {"pool": object() if postgres else None})()
    conn = _PlatformAdminWriterConnection(
        postgres=postgres,
        role_rows=role_rows,
        role_permission_rows=role_permission_rows,
        direct_rows=direct_rows,
    )
    result = await MembershipWriter(pool).apply_membership_mutations(
        conn=conn,
        context=ActorMembershipWriteContext(
            actor_user_id=11,
            required_authority=MembershipAuthority.PLATFORM_ADMIN,
        ),
        mutations=(_mutation(),),
        anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
        operation_time=BASE,
    )
    return result, conn


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.parametrize("grant_source", ["admin_role", "system_permission"])
@pytest.mark.asyncio
async def test_writer_path_accepts_active_persisted_platform_admin_grant(
    monkeypatch,
    postgres,
    grant_source,
) -> None:
    kwargs = (
        {"role_rows": ({"name": "admin"},)}
        if grant_source == "admin_role"
        else {"role_permission_rows": ({"name": "system.configure"},)}
    )

    result, conn = await _apply_as_persisted_platform_admin(
        monkeypatch,
        postgres=postgres,
        **kwargs,
    )

    assert result.mutation_results[0].found is True
    assert result.mutation_results[0].changed is False
    assert any("user_roles" in statement for statement in conn.statements)
    if postgres:
        lock_index = max(
            index
            for index, statement in enumerate(conn.statements)
            if "public.users" in statement and "FOR UPDATE" in statement
        )
        authority_index = min(
            index
            for index, statement in enumerate(conn.statements)
            if "user_roles" in statement
        )
        assert authority_index > lock_index


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_writer_path_rejects_claim_only_platform_admin(monkeypatch, postgres) -> None:
    with pytest.raises(MembershipAuthorizationError):
        await _apply_as_persisted_platform_admin(
            monkeypatch,
            postgres=postgres,
        )


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.parametrize("actor_state", ["missing", "inactive"])
@pytest.mark.asyncio
async def test_organization_creation_authorization_rejects_unusable_actor(
    postgres: bool,
    actor_state: str,
) -> None:
    class _Cursor:
        def __init__(self, row) -> None:
            self._row = row

        async def fetchone(self):
            return self._row

    class _Connection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def _actor_row(self):
            if actor_state == "missing":
                return None
            return {
                "id": 11,
                "is_active": False,
                "is_superuser": True,
                "role": "admin",
            }

        async def fetchrow(self, sql, *_parameters):
            self.statements.append(str(sql))
            return self._actor_row()

        async def fetch(self, sql, *_parameters):
            self.statements.append(str(sql))
            return []

        async def execute(self, sql, _parameters):
            self.statements.append(str(sql))
            return _Cursor(self._actor_row())

    pool = type("Pool", (), {"pool": object() if postgres else None})()
    conn = _Connection()

    with pytest.raises(MembershipAuthorizationError):
        await MembershipWriter(pool).authorize_organization_creation(
            conn=conn,
            context=ActorMembershipWriteContext(
                actor_user_id=11,
                required_authority=MembershipAuthority.PLATFORM_ADMIN,
            ),
            owner_user_id=None,
        )

    assert any("users" in statement for statement in conn.statements)
    if postgres:
        assert "FOR UPDATE" in conn.statements[0]


@pytest.mark.asyncio
async def test_organization_creation_authorization_locks_actor_and_owner_in_order(
) -> None:
    class _Connection:
        def __init__(self) -> None:
            self.locked_user_ids: list[int] = []
            self.authority_statements: list[str] = []

        async def fetchrow(self, sql, user_id):
            if "FOR UPDATE" in str(sql):
                self.locked_user_ids.append(user_id)
            return {
                "id": user_id,
                "is_active": True,
                "is_superuser": user_id == 11,
                "role": "admin" if user_id == 11 else "user",
            }

        async def fetch(self, sql, actor_user_id):
            assert actor_user_id == 11
            self.authority_statements.append(sql)
            return []

    pool = type("Pool", (), {"pool": object()})()
    conn = _Connection()

    await MembershipWriter(pool).authorize_organization_creation(
        conn=conn,
        context=ActorMembershipWriteContext(
            actor_user_id=11,
            required_authority=MembershipAuthority.PLATFORM_ADMIN,
        ),
        owner_user_id=7,
    )

    assert conn.locked_user_ids == [7, 11]
    assert len(conn.authority_statements) == 5
    assert conn.authority_statements[0].startswith(
        "SELECT r.id FROM public.roles"
    )
    assert conn.authority_statements[-1].startswith(
        "SELECT up.permission_id FROM public.user_permissions"
    )


@pytest.mark.asyncio
async def test_scope_deletion_locks_platform_authority_after_membership_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = MembershipScopeDeletionSnapshot(
        scope_type=MembershipScopeType.ORGANIZATION,
        scope_id=5,
        organization_ids=(5,),
        team_parents=(),
        membership_rows=(
            MembershipRowSnapshot(
                scope_type=MembershipScopeType.ORGANIZATION,
                scope_id=5,
                user_id=7,
                role="member",
                status="active",
            ),
        ),
    )
    context = ActorMembershipWriteContext(
        actor_user_id=11,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    writer = MembershipWriter(type("Pool", (), {"pool": object()})())
    lock_set = writer._build_scope_deletion_lock_set(
        context=context,
        snapshot=snapshot,
    )

    async def _skip_snapshot_recheck(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        writer,
        "_recheck_scope_deletion_snapshot",
        _skip_snapshot_recheck,
    )

    class _Connection:
        def __init__(self) -> None:
            self.events: list[tuple[str, str]] = []

        async def fetchrow(self, sql, *_parameters):
            self.events.append(("row", sql))
            return {"id": 1}

        async def fetch(self, sql, actor_user_id):
            assert actor_user_id == 11
            self.events.append(("authority", sql))
            return []

    conn = _Connection()
    await writer._execute_scope_deletion_locks(
        conn,
        context=context,
        snapshot=snapshot,
        lock_set=lock_set,
    )

    assert [kind for kind, _sql in conn.events][-5:] == ["authority"] * 5


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_organization_creation_missing_actor_takes_precedence_over_owner(
    postgres: bool,
) -> None:
    class _Cursor:
        async def fetchone(self):
            return None

    class _Connection:
        def __init__(self) -> None:
            self.read_user_ids: list[int] = []

        async def fetchrow(self, _sql, user_id):
            self.read_user_ids.append(user_id)
            return None

        async def execute(self, _sql, parameters):
            self.read_user_ids.append(parameters[0])
            return _Cursor()

    pool = type("Pool", (), {"pool": object() if postgres else None})()
    conn = _Connection()

    with pytest.raises(MembershipAuthorizationError):
        await MembershipWriter(pool).authorize_organization_creation(
            conn=conn,
            context=ActorMembershipWriteContext(
                actor_user_id=11,
                required_authority=MembershipAuthority.PLATFORM_ADMIN,
            ),
            owner_user_id=7,
        )

    assert conn.read_user_ids == [7, 11]


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_scoped_actor_can_create_self_owned_organization(
    postgres: bool,
) -> None:
    class _Cursor:
        async def fetchone(self):
            return (11, 1, 0, "user")

    class _Connection:
        async def fetchrow(self, _sql, user_id):
            return {
                "id": user_id,
                "is_active": True,
                "is_superuser": False,
                "role": "user",
            }

        async def execute(self, _sql, _parameters):
            return _Cursor()

    pool = type("Pool", (), {"pool": object() if postgres else None})()

    await MembershipWriter(pool).authorize_organization_creation(
        conn=_Connection(),
        context=ActorMembershipWriteContext(
            actor_user_id=11,
            required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
        ),
        owner_user_id=11,
    )


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.parametrize("owner_user_id", [None, 7])
@pytest.mark.asyncio
async def test_scoped_actor_cannot_create_ownerless_or_different_owner_organization(
    postgres: bool,
    owner_user_id: int | None,
) -> None:
    class _Cursor:
        def __init__(self, user_id: int) -> None:
            self._user_id = user_id

        async def fetchone(self):
            return (self._user_id, 1, 0, "user")

    class _Connection:
        async def fetchrow(self, _sql, user_id):
            return {
                "id": user_id,
                "is_active": True,
                "is_superuser": False,
                "role": "user",
            }

        async def execute(self, _sql, parameters):
            return _Cursor(parameters[0])

    pool = type("Pool", (), {"pool": object() if postgres else None})()

    with pytest.raises(MembershipAuthorizationError):
        await MembershipWriter(pool).authorize_organization_creation(
            conn=_Connection(),
            context=ActorMembershipWriteContext(
                actor_user_id=11,
                required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
            ),
            owner_user_id=owner_user_id,
        )


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_writer_path_honors_direct_deny_over_role_permission(
    monkeypatch,
    postgres,
) -> None:
    with pytest.raises(MembershipAuthorizationError):
        await _apply_as_persisted_platform_admin(
            monkeypatch,
            postgres=postgres,
            role_permission_rows=({"name": "system.configure"},),
            direct_rows=({"name": "system.configure", "granted": False},),
        )


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_writer_path_filters_expired_platform_admin_grants(
    monkeypatch,
    postgres,
) -> None:
    with pytest.raises(MembershipAuthorizationError):
        await _apply_as_persisted_platform_admin(
            monkeypatch,
            postgres=postgres,
        )

    pool = type("Pool", (), {"pool": object() if postgres else None})()
    conn = _RbacConnection()
    assert not await MembershipWriter(pool)._has_persisted_platform_admin(conn, 11)
    if postgres:
        assert len(conn.statements) == 5
        assert all("FOR UPDATE" not in statement for statement in conn.statements)
    else:
        assert all(
            "expires_at IS NULL" in statement
            and "expires_at > CURRENT_TIMESTAMP" in statement
            for statement in conn.statements
        )


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_platform_admin_accepts_active_persisted_rbac_role(postgres) -> None:
    pool = type("Pool", (), {"pool": object() if postgres else None})()
    conn = _RbacConnection(role_rows=({"name": "admin"},))

    assert await MembershipWriter(pool)._has_persisted_platform_admin(conn, 11)
    assert any("expires_at" in statement for statement in conn.statements)
    expected_schema = "public." if postgres else "main."
    assert all(expected_schema in statement for statement in conn.statements)


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_platform_admin_accepts_active_role_permission(postgres) -> None:
    pool = type("Pool", (), {"pool": object() if postgres else None})()
    conn = _RbacConnection(
        role_permission_rows=({"name": "system.configure"},),
    )

    assert await MembershipWriter(pool)._has_persisted_platform_admin(conn, 11)


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_platform_admin_honors_permission_deny_and_direct_allow(postgres) -> None:
    pool = type("Pool", (), {"pool": object() if postgres else None})()
    denied = _RbacConnection(
        role_permission_rows=({"name": "system.configure"},),
        direct_rows=({"name": "system.configure", "granted": False},),
    )
    allowed = _RbacConnection(
        direct_rows=({"name": "*", "granted": True},),
    )

    writer = MembershipWriter(pool)
    assert not await writer._has_persisted_platform_admin(denied, 11)
    assert await writer._has_persisted_platform_admin(allowed, 11)


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_platform_admin_filters_expired_role_and_direct_grants(postgres) -> None:
    pool = type("Pool", (), {"pool": object() if postgres else None})()
    conn = _RbacConnection()

    assert not await MembershipWriter(pool)._has_persisted_platform_admin(conn, 11)
    assert len(conn.statements) == (5 if postgres else 3)
    if postgres:
        assert all("FOR UPDATE" not in statement for statement in conn.statements)
    else:
        assert all(
            "expires_at IS NULL" in statement
            and "expires_at > CURRENT_TIMESTAMP" in statement
            for statement in conn.statements
        )


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.asyncio
async def test_platform_admin_rejects_claim_only_actor_without_persisted_grant(
    postgres,
) -> None:
    pool = type("Pool", (), {"pool": object() if postgres else None})()

    assert not await MembershipWriter(pool)._has_persisted_platform_admin(
        _RbacConnection(),
        11,
    )
