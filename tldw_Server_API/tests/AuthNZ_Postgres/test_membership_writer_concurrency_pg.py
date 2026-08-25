from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    AnchorOwnership,
    MembershipAuthority,
    MembershipAuthorizationError,
    MembershipMutation,
    MembershipMutationKind,
    MembershipPreflightChanged,
    MembershipScopeType,
    MembershipWriter,
    MembershipWriteResult,
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    VersionedUserWriteGateway,
)
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
    DEFAULT_BASE_TEAM_NAME,
    AuthnzOrgsTeamsRepo,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_BOOTSTRAP = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)
_OPERATION_TIME = datetime(2026, 8, 8, tzinfo=timezone.utc)


async def _create_user(pool: Any, label: str) -> int:
    async with pool.transaction() as conn:
        result = await VersionedUserWriteGateway("postgres").insert_user(
            conn,
            values={
                "uuid": str(uuid.uuid4()),
                "username": label,
                "email": f"{label}@example.com",
                "password_hash": "x",
                "role": "user",
                "is_active": True,
                "is_verified": True,
            },
        )
    return result.affected_user_ids[0]


async def _create_membership_fixture(pool: Any, prefix: str) -> dict[str, int]:
    owner_id = await _create_user(pool, f"{prefix}_owner")
    first_id = await _create_user(pool, f"{prefix}_first")
    second_id = await _create_user(pool, f"{prefix}_second")
    repo = AuthnzOrgsTeamsRepo(pool)
    organization = await repo.create_organization(
        name=f"{prefix} organization",
        owner_user_id=owner_id,
    )
    org_id = int(organization["id"])
    for user_id, role in (
        (owner_id, "owner"),
        (first_id, "member"),
        (second_id, "member"),
    ):
        await repo.add_org_member(
            org_id=org_id,
            user_id=user_id,
            role=role,
            context=_BOOTSTRAP,
        )
    team = await repo.create_team(org_id=org_id, name=f"{prefix} team")
    team_id = int(team["id"])
    for user_id in (first_id, second_id):
        await repo.add_team_member(
            team_id=team_id,
            user_id=user_id,
            role="member",
            context=_BOOTSTRAP,
        )
    default_team_id = await pool.fetchval(
        "SELECT id FROM public.teams WHERE org_id = $1 AND name = $2",
        org_id,
        DEFAULT_BASE_TEAM_NAME,
    )
    return {
        "owner_id": owner_id,
        "first_id": first_id,
        "second_id": second_id,
        "org_id": org_id,
        "team_id": team_id,
        "default_team_id": int(default_team_id),
    }


@pytest.mark.asyncio
async def test_postgres_writer_uses_public_relations_under_shadow_search_path(
    test_db_pool,
) -> None:
    ids = await _create_membership_fixture(test_db_pool, f"shadow_{uuid.uuid4().hex[:8]}")
    schema = f"membership_shadow_{uuid.uuid4().hex[:8]}"

    async with test_db_pool.transaction() as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(
            f'CREATE TABLE "{schema}".organizations '
            "(id INTEGER PRIMARY KEY, owner_user_id INTEGER, is_active BOOLEAN)"
        )
        await conn.execute(
            f'CREATE TABLE "{schema}".teams '
            "(id INTEGER PRIMARY KEY, org_id INTEGER, name TEXT, is_active BOOLEAN)"
        )
        await conn.execute(
            f'CREATE TABLE "{schema}".org_members '
            "(org_id INTEGER, user_id INTEGER, role TEXT, status TEXT)"
        )
        await conn.execute(
            f'CREATE TABLE "{schema}".team_members '
            "(team_id INTEGER, user_id INTEGER, role TEXT, status TEXT)"
        )
        await conn.execute(
            f'INSERT INTO "{schema}".organizations VALUES ($1, $2, FALSE)',
            ids["org_id"],
            ids["owner_id"],
        )
        await conn.execute(
            f'INSERT INTO "{schema}".teams VALUES ($1, $2, $3, FALSE)',
            ids["team_id"],
            ids["org_id"],
            "shadow team",
        )
        await conn.execute(
            f'INSERT INTO "{schema}".teams VALUES ($1, $2, $3, TRUE)',
            999_999,
            ids["org_id"],
            DEFAULT_BASE_TEAM_NAME,
        )
        await conn.execute(
            f'INSERT INTO "{schema}".org_members VALUES ($1, $2, $3, $4)',
            ids["org_id"],
            ids["first_id"],
            "shadow-role",
            "active",
        )
        await conn.execute(
            f'INSERT INTO "{schema}".team_members VALUES ($1, $2, $3, $4)',
            ids["team_id"],
            ids["second_id"],
            "shadow-role",
            "active",
        )
        await conn.execute(f'SET LOCAL search_path TO "{schema}", public')

        default_team_id = await AuthnzOrgsTeamsRepo(
            test_db_pool
        )._get_or_create_default_team_id(  # noqa: SLF001
            conn,
            ids["org_id"],
            create=False,
        )
        result = await MembershipWriter(test_db_pool).apply_membership_mutations(
            conn=conn,
            context=_BOOTSTRAP,
            mutations=(
                MembershipMutation(
                    scope_type=MembershipScopeType.ORGANIZATION,
                    scope_id=ids["org_id"],
                    user_id=ids["first_id"],
                    kind=MembershipMutationKind.UPDATE_ROLE,
                    role="admin",
                ),
                MembershipMutation(
                    scope_type=MembershipScopeType.TEAM,
                    scope_id=ids["team_id"],
                    user_id=ids["second_id"],
                    kind=MembershipMutationKind.UPDATE_ROLE,
                    role="lead",
                ),
            ),
            anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
            operation_time=_OPERATION_TIME,
        )

        assert default_team_id == ids["default_team_id"]
        assert result.affected_user_ids == (ids["first_id"], ids["second_id"])
        assert await conn.fetchval(
            "SELECT role FROM public.org_members WHERE org_id = $1 AND user_id = $2",
            ids["org_id"],
            ids["first_id"],
        ) == "admin"
        assert await conn.fetchval(
            "SELECT role FROM public.team_members WHERE team_id = $1 AND user_id = $2",
            ids["team_id"],
            ids["second_id"],
        ) == "lead"
        assert await conn.fetchval(
            f'SELECT role FROM "{schema}".org_members'
        ) == "shadow-role"
        assert await conn.fetchval(
            f'SELECT role FROM "{schema}".team_members'
        ) == "shadow-role"


class _TwoPartyGate:
    def __init__(self) -> None:
        self._arrivals = 0
        self._ready = asyncio.Event()

    async def arrive(self) -> None:
        self._arrivals += 1
        if self._arrivals == 2:
            self._ready.set()
        await self._ready.wait()


class _FirstUserLockBarrierConnection:
    def __init__(self, conn: Any, gate: _TwoPartyGate) -> None:
        self._conn = conn
        self._gate = gate
        self._entered = False

    async def fetchrow(self, sql: str, *parameters: Any) -> Any:
        if (
            not self._entered
            and sql == "SELECT id FROM public.users WHERE id = $1 FOR UPDATE"
        ):
            self._entered = True
            await self._gate.arrive()
        return await self._conn.fetchrow(sql, *parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _RevocationHoldConnection:
    def __init__(
        self,
        conn: Any,
        *,
        revocation_written: asyncio.Event,
        allow_commit: asyncio.Event,
    ) -> None:
        self._conn = conn
        self._revocation_written = revocation_written
        self._allow_commit = allow_commit

    async def execute(self, sql: str, *parameters: Any) -> Any:
        result = await self._conn.execute(sql, *parameters)
        if sql.lstrip().upper().startswith("DELETE FROM"):
            self._revocation_written.set()
            await self._allow_commit.wait()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _RevocationHoldPool:
    def __init__(
        self,
        pool: Any,
        *,
        revocation_written: asyncio.Event,
        allow_commit: asyncio.Event,
    ) -> None:
        self._pool = pool
        self.pool = pool.pool
        self._revocation_written = revocation_written
        self._allow_commit = allow_commit

    @asynccontextmanager
    async def transaction(self):
        async with self._pool.transaction() as conn:
            yield _RevocationHoldConnection(
                conn,
                revocation_written=self._revocation_written,
                allow_commit=self._allow_commit,
            )


class _ActorUserLockProbeConnection:
    def __init__(
        self,
        conn: Any,
        *,
        actor_user_id: int,
        lock_attempted: asyncio.Event,
        acquired_before_revocation_commit: asyncio.Event,
        allow_commit: asyncio.Event,
    ) -> None:
        self._conn = conn
        self._actor_user_id = actor_user_id
        self._lock_attempted = lock_attempted
        self._acquired_before_revocation_commit = acquired_before_revocation_commit
        self._allow_commit = allow_commit

    async def fetchrow(self, sql: str, *parameters: Any) -> Any:
        actor_lock = (
            sql == "SELECT id FROM public.users WHERE id = $1 FOR UPDATE"
            and int(parameters[0]) == self._actor_user_id
        )
        if actor_lock:
            self._lock_attempted.set()
        result = await self._conn.fetchrow(sql, *parameters)
        if actor_lock and not self._allow_commit.is_set():
            self._acquired_before_revocation_commit.set()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


@pytest.mark.asyncio
async def test_postgres_opposite_request_orders_share_one_canonical_lock_order(
    test_db_pool,
) -> None:
    ids = await _create_membership_fixture(test_db_pool, f"order_{uuid.uuid4().hex[:8]}")
    gate = _TwoPartyGate()

    first_order = (
        MembershipMutation(
            MembershipScopeType.TEAM,
            ids["team_id"],
            ids["first_id"],
            MembershipMutationKind.UPDATE_ROLE,
            "lead",
        ),
        MembershipMutation(
            MembershipScopeType.TEAM,
            ids["team_id"],
            ids["second_id"],
            MembershipMutationKind.UPDATE_ROLE,
            "member",
        ),
    )
    opposite_order = (
        MembershipMutation(
            MembershipScopeType.TEAM,
            ids["team_id"],
            ids["second_id"],
            MembershipMutationKind.UPDATE_ROLE,
            "admin",
        ),
        MembershipMutation(
            MembershipScopeType.TEAM,
            ids["team_id"],
            ids["first_id"],
            MembershipMutationKind.UPDATE_ROLE,
            "member",
        ),
    )

    async def _apply(mutations: tuple[MembershipMutation, ...]):
        async with test_db_pool.transaction() as conn:
            return await MembershipWriter(test_db_pool).apply_membership_mutations(
                conn=_FirstUserLockBarrierConnection(conn, gate),
                context=_BOOTSTRAP,
                mutations=mutations,
                anchor_ownership=AnchorOwnership.WRITER_OWNS_ANCHOR,
                operation_time=_OPERATION_TIME,
            )

    results = await asyncio.wait_for(
        asyncio.gather(_apply(first_order), _apply(opposite_order)),
        timeout=10,
    )

    assert all(
        result.affected_user_ids == (ids["first_id"], ids["second_id"])
        for result in results
    )
    rows = await test_db_pool.fetchall(
        "SELECT user_id, role FROM public.team_members "
        "WHERE team_id = $1 AND user_id = ANY($2::int[]) ORDER BY user_id",
        ids["team_id"],
        [ids["first_id"], ids["second_id"]],
    )
    assert tuple((int(row["user_id"]), str(row["role"])) for row in rows) in {
        ((ids["first_id"], "lead"), (ids["second_id"], "member")),
        ((ids["first_id"], "member"), (ids["second_id"], "admin")),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "grant_source",
    ["user_role", "role_permission", "direct_permission"],
)
async def test_postgres_platform_admin_revocation_serializes_before_authorization(
    test_db_pool,
    grant_source: str,
) -> None:
    ids = await _create_membership_fixture(test_db_pool, f"auth_{uuid.uuid4().hex[:8]}")
    async with test_db_pool.transaction() as conn:
        permission_id = await conn.fetchval(
            "SELECT id FROM public.permissions WHERE name = 'system.configure'"
        )
        if grant_source == "user_role":
            role_id = await conn.fetchval(
                "SELECT id FROM public.roles WHERE name = 'admin'"
            )
            await conn.execute(
                "INSERT INTO public.user_roles (user_id, role_id) VALUES ($1, $2) "
                "ON CONFLICT (user_id, role_id) DO NOTHING",
                ids["first_id"],
                role_id,
            )
            revoke_sql = "DELETE FROM user_roles WHERE user_id = $1 AND role_id = $2"
            revoke_parameters = (ids["first_id"], int(role_id))
        elif grant_source == "role_permission":
            role_id = await conn.fetchval(
                "SELECT id FROM public.roles WHERE name = 'member'"
            )
            await conn.execute(
                "INSERT INTO public.user_roles (user_id, role_id) VALUES ($1, $2) "
                "ON CONFLICT (user_id, role_id) DO NOTHING",
                ids["first_id"],
                role_id,
            )
            await conn.execute(
                "INSERT INTO public.role_permissions (role_id, permission_id) "
                "VALUES ($1, $2) ON CONFLICT (role_id, permission_id) DO NOTHING",
                role_id,
                permission_id,
            )
            revoke_sql = (
                "DELETE FROM role_permissions WHERE role_id = $1 AND permission_id = $2"
            )
            revoke_parameters = (int(role_id), int(permission_id))
        else:
            await conn.execute(
                "INSERT INTO public.user_permissions (user_id, permission_id, granted) "
                "VALUES ($1, $2, TRUE) "
                "ON CONFLICT (user_id, permission_id) DO UPDATE SET granted = TRUE",
                ids["first_id"],
                permission_id,
            )
            revoke_sql = (
                "DELETE FROM user_permissions WHERE user_id = $1 AND permission_id = $2"
            )
            revoke_parameters = (ids["first_id"], int(permission_id))
    revocation_written = asyncio.Event()
    allow_commit = asyncio.Event()
    lock_attempted = asyncio.Event()
    acquired_before_revocation_commit = asyncio.Event()
    revocation_pool = _RevocationHoldPool(
        test_db_pool,
        revocation_written=revocation_written,
        allow_commit=allow_commit,
    )

    async def _revoke() -> bool:
        async with revocation_pool.transaction() as conn:
            status = await conn.execute(revoke_sql, *revoke_parameters)
            return status == "DELETE 1"

    async def _mutate() -> MembershipWriteResult | Exception:
        try:
            async with test_db_pool.transaction() as conn:
                return await MembershipWriter(test_db_pool).apply_membership_mutations(
                    conn=_ActorUserLockProbeConnection(
                        conn,
                        actor_user_id=ids["first_id"],
                        lock_attempted=lock_attempted,
                        acquired_before_revocation_commit=(
                            acquired_before_revocation_commit
                        ),
                        allow_commit=allow_commit,
                    ),
                    context=ActorMembershipWriteContext(
                        actor_user_id=ids["first_id"],
                        required_authority=MembershipAuthority.PLATFORM_ADMIN,
                    ),
                    mutations=(
                        MembershipMutation(
                            MembershipScopeType.TEAM,
                            ids["team_id"],
                            ids["second_id"],
                            MembershipMutationKind.UPDATE_ROLE,
                            "lead",
                        ),
                    ),
                    anchor_ownership=AnchorOwnership.WRITER_OWNS_ANCHOR,
                    operation_time=_OPERATION_TIME,
                )
        except Exception as exc:  # noqa: BLE001 - outcome is asserted below
            return exc

    revoke_task = asyncio.create_task(_revoke())
    await asyncio.wait_for(revocation_written.wait(), timeout=5)
    mutation_task = asyncio.create_task(_mutate())
    await asyncio.wait_for(lock_attempted.wait(), timeout=5)
    try:
        await asyncio.wait_for(
            acquired_before_revocation_commit.wait(),
            timeout=0.25,
        )
        acquired_early = True
    except TimeoutError:
        acquired_early = False
    finally:
        allow_commit.set()

    revoked, mutation_outcome = await asyncio.wait_for(
        asyncio.gather(revoke_task, mutation_task),
        timeout=10,
    )

    assert revoked is True
    assert acquired_early is False
    assert isinstance(mutation_outcome, MembershipAuthorizationError)
    assert await test_db_pool.fetchval(
        "SELECT role FROM public.team_members WHERE team_id = $1 AND user_id = $2",
        ids["team_id"],
        ids["second_id"],
    ) == "member"


@pytest.mark.asyncio
async def test_postgres_concurrent_last_owner_removals_serialize_and_recheck(
    test_db_pool,
) -> None:
    ids = await _create_membership_fixture(test_db_pool, f"owner_{uuid.uuid4().hex[:8]}")
    repo = AuthnzOrgsTeamsRepo(test_db_pool)
    await repo.update_org_member_role(
        org_id=ids["org_id"],
        user_id=ids["first_id"],
        role="owner",
        context=_BOOTSTRAP,
    )
    gate = _TwoPartyGate()

    async def _remove_owner(user_id: int):
        async with test_db_pool.transaction() as conn:
            return await MembershipWriter(test_db_pool).apply_membership_mutations(
                conn=_FirstUserLockBarrierConnection(conn, gate),
                context=_BOOTSTRAP,
                mutations=(
                    MembershipMutation(
                        MembershipScopeType.ORGANIZATION,
                        ids["org_id"],
                        user_id,
                        MembershipMutationKind.REMOVE,
                    ),
                ),
                anchor_ownership=AnchorOwnership.WRITER_OWNS_ANCHOR,
                operation_time=_OPERATION_TIME,
            )

    outcomes = await asyncio.wait_for(
        asyncio.gather(
            _remove_owner(ids["owner_id"]),
            _remove_owner(ids["first_id"]),
            return_exceptions=True,
        ),
        timeout=10,
    )

    assert sum(isinstance(outcome, MembershipWriteResult) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, MembershipPreflightChanged) for outcome in outcomes) == 1
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.org_members "
        "WHERE org_id = $1 AND role = 'owner' AND status = 'active'",
        ids["org_id"],
    ) == 1
