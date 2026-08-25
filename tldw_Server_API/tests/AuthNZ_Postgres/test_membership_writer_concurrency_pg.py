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
    MembershipParentRequired,
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
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
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
    organization = await repo.create_organization_with_owner_membership(
        name=f"{prefix} organization",
        owner_user_id=owner_id,
        context=_BOOTSTRAP,
    )
    org_id = int(organization["id"])
    for user_id, role in (
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
        await conn.fetchval(
            "SELECT set_config('search_path', $1, true)",
            f"{schema}, public",
        )

        repo = AuthnzOrgsTeamsRepo(test_db_pool)
        org_memberships = await repo.list_org_memberships_for_user(
            ids["first_id"],
            conn=conn,
        )
        team_memberships = await repo.list_memberships_for_user(
            ids["first_id"],
            conn=conn,
        )
        default_team_id = await repo._get_or_create_default_team_id(  # noqa: SLF001
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
        assert org_memberships == [
            {"org_id": ids["org_id"], "role": "member", "status": "active"}
        ]
        assert {row["team_id"] for row in team_memberships} == {
            ids["default_team_id"],
            ids["team_id"],
        }
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
            f'SELECT COUNT(*) FROM "{schema}".org_members'
        ) == 0
        assert await conn.fetchval(
            f'SELECT COUNT(*) FROM "{schema}".team_members'
        ) == 0


class _TwoPartyGate:
    def __init__(self) -> None:
        self._arrivals = 0
        self._ready = asyncio.Event()

    async def arrive(self) -> None:
        self._arrivals += 1
        if self._arrivals == 2:
            self._ready.set()
        await self._ready.wait()


class _ManagedPostgresConnectionProxy:
    _authnz_profile_user_backend = "postgres"

    @property
    def _authnz_profile_user_guard_identity(self) -> object:
        return self._conn._authnz_profile_user_guard_identity


class _FirstUserLockBarrierConnection(_ManagedPostgresConnectionProxy):
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


class _RevocationHoldConnection(_ManagedPostgresConnectionProxy):
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


class _AuthorityLockProbeConnection(_ManagedPostgresConnectionProxy):
    def __init__(
        self,
        conn: Any,
        *,
        authority_sql_prefix: str,
        authority_lock_attempted: asyncio.Event,
    ) -> None:
        self._conn = conn
        self._authority_sql_prefix = authority_sql_prefix
        self._authority_lock_attempted = authority_lock_attempted

    async def fetch(self, sql: str, *parameters: Any) -> Any:
        if sql.startswith(self._authority_sql_prefix):
            self._authority_lock_attempted.set()
        return await self._conn.fetch(sql, *parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _AuthorityLockProbePool:
    def __init__(
        self,
        pool: Any,
        *,
        authority_sql_prefix: str,
        authority_lock_attempted: asyncio.Event,
    ) -> None:
        self._pool = pool
        self.pool = pool.pool
        self._authority_sql_prefix = authority_sql_prefix
        self._authority_lock_attempted = authority_lock_attempted

    def _proxy(self, conn: Any) -> _AuthorityLockProbeConnection:
        return _AuthorityLockProbeConnection(
            conn,
            authority_sql_prefix=self._authority_sql_prefix,
            authority_lock_attempted=self._authority_lock_attempted,
        )

    @asynccontextmanager
    async def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ):
        async with self._pool.transaction(
            acquire_timeout_seconds=acquire_timeout_seconds,
        ) as conn:
            yield self._proxy(conn)

    @asynccontextmanager
    async def acquire(self, *, timeout: float | None = None):
        async with self._pool.acquire(timeout=timeout) as conn:
            yield self._proxy(conn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


class _HoldOrganizationLockConnection(_ManagedPostgresConnectionProxy):
    def __init__(
        self,
        conn: Any,
        *,
        organization_id: int,
        organization_locked: asyncio.Event,
        allow_removal: asyncio.Event,
    ) -> None:
        self._conn = conn
        self._organization_id = organization_id
        self._organization_locked = organization_locked
        self._allow_removal = allow_removal

    async def fetchrow(self, sql: str, *parameters: Any) -> Any:
        result = await self._conn.fetchrow(sql, *parameters)
        if (
            sql == "SELECT id FROM public.organizations WHERE id = $1 FOR UPDATE"
            and int(parameters[0]) == self._organization_id
        ):
            self._organization_locked.set()
            await self._allow_removal.wait()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _TargetUserLockAttemptConnection(_ManagedPostgresConnectionProxy):
    def __init__(
        self,
        conn: Any,
        *,
        user_id: int,
        lock_attempted: asyncio.Event,
    ) -> None:
        self._conn = conn
        self._user_id = user_id
        self._lock_attempted = lock_attempted

    async def fetchrow(self, sql: str, *parameters: Any) -> Any:
        if (
            sql == "SELECT id FROM public.users WHERE id = $1 FOR UPDATE"
            and int(parameters[0]) == self._user_id
        ):
            self._lock_attempted.set()
        return await self._conn.fetchrow(sql, *parameters)

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

    assert {
        user_id
        for result in results
        for user_id in result.affected_user_ids
    } == {ids["first_id"], ids["second_id"]}
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
            "INSERT INTO public.permissions (name, description, category) "
            "VALUES ('system.configure', 'Configure system', 'system') "
            "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id"
        )
        if grant_source == "user_role":
            role_id = await conn.fetchval(
                "INSERT INTO public.roles (name, description, is_system) "
                "VALUES ('admin', 'Administrator', TRUE) "
                "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id"
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
                "INSERT INTO public.roles (name, description, is_system) "
                "VALUES ('test-platform-role', 'Test role', FALSE) "
                "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id"
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
    authority_lock_attempted = asyncio.Event()
    authority_sql_prefix = {
        "user_role": "SELECT ur.role_id FROM public.user_roles",
        "role_permission": "SELECT rp.role_id, rp.permission_id FROM public.role_permissions",
        "direct_permission": "SELECT up.permission_id FROM public.user_permissions",
    }[grant_source]
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
                    conn=_AuthorityLockProbeConnection(
                        conn,
                        authority_sql_prefix=authority_sql_prefix,
                        authority_lock_attempted=authority_lock_attempted,
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
    await asyncio.wait_for(authority_lock_attempted.wait(), timeout=5)
    try:
        await asyncio.wait_for(asyncio.shield(mutation_task), timeout=0.25)
        completed_before_revocation_commit = True
    except TimeoutError:
        completed_before_revocation_commit = False
    finally:
        allow_commit.set()

    revoked, mutation_outcome = await asyncio.wait_for(
        asyncio.gather(revoke_task, mutation_task),
        timeout=10,
    )

    assert revoked is True
    assert completed_before_revocation_commit is False
    assert isinstance(mutation_outcome, MembershipAuthorizationError)
    assert await test_db_pool.fetchval(
        "SELECT role FROM public.team_members WHERE team_id = $1 AND user_id = $2",
        ids["team_id"],
        ids["second_id"],
    ) == "member"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["organization_creation", "scope_deletion", "provider_secret"],
)
async def test_postgres_specialized_platform_admin_paths_serialize_revocation(
    test_db_pool,
    operation: str,
) -> None:
    prefix = f"special_auth_{uuid.uuid4().hex[:8]}"
    ids = await _create_membership_fixture(test_db_pool, prefix)
    if operation == "provider_secret":
        await AuthnzOrgProviderSecretsRepo(test_db_pool).ensure_tables()

    async with test_db_pool.transaction() as conn:
        permission_id = await conn.fetchval(
            "INSERT INTO public.permissions (name, description, category) "
            "VALUES ('system.configure', 'Configure system', 'system') "
            "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id"
        )
        await conn.execute(
            "INSERT INTO public.user_permissions (user_id, permission_id, granted) "
            "VALUES ($1, $2, TRUE) "
            "ON CONFLICT (user_id, permission_id) DO UPDATE SET granted = TRUE",
            ids["first_id"],
            permission_id,
        )

    revocation_written = asyncio.Event()
    allow_commit = asyncio.Event()
    authority_lock_attempted = asyncio.Event()
    revocation_pool = _RevocationHoldPool(
        test_db_pool,
        revocation_written=revocation_written,
        allow_commit=allow_commit,
    )
    operation_pool = _AuthorityLockProbePool(
        test_db_pool,
        authority_sql_prefix="SELECT up.permission_id FROM public.user_permissions",
        authority_lock_attempted=authority_lock_attempted,
    )
    context = ActorMembershipWriteContext(
        actor_user_id=ids["first_id"],
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    created_name = f"{prefix} created organization"

    async def _revoke() -> bool:
        async with revocation_pool.transaction() as conn:
            status = await conn.execute(
                "DELETE FROM public.user_permissions "
                "WHERE user_id = $1 AND permission_id = $2",
                ids["first_id"],
                permission_id,
            )
            return status == "DELETE 1"

    async def _operate() -> dict[str, Any] | None | Exception:
        try:
            if operation == "organization_creation":
                return await AuthnzOrgsTeamsRepo(
                    operation_pool
                ).create_organization_as_actor(
                    name=created_name,
                    context=context,
                )
            if operation == "scope_deletion":
                await AuthnzOrgsTeamsRepo(
                    operation_pool
                ).delete_team_with_provider_secrets(
                    team_id=ids["team_id"],
                    context=context,
                )
                return None
            return await AuthnzOrgProviderSecretsRepo(operation_pool).upsert_secret(
                scope_type="team",
                scope_id=ids["team_id"],
                provider="openai",
                encrypted_blob="ciphertext",
                key_hint="test",
                metadata=None,
                updated_at=_OPERATION_TIME,
                updated_by=ids["first_id"],
                authorization_context=context,
            )
        except Exception as exc:  # noqa: BLE001 - outcome is asserted below
            return exc

    revoke_task = asyncio.create_task(_revoke())
    await asyncio.wait_for(revocation_written.wait(), timeout=5)
    operation_task = asyncio.create_task(_operate())
    await asyncio.wait_for(authority_lock_attempted.wait(), timeout=5)
    try:
        await asyncio.wait_for(asyncio.shield(operation_task), timeout=0.25)
        completed_before_revocation_commit = True
    except TimeoutError:
        completed_before_revocation_commit = False
    finally:
        allow_commit.set()

    revoked, operation_outcome = await asyncio.wait_for(
        asyncio.gather(revoke_task, operation_task),
        timeout=10,
    )

    assert revoked is True
    assert completed_before_revocation_commit is False
    assert isinstance(operation_outcome, MembershipAuthorizationError)
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.organizations WHERE name = $1",
        created_name,
    ) == 0
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.teams WHERE id = $1",
        ids["team_id"],
    ) == 1
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.org_provider_secrets "
        "WHERE scope_type = 'team' AND scope_id = $1 AND provider = 'openai'",
        ids["team_id"],
    ) == 0


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


@pytest.mark.asyncio
async def test_postgres_org_member_removal_serializes_concurrent_team_add(
    test_db_pool,
) -> None:
    ids = await _create_membership_fixture(test_db_pool, f"cascade_{uuid.uuid4().hex[:8]}")
    repo = AuthnzOrgsTeamsRepo(test_db_pool)
    organization_locked = asyncio.Event()
    allow_removal = asyncio.Event()
    add_lock_attempted = asyncio.Event()

    async def _remove_org_member() -> dict[str, Any] | Exception:
        try:
            async with test_db_pool.transaction() as conn:
                return await repo.remove_org_member_on_connection(
                    conn=_HoldOrganizationLockConnection(
                        conn,
                        organization_id=ids["org_id"],
                        organization_locked=organization_locked,
                        allow_removal=allow_removal,
                    ),
                    org_id=ids["org_id"],
                    user_id=ids["first_id"],
                    context=_BOOTSTRAP,
                    anchor_ownership=AnchorOwnership.WRITER_OWNS_ANCHOR,
                    operation_time=_OPERATION_TIME,
                )
        except Exception as exc:  # noqa: BLE001 - outcome is asserted below
            return exc

    async def _add_team_member() -> dict[str, Any] | Exception:
        try:
            async with test_db_pool.transaction() as conn:
                return await repo.add_team_member_on_connection(
                    conn=_TargetUserLockAttemptConnection(
                        conn,
                        user_id=ids["first_id"],
                        lock_attempted=add_lock_attempted,
                    ),
                    team_id=ids["team_id"],
                    user_id=ids["first_id"],
                    role="member",
                    context=_BOOTSTRAP,
                    anchor_ownership=AnchorOwnership.WRITER_OWNS_ANCHOR,
                    operation_time=_OPERATION_TIME,
                )
        except Exception as exc:  # noqa: BLE001 - outcome is asserted below
            return exc

    removal_task = asyncio.create_task(_remove_org_member())
    await asyncio.wait_for(organization_locked.wait(), timeout=5)
    add_task = asyncio.create_task(_add_team_member())
    await asyncio.wait_for(add_lock_attempted.wait(), timeout=5)
    await asyncio.sleep(0)
    assert not add_task.done()
    allow_removal.set()

    removal_outcome, add_outcome = await asyncio.wait_for(
        asyncio.gather(removal_task, add_task),
        timeout=10,
    )

    assert isinstance(removal_outcome, dict)
    assert isinstance(add_outcome, MembershipParentRequired)
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.org_members WHERE org_id = $1 AND user_id = $2",
        ids["org_id"],
        ids["first_id"],
    ) == 0
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.team_members tm "
        "JOIN public.teams t ON t.id = tm.team_id "
        "WHERE t.org_id = $1 AND tm.user_id = $2",
        ids["org_id"],
        ids["first_id"],
    ) == 0
