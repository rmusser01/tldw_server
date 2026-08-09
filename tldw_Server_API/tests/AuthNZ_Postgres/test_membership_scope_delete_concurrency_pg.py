from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    MembershipAuthority,
    MembershipPreflightChanged,
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    VersionedUserWriteGateway,
)
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
    AuthnzOrgsTeamsRepo,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_BOOTSTRAP = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


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


async def _create_org_fixture(pool: Any, prefix: str) -> dict[str, int]:
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
    team = await repo.create_team(org_id=org_id, name=f"{prefix} populated")
    team_id = int(team["id"])
    await repo.add_team_member(
        team_id=team_id,
        user_id=first_id,
        role="member",
        context=_BOOTSTRAP,
    )
    return {
        "owner": owner_id,
        "first": first_id,
        "second": second_id,
        "org": org_id,
        "team": team_id,
    }


def _owner_context(owner_id: int) -> ActorMembershipWriteContext:
    return ActorMembershipWriteContext(
        actor_user_id=owner_id,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )


class _ParentLockGate:
    def __init__(self, *, pause_attempts: int) -> None:
        self.pause_attempts = pause_attempts
        self.arrivals: asyncio.Queue[int] = asyncio.Queue()
        self._releases: dict[int, asyncio.Event] = {}

    async def pause(self, attempt: int) -> None:
        if attempt > self.pause_attempts:
            return
        release = asyncio.Event()
        self._releases[attempt] = release
        await self.arrivals.put(attempt)
        await release.wait()

    def release(self, attempt: int) -> None:
        self._releases[attempt].set()


class _ManagedPostgresConnectionProxy:
    _authnz_profile_user_backend = "postgres"

    @property
    def _authnz_profile_user_guard_identity(self) -> object:
        return self._conn._authnz_profile_user_guard_identity


class _ParentLockGateConnection(_ManagedPostgresConnectionProxy):
    def __init__(
        self,
        conn: Any,
        *,
        attempt: int,
        org_id: int,
        gate: _ParentLockGate,
    ) -> None:
        self._conn = conn
        self._attempt = attempt
        self._org_id = org_id
        self._gate = gate
        self._paused = False

    async def fetchrow(self, sql: str, *parameters: Any) -> Any:
        if (
            not self._paused
            and sql == "SELECT id FROM public.organizations WHERE id = $1 FOR UPDATE"
            and int(parameters[0]) == self._org_id
        ):
            self._paused = True
            await self._gate.pause(self._attempt)
        return await self._conn.fetchrow(sql, *parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _ObservedDeletionPool:
    def __init__(self, pool: Any, *, org_id: int, gate: _ParentLockGate) -> None:
        self._pool = pool
        self.pool = pool.pool
        self._org_id = org_id
        self._gate = gate
        self.transaction_count = 0
        self.escaped_failures: list[type[BaseException]] = []

    @asynccontextmanager
    async def transaction(self):
        self.transaction_count += 1
        attempt = self.transaction_count
        try:
            async with self._pool.transaction() as conn:
                yield _ParentLockGateConnection(
                    conn,
                    attempt=attempt,
                    org_id=self._org_id,
                    gate=self._gate,
                )
        except BaseException as exc:
            self.escaped_failures.append(type(exc))
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


class _TwoPartyGate:
    def __init__(self) -> None:
        self._arrivals = 0
        self._ready = asyncio.Event()

    async def arrive(self) -> None:
        self._arrivals += 1
        if self._arrivals == 2:
            self._ready.set()
        await self._ready.wait()


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


class _FirstUserLockBarrierPool:
    def __init__(self, pool: Any, gate: _TwoPartyGate) -> None:
        self._pool = pool
        self.pool = pool.pool
        self._gate = gate

    @asynccontextmanager
    async def transaction(self):
        async with self._pool.transaction() as conn:
            yield _FirstUserLockBarrierConnection(conn, self._gate)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


class _ProviderSecretDeleteGateConnection(_ManagedPostgresConnectionProxy):
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

    async def execute(self, query: Any, *parameters: Any) -> Any:
        query_text = query if type(query) is str else getattr(query, "text", "")
        if type(query_text) is str:
            normalized = " ".join(query_text.split())
            parent_table = (
                "organizations" if self._scope_type == "org" else "teams"
            )
            if (
                not self._paused
                and normalized.startswith(f"DELETE FROM public.{parent_table}")
            ):
                self._paused = True
                self._ready.set()
                await self._release.wait()
        return await self._conn.execute(query, *parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _ProviderSecretDeleteGatePool:
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
    async def transaction(self):
        async with self._delegate.transaction() as conn:
            yield _ProviderSecretDeleteGateConnection(
                conn,
                scope_type=self._scope_type,
                ready=self._ready,
                release=self._release,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _ProviderUpsertRaceConnection(_ManagedPostgresConnectionProxy):
    def __init__(
        self,
        conn: Any,
        *,
        parent_lock_attempted: asyncio.Event,
        insert_attempted: asyncio.Event,
        allow_insert: asyncio.Event,
    ) -> None:
        self._conn = conn
        self._parent_lock_attempted = parent_lock_attempted
        self._insert_attempted = insert_attempted
        self._allow_insert = allow_insert

    async def execute(self, query: Any, *parameters: Any) -> Any:
        if type(query) is str and " ".join(query.split()).startswith(
            "INSERT INTO public.org_provider_secrets"
        ):
            self._insert_attempted.set()
            await self._allow_insert.wait()
        return await self._conn.execute(query, *parameters)

    async def fetchrow(self, query: Any, *parameters: Any) -> Any:
        if type(query) is str:
            normalized = " ".join(query.split())
            if (
                normalized
                == "SELECT id, is_active FROM public.organizations "
                "WHERE id = $1 FOR UPDATE"
            ):
                self._parent_lock_attempted.set()
            if normalized.startswith("INSERT INTO public.org_provider_secrets"):
                self._insert_attempted.set()
                await self._allow_insert.wait()
        return await self._conn.fetchrow(query, *parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _ProviderUpsertRacePool:
    def __init__(
        self,
        delegate: Any,
        *,
        parent_lock_attempted: asyncio.Event,
        insert_attempted: asyncio.Event,
        allow_insert: asyncio.Event,
    ) -> None:
        self._delegate = delegate
        self.pool = delegate.pool
        self._parent_lock_attempted = parent_lock_attempted
        self._insert_attempted = insert_attempted
        self._allow_insert = allow_insert

    @asynccontextmanager
    async def transaction(self):
        async with self._delegate.transaction() as conn:
            yield _ProviderUpsertRaceConnection(
                conn,
                parent_lock_attempted=self._parent_lock_attempted,
                insert_attempted=self._insert_attempted,
                allow_insert=self._allow_insert,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _ProviderUpsertHoldConnection(_ManagedPostgresConnectionProxy):
    def __init__(
        self,
        conn: Any,
        *,
        org_locked: asyncio.Event,
        team_locked: asyncio.Event,
        insert_ready: asyncio.Event,
        release_insert: asyncio.Event,
    ) -> None:
        self._conn = conn
        self._org_locked = org_locked
        self._team_locked = team_locked
        self._insert_ready = insert_ready
        self._release_insert = release_insert

    async def fetchrow(self, query: Any, *parameters: Any) -> Any:
        if type(query) is not str:
            return await self._conn.fetchrow(query, *parameters)
        normalized = " ".join(query.split())
        if normalized.startswith("INSERT INTO public.org_provider_secrets"):
            self._insert_ready.set()
            await self._release_insert.wait()
            return await self._conn.fetchrow(query, *parameters)
        row = await self._conn.fetchrow(query, *parameters)
        if (
            normalized
            == "SELECT id, is_active FROM public.organizations "
            "WHERE id = $1 FOR UPDATE"
        ):
            self._org_locked.set()
        if (
            normalized
            == "SELECT id, org_id, is_active FROM public.teams "
            "WHERE id = $1 FOR UPDATE"
        ):
            self._team_locked.set()
        return row

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _ProviderUpsertHoldPool:
    def __init__(
        self,
        delegate: Any,
        *,
        org_locked: asyncio.Event,
        team_locked: asyncio.Event,
        insert_ready: asyncio.Event,
        release_insert: asyncio.Event,
    ) -> None:
        self._delegate = delegate
        self.pool = delegate.pool
        self._org_locked = org_locked
        self._team_locked = team_locked
        self._insert_ready = insert_ready
        self._release_insert = release_insert
        self.backend_pid: int | None = None

    @asynccontextmanager
    async def transaction(self):
        async with self._delegate.transaction() as conn:
            self.backend_pid = int(await conn.fetchval("SELECT pg_backend_pid()"))
            yield _ProviderUpsertHoldConnection(
                conn,
                org_locked=self._org_locked,
                team_locked=self._team_locked,
                insert_ready=self._insert_ready,
                release_insert=self._release_insert,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _DeletionParentAttemptConnection(_ManagedPostgresConnectionProxy):
    def __init__(self, conn: Any, attempted: asyncio.Event) -> None:
        self._conn = conn
        self._attempted = attempted

    async def fetchrow(self, query: Any, *parameters: Any) -> Any:
        if (
            type(query) is str
            and query == "SELECT id FROM public.organizations WHERE id = $1 FOR UPDATE"
        ):
            self._attempted.set()
        return await self._conn.fetchrow(query, *parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _DeletionParentAttemptPool:
    def __init__(self, delegate: Any, attempted: asyncio.Event) -> None:
        self._delegate = delegate
        self.pool = delegate.pool
        self._attempted = attempted
        self.backend_pid: int | None = None

    @asynccontextmanager
    async def transaction(self):
        async with self._delegate.transaction() as conn:
            self.backend_pid = int(await conn.fetchval("SELECT pg_backend_pid()"))
            yield _DeletionParentAttemptConnection(conn, self._attempted)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _ParentLockMatcherDelegate:
    async def fetchrow(self, _query: Any, *_parameters: Any) -> dict[str, Any]:
        return {"id": 1, "org_id": 2, "is_active": True}


@pytest.mark.asyncio
async def test_provider_upsert_race_proxy_matches_current_org_lock_projection() -> None:
    parent_lock_attempted = asyncio.Event()
    proxy = _ProviderUpsertRaceConnection(
        _ParentLockMatcherDelegate(),
        parent_lock_attempted=parent_lock_attempted,
        insert_attempted=asyncio.Event(),
        allow_insert=asyncio.Event(),
    )

    await proxy.fetchrow(
        "SELECT id FROM public.organizations WHERE id = $1 FOR UPDATE",
        1,
    )
    assert not parent_lock_attempted.is_set()

    await proxy.fetchrow(
        "SELECT id, is_active FROM public.organizations "
        "WHERE id = $1 FOR UPDATE",
        1,
    )
    assert parent_lock_attempted.is_set()


@pytest.mark.parametrize(
    ("scope_type", "stale_query", "current_query"),
    (
        (
            "org",
            "SELECT id FROM public.organizations WHERE id = $1 FOR UPDATE",
            "SELECT id, is_active FROM public.organizations "
            "WHERE id = $1 FOR UPDATE",
        ),
        (
            "team",
            "SELECT id, org_id FROM public.teams WHERE id = $1 FOR UPDATE",
            "SELECT id, org_id, is_active FROM public.teams "
            "WHERE id = $1 FOR UPDATE",
        ),
    ),
)
@pytest.mark.asyncio
async def test_provider_upsert_hold_proxy_matches_current_parent_lock_projection(
    scope_type: str,
    stale_query: str,
    current_query: str,
) -> None:
    org_locked = asyncio.Event()
    team_locked = asyncio.Event()
    proxy = _ProviderUpsertHoldConnection(
        _ParentLockMatcherDelegate(),
        org_locked=org_locked,
        team_locked=team_locked,
        insert_ready=asyncio.Event(),
        release_insert=asyncio.Event(),
    )
    expected_event = org_locked if scope_type == "org" else team_locked

    await proxy.fetchrow(stale_query, 1)
    assert not expected_event.is_set()

    await proxy.fetchrow(current_query, 1)
    assert expected_event.is_set()


@pytest.mark.asyncio
async def test_provider_upsert_hold_proxy_matches_public_insert_projection() -> None:
    insert_ready = asyncio.Event()
    release_insert = asyncio.Event()
    release_insert.set()
    proxy = _ProviderUpsertHoldConnection(
        _ParentLockMatcherDelegate(),
        org_locked=asyncio.Event(),
        team_locked=asyncio.Event(),
        insert_ready=insert_ready,
        release_insert=release_insert,
    )

    await proxy.fetchrow(
        "INSERT INTO public.org_provider_secrets (scope_type) VALUES ($1)",
        "org",
    )

    assert insert_ready.is_set()


async def _require_pg_blocker(
    pool: Any,
    *,
    blocked_pid: int,
    blocker_pid: int,
) -> None:
    for _attempt in range(100):
        blocked = await pool.fetchval(
            "SELECT $1 = ANY(pg_blocking_pids($2))",
            blocker_pid,
            blocked_pid,
        )
        if bool(blocked):
            return
    raise AssertionError("expected PostgreSQL parent-row lock blocker")


@pytest.mark.asyncio
async def test_org_delete_retries_in_fresh_transaction_after_new_membership_drift(
    test_db_pool,
) -> None:
    prefix = f"delete_member_{uuid.uuid4().hex[:8]}"
    ids = await _create_org_fixture(test_db_pool, prefix)
    late_user_id = await _create_user(test_db_pool, f"{prefix}_late")
    gate = _ParentLockGate(pause_attempts=1)
    observed_pool = _ObservedDeletionPool(
        test_db_pool,
        org_id=ids["org"],
        gate=gate,
    )
    delete_task = asyncio.create_task(
        AuthnzOrgsTeamsRepo(observed_pool).delete_organization_with_provider_secrets(
            org_id=ids["org"],
            context=_owner_context(ids["owner"]),
        )
    )

    attempt = await asyncio.wait_for(gate.arrivals.get(), timeout=5)
    await AuthnzOrgsTeamsRepo(test_db_pool).add_org_member(
        org_id=ids["org"],
        user_id=late_user_id,
        role="member",
        context=_BOOTSTRAP,
    )
    gate.release(attempt)
    await asyncio.wait_for(delete_task, timeout=10)

    assert observed_pool.transaction_count == 2
    assert len(observed_pool.escaped_failures) == 1
    assert observed_pool.escaped_failures[0].__name__.startswith("_")
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.organizations WHERE id = $1",
        ids["org"],
    ) == 0


@pytest.mark.asyncio
async def test_org_delete_retries_in_fresh_transaction_after_empty_team_drift(
    test_db_pool,
) -> None:
    prefix = f"delete_team_{uuid.uuid4().hex[:8]}"
    ids = await _create_org_fixture(test_db_pool, prefix)
    gate = _ParentLockGate(pause_attempts=1)
    observed_pool = _ObservedDeletionPool(
        test_db_pool,
        org_id=ids["org"],
        gate=gate,
    )
    delete_task = asyncio.create_task(
        AuthnzOrgsTeamsRepo(observed_pool).delete_organization_with_provider_secrets(
            org_id=ids["org"],
            context=_owner_context(ids["owner"]),
        )
    )

    attempt = await asyncio.wait_for(gate.arrivals.get(), timeout=5)
    await AuthnzOrgsTeamsRepo(test_db_pool).create_team(
        org_id=ids["org"],
        name=f"{prefix} empty child",
    )
    gate.release(attempt)
    await asyncio.wait_for(delete_task, timeout=10)

    assert observed_pool.transaction_count == 2
    assert len(observed_pool.escaped_failures) == 1
    assert observed_pool.escaped_failures[0].__name__.startswith("_")
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.teams WHERE org_id = $1",
        ids["org"],
    ) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ("org", "team"))
async def test_scope_delete_serializes_against_provider_secret_upsert(
    test_db_pool,
    scope_type: str,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.membership_writer import (
        MembershipScopeNotFound,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )

    prefix = f"secret_race_{scope_type}_{uuid.uuid4().hex[:8]}"
    ids = await _create_org_fixture(test_db_pool, prefix)
    scope_id = ids[scope_type]
    delete_ready = asyncio.Event()
    release_delete = asyncio.Event()
    parent_lock_attempted = asyncio.Event()
    insert_attempted = asyncio.Event()
    allow_insert = asyncio.Event()
    delete_repo = AuthnzOrgsTeamsRepo(
        _ProviderSecretDeleteGatePool(
            test_db_pool,
            scope_type=scope_type,
            ready=delete_ready,
            release=release_delete,
        )
    )
    upsert_repo = AuthnzOrgProviderSecretsRepo(
        _ProviderUpsertRacePool(
            test_db_pool,
            parent_lock_attempted=parent_lock_attempted,
            insert_attempted=insert_attempted,
            allow_insert=allow_insert,
        )
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
    waiters: tuple[asyncio.Task[bool], asyncio.Task[bool]] | None = None
    try:
        await asyncio.wait_for(delete_ready.wait(), timeout=5)
        upsert_task = asyncio.create_task(
            upsert_repo.upsert_secret(
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
        waiters = (
            asyncio.create_task(parent_lock_attempted.wait()),
            asyncio.create_task(insert_attempted.wait()),
        )
        arrived, waiting = await asyncio.wait(
            waiters,
            timeout=5,
            return_when=asyncio.FIRST_COMPLETED,
        )
        assert arrived
        for waiter in waiting:
            waiter.cancel()
        await asyncio.gather(*waiting, return_exceptions=True)

        release_delete.set()
        deleted = await asyncio.wait_for(delete_task, timeout=10)
        allow_insert.set()
        upserted = (
            await asyncio.gather(upsert_task, return_exceptions=True)
        )[0]
    finally:
        release_delete.set()
        allow_insert.set()
        if waiters is not None:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
        pending = [
            task
            for task in (delete_task, upsert_task)
            if task is not None and not task.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert deleted is None
    assert parent_lock_attempted.is_set()
    assert not insert_attempted.is_set()
    assert isinstance(upserted, MembershipScopeNotFound)
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.org_provider_secrets "
        "WHERE scope_type = $1 AND scope_id = $2",
        scope_type,
        scope_id,
    ) == 0

    if scope_type == "org":
        await test_db_pool.execute(
            "INSERT INTO public.organizations "
            "(id, name, slug, owner_user_id, is_active) "
            "VALUES ($1, $2, $3, $4, TRUE)",
            scope_id,
            f"{prefix} reused organization",
            f"{prefix}-reused-org",
            ids["owner"],
        )
    else:
        await test_db_pool.execute(
            "INSERT INTO public.teams (id, org_id, name, slug, is_active) "
            "VALUES ($1, $2, $3, $4, TRUE)",
            scope_id,
            ids["org"],
            f"{prefix} reused team",
            f"{prefix}-reused-team",
        )

    assert (
        await AuthnzOrgProviderSecretsRepo(test_db_pool).fetch_secret(
            scope_type,
            scope_id,
            "openai",
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ("org", "team"))
async def test_provider_secret_upsert_holds_parent_lock_until_scope_delete(
    test_db_pool,
    scope_type: str,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )

    prefix = f"secret_first_{scope_type}_{uuid.uuid4().hex[:8]}"
    ids = await _create_org_fixture(test_db_pool, prefix)
    scope_id = ids[scope_type]
    org_locked = asyncio.Event()
    team_locked = asyncio.Event()
    insert_ready = asyncio.Event()
    release_insert = asyncio.Event()
    delete_parent_attempted = asyncio.Event()
    upsert_pool = _ProviderUpsertHoldPool(
        test_db_pool,
        org_locked=org_locked,
        team_locked=team_locked,
        insert_ready=insert_ready,
        release_insert=release_insert,
    )
    delete_pool = _DeletionParentAttemptPool(
        test_db_pool,
        delete_parent_attempted,
    )
    upsert_task = asyncio.create_task(
        AuthnzOrgProviderSecretsRepo(upsert_pool).upsert_secret(
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
        delete_repo = AuthnzOrgsTeamsRepo(delete_pool)
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
        await asyncio.wait_for(delete_parent_attempted.wait(), timeout=5)
        assert upsert_pool.backend_pid is not None
        assert delete_pool.backend_pid is not None
        await _require_pg_blocker(
            test_db_pool,
            blocked_pid=delete_pool.backend_pid,
            blocker_pid=upsert_pool.backend_pid,
        )
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

    assert org_locked.is_set()
    assert team_locked.is_set() is (scope_type == "team")
    assert upserted["provider"] == "openai"
    assert deleted is None
    parent_table = "organizations" if scope_type == "org" else "teams"
    assert await test_db_pool.fetchval(
        f"SELECT COUNT(*) FROM public.{parent_table} WHERE id = $1",  # nosec B608
        scope_id,
    ) == 0
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.org_provider_secrets "
        "WHERE scope_type = $1 AND scope_id = $2",
        scope_type,
        scope_id,
    ) == 0


@pytest.mark.asyncio
async def test_org_delete_bounds_retries_and_maps_only_after_each_rollback(
    test_db_pool,
) -> None:
    prefix = f"delete_retry_{uuid.uuid4().hex[:8]}"
    ids = await _create_org_fixture(test_db_pool, prefix)
    gate = _ParentLockGate(pause_attempts=3)
    observed_pool = _ObservedDeletionPool(
        test_db_pool,
        org_id=ids["org"],
        gate=gate,
    )
    delete_task = asyncio.create_task(
        AuthnzOrgsTeamsRepo(observed_pool).delete_organization_with_provider_secrets(
            org_id=ids["org"],
            context=_owner_context(ids["owner"]),
        )
    )

    for index in range(3):
        attempt = await asyncio.wait_for(gate.arrivals.get(), timeout=5)
        await AuthnzOrgsTeamsRepo(test_db_pool).create_team(
            org_id=ids["org"],
            name=f"{prefix} drift {index}",
        )
        gate.release(attempt)

    with pytest.raises(MembershipPreflightChanged):
        await asyncio.wait_for(delete_task, timeout=10)

    assert observed_pool.transaction_count == 3
    assert len(observed_pool.escaped_failures) == 3
    assert all(
        failure.__name__.startswith("_")
        for failure in observed_pool.escaped_failures
    )
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.organizations WHERE id = $1",
        ids["org"],
    ) == 1


@pytest.mark.asyncio
async def test_opposite_ownership_transfers_recheck_locked_owner_without_deadlock(
    test_db_pool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = f"owner_race_{uuid.uuid4().hex[:8]}"
    ids = await _create_org_fixture(test_db_pool, prefix)
    repo = AuthnzOrgsTeamsRepo(
        _FirstUserLockBarrierPool(test_db_pool, _TwoPartyGate())
    )
    touched: list[int] = []
    original_touch = VersionedUserWriteGateway.final_touch

    async def _record_touch(self, conn, *, user_id, version_floor):
        touched.append(int(user_id))
        return await original_touch(
            self,
            conn,
            user_id=user_id,
            version_floor=version_floor,
        )

    monkeypatch.setattr(VersionedUserWriteGateway, "final_touch", _record_touch)

    async def _transfer(candidate_id: int):
        return await repo.transfer_organization_ownership(
            org_id=ids["org"],
            new_owner_user_id=candidate_id,
            current_owner_user_id=ids["owner"],
            context=_owner_context(ids["owner"]),
        )

    outcomes = await asyncio.wait_for(
        asyncio.gather(
            _transfer(ids["first"]),
            _transfer(ids["second"]),
            return_exceptions=True,
        ),
        timeout=10,
    )

    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, MembershipPreflightChanged) for outcome in outcomes) == 1
    winning_owner = int(
        await test_db_pool.fetchval(
            "SELECT owner_user_id FROM public.organizations WHERE id = $1",
            ids["org"],
        )
    )
    assert winning_owner in (ids["first"], ids["second"])
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.org_members "
        "WHERE org_id = $1 AND role = 'owner' AND status = 'active'",
        ids["org"],
    ) == 1
    assert await test_db_pool.fetchval(
        "SELECT role FROM public.org_members WHERE org_id = $1 AND user_id = $2",
        ids["org"],
        ids["owner"],
    ) == "admin"
    assert touched == sorted((ids["owner"], winning_owner))
