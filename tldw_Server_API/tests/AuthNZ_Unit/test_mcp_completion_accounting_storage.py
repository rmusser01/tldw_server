"""End-to-end admission and settlement contracts on real accounting storage."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.repos import usage_repo
from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import execute, fetch_all
from tldw_Server_API.app.core.AuthNZ.repos.provider_usage_reservations_repo import (
    MAX_INT,
    BillingScope,
    ProviderUsageReservation,
    ProviderUsageReservationsRepo,
    ReservationActuals,
    ReservationQuotaSnapshot,
)
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.accounting import (
    CompletionAccountingPolicy,
    ModelCompletionAccounting,
)
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionFailure,
    ModelCompletionRequest,
    ModelInvocationIdentity,
)
from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo
from tldw_Server_API.tests.AuthNZ_Unit.test_provider_usage_reservations_repo import (
    reservation_pool as reservation_pool,
)
from tldw_Server_API.tests.AuthNZ_Unit.test_provider_usage_reservations_repo import (
    reservation_schema as reservation_schema,
)

pytestmark = pytest.mark.unit
NATIVE_LIMITS = (
    ("limits.llm_tokens_per_month", 47),
    ("limits.llm_cost_per_day_usd", 0.000000104),
    ("limits.llm_cost_per_month_usd", 0.000000104),
)


async def native_limit(pool, monkeypatch, key, value):
    """Exercise persisted native profile precedence without a quota-reader stub."""
    monkeypatch.setattr(quota_resolver, "usage_quotas_enabled", lambda: True)
    repo = UserProfileOverridesRepo(pool)
    await repo.ensure_tables()
    await repo.upsert_override(user_id=1, key=key, value=value, updated_by=None)


def make_service(pool, *, token_limit=47, usage=None, governor=None):
    policy = CompletionAccountingPolicy(
        provider="openai",
        model="fixed",
        input_cost_units_per_token=2,
        output_cost_units_per_token=3,
        monthly_token_limit=token_limit,
        monthly_cost_limit=1000,
        governor_policy_id="mcp.fixed",
    )
    if governor is None:
        governor = MemoryResourceGovernor(
            policies={
                "mcp.fixed": {
                    "requests": {"rpm": 1000, "burst": 1.0},
                    "tokens": {"per_min": 100000, "burst": 1.0},
                    "jobs": {"max_concurrent": 2, "ttl_sec": 90},
                    "scopes": ["user"],
                    "fail_mode": "fail_closed",
                }
            }
        )
    return ModelCompletionAccounting(
        policy=policy,
        reservations=ProviderUsageReservationsRepo(pool),
        usage=usage or AuthnzUsageRepo(pool),
        billing=BillingEnforcer(),
        governor=governor,
    )


def request():
    return ModelCompletionRequest("sys", "\u00fc", 10, 100, 200, 2000)


def identity():
    return ModelInvocationIdentity(1, None, None, str(uuid4()))


async def canonical_rows(pool):
    async with pool.acquire() as conn:
        return await fetch_all(
            conn,
            pool.pool is not None,
            """
            SELECT user_id, billing_org_id, provider, model, request_id,
                   prompt_tokens, completion_tokens, total_tokens, estimated
            FROM llm_usage_log WHERE operation = 'mcp_model_completion'
        """,
            (),
            (
                "user_id",
                "billing_org_id",
                "provider",
                "model",
                "request_id",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "estimated",
            ),
        )


class AccountingStorageContract:
    """Run the identical composed lifecycle against SQLite and PostgreSQL."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key,allowance", NATIVE_LIMITS)
    @pytest.mark.parametrize("different_scope", [False, True])
    async def test_native_positive_allowance_counts_ambiguous_paid_exposure(
        self, reservation_pool, monkeypatch, key, allowance, different_scope
    ):
        await native_limit(reservation_pool, monkeypatch, key, allowance)
        service = make_service(reservation_pool, token_limit=10000)
        first = ModelInvocationIdentity(1, None, 12, str(uuid4()))
        handle = await service.reserve(request(), first, credential_source="server_default")
        await service.mark_dispatched(handle)
        assert await service.retain_ambiguous(handle)
        next_scope = 13 if different_scope else 12
        second = ModelInvocationIdentity(1, None, next_scope, str(uuid4()))
        with pytest.raises(ModelCompletionFailure) as error:
            await service.reserve(request(), second, credential_source="server_default")
        assert error.value.code == "model_quota_exceeded"
        assert await ProviderUsageReservationsRepo(reservation_pool).get(second.execution_id) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "key,allowance",
        [
            ("limits.llm_tokens_per_month", 46),
            ("limits.llm_cost_per_day_usd", 0.000000103),
            ("limits.llm_cost_per_month_usd", 0.000000103),
        ],
    )
    async def test_native_admission_includes_proposed_worst_case(self, reservation_pool, monkeypatch, key, allowance):
        await native_limit(reservation_pool, monkeypatch, key, allowance)
        service = make_service(reservation_pool, token_limit=10000)
        invocation = identity()
        with pytest.raises(ModelCompletionFailure) as error:
            await service.reserve(request(), invocation, credential_source="server_default")
        assert error.value.code == "model_quota_exceeded"
        assert await ProviderUsageReservationsRepo(reservation_pool).get(invocation.execution_id) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key,allowance", NATIVE_LIMITS)
    async def test_concurrent_native_admission_different_scopes_has_one_winner(
        self, reservation_pool, monkeypatch, key, allowance
    ):
        await native_limit(reservation_pool, monkeypatch, key, allowance)
        service = make_service(reservation_pool, token_limit=10000)
        entered, release = asyncio.Event(), asyncio.Event()
        original_read = service._usage.read_provider_scope_usage
        block_once = True

        async def held_scope_read(*args, **kwargs):
            nonlocal block_once
            if block_once:
                block_once = False
                entered.set()
                await release.wait()
            return await original_read(*args, **kwargs)

        monkeypatch.setattr(service._usage, "read_provider_scope_usage", held_scope_read)
        first = asyncio.create_task(service.reserve(request(), ModelInvocationIdentity(1, None, 12, str(uuid4()))))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            second = asyncio.create_task(service.reserve(request(), ModelInvocationIdentity(1, 5, None, str(uuid4()))))
            await asyncio.sleep(0.05)
        finally:
            release.set()
            results = await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
        assert sum(isinstance(result, ModelCompletionFailure) for result in results) == 1
        failures = [result for result in results if isinstance(result, ModelCompletionFailure)]
        assert failures[0].code == "model_quota_exceeded"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key,allowance", NATIVE_LIMITS)
    async def test_native_settlement_cannot_move_exposure_between_snapshot_reads(
        self, reservation_pool, monkeypatch, key, allowance
    ):
        await native_limit(reservation_pool, monkeypatch, key, allowance)
        service = make_service(reservation_pool, token_limit=10000)
        first = await service.reserve(request(), ModelInvocationIdentity(1, None, 12, str(uuid4())))
        await service.mark_dispatched(first)
        read_complete, continue_read = asyncio.Event(), asyncio.Event()
        original_fetch = usage_repo.fetch_all
        hold_once = True

        async def paused_actuals(conn, postgres, sql, args, columns):
            nonlocal hold_once
            result = await original_fetch(conn, postgres, sql, args, columns)
            if hold_once and "WHERE l.user_id = ?" in sql:
                hold_once = False
                read_complete.set()
                await continue_read.wait()
            return result

        monkeypatch.setattr(usage_repo, "fetch_all", paused_actuals)
        admission = asyncio.create_task(service.reserve(request(), ModelInvocationIdentity(1, 5, None, str(uuid4()))))
        settlement = None
        try:
            await asyncio.wait_for(read_complete.wait(), 5)
            settlement = asyncio.create_task(service.reconcile(first, input_tokens=6, output_tokens=4))
            await asyncio.sleep(0.05)
            assert not settlement.done()
        finally:
            continue_read.set()
            outcomes = await asyncio.gather(admission, *([settlement] if settlement else []), return_exceptions=True)
        assert isinstance(outcomes[0], ModelCompletionFailure)
        assert outcomes[0].code == "model_quota_exceeded"
        assert outcomes[1] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key,allowance", NATIVE_LIMITS)
    @pytest.mark.parametrize("source", ["user", "user_unmetered"])
    @pytest.mark.parametrize("settlement", ["ambiguous", "reconciled"])
    async def test_native_exposure_preserves_own_key_exclusion(
        self, reservation_pool, monkeypatch, key, allowance, source, settlement
    ):
        await native_limit(reservation_pool, monkeypatch, key, allowance)
        service = make_service(reservation_pool, token_limit=10000)
        handle = await service.reserve(request(), identity(), credential_source=source)
        await service.mark_dispatched(handle)
        if settlement == "ambiguous":
            assert await service.retain_ambiguous(handle)
        else:
            assert await service.reconcile(handle, input_tokens=6, output_tokens=4)
        if source == "user":
            paid = await service.reserve(request(), identity(), credential_source="server_default")
            await service.release_before_dispatch(paid)
        else:
            with pytest.raises(ModelCompletionFailure) as error:
                await service.reserve(request(), identity(), credential_source="server_default")
            assert error.value.code == "model_quota_exceeded"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "key,allowance",
        [
            ("limits.llm_tokens_per_month", 57),
            ("limits.llm_cost_per_day_usd", 0.000000128),
            ("limits.llm_cost_per_month_usd", 0.000000128),
        ],
    )
    async def test_native_reconciliation_counts_actuals_without_reservation_double_count(
        self, reservation_pool, monkeypatch, key, allowance
    ):
        await native_limit(reservation_pool, monkeypatch, key, allowance)
        service = make_service(reservation_pool, token_limit=10000)
        first = await service.reserve(request(), identity(), credential_source="server_default")
        await service.mark_dispatched(first)
        assert await service.reconcile(first, input_tokens=6, output_tokens=4)
        second = await service.reserve(request(), identity(), credential_source="server_default")
        with pytest.raises(ModelCompletionFailure) as error:
            await service.reserve(request(), identity(), credential_source="server_default")
        assert error.value.code == "model_quota_exceeded"
        await service.release_before_dispatch(second)
        after_release = await service.reserve(request(), identity(), credential_source="server_default")
        await service.release_before_dispatch(after_release)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key,allowance", NATIVE_LIMITS)
    async def test_native_exposure_excludes_other_users_and_embeddings(
        self, reservation_pool, monkeypatch, key, allowance
    ):
        await native_limit(reservation_pool, monkeypatch, key, allowance)
        other = ProviderUsageReservation(
            str(uuid4()), 2, None, 12, BillingScope("org", 12), "openai", "fixed", 47, 0, 104, "server_default"
        )

        async def unlimited(_conn):
            return ReservationQuotaSnapshot(0, None, 0, None)

        await ProviderUsageReservationsRepo(reservation_pool).reserve(other, snapshot_reader=unlimited)
        async with reservation_pool.transaction() as conn:
            await execute(
                conn,
                reservation_pool.pool is not None,
                "INSERT INTO llm_usage_log (user_id, operation, total_tokens, total_cost_usd) "
                "VALUES (1, 'embeddings', 1000000, 0.000000001)",
                (),
            )
        service = make_service(reservation_pool, token_limit=2000000)
        own = await service.reserve(request(), identity(), credential_source="server_default")
        await service.release_before_dispatch(own)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key,allowance", NATIVE_LIMITS)
    async def test_native_period_actuals_expire_but_unresolved_exposure_carries_forward(
        self, reservation_pool, monkeypatch, key, allowance
    ):
        await native_limit(reservation_pool, monkeypatch, key, allowance)
        service = make_service(reservation_pool, token_limit=10000)
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        since = today if key == "limits.llm_cost_per_day_usd" else today.replace(day=1)
        old = since - timedelta(seconds=1)
        timestamp = old.replace(tzinfo=None) if reservation_pool.pool is not None else old.isoformat()
        async with reservation_pool.transaction() as conn:
            await execute(
                conn,
                reservation_pool.pool is not None,
                "INSERT INTO llm_usage_log (ts, user_id, operation, total_tokens, total_cost_usd) "
                "VALUES (?, 1, 'chat', 47, 0.000000104)",
                (timestamp,),
            )
        handle = await service.reserve(request(), identity(), credential_source="server_default")
        async with reservation_pool.transaction() as conn:
            await execute(
                conn,
                reservation_pool.pool is not None,
                "UPDATE provider_usage_reservations SET created_at = ? WHERE execution_id = ?",
                (old if reservation_pool.pool is not None else old.isoformat(), handle.reservation.execution_id),
            )
        with pytest.raises(ModelCompletionFailure) as error:
            await service.reserve(request(), identity(), credential_source="server_default")
        assert error.value.code == "model_quota_exceeded"

    @pytest.mark.asyncio
    async def test_native_profile_reads_borrow_the_admission_transaction(self, reservation_pool, monkeypatch):
        await native_limit(reservation_pool, monkeypatch, "limits.llm_tokens_per_month", 47)
        service = make_service(reservation_pool, token_limit=10000)
        transaction = reservation_pool.transaction
        resolve_limit = quota_resolver.uncached_user_quota
        held_connection = None
        observations = []

        @asynccontextmanager
        async def observed_transaction():
            nonlocal held_connection
            async with transaction() as conn:
                held_connection = conn
                try:
                    yield conn
                finally:
                    held_connection = None

        async def observed_limit(*args, db_pool, **kwargs):
            async with db_pool.acquire() as conn:
                observations.append(conn is held_connection)
            return await resolve_limit(*args, db_pool=db_pool, **kwargs)

        async def canonical_pool():
            assert held_connection is None
            return reservation_pool

        monkeypatch.setattr(reservation_pool, "transaction", observed_transaction)
        monkeypatch.setattr(quota_resolver, "uncached_user_quota", observed_limit)
        service._quota_pool_factory = canonical_pool
        await service.reserve(request(), identity(), credential_source="server_default")
        assert observations == [True, True, True]

    @pytest.mark.asyncio
    async def test_admission_commit_then_release_failure_releases_owned_exposure(self, reservation_pool, monkeypatch):
        service = make_service(reservation_pool)
        invocation = identity()
        transaction = reservation_pool.transaction
        fail_release = True

        @asynccontextmanager
        async def committed_then_failed_release():
            nonlocal fail_release
            async with transaction() as conn:
                yield conn
            if fail_release:
                fail_release = False
                raise OSError("private connection-release failure")

        if reservation_pool.pool is None:
            monkeypatch.setattr(reservation_pool, "transaction", committed_then_failed_release)
        else:
            native_pool = reservation_pool.pool

            class ReleaseFailurePool:
                """Fail release after the native commit through DatabasePool's real wrapper."""

                def __getattr__(self, name):
                    return getattr(native_pool, name)

                @asynccontextmanager
                async def acquire(self, *args, **kwargs):
                    nonlocal fail_release
                    async with native_pool.acquire(*args, **kwargs) as conn:
                        yield conn
                    if fail_release:
                        fail_release = False
                        raise OSError("private connection-release failure")

            monkeypatch.setattr(reservation_pool, "pool", ReleaseFailurePool())
        with pytest.raises(ModelCompletionFailure) as error:
            await service.reserve(request(), invocation)
        repo = ProviderUsageReservationsRepo(reservation_pool)
        assert (await repo.get(invocation.execution_id))["state"] == "released"
        assert await repo.outstanding(BillingScope("user", 1)) == {"tokens": 0, "cost_units": 0}
        assert error.value.code == "accounting_unavailable"
        assert error.value.__context__ is None and error.value.__cause__ is None

    @pytest.mark.asyncio
    async def test_duplicate_admission_never_attempts_release_of_existing_row(self, reservation_pool, monkeypatch):
        service = make_service(reservation_pool)
        invocation = identity()
        await service.reserve(request(), invocation)
        releases = []

        async def forbidden_release(execution_id):
            releases.append(execution_id)
            raise AssertionError("Duplicate does not own the existing reservation")

        monkeypatch.setattr(service._reservations, "release_before_dispatch", forbidden_release)
        with pytest.raises(ModelCompletionFailure):
            await service.reserve(request(), invocation)
        assert releases == []
        assert (await ProviderUsageReservationsRepo(reservation_pool).get(invocation.execution_id))[
            "state"
        ] == "reserved"

    @pytest.mark.asyncio
    async def test_cancelled_duplicate_cannot_release_another_invocations_row(self, reservation_pool, monkeypatch):
        service = make_service(reservation_pool)
        invocation = identity()
        await service.reserve(request(), invocation)
        entered = asyncio.Event()
        original_get = service._reservations._get
        block_once = True

        async def blocked_duplicate_read(conn, execution_id):
            nonlocal block_once
            if block_once:
                block_once = False
                entered.set()
                await asyncio.Event().wait()
            return await original_get(conn, execution_id)

        monkeypatch.setattr(service._reservations, "_get", blocked_duplicate_read)
        admission = asyncio.create_task(service.reserve(request(), invocation))
        await asyncio.wait_for(entered.wait(), 5)
        admission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(admission, 5)
        assert (await ProviderUsageReservationsRepo(reservation_pool).get(invocation.execution_id))[
            "state"
        ] == "reserved"

    @pytest.mark.asyncio
    async def test_cancelled_insert_rollback_cleanup_cannot_release_replacement(self, reservation_pool, monkeypatch):
        first_service = make_service(reservation_pool)
        replacement_service = make_service(reservation_pool)
        invocation = identity()
        inserted, cleanup_entered, finish_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_get = first_service._reservations._get
        original_cleanup = first_service._release_pre_dispatch
        acknowledged_owners = []

        async def paused_insert_reread(conn, execution_id):
            row = await original_get(conn, execution_id)
            if row is not None and not inserted.is_set():
                acknowledged_owners.append(row["insertion_owner"])
                inserted.set()
                await asyncio.Event().wait()
            return row

        async def paused_cleanup(*args, **kwargs):
            cleanup_entered.set()
            await finish_cleanup.wait()
            return await original_cleanup(*args, **kwargs)

        monkeypatch.setattr(first_service._reservations, "_get", paused_insert_reread)
        monkeypatch.setattr(first_service, "_release_pre_dispatch", paused_cleanup)
        admission = asyncio.create_task(first_service.reserve(request(), invocation))
        repo = ProviderUsageReservationsRepo(reservation_pool)
        try:
            await asyncio.wait_for(inserted.wait(), 5)
            admission.cancel()
            await asyncio.wait_for(cleanup_entered.wait(), 5)
            # Cleanup is outside the transaction: A's acknowledged insert rolled back.
            assert await repo.get(invocation.execution_id) is None
            replacement = await asyncio.wait_for(replacement_service.reserve(request(), invocation), 5)
            replacement_owner = (await repo.get(invocation.execution_id))["insertion_owner"]
        finally:
            finish_cleanup.set()
            if not admission.done():
                admission.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(admission, 5)
        assert (await repo.get(invocation.execution_id))["state"] == "reserved"
        assert acknowledged_owners[0] != replacement_owner
        for owner in (*acknowledged_owners, replacement_owner):
            assert str(UUID(owner)) == owner and UUID(owner).version == 4
            assert owner != invocation.execution_id
        assert await repo.outstanding(BillingScope("user", 1)) == {"tokens": 47, "cost_units": 104}
        await replacement_service.mark_dispatched(replacement)
        assert (await repo.get(invocation.execution_id))["state"] == "dispatched"

    @pytest.mark.asyncio
    async def test_normal_handle_release_owner_mismatch_retains_exposure_and_governor(self, reservation_pool):
        governor = MemoryResourceGovernor(
            policies={
                "mcp.fixed": {
                    "requests": {"rpm": 1000},
                    "tokens": {"per_min": 100000},
                    "jobs": {"max_concurrent": 1, "ttl_sec": 90},
                    "scopes": ["user"],
                }
            }
        )
        service = make_service(reservation_pool, token_limit=1000, governor=governor)
        handle = await service.reserve(request(), identity())
        repo = ProviderUsageReservationsRepo(reservation_pool)
        owner = (await repo.get(handle.reservation.execution_id))["insertion_owner"]
        wrong_owner = str(uuid4())
        foreign_handle = replace(handle, _insertion_owner=wrong_owner)
        messages = []
        sink = logger.add(messages.append, format="{message}")
        try:
            with pytest.raises(ModelCompletionFailure, match="accounting_unavailable") as error:
                await service.release_before_dispatch(foreign_handle)
            assert (await repo.get(handle.reservation.execution_id))["state"] == "reserved"
            assert await repo.outstanding(BillingScope("user", 1)) == {"tokens": 47, "cost_units": 104}
            # The explicit ownership fence must not refund the live concurrency lease.
            with pytest.raises(ModelCompletionFailure, match="model_quota_exceeded"):
                await service.reserve(request(), identity())
            for token in (owner, wrong_owner):
                assert token not in "".join(messages)
                assert token not in repr(error.value)
                assert token not in repr(foreign_handle)
            assert error.value.__context__ is None and error.value.__cause__ is None
        finally:
            logger.remove(sink)
        await service.release_before_dispatch(handle)

    @pytest.mark.asyncio
    async def test_none_handle_owner_cannot_enter_legacy_wildcard_release(self, reservation_pool, monkeypatch):
        service = make_service(reservation_pool)
        handle = await service.reserve(request(), identity())
        invalid_handle = replace(handle, _insertion_owner=None)
        release = AsyncMock(wraps=service._reservations.release_before_dispatch)
        refund = AsyncMock(wraps=service._governor.release)
        monkeypatch.setattr(service._reservations, "release_before_dispatch", release)
        monkeypatch.setattr(service._governor, "release", refund)
        with pytest.raises(ModelCompletionFailure, match="invalid_accounting_handle"):
            await service.release_before_dispatch(invalid_handle)
        assert release.call_count == refund.call_count == 0
        repo = ProviderUsageReservationsRepo(reservation_pool)
        assert (await repo.get(handle.reservation.execution_id))["state"] == "reserved"
        assert await repo.outstanding(BillingScope("user", 1)) == {"tokens": 47, "cost_units": 104}
        await service.release_before_dispatch(handle)

    @pytest.mark.asyncio
    async def test_hosted_limit_excludes_own_key_pending_and_settled_usage(self, reservation_pool, monkeypatch):
        service = make_service(reservation_pool, token_limit=2000)
        async with reservation_pool.acquire() as conn:
            await execute(
                conn,
                reservation_pool.pool is not None,
                "INSERT INTO llm_usage_log (user_id, billing_org_id, operation, total_tokens, total_cost_usd, credential_source) "
                "VALUES (1, 12, 'chat', 1000, 0, 'user')",
                (),
            )

        async def hosted_limit(self, scope, *, operator_limit, conn):
            assert operator_limit is None
            return 47

        monkeypatch.setattr(BillingEnforcer, "get_mcp_token_limit", hosted_limit)
        own = await service.reserve(
            request(), ModelInvocationIdentity(1, None, 12, str(uuid4())), credential_source="user"
        )
        paid = await service.reserve(
            request(), ModelInvocationIdentity(1, None, 12, str(uuid4())), credential_source="server_default"
        )
        usage = AuthnzUsageRepo(reservation_pool)
        since = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        async with reservation_pool.acquire() as conn:
            assert await usage.read_org_token_exposure(conn, 12, since) == 47
        for handle in (own, paid):
            await service.mark_dispatched(handle)
            assert await service.reconcile(handle, input_tokens=6, output_tokens=4)
        async with reservation_pool.acquire() as conn:
            assert await usage.read_org_token_exposure(conn, 12, since) == 10
        with pytest.raises(ModelCompletionFailure) as error:
            await service.reserve(
                request(), ModelInvocationIdentity(1, None, 12, str(uuid4())), credential_source="server_default"
            )
        assert error.value.code == "model_quota_exceeded"

    @pytest.mark.asyncio
    async def test_explicit_mcp_ceiling_still_counts_own_key_pending_usage(self, reservation_pool):
        service = make_service(reservation_pool, token_limit=47)
        await service.reserve(request(), identity(), credential_source="user")
        with pytest.raises(ModelCompletionFailure) as error:
            await service.reserve(request(), identity(), credential_source="user")
        assert error.value.code == "model_quota_exceeded"

    @pytest.mark.asyncio
    async def test_org_billing_exposure_counts_pending_and_canonical_without_double_counting(self, reservation_pool):
        service = make_service(reservation_pool)
        handle = await service.reserve(request(), ModelInvocationIdentity(1, None, 12, str(uuid4())))
        usage = AuthnzUsageRepo(reservation_pool)
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        async with reservation_pool.acquire() as conn:
            assert await usage.read_org_token_exposure(conn, 12, month_start) == 47
        await service.mark_dispatched(handle)
        assert await service.reconcile(handle, input_tokens=6, output_tokens=4)
        async with reservation_pool.acquire() as conn:
            assert await usage.read_org_token_exposure(conn, 12, month_start) == 10

    @pytest.mark.asyncio
    async def test_canonical_usage_timestamp_is_utc_independent_of_session_timezone(self, reservation_pool):
        repo = ProviderUsageReservationsRepo(reservation_pool)
        usage = AuthnzUsageRepo(reservation_pool)
        item = ProviderUsageReservation(
            str(uuid4()),
            1,
            None,
            None,
            BillingScope("user", 1),
            "openai",
            "fixed",
            1,
            1,
            2,
        )

        async def snapshot(_conn):
            return ReservationQuotaSnapshot(0, None, 0, None)

        async def writer(conn, row, amounts):
            if reservation_pool.pool is not None:
                await execute(conn, True, "SELECT set_config('TimeZone', 'America/Los_Angeles', true)")
            await usage.insert_mcp_completion_usage(conn, row, amounts, estimated=False)

        await repo.reserve(item, snapshot_reader=snapshot)
        await repo.mark_dispatched(item.execution_id)
        before = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)
        await repo.reconcile(item.execution_id, ReservationActuals(1, 1, 2), usage_writer=writer)
        async with reservation_pool.acquire() as conn:
            row = (
                await fetch_all(
                    conn,
                    reservation_pool.pool is not None,
                    "SELECT ts FROM llm_usage_log WHERE request_id = ?",
                    (item.execution_id,),
                    ("ts",),
                )
            )[0]
        timestamp = row["ts"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        assert before <= timestamp <= datetime.now(timezone.utc).replace(tzinfo=None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("settlement", ["release", "ambiguous", "reconcile"])
    async def test_real_governor_concurrency_denies_then_restores_capacity(self, reservation_pool, settlement):
        governor = MemoryResourceGovernor(
            policies={
                "mcp.fixed": {
                    "requests": {"rpm": 1000},
                    "tokens": {"per_min": 100000},
                    "jobs": {"max_concurrent": 1, "ttl_sec": 90},
                    "scopes": ["user"],
                }
            }
        )
        service = make_service(reservation_pool, token_limit=1000, governor=governor)
        handle = await service.reserve(request(), identity())
        with pytest.raises(ModelCompletionFailure, match="model_quota_exceeded"):
            await service.reserve(request(), identity())
        if settlement == "release":
            await service.release_before_dispatch(handle)
        else:
            await service.mark_dispatched(handle)
            if settlement == "ambiguous":
                assert await service.retain_ambiguous(handle)
            else:
                assert await service.reconcile(handle, input_tokens=6, output_tokens=4)
        subsequent = await service.reserve(request(), identity())
        await service.release_before_dispatch(subsequent)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("team,org", [(11, None), (11, 12)])
    async def test_actuals_stay_in_exact_active_scope(self, reservation_pool, team, org):
        service = make_service(reservation_pool)
        invocation = ModelInvocationIdentity(1, team, org, str(uuid4()))
        handle = await service.reserve(request(), invocation)
        await service.mark_dispatched(handle)
        assert await service.reconcile(handle, input_tokens=6, output_tokens=4)
        with pytest.raises(ModelCompletionFailure, match="model_quota_exceeded"):
            await service.reserve(request(), replace(invocation, execution_id=str(uuid4())))
        personal = await service.reserve(request(), identity())
        assert (await canonical_rows(reservation_pool))[0]["billing_org_id"] == org
        await service.release_before_dispatch(personal)

    @pytest.mark.asyncio
    async def test_canonical_cost_read_preserves_exact_integer_actuals(self, reservation_pool):
        repo = ProviderUsageReservationsRepo(reservation_pool)
        usage = AuthnzUsageRepo(reservation_pool)
        item = ProviderUsageReservation(
            str(uuid4()),
            1,
            None,
            None,
            BillingScope("user", 1),
            "openai",
            "fixed",
            1,
            1,
            MAX_INT,
        )

        async def snapshot(_conn):
            return ReservationQuotaSnapshot(0, None, 0, None)

        async def writer(conn, row, amounts):
            await usage.insert_mcp_completion_usage(conn, row, amounts, estimated=False)

        await repo.reserve(item, snapshot_reader=snapshot)
        await repo.mark_dispatched(item.execution_id)
        await repo.reconcile(item.execution_id, ReservationActuals(1, 1, MAX_INT - 1), usage_writer=writer)
        async with reservation_pool.acquire() as conn:
            actuals = await usage.read_provider_scope_usage(
                conn,
                item.billing_scope,
                datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            )
        assert actuals == (2, MAX_INT - 1)

    @pytest.mark.asyncio
    async def test_success_reconciles_once_and_replays_without_double_billing(self, reservation_pool):
        service = make_service(reservation_pool)
        invocation = identity()
        handle = await service.reserve(request(), invocation)
        await service.mark_dispatched(handle)
        for _ in range(2):
            assert await service.reconcile(handle, input_tokens=6, output_tokens=4)
        rows = await canonical_rows(reservation_pool)
        assert rows == [
            {
                "user_id": 1,
                "billing_org_id": None,
                "provider": "openai",
                "model": "fixed",
                "request_id": invocation.execution_id,
                "prompt_tokens": 6,
                "completion_tokens": 4,
                "total_tokens": 10,
                "estimated": False,
            }
        ]
        repo = ProviderUsageReservationsRepo(reservation_pool)
        assert (await repo.get(invocation.execution_id))["state"] == "reconciled"
        assert await repo.outstanding(BillingScope("user", 1)) == {"tokens": 0, "cost_units": 0}

    @pytest.mark.asyncio
    async def test_new_admission_includes_committed_canonical_actuals(self, reservation_pool):
        service = make_service(reservation_pool)
        handle = await service.reserve(request(), identity())
        await service.mark_dispatched(handle)
        assert await service.reconcile(handle, input_tokens=6, output_tokens=4)
        with pytest.raises(ModelCompletionFailure, match="model_quota_exceeded"):
            await service.reserve(request(), identity())

    @pytest.mark.asyncio
    async def test_concurrent_final_capacity_admits_only_one_execution(self, reservation_pool):
        service = make_service(reservation_pool)
        results = await asyncio.gather(
            service.reserve(request(), identity()),
            service.reserve(request(), identity()),
            return_exceptions=True,
        )
        assert sum(not isinstance(result, BaseException) for result in results) == 1
        assert sum(isinstance(result, ModelCompletionFailure) for result in results) == 1
        assert await ProviderUsageReservationsRepo(reservation_pool).outstanding(BillingScope("user", 1)) == {
            "tokens": 47,
            "cost_units": 104,
        }

    @pytest.mark.asyncio
    async def test_writer_failure_rolls_back_actuals_and_retains_worst_case(self, reservation_pool):
        class FailingUsage(AuthnzUsageRepo):
            async def insert_mcp_completion_usage(self, *args, **kwargs):
                await super().insert_mcp_completion_usage(*args, **kwargs)
                raise RuntimeError("private-provider-data")

        service = make_service(reservation_pool, usage=FailingUsage(reservation_pool))
        handle = await service.reserve(request(), identity())
        await service.mark_dispatched(handle)
        assert not await service.reconcile(handle, input_tokens=6, output_tokens=4)
        assert await canonical_rows(reservation_pool) == []
        repo = ProviderUsageReservationsRepo(reservation_pool)
        assert (await repo.get(handle.reservation.execution_id))["state"] == "ambiguous"
        assert await repo.outstanding(BillingScope("user", 1)) == {"tokens": 47, "cost_units": 104}
        with pytest.raises(ModelCompletionFailure, match="model_quota_exceeded"):
            await service.reserve(request(), identity())

    @pytest.mark.asyncio
    async def test_pre_dispatch_release_restores_durable_capacity(self, reservation_pool):
        service = make_service(reservation_pool)
        handle = await service.reserve(request(), identity())
        await service.release_before_dispatch(handle)
        second = await service.reserve(request(), identity())
        assert second.reservation.execution_id != handle.reservation.execution_id
        await service.release_before_dispatch(second)

    @pytest.mark.asyncio
    async def test_foreign_service_handle_cannot_change_durable_state(self, reservation_pool):
        service = make_service(reservation_pool)
        handle = await service.reserve(request(), identity())
        other_service = make_service(reservation_pool)
        with pytest.raises(ModelCompletionFailure, match="invalid_accounting_handle"):
            await other_service.release_before_dispatch(handle)
        forged = replace(handle, _owner=object())
        with pytest.raises(ModelCompletionFailure, match="invalid_accounting_handle"):
            await service.reconcile(forged, input_tokens=1, output_tokens=1)
        assert (await ProviderUsageReservationsRepo(reservation_pool).get(handle.reservation.execution_id))[
            "state"
        ] == "reserved"


class TestSQLiteAccountingStorage(AccountingStorageContract):
    pass
