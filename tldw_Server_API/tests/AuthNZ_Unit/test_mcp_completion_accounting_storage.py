"""End-to-end admission and settlement contracts on real accounting storage."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

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
from tldw_Server_API.tests.AuthNZ_Unit.test_provider_usage_reservations_repo import (
    reservation_pool as reservation_pool,
)
from tldw_Server_API.tests.AuthNZ_Unit.test_provider_usage_reservations_repo import (
    reservation_schema as reservation_schema,
)

pytestmark = pytest.mark.unit


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
