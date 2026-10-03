"""Admission and conservative settlement contracts for bounded completions."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionFailure,
    ModelCompletionRequest,
    ModelFailureDomain,
    ModelInvocationIdentity,
)


def request(system="sys", user="\u00fc", output=10):
    return ModelCompletionRequest(system, user, output, 100, 200, 2000)


def identity(team=None, org=None):
    return ModelInvocationIdentity(7, team, org, str(uuid4()))


@pytest.fixture
def accounting():
    from tldw_Server_API.app.core.MCP_unified.adapters.model_completion.accounting import (
        CompletionAccountingPolicy,
        ModelCompletionAccounting,
    )

    events = []

    class Repo:
        failure = None
        release_failure = None
        row = None
        state = None
        actuals = None

        async def reserve(self, reservation, *, snapshot_reader):
            if self.failure:
                raise self.failure
            self.snapshot = await snapshot_reader("locked-connection")
            self.row = reservation
            self.state = "reserved"
            events.append("durable_reserved")

        async def mark_dispatched(self, execution_id):
            self.state = "dispatched"

        async def release_before_dispatch(self, execution_id):
            if self.release_failure:
                raise self.release_failure
            self.state = "released"
            events.append("durable_released")

        async def retain_ambiguous(self, execution_id):
            self.state = "ambiguous"
            events.append("durable_ambiguous")

        async def reconcile(self, execution_id, actuals, *, usage_writer):
            if self.failure:
                raise self.failure
            if self.state != "dispatched":
                raise RuntimeError("late settlement denied")
            await usage_writer("locked-connection", self.row, actuals)
            self.actuals = actuals
            self.state = "reconciled"

    class Usage:
        async def read_provider_scope_usage(self, conn, scope, since):
            assert conn == "locked-connection"
            events.append("canonical_usage")
            return 0, 0

        async def insert_mcp_completion_usage(self, conn, reservation, actuals, **kwargs):
            assert conn == "locked-connection"
            self.estimated = kwargs["estimated"]
            events.append("canonical_inserted")

    class Billing:
        async def get_mcp_token_limit(self, scope, *, operator_limit, conn):
            assert conn == "locked-connection"
            events.append("fresh_limits")
            return operator_limit

    class Governor:
        allowed = True
        failure = None
        cancelled = False

        async def reserve(self, req, op_id=None):
            events.append("governor_reserved")
            self.request = req
            if self.cancelled:
                raise asyncio.CancelledError
            if self.failure:
                raise self.failure
            return SimpleNamespace(allowed=self.allowed), "governor-handle" if self.allowed else None

        async def release(self, handle):
            events.append("governor_released")

        async def commit(self, handle, actuals=None, op_id=None):
            self.actuals = actuals
            events.append("governor_committed")
            if self.failure:
                raise self.failure

    policy = CompletionAccountingPolicy(
        provider="openai",
        model="fixed",
        input_cost_units_per_token=2,
        output_cost_units_per_token=3,
        monthly_token_limit=1000,
        monthly_cost_limit=2000,
        governor_policy_id="mcp.default",
    )
    repo, usage, billing, governor = Repo(), Usage(), Billing(), Governor()
    service = ModelCompletionAccounting(
        policy=policy, reservations=repo, usage=usage, billing=billing, governor=governor
    )
    return SimpleNamespace(service=service, repo=repo, usage=usage, governor=governor, events=events)


@pytest.mark.asyncio
@pytest.mark.parametrize("team,org,kind,value", [(None, None, "user", 7), (11, None, "team", 11), (11, 13, "org", 13)])
async def test_reserves_worst_case_before_governor_with_exact_scope(accounting, team, org, kind, value):
    handle = await accounting.service.reserve(request(), identity(team, org))
    row = handle.reservation
    assert (row.billing_scope.kind, row.billing_scope.value) == (kind, value)
    assert (row.reserved_input_tokens, row.reserved_output_tokens, row.reserved_cost_units) == (37, 10, 104)
    assert accounting.events == ["fresh_limits", "canonical_usage", "durable_reserved", "governor_reserved"]


@pytest.mark.asyncio
async def test_storage_failure_cannot_reach_governor_or_leak_exception(accounting):
    accounting.repo.failure = RuntimeError("private-prompt-and-key")
    with pytest.raises(ModelCompletionFailure) as exc:
        await accounting.service.reserve(request(), identity())
    assert exc.value.domain is ModelFailureDomain.SHARED_INFRASTRUCTURE
    assert exc.value.__cause__ is None and exc.value.__context__ is None
    assert "private" not in str(exc.value)
    assert accounting.events == []


@pytest.mark.asyncio
async def test_quota_denial_is_request_local(accounting):
    from tldw_Server_API.app.core.AuthNZ.repos.provider_usage_reservations_repo import ProviderUsageReservationError

    accounting.repo.failure = ProviderUsageReservationError("quota_exceeded")
    with pytest.raises(ModelCompletionFailure) as exc:
        await accounting.service.reserve(request(), identity())
    assert exc.value.domain is ModelFailureDomain.REQUEST


@pytest.mark.asyncio
async def test_governor_denial_releases_durable_pre_dispatch_reservation(accounting):
    accounting.governor.allowed = False
    with pytest.raises(ModelCompletionFailure) as exc:
        await accounting.service.reserve(request(), identity())
    assert exc.value.domain is ModelFailureDomain.REQUEST
    assert accounting.repo.state == "released"


@pytest.mark.asyncio
async def test_cancellation_during_governor_admission_releases_before_propagation(accounting):
    accounting.governor.cancelled = True
    with pytest.raises(asyncio.CancelledError):
        await accounting.service.reserve(request(), identity())
    assert accounting.repo.state == "released"


@pytest.mark.asyncio
async def test_release_is_explicitly_pre_dispatch_and_releases_governor(accounting):
    handle = await accounting.service.reserve(request(), identity())
    await accounting.service.release_before_dispatch(handle)
    assert accounting.repo.state == "released"
    assert accounting.events[-2:] == ["durable_released", "governor_released"]


@pytest.mark.asyncio
async def test_release_attempts_governor_cleanup_when_durable_storage_fails(accounting):
    handle = await accounting.service.reserve(request(), identity())
    accounting.repo.release_failure = RuntimeError("private-storage-data")
    with pytest.raises(ModelCompletionFailure, match="accounting_unavailable"):
        await accounting.service.release_before_dispatch(handle)
    assert "governor_released" in accounting.events
    assert accounting.repo.state == "reserved"


@pytest.mark.asyncio
async def test_dispatch_attempt_blocks_refund_even_when_storage_is_unavailable(accounting):
    handle = await accounting.service.reserve(request(), identity())
    await accounting.service.mark_dispatched(handle)
    accounting.repo.release_failure = RuntimeError("private-storage-data")
    with pytest.raises(ModelCompletionFailure):
        await accounting.service.release_before_dispatch(handle)
    assert "governor_released" not in accounting.events
    assert accounting.repo.state == "dispatched"


@pytest.mark.asyncio
async def test_degraded_release_racing_dispatch_blocks_new_dispatch(accounting, monkeypatch):
    handle = await accounting.service.reserve(request(), identity())
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def delayed_failure(execution_id):
        entered.set()
        await proceed.wait()
        raise RuntimeError("private-storage-data")

    monkeypatch.setattr(accounting.repo, "release_before_dispatch", delayed_failure)
    cleanup = asyncio.create_task(accounting.service.release_before_dispatch(handle))
    await entered.wait()
    with pytest.raises(ModelCompletionFailure, match="invalid_accounting_state"):
        await accounting.service.mark_dispatched(handle)
    proceed.set()
    with pytest.raises(ModelCompletionFailure, match="accounting_unavailable"):
        await cleanup
    assert "governor_released" in accounting.events


@pytest.mark.asyncio
async def test_failed_release_permanently_blocks_dispatch_on_refunded_handle(accounting):
    handle = await accounting.service.reserve(request(), identity())
    accounting.repo.release_failure = RuntimeError("private-storage-data")
    with pytest.raises(ModelCompletionFailure, match="accounting_unavailable"):
        await accounting.service.release_before_dispatch(handle)
    with pytest.raises(ModelCompletionFailure, match="invalid_accounting_state"):
        await accounting.service.mark_dispatched(handle)
    assert accounting.repo.state == "reserved"


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_pre_dispatch_release(accounting, monkeypatch):
    handle = await accounting.service.reserve(request(), identity())
    entered, proceed = asyncio.Event(), asyncio.Event()
    original = accounting.repo.release_before_dispatch

    async def delayed_release(execution_id):
        entered.set()
        await proceed.wait()
        await original(execution_id)

    monkeypatch.setattr(accounting.repo, "release_before_dispatch", delayed_release)
    task = asyncio.create_task(accounting.service.release_before_dispatch(handle))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert accounting.repo.state == "released"
    assert "governor_released" in accounting.events


@pytest.mark.asyncio
async def test_ambiguous_retains_ceiling_but_releases_concurrency(accounting):
    handle = await accounting.service.reserve(request(), identity())
    await accounting.service.mark_dispatched(handle)
    await accounting.service.retain_ambiguous(handle)
    assert accounting.repo.state == "ambiguous"
    assert accounting.governor.actuals is None


@pytest.mark.asyncio
async def test_valid_counts_reconcile_canonical_usage_in_same_transaction(accounting):
    handle = await accounting.service.reserve(request(), identity())
    await accounting.service.mark_dispatched(handle)
    assert await accounting.service.reconcile(handle, input_tokens=6, output_tokens=4)
    assert (
        accounting.repo.actuals.input_tokens,
        accounting.repo.actuals.output_tokens,
        accounting.repo.actuals.cost_units,
    ) == (6, 4, 24)
    assert not accounting.usage.estimated
    assert accounting.repo.state == "reconciled"


@pytest.mark.asyncio
@pytest.mark.parametrize("input_tokens,output_tokens", [(None, None), (True, 1), (1, -1), (38, 1), (1, 11), (1.0, 1)])
async def test_untrusted_usage_counts_use_conservative_ceiling(accounting, input_tokens, output_tokens):
    handle = await accounting.service.reserve(request(), identity())
    await accounting.service.mark_dispatched(handle)
    assert await accounting.service.reconcile(handle, input_tokens=input_tokens, output_tokens=output_tokens)
    assert (
        accounting.repo.actuals.input_tokens,
        accounting.repo.actuals.output_tokens,
        accounting.repo.actuals.cost_units,
    ) == (37, 10, 104)
    assert accounting.usage.estimated


@pytest.mark.asyncio
async def test_settlement_failure_does_not_turn_valid_output_into_public_error(accounting):
    handle = await accounting.service.reserve(request(), identity())
    await accounting.service.mark_dispatched(handle)
    accounting.repo.failure = RuntimeError("private-accounting-data")
    assert not await accounting.service.reconcile(handle, input_tokens=6, output_tokens=4)
    assert accounting.repo.state == "ambiguous"
    assert "canonical_inserted" not in accounting.events


@pytest.mark.asyncio
async def test_late_reconciliation_cannot_publish_after_ambiguity(accounting):
    handle = await accounting.service.reserve(request(), identity())
    await accounting.service.mark_dispatched(handle)
    await accounting.service.retain_ambiguous(handle)
    assert not await accounting.service.reconcile(handle, input_tokens=6, output_tokens=4)
    assert "canonical_inserted" not in accounting.events


@pytest.mark.asyncio
async def test_governor_commit_failure_preserves_conservative_reservation(accounting):
    handle = await accounting.service.reserve(request(), identity())
    await accounting.service.mark_dispatched(handle)
    accounting.governor.failure = RuntimeError("private-governor-data")
    assert not await accounting.service.reconcile(handle, input_tokens=6, output_tokens=4)
    assert accounting.repo.state == "ambiguous"


@pytest.mark.asyncio
@pytest.mark.parametrize("system,user,output", [("\ud800", "ok", 10), ("ok", "ok", (1 << 63) - 1)])
async def test_invalid_or_overflowing_exposure_cannot_admit(accounting, system, user, output):
    with pytest.raises(ModelCompletionFailure) as exc:
        await accounting.service.reserve(request(system, user, output), identity())
    assert exc.value.domain is ModelFailureDomain.REQUEST
    assert accounting.events == []
