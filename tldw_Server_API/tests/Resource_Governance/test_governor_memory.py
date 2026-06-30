import asyncio
import math
import pytest

from tldw_Server_API.app.core.Resource_Governance import (
    MemoryResourceGovernor,
    RGRequest,
)


class FakeTime:
    def __init__(self, t0: float = 0.0):
        self._t = t0

    def __call__(self) -> float:

        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


@pytest.mark.asyncio
async def test_requests_token_bucket_allow_then_deny_then_allow_after_refill():
    ft = FakeTime(0.0)
    policies = {
        "test.policy": {
            "requests": {"rpm": 2, "burst": 1.0},
            "scopes": ["global", "user"],
        }
    }
    rg = MemoryResourceGovernor(policies=policies, time_source=ft)

    # Two requests allowed within same minute
    req = RGRequest(entity="user:42", categories={"requests": {"units": 1}}, tags={"policy_id": "test.policy"})
    d1, h1 = await rg.reserve(req, op_id="op1")
    assert d1.allowed and h1
    d2, h2 = await rg.reserve(req, op_id="op2")
    assert d2.allowed and h2 and h2 != h1

    # Third should be denied until refill
    d3, h3 = await rg.reserve(req, op_id="op3")
    assert not d3.allowed and h3 is None
    assert d3.retry_after and d3.retry_after > 0

    # Advance 60s → refill to capacity; next allowed
    ft.advance(60.0)
    d4, h4 = await rg.reserve(req, op_id="op4")
    assert d4.allowed and h4


@pytest.mark.asyncio
async def test_reserve_idempotency_returns_same_handle():
    ft = FakeTime(0.0)
    policies = {"p": {"requests": {"rpm": 5, "burst": 1.0}, "scopes": ["global", "user"]}}
    rg = MemoryResourceGovernor(policies=policies, time_source=ft)
    req = RGRequest(entity="user:1", categories={"requests": {"units": 1}}, tags={"policy_id": "p"})
    d1, h1 = await rg.reserve(req, op_id="A")
    d2, h2 = await rg.reserve(req, op_id="A")
    assert d1.allowed and d2.allowed
    assert h1 == h2


@pytest.mark.asyncio
async def test_commit_with_same_op_id_does_not_corrupt_reserve_replay():
    ft = FakeTime(0.0)
    policies = {"p": {"requests": {"rpm": 5, "burst": 1.0}, "scopes": ["user"]}}
    rg = MemoryResourceGovernor(policies=policies, time_source=ft)
    req = RGRequest(entity="user:1", categories={"requests": {"units": 1}}, tags={"policy_id": "p"})

    decision, handle_id = await rg.reserve(req, op_id="shared-op")
    assert decision.allowed and handle_id

    await rg.commit(handle_id, op_id="shared-op")

    replay_decision, replay_handle = await rg.reserve(req, op_id="shared-op")
    assert replay_decision.allowed is True
    assert replay_handle == handle_id


@pytest.mark.asyncio
async def test_concurrency_streams_limit_and_renew_release():
    ft = FakeTime(0.0)
    policies = {"p": {"streams": {"max_concurrent": 1, "ttl_sec": 30}, "scopes": ["global", "user"]}}
    rg = MemoryResourceGovernor(policies=policies, time_source=ft)
    req = RGRequest(entity="user:9", categories={"streams": {"units": 1}}, tags={"policy_id": "p"})

    d1, h1 = await rg.reserve(req, op_id="s1")
    assert d1.allowed and h1

    # Second should be denied while first holds lease
    d2, h2 = await rg.reserve(req, op_id="s2")
    assert not d2.allowed and h2 is None

    # Renew lease keeps it active
    await rg.renew(h1, ttl_s=30)
    ft.advance(20.0)
    d3, h3 = await rg.reserve(req, op_id="s3")
    assert not d3.allowed and h3 is None

    # Release and try again → allowed
    await rg.release(h1)
    d4, h4 = await rg.reserve(req, op_id="s4")
    assert d4.allowed and h4


@pytest.mark.asyncio
async def test_concurrency_streams_units_enforced():
    ft = FakeTime(0.0)
    policies = {"p": {"streams": {"max_concurrent": 2, "ttl_sec": 30}, "scopes": ["user"]}}
    rg = MemoryResourceGovernor(policies=policies, time_source=ft)
    req_two = RGRequest(entity="user:9", categories={"streams": {"units": 2}}, tags={"policy_id": "p"})
    req_one = RGRequest(entity="user:9", categories={"streams": {"units": 1}}, tags={"policy_id": "p"})

    d1, h1 = await rg.reserve(req_two, op_id="u1")
    assert d1.allowed and h1

    d2, h2 = await rg.reserve(req_one, op_id="u2")
    assert not d2.allowed and h2 is None

    await rg.release(h1)
    d3, h3 = await rg.reserve(req_one, op_id="u3")
    assert d3.allowed and h3


@pytest.mark.asyncio
async def test_commit_refund_difference_returns_tokens():
    ft = FakeTime(0.0)
    # tokens per minute 1000; reserve 800 and commit 200 → refund 600
    policies = {"p": {"tokens": {"per_min": 1000, "burst": 1.0}, "scopes": ["global", "user"]}}
    rg = MemoryResourceGovernor(policies=policies, time_source=ft)
    req = RGRequest(entity="user:5", categories={"tokens": {"units": 800}}, tags={"policy_id": "p"})
    d1, h1 = await rg.reserve(req, op_id="t1")
    assert d1.allowed and h1

    # Commit with fewer actual tokens; should refund the difference
    await rg.commit(h1, actuals={"tokens": 200}, op_id="t1c")

    # Immediately reserve more, should have at least 600 capacity restored
    req2 = RGRequest(entity="user:5", categories={"tokens": {"units": 600}}, tags={"policy_id": "p"})
    d2, h2 = await rg.reserve(req2, op_id="t2")
    assert d2.allowed and h2
