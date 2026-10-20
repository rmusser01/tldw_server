"""On real Redis a request's own max_concurrent holds across workers (TASK-13435)."""

import pytest

from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest

pytestmark = pytest.mark.rate_limit


class _NoJobsBlock:
    """A policy without a jobs block, as the usage-quota lease store uses."""

    def get_policy(self, _policy_id: str) -> dict:
        """The same block-less policy for any id."""
        return {"scopes": ["user"]}


def _lease() -> RGRequest:
    """One jobs lease held to a limit of 1."""
    return RGRequest(
        entity="user:42",
        categories={"jobs": {"units": 1, "max_concurrent": 1, "ttl_sec": 60}},
        tags={"policy_id": "limits.test_concurrent"},
    )


async def test_request_limit_holds_across_two_workers(real_redis, rg_unique_ns) -> None:
    """Two governors on one Redis namespace (two workers) share the user's limit; a release frees it for the other."""
    worker_a = RedisResourceGovernor(policy_loader=_NoJobsBlock(), ns=rg_unique_ns)
    worker_b = RedisResourceGovernor(policy_loader=_NoJobsBlock(), ns=rg_unique_ns)
    if not await worker_a._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    d1, h1 = await worker_a.reserve(_lease())
    d2, h2 = await worker_b.reserve(_lease())
    assert d1.allowed and h1
    assert not d2.allowed and h2 is None
    await worker_a.release(h1)
    d3, h3 = await worker_b.reserve(_lease())
    assert d3.allowed and h3


class _Policy:
    """One policy for any id: ``body``, with the scope and fail mode the test names."""

    def __init__(self, fail_mode: str, **body: dict) -> None:
        """Remember the policy's fail mode and its category blocks."""
        self._policy = {"scopes": ["user"], "fail_mode": fail_mode, **body}

    def get_policy(self, _policy_id: str) -> dict:
        """The same policy for any id."""
        return dict(self._policy)


# What a Redis restart or failover leaves in a worker: a cached SHA the server no longer knows (NOSCRIPT).
# Never SCRIPT FLUSH a shared Redis to get there.
_LOST_SHA = "f" * 40


@pytest.mark.parametrize("fail_mode", ["fallback_memory", "fail_closed", "fail_open"])
async def test_a_lost_lease_script_falls_back_to_counting_in_python(real_redis, rg_unique_ns, fail_mode) -> None:
    """A lease reserve that cannot run its script still counts leases, then loads the script again (TASK-13859)."""
    gov = RedisResourceGovernor(policy_loader=_Policy(fail_mode), ns=rg_unique_ns)
    if not await gov._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    lease = RGRequest(
        entity="user:42",
        categories={"jobs": {"units": 1, "max_concurrent": 2, "ttl_sec": 60}},
        tags={"policy_id": "limits.test_concurrent"},
    )
    first, _ = await gov.reserve(lease)  # loads the script
    gov._concurrency_lua_sha = _LOST_SHA

    second, _ = await gov.reserve(lease)
    third, _ = await gov.reserve(lease)

    assert [first.allowed, second.allowed, third.allowed] == [True, True, False]


async def test_a_lost_rate_script_falls_back_once_then_reloads(real_redis, rg_unique_ns) -> None:
    """A rate reserve that cannot run its script uses the Python windows once, then loads the script again (TASK-13859)."""
    gov = RedisResourceGovernor(policy_loader=_Policy("fallback_memory", requests={"rpm": 5}), ns=rg_unique_ns)
    if not await gov._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    req = RGRequest(entity="user:42", categories={"requests": {"units": 1}}, tags={"policy_id": "rate.test"})
    first, _ = await gov.reserve(req)  # loads the script
    gov._multi_lua_sha = _LOST_SHA

    second, _ = await gov.reserve(req)
    third, _ = await gov.reserve(req)

    assert [first.allowed, second.allowed, third.allowed] == [True, True, True]
    assert gov._multi_lua_sha not in (None, _LOST_SHA)
