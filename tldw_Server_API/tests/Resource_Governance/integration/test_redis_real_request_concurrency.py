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
