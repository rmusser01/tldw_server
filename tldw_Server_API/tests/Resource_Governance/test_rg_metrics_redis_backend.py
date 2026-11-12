import uuid

import pytest

pytestmark = pytest.mark.rate_limit

from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry


@pytest.mark.asyncio
async def test_redis_backend_metrics_allow_deny_refund_paths():
    class _Loader:
        def get_policy(self, pid):
            return {"requests": {"rpm": 1}, "tokens": {"per_min": 2}, "scopes": ["global", "user"]}

    # The factory uses a reachable local Redis (else the in-memory stub). Real time
    # makes test_force_clear_windows a no-op, so a fixed namespace would inherit the
    # previous run's window members and deny the first reserve: use a fresh one.
    rg = RedisResourceGovernor(policy_loader=_Loader(), ns=f"rg_m_redis_{uuid.uuid4().hex[:8]}")
    reg = get_metrics_registry()

    # Baselines filtered by backend=redis
    before_dec = reg.get_metric_stats("rg_decisions_total", labels={"backend": "redis"})
    before_den = reg.get_metric_stats("rg_denials_total")
    before_ref = reg.get_metric_stats("rg_refunds_total")

    e = "user:rmet"
    # Allow tokens once, then deny next combined (requests)
    d1, h1 = await rg.reserve(RGRequest(entity=e, categories={"tokens": {"units": 2}}, tags={"policy_id": "p"}))
    assert d1.allowed and h1
    d2, h2 = await rg.reserve(RGRequest(entity=e, categories={"requests": {"units": 1}}, tags={"policy_id": "p"}))
    assert d2.allowed and h2
    # Deny next request (rpm=1)
    d3, h3 = await rg.reserve(RGRequest(entity=e, categories={"requests": {"units": 1}}, tags={"policy_id": "p"}))
    assert not d3.allowed and h3 is None

    # Trigger refund by committing fewer tokens than reserved on the first handle
    await rg.commit(h1, actuals={"tokens": 1})

    # Post metrics
    after_dec = reg.get_metric_stats("rg_decisions_total", labels={"backend": "redis"})
    after_den = reg.get_metric_stats("rg_denials_total")
    after_ref = reg.get_metric_stats("rg_refunds_total")

    # Ensure counters increased (allow + allow + deny) and refunds observed
    if before_dec:
        assert after_dec["count"] >= before_dec["count"] + 3
    else:
        assert after_dec["count"] >= 3
    if before_den:
        assert after_den["count"] >= before_den["count"] + 1
    else:
        assert after_den["count"] >= 1
    if before_ref:
        assert after_ref["count"] >= before_ref["count"] + 1
    else:
        assert after_ref["count"] >= 1
