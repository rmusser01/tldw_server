import pytest
from hypothesis import given, strategies as st, settings

pytestmark = pytest.mark.rate_limit

from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest


class FakeTime:
    def __init__(self, t0: float = 0.0):
        self.t = t0

    def __call__(self) -> float:

        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.mark.asyncio
@settings(deadline=None, max_examples=10)
@given(
    per_min=st.integers(min_value=2, max_value=10),
    first_units=st.integers(min_value=1, max_value=5),
    commit_actual=st.integers(min_value=0, max_value=5),
)
async def test_tokens_refund_parity_and_steady_no_denials(per_min, first_units, commit_actual):
    first_units = min(int(first_units), int(per_min))

    class _Loader:
        def get_policy(self, pid):
            return {"tokens": {"per_min": int(per_min)}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    ns = "rg_p_tok_refund"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    # Ensure a clean slate for sliding-window keys when FakeTime≈0.0
    await rg.test_force_clear_windows(policy_id="p")
    e = "user:tokp"

    # Reserve first batch
    d1, h1 = await rg.reserve(RGRequest(entity=e, categories={"tokens": {"units": first_units}}, tags={"policy_id": "p"}))
    assert d1.allowed and h1

    # Commit with actual less/equal than reserved (never greater)
    actual = min(int(commit_actual), int(first_units))
    await rg.commit(h1, actuals={"tokens": int(actual)})

    # Immediately attempt to consume the remaining budget within the minute
    remaining = max(0, int(per_min) - int(actual))
    if remaining > 0:
        d2, h2 = await rg.reserve(RGRequest(entity=e, categories={"tokens": {"units": int(remaining)}}, tags={"policy_id": "p"}))
        assert d2.allowed and h2

    # Steady rate: 1 token every 10 seconds for per_min>=6 should always pass
    if per_min >= 6:
        ft2 = FakeTime(0.0)
        rg2 = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft2, ns=ns + "_steady")
        for _ in range(12):
            d, h = await rg2.reserve(RGRequest(entity=e, categories={"tokens": {"units": 1}}, tags={"policy_id": "p"}))
            assert d.allowed and h
            ft2.advance(10.0)
