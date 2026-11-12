"""check_usage fails open on counter errors, and the ledger is cached (spec 2 review A3)."""

import pytest

from tldw_Server_API.app.core.Usage import quota_checks

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_ledger_cache() -> None:
    """Leave the module-level ledger cache as it was found."""
    quota_checks.reset_ledger_cache()
    yield
    quota_checks.reset_ledger_cache()


async def test_check_usage_fails_open_when_the_counter_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A counter() that raises must allow the request (fail open), not 500 it."""

    async def _limit(_uid: int, _key: str) -> int:
        """A real limit is set, so the counter would normally run."""
        return 10

    async def _raising_counter() -> float:
        """Simulates a ledger/DB hiccup."""
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(quota_checks, "user_quota", _limit)
    decision = await quota_checks.check_usage(7, "limits.rag_queries_per_day", 1, _raising_counter)
    assert decision.allowed is True


async def test_ledger_is_cached_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two _ledger() calls return the same instance; initialize() runs once."""
    created: list[object] = []
    init_calls: list[object] = []

    class _FakeLedger:
        """Records construction and initialize() calls."""

        def __init__(self) -> None:
            created.append(self)

        async def initialize(self) -> None:
            init_calls.append(self)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger.ResourceDailyLedger",
        _FakeLedger,
    )

    first = await quota_checks._ledger()
    second = await quota_checks._ledger()

    assert first is second
    assert len(created) == 1
    assert len(init_calls) == 1
