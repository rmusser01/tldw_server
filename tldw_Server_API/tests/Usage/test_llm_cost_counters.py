"""The LLM cost counters sum stored cost, leave embeddings out and bind naive UTC on Postgres (spec §6)."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.Usage import quota_checks
from tldw_Server_API.tests.Usage.test_usage_tracker_sqlite import _ensure_llm_tables

pytestmark = pytest.mark.unit


async def test_cost_sum_leaves_out_embeddings_older_rows_and_other_users(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Chat 0.25 + NULL-operation 0.05 + NULL cost count; embeddings, an old row and another user do not."""
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.sqlite'}")

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()
    pool = await get_db_pool()
    await _ensure_llm_tables(pool)

    now = datetime.now(timezone.utc)
    recent = now.strftime("%Y-%m-%d %H:%M:%S")
    old = (now - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
    rows = (
        (1, "chat", 0.25, recent),
        (1, None, 0.05, recent),
        (1, "chat", None, recent),
        (1, "embeddings", 9.0, recent),
        (1, "chat", 7.0, old),
        (2, "chat", 3.0, recent),
    )
    for user_id, operation, cost, ts in rows:
        await pool.execute(
            "INSERT INTO llm_usage_log (user_id, operation, total_cost_usd, ts) VALUES (?, ?, ?, ?)",
            user_id, operation, cost, ts,
        )

    since = now - timedelta(days=1)
    assert await AuthnzUsageRepo(pool).sum_user_llm_cost_since(user_id=1, since=since) == pytest.approx(0.30)
    assert await AuthnzUsageRepo(pool).sum_user_llm_cost_since(user_id=3, since=since) == 0.0


class _RecordingPool:
    """A Postgres-shaped pool that records the query and its parameters."""

    pool = object()

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    async def fetchval(self, query: str, *args: object) -> float:
        """Record the call and return a total."""
        self.calls.append((query, args))
        return 1.5


@pytest.mark.parametrize(
    ("counter", "column", "is_month"),
    [
        (quota_checks.llm_cost_today, "total_cost_usd", False),
        (quota_checks.llm_cost_this_month, "total_cost_usd", True),
        (quota_checks.llm_tokens_this_month, "total_tokens", True),
    ],
)
async def test_counters_bind_naive_utc_period_starts(monkeypatch: pytest.MonkeyPatch, counter, column, is_month) -> None:
    """Each counter sums its column from the start of the UTC day or month, bound naive for Postgres."""
    pool = _RecordingPool()

    async def _get_pool() -> _RecordingPool:
        """Hand back the recording pool."""
        return pool

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", _get_pool)
    assert await counter(9) == 1.5
    (query, (user_id, start)), = pool.calls
    today = datetime.now(timezone.utc).date()
    assert f"SUM({column})" in query and "embeddings" in query
    assert user_id == 9 and start.tzinfo is None
    assert (start.hour, start.minute, start.second) == (0, 0, 0)
    assert start.date() == (today.replace(day=1) if is_month else today)
