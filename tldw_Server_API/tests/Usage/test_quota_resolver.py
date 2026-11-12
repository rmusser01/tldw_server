"""Usage-quota precedence, resolver and checks (spec 2 §2, §4)."""

from datetime import datetime

import pytest

from tldw_Server_API.app.core.Usage import quota_checks, quota_resolver
from tldw_Server_API.app.core.UserProfiles.limits_precedence import effective_limits, most_generous_values

pytestmark = pytest.mark.unit


def _row(key: str, value: object, **ids: int) -> dict:
    """An override row as the overrides repos return it."""
    return {"key": key, "value": value, **ids}


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with an empty resolver cache and quotas switched on."""
    quota_resolver.invalidate_all()
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")


def test_most_generous_value_wins_among_rows_that_set_the_key() -> None:
    """The largest numeric value per limits.* key wins; other keys and non-numbers are ignored."""
    rows = [
        _row("limits.rag_queries_per_day", 10, team_id=1),
        _row("limits.rag_queries_per_day", 50, team_id=2),
        _row("preferences.ui.theme", "dark", team_id=1),
        _row("limits.audio_daily_minutes", "lots", team_id=1),
        _row("limits.audio_daily_minutes", True, team_id=2),
    ]
    assert most_generous_values(rows) == {"limits.rag_queries_per_day": 50}


def test_team_beats_org_even_when_org_is_larger() -> None:
    """Team values outrank org values; the user's own value outranks both, even when lower."""
    teams = [_row("limits.rag_queries_per_day", 10, team_id=1), _row("limits.rag_queries_per_day", 50, team_id=2)]
    orgs = [_row("limits.rag_queries_per_day", 100, org_id=1), _row("limits.audio_daily_minutes", 30, org_id=1)]
    assert effective_limits([], teams, orgs) == {"limits.rag_queries_per_day": 50, "limits.audio_daily_minutes": 30}
    user = [_row("limits.rag_queries_per_day", 5)]
    assert effective_limits(user, teams, orgs)["limits.rag_queries_per_day"] == 5


def test_team_without_the_key_does_not_unlimit() -> None:
    """A team that never set the key leaves the org value in force."""
    teams = [_row("limits.audio_daily_minutes", 10, team_id=1)]
    orgs = [_row("limits.rag_queries_per_day", 7, org_id=1)]
    assert effective_limits([], teams, orgs)["limits.rag_queries_per_day"] == 7


async def test_resolver_returns_none_when_quotas_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the switch off the resolver never touches the DB and reports unlimited."""
    loads: list[int] = []

    async def _load(user_id: int) -> dict:
        """Record that a DB load happened."""
        loads.append(user_id)
        return {"limits.rag_queries_per_day": 1}

    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)
    monkeypatch.setattr(quota_resolver, "_load_limits", _load)
    assert await quota_resolver.user_quota(1, "limits.rag_queries_per_day") is None
    assert loads == []


async def test_resolver_caches_until_invalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    """One DB load serves repeat lookups; invalidating the user forces a reload."""
    loads: list[int] = []

    async def _load(user_id: int) -> dict:
        """Record each DB load and return a fixed limit."""
        loads.append(user_id)
        return {"limits.rag_queries_per_day": 3}

    monkeypatch.setattr(quota_resolver, "_load_limits", _load)
    assert await quota_resolver.user_quota(1, "limits.rag_queries_per_day") == 3
    assert await quota_resolver.user_quota(1, "limits.audio_daily_minutes") is None
    assert loads == [1]
    quota_resolver.invalidate_user(1)
    await quota_resolver.user_quota(1, "limits.rag_queries_per_day")
    assert loads == [1, 1]


async def test_lookup_failure_fails_open_and_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing lookup returns unlimited and is retried on the next call."""
    calls: list[int] = []

    async def _boom(user_id: int) -> dict:
        """Fail like a broken DB."""
        calls.append(user_id)
        raise RuntimeError("db down")

    monkeypatch.setattr(quota_resolver, "_load_limits", _boom)
    assert await quota_resolver.user_quota(1, "limits.rag_queries_per_day") is None
    assert await quota_resolver.user_quota(1, "limits.rag_queries_per_day") is None
    assert calls == [1, 1]


async def test_counter_runs_only_when_a_limit_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlimited users cost no counter query; limited users are compared against it."""
    counted: list[str] = []

    async def _counter() -> float:
        """Record that the counter ran."""
        counted.append("ran")
        return 4.0

    async def _no_limit(_user_id: int, _key: str) -> None:
        """No limit set for this user."""
        return None

    monkeypatch.setattr(quota_checks, "user_quota", _no_limit)
    assert (await quota_checks.check_usage(1, "limits.x", 1, _counter)).allowed is True
    assert counted == []

    async def _limit_five(_user_id: int, _key: str) -> int:
        """A limit of five."""
        return 5

    monkeypatch.setattr(quota_checks, "user_quota", _limit_five)
    assert (await quota_checks.check_usage(1, "limits.x", 1, _counter)).allowed is True
    assert (await quota_checks.check_usage(1, "limits.x", 2, _counter)).allowed is False
    assert counted == ["ran", "ran"]


async def test_zero_limit_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A limit of 0 means none allowed, not unlimited."""

    async def _zero(_user_id: int, _key: str) -> int:
        """A zero limit."""
        return 0

    async def _none_used() -> float:
        """Nothing used yet."""
        return 0.0

    monkeypatch.setattr(quota_checks, "user_quota", _zero)
    decision = await quota_checks.check_usage(1, "limits.x", 1, _none_used)
    assert decision.allowed is False and decision.limit == 0


def test_as_quota_user_id_tolerates_non_numeric_ids() -> None:
    """Numeric ids become ints; anything else becomes None, which means unlimited."""
    assert quota_checks.as_quota_user_id("7") == 7
    assert quota_checks.as_quota_user_id(7) == 7
    assert quota_checks.as_quota_user_id("single-user") is None
    assert quota_checks.as_quota_user_id(None) is None


async def test_llm_tokens_month_query_passes_naive_utc_on_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    """Postgres stores ts as TIMESTAMP without time zone, so the bound must be naive (TASK-13433 class)."""
    seen: dict = {}

    class _Pool:
        """A Postgres-shaped pool that records the query parameters."""

        pool = object()

        async def fetchval(self, query: str, *args: object) -> int:
            """Record the call and return a token total."""
            seen["args"] = args
            return 42

    async def _get_pool() -> _Pool:
        """Hand back the recording pool."""
        return _Pool()

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", _get_pool)
    assert await quota_checks.llm_tokens_this_month(9) == 42.0
    user_id, month_start = seen["args"]
    assert user_id == 9 and isinstance(month_start, datetime) and month_start.tzinfo is None and month_start.day == 1


def test_seconds_until_utc_midnight_is_positive() -> None:
    """Retry-After is always at least one second."""
    assert 1 <= quota_checks.seconds_until_utc_midnight() <= 86400


def test_seconds_until_utc_month_start_is_positive() -> None:
    """Retry-After for the monthly LLM-token quota is at least one second, at most a month."""
    assert 1 <= quota_checks.seconds_until_utc_month_start() <= 31 * 86400
