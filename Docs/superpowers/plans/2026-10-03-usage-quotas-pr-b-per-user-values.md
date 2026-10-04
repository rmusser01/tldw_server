# Usage Quotas PR B (Per-User Values) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With `USAGE_QUOTAS_ENABLED` on, every non-storage usage quota applies only to the users, teams and orgs a platform admin assigned it to. Values are UserProfiles `limits.*` keys, and an unset key means unlimited.

**Architecture:**
- **Precedence helper.** `core/UserProfiles/limits_precedence.py` applies one rule: the user's own value wins, else the most generous team value among teams that set the key, else the most generous org value.
- **Resolver.** `core/Usage/quota_resolver.user_quota(user_id, key)` caches each user's effective limits for 60 s and returns `None` for unlimited.
- **Checks.** `core/Usage/quota_checks.check_usage(...)` runs a site's counter only when a limit applies. Each enforcement site calls it and raises its existing error.
- **Writes.** Admins set values through the existing user-profile PATCH and new platform-admin team/org override routes; `null` deletes.

**Tech Stack:** Python 3.12, FastAPI, pytest (`asyncio_mode = "auto"`), loguru, PyYAML, the Backlog CLI, `gh`.

**Spec:** `Docs/Design/2026-10-02-usage-quota-posture-design.md`, PR B of four. PR A (#3098) is merged. Storage (PR C) and reporting, docs and ADR (PR D) get their own plans.

## Global Constraints

- **Branches and PRs:** all PRs target `dev`, never `main`. Never use `git stash`; never pass `--no-verify`.
- **Commit and PR text:**
  - Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and include the owner's waiver line: `> **Waived by the repository owner (@rmusser01) on 2026-09-22**, by explicit instruction in the session that produced this PR.`
- **Backlog:** the Backlog CLI is the ledger. Never hand-edit task files. Use `--append-notes`, never `--notes`.
- **Code conventions:** loguru only, and no new dependencies. New tests carry `pytestmark = pytest.mark.unit`, plus a one-line docstring per test and helper.
- **Running tests:**
  - Use `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (there is no bare `python`), with `TLDW_TEST_NO_DOCKER=1` for suites.
  - A failure counts as environmental only if it also occurs on `origin/dev`.
- **The test session runs with quotas ON:** the root conftest sets `LIMIT_ENFORCEMENT_ENABLED=true`. Tests that need quotas off clear both `USAGE_QUOTAS_ENABLED` and `LIMIT_ENFORCEMENT_ENABLED`.
- **Fakes that can't fail.** Never assert with a fake that *raises* inside a module whose noncritical-exception tuple contains `AssertionError`: audio_quota, workflows, billing enforcement, user_rate_limiter, and any broad `except Exception`. Use call-recording fakes and assert on the record.
- **Gate the check, never the record.** Usage counters are written whether or not quotas are on.
- **Key catalog.** Every `limits.*` key: `default: null`, `minimum: 0`, `editable_by: [platform_admin]`. Months are calendar months in UTC.
- **Storage stays as it is in this PR.** `limits.storage_quota_mb` keeps its current users-column write path, and group overrides of it are rejected until PR C.
- **Mirrored docs.** Any edit to a doc mirrored under `Docs/Published` needs `bash Helper_Scripts/refresh_docs_published.sh`, with the Published diff in the same commit.
- **Merging.** Merge into `dev` only after the peer session `tldw-server-03` gives the go-ahead, and message it when the merge lands. No hosted deployments exist, so there are no deploy prerequisites.

## Rulings carried into this plan

- **Deferred to follow-up tasks** (spec Non-goals, review ruling R1):
  - per-user concurrency on synchronous paths (media ingest, audio streams, synchronous transcription);
  - evaluation cost caps;
  - LLM-token gating outside `/chat/completions`.

  Those paths become unlimited: they stop reserving RG leases.
- **Evaluations.** The `evaluations` RG category carries the per-minute rate as well as any `daily_cap`, so it keeps being sent. An operator on `RG_POLICY_STORE=db` whose stored `evals.*` policy has a `daily_cap` keeps that cap until they edit the stored policy; this is documented.
- **Chatbooks.** The `QuotaManager` tier tables stay in this PR, because `check_file_size` and the usage summary still read them. Admission stops reading them. PR D deletes the tables along with the reporting change.
- **Audio.** `TIER_LIMITS` stays, because the deprecated tier admin API validates tier names against it. `get_limits_for_user` stops reading it.

## Review Focus

These are inputs the spec implies but no task's main tests exercise. Each one has a pinning test, named at the end of its line.

1. **A team value beats an org value even when the org's is larger.** With teams set to 10 and 50 and the org to 100, the effective value is 50. Task 1, `test_team_beats_org_even_when_org_is_larger`.
2. **A value of 0 blocks.** It does not mean unlimited. Task 1, `test_zero_limit_blocks`.
3. **A broken DB during a quota lookup fails open:** the request proceeds unlimited and a warning is logged. Task 1, `test_lookup_failure_fails_open_and_is_not_cached`.
4. **An org admin who is not a platform admin cannot set any `limits.*` value,** through either the profile PATCH or the group routes. Tasks 2 and 3, `test_org_admin_cannot_edit_limits` and `test_group_override_requires_platform_admin`.
5. **Deleting a team override drops the member back to the org value,** not to unlimited. Task 3, `test_deleting_team_override_falls_back_to_org`.

---

## PR B — Per-user values, group routes, enforcement

Branch: `fix/usage-quotas-per-user` from `origin/dev`.

### Task 0: Ledger and branch

**Files:** none (git and the Backlog CLI).

- [ ] **Step 1: Create the branch**

```bash
git fetch origin
git checkout -b fix/usage-quotas-per-user origin/dev
```

- [ ] **Step 2: Record PR A on the parent task**

```bash
backlog task edit 13434 --check-ac 1 --append-notes "PR A merged as #3098 (ea1eda99f9) on 2026-10-03. PR B in progress: plan Docs/superpowers/plans/2026-10-03-usage-quotas-pr-b-per-user-values.md"
git add backlog/tasks/ Docs/superpowers/plans/2026-10-03-usage-quotas-pr-b-per-user-values.md
git commit -m "docs(plan): usage quotas PR B plan; PR A merged on TASK-13434

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 1: Precedence, resolver and checks

**Files:**
- Create: `tldw_Server_API/app/core/UserProfiles/limits_precedence.py`
- Create: `tldw_Server_API/app/core/Usage/quota_resolver.py`
- Create: `tldw_Server_API/app/core/Usage/quota_checks.py`
- Create: `tldw_Server_API/tests/Usage/test_quota_resolver.py`

**Interfaces:**
- **Produces, `limits_precedence`:** `LIMITS_PREFIX = "limits."`, `most_generous_values(rows: list[dict]) -> dict[str, int | float]` and `effective_limits(user_rows, team_rows, org_rows) -> dict[str, int | float]`.
- **Produces, `quota_resolver`:** `async user_quota(user_id: int | None, key: str) -> int | float | None`, `invalidate_user(user_id: int) -> None` and `invalidate_all() -> None`.
- **Produces, `quota_checks`:**
  - `QuotaDecision(allowed: bool, limit: int | float | None, used: float)`;
  - `as_quota_user_id(value) -> int | None`;
  - `async check_usage(user_id, key, requested, counter) -> QuotaDecision`;
  - `seconds_until_utc_midnight() -> int`;
  - `async ledger_used_today(entity_value: str, category: str) -> float`;
  - `async ledger_used_this_month(entity_value: str, category: str) -> float`;
  - `async llm_tokens_this_month(user_id: int) -> float`;
  - `async rag_queries_decision(user_id, units: int) -> QuotaDecision`;
  - `async workflows_runs_decision(user_id) -> QuotaDecision`;
  - the constant `RAG_QUERIES_CATEGORY = "rag_queries"`.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_quota_resolver.py`:

```python
"""Usage-quota precedence, resolver and checks (spec 2 §2, §4)."""

from datetime import datetime
from types import SimpleNamespace

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
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_quota_resolver.py -q`
Expected: FAIL with `ModuleNotFoundError: ... quota_resolver`.

- [ ] **Step 3: Implement the three modules**

`tldw_Server_API/app/core/UserProfiles/limits_precedence.py`:

```python
"""Precedence for UserProfiles ``limits.*`` keys (spec 2 §2).

The user's own value wins; otherwise the most generous value among the user's
teams that set the key; otherwise the most generous among their orgs. 0 is the
least generous value. Joining a group never lowers an allowance.
"""

from __future__ import annotations

from typing import Any

LIMITS_PREFIX = "limits."


def _numeric(value: Any) -> int | float | None:
    """The value when it is a real number (bools are not), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def most_generous_values(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    """For each ``limits.*`` key, the largest numeric value among the rows that set it."""
    best: dict[str, int | float] = {}
    for row in rows:
        key = str(row.get("key") or "")
        value = _numeric(row.get("value"))
        if not key.startswith(LIMITS_PREFIX) or value is None:
            continue
        if key not in best or value > best[key]:
            best[key] = value
    return best


def effective_limits(
    user_rows: list[dict[str, Any]],
    team_rows: list[dict[str, Any]],
    org_rows: list[dict[str, Any]],
) -> dict[str, int | float]:
    """The user's effective ``limits.*`` values: user over team over org."""
    effective = most_generous_values(org_rows)
    effective.update(most_generous_values(team_rows))
    for row in user_rows:
        key = str(row.get("key") or "")
        value = _numeric(row.get("value"))
        if key.startswith(LIMITS_PREFIX) and value is not None:
            effective[key] = value
    return effective
```

`tldw_Server_API/app/core/Usage/quota_resolver.py`:

```python
"""Per-user usage-quota values (spec 2 §2). None means unlimited."""

from __future__ import annotations

import time

from loguru import logger

from tldw_Server_API.app.core.config import usage_quotas_enabled
from tldw_Server_API.app.core.UserProfiles.limits_precedence import effective_limits

_CACHE_TTL_SECONDS = 60.0
_CACHE_MAX_USERS = 4096
_cache: dict[int, tuple[float, dict[str, int | float]]] = {}


async def _load_limits(user_id: int) -> dict[str, int | float]:
    """Read the user's, their active teams' and active orgs' overrides and apply precedence."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.core.UserProfiles.overrides_repo import (
        OrgProfileOverridesRepo,
        TeamProfileOverridesRepo,
        UserProfileOverridesRepo,
    )

    pool = await get_db_pool()
    user_repo = UserProfileOverridesRepo(pool)
    await user_repo.ensure_tables()
    user_rows = await user_repo.list_overrides_for_user(user_id)

    memberships = AuthnzOrgsTeamsRepo(db_pool=pool)
    org_ids = sorted(
        {
            int(row["org_id"])
            for row in await memberships.list_org_memberships_for_user(user_id)
            if row.get("org_id") is not None and row.get("status") in (None, "active")
        }
    )
    team_ids = sorted(
        {
            int(row["team_id"])
            for row in await memberships.list_active_team_memberships_for_user(user_id)
            if row.get("team_id") is not None
        }
    )
    org_rows: list[dict] = []
    team_rows: list[dict] = []
    if org_ids:
        org_repo = OrgProfileOverridesRepo(pool)
        await org_repo.ensure_tables()
        org_rows = await org_repo.list_overrides_for_orgs(org_ids)
    if team_ids:
        team_repo = TeamProfileOverridesRepo(pool)
        await team_repo.ensure_tables()
        team_rows = await team_repo.list_overrides_for_teams(team_ids)
    return effective_limits(user_rows, team_rows, org_rows)


async def user_quota(user_id: int | None, key: str) -> int | float | None:
    """The user's effective ``limits.<name>`` value, or None for unlimited.

    None when usage quotas are off, the user is unknown, nothing is set at any
    level, or the lookup fails (fail open, logged). Cached per user for 60 s;
    other workers see a change within that window.
    """
    if user_id is None or not usage_quotas_enabled():
        return None
    now = time.monotonic()
    hit = _cache.get(user_id)
    if hit is not None and hit[0] > now:
        return hit[1].get(key)
    try:
        limits = await _load_limits(user_id)
    except Exception:  # noqa: BLE001 - a quota lookup failure must not block requests (spec 2 §2)
        logger.opt(exception=True).warning("Usage quota lookup failed for user {}; treating as unlimited", user_id)
        return None
    if len(_cache) >= _CACHE_MAX_USERS:
        # ponytail: wholesale clear bounds memory; switch to LRU if the churn shows up in profiles.
        _cache.clear()
    _cache[user_id] = (now + _CACHE_TTL_SECONDS, limits)
    return limits.get(key)


def invalidate_user(user_id: int) -> None:
    """Forget one user's cached limits (after a write to their own overrides)."""
    _cache.pop(int(user_id), None)


def invalidate_all() -> None:
    """Forget every cached user (after a team or org override write)."""
    _cache.clear()
```

`tldw_Server_API/app/core/Usage/quota_checks.py`:

```python
"""Usage-quota checks shared by enforcement sites (spec 2 §4).

``check_usage`` runs a site's counter only when the user has a limit, so
unlimited users cost no query. Sites raise their own existing errors.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.Usage.quota_resolver import user_quota

RAG_QUERIES_CATEGORY = "rag_queries"


@dataclass(frozen=True)
class QuotaDecision:
    """Whether a request fits the user's quota; ``limit`` None means unlimited."""

    allowed: bool
    limit: int | float | None
    used: float


UNLIMITED = QuotaDecision(allowed=True, limit=None, used=0.0)


def as_quota_user_id(value: Any) -> int | None:
    """The numeric user id quotas are keyed by, or None (treated as unlimited)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def check_usage(
    user_id: int | None,
    key: str,
    requested: float,
    counter: Callable[[], Awaitable[float]],
) -> QuotaDecision:
    """Admit when the user has no ``key`` limit, or when used + requested stays within it."""
    limit = await user_quota(user_id, key)
    if limit is None:
        return UNLIMITED
    used = float(await counter())
    return QuotaDecision(allowed=used + float(requested) <= float(limit), limit=limit, used=used)


def seconds_until_utc_midnight() -> int:
    """Seconds until the daily counters reset (UTC), at least 1."""
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


async def _ledger() -> Any:
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger

    ledger = ResourceDailyLedger()
    await ledger.initialize()
    return ledger


async def ledger_used_today(entity_value: str, category: str) -> float:
    """Today's (UTC) ledger total for ``user:<entity_value>`` in ``category``."""
    ledger = await _ledger()
    return float(await ledger.total_for_day("user", str(entity_value), category))


async def ledger_used_this_month(entity_value: str, category: str) -> float:
    """This calendar month's (UTC) ledger total for ``user:<entity_value>`` in ``category``."""
    ledger = await _ledger()
    month_start = datetime.now(timezone.utc).date().replace(day=1).isoformat()
    result = await ledger.peek_range("user", str(entity_value), category, month_start, _utc_today())
    return float(result.get("total") or 0)


async def llm_tokens_this_month(user_id: int) -> float:
    """This calendar month's (UTC) ``llm_usage_log`` token total for the user."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # llm_usage_log.ts is TIMESTAMP without time zone on Postgres: bind naive UTC there.
    bound: Any = (
        month_start.replace(tzinfo=None)
        if getattr(pool, "pool", None) is not None
        else month_start.strftime("%Y-%m-%d %H:%M:%S")
    )
    value = await pool.fetchval(
        "SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage_log WHERE user_id = ? AND ts >= ?",
        int(user_id),
        bound,
    )
    return float(value or 0)


async def rag_queries_decision(user_id: Any, units: int) -> QuotaDecision:
    """The user's daily RAG-query allowance (``limits.rag_queries_per_day``)."""
    uid = as_quota_user_id(user_id)
    return await check_usage(
        uid,
        "limits.rag_queries_per_day",
        units,
        lambda: ledger_used_today(str(uid), RAG_QUERIES_CATEGORY),
    )


async def workflows_runs_decision(user_id: Any) -> QuotaDecision:
    """The user's daily workflow-run allowance (``limits.workflows_runs_per_day``)."""
    from tldw_Server_API.app.core.Workflows.daily_ledger import workflows_ledger_category

    uid = as_quota_user_id(user_id)
    return await check_usage(
        uid,
        "limits.workflows_runs_per_day",
        1,
        lambda: ledger_used_today(str(uid), workflows_ledger_category()),
    )
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_quota_resolver.py -q`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/UserProfiles/limits_precedence.py tldw_Server_API/app/core/Usage/quota_resolver.py tldw_Server_API/app/core/Usage/quota_checks.py tldw_Server_API/tests/Usage/test_quota_resolver.py
git commit -m "feat(quotas): limits precedence, per-user quota resolver and checks (spec 2 §2)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: Catalog keys, the generic write path and the profile view

**Files:**
- Modify: `tldw_Server_API/Config_Files/user_profile_catalog.yaml`: the `limits.*` entries (near lines 188-268).
- Modify: `tldw_Server_API/app/core/UserProfiles/update_service.py`: the null branch (near lines 143-157), and the audio and evaluations branches of `_apply_key_update` (near lines 311-378).
- Modify: `tldw_Server_API/app/core/UserProfiles/service.py`: `_build_effective_config` (near lines 553-618).
- Create: `tldw_Server_API/tests/UserProfile/test_user_profile_limits.py`.

**Interfaces:**
- Consumes: `quota_resolver.invalidate_user`, `quota_resolver.user_quota` and `limits_precedence.most_generous_values` / `LIMITS_PREFIX` (Task 1).
- Produces: every catalog `limits.*` key except `limits.storage_quota_mb` is a plain user override, and `null` deletes it.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/UserProfile/test_user_profile_limits.py`:

```python
"""limits.* keys: platform-admin-only, plain overrides, null deletes, most-generous profile view (spec 2 §3)."""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.orgs_teams import add_team_member, create_organization, create_team
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.core.UserProfiles.overrides_repo import TeamProfileOverridesRepo
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
from tldw_Server_API.app.core.UserProfiles.update_service import _can_edit
from tldw_Server_API.app.core.UserProfiles.user_profile_catalog import load_user_profile_catalog
from tldw_Server_API.app.main import app

pytestmark = pytest.mark.unit

NEW_KEYS = {
    "limits.transcription_minutes_per_month",
    "limits.llm_tokens_per_month",
    "limits.rag_queries_per_day",
    "limits.media_ingest_mb_per_day",
    "limits.workflows_runs_per_day",
    "limits.evaluation_tokens_per_day",
    "limits.chatbooks_exports_per_day",
    "limits.chatbooks_imports_per_day",
    "limits.chatbooks_concurrent_jobs",
}


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas switched on, resolver cache empty."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()


def _user_id(client: TestClient, auth_headers: dict) -> int:
    """The single user's id."""
    resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert resp.status_code == 200
    return int(resp.json()["user"]["id"])


def test_catalog_has_every_limit_key_platform_admin_only() -> None:
    """All limits.* keys exist, default to null, minimum 0, and only platform admins may edit them."""
    entries = {e.key: e for e in load_user_profile_catalog().entries if e.key.startswith("limits.")}
    assert NEW_KEYS <= set(entries)
    for entry in entries.values():
        assert entry.default is None
        assert list(entry.editable_by) == ["platform_admin"]


def test_org_admin_cannot_edit_limits() -> None:
    """An org admin without platform-admin rights is refused every limits.* key."""
    for entry in load_user_profile_catalog().entries:
        if entry.key.startswith("limits."):
            assert _can_edit(entry, {"org_admin", "team_admin"}) is False
            assert _can_edit(entry, {"platform_admin"}) is True


def test_generic_limit_write_and_null_delete(auth_headers: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new limits.* key is stored as a user override the resolver sees; null removes it."""
    flips: list[str] = []
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.user_rate_limiter.UserRateLimiter.upgrade_user_tier",
        lambda *a, **k: flips.append("tier"),
    )
    with TestClient(app) as client:
        user_id = _user_id(client, auth_headers)
        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={"updates": [{"key": "limits.rag_queries_per_day", "value": 5}, {"key": "limits.evaluations_per_day", "value": 9}]},
        )
        assert resp.status_code == 200
        assert set(resp.json()["applied"]) == {"limits.rag_queries_per_day", "limits.evaluations_per_day"}
        assert asyncio.run(quota_resolver.user_quota(user_id, "limits.rag_queries_per_day")) == 5
        assert asyncio.run(quota_resolver.user_quota(user_id, "limits.evaluations_per_day")) == 9
        assert flips == []  # writing an evaluations limit no longer flips the user to the CUSTOM tier

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={"updates": [{"key": "limits.rag_queries_per_day", "value": None}]},
        )
        assert resp.status_code == 200
        assert "limits.rag_queries_per_day" in resp.json()["applied"]
        assert asyncio.run(quota_resolver.user_quota(user_id, "limits.rag_queries_per_day")) is None


def test_profile_view_shows_most_generous_team_value(auth_headers: dict) -> None:
    """The effective-config view applies the same precedence the resolver enforces."""
    with TestClient(app) as client:
        user_id = _user_id(client, auth_headers)
        suffix = uuid.uuid4().hex[:8]

        async def _setup() -> dict:
            """Two teams with different values; read the effective profile."""
            org = await create_organization(name=f"Limits Org {suffix}", owner_user_id=None)
            low = await create_team(org_id=int(org["id"]), name=f"Low {suffix}")
            high = await create_team(org_id=int(org["id"]), name=f"High {suffix}")
            await add_team_member(team_id=int(low["id"]), user_id=user_id)
            await add_team_member(team_id=int(high["id"]), user_id=user_id)
            pool = await get_db_pool()
            repo = TeamProfileOverridesRepo(pool)
            await repo.ensure_tables()
            await repo.upsert_override(team_id=int(low["id"]), key="limits.workflows_runs_per_day", value=10, updated_by=None)
            await repo.upsert_override(team_id=int(high["id"]), key="limits.workflows_runs_per_day", value=50, updated_by=None)
            return await UserProfileService(pool)._build_effective_config(user_id, include_sources=True, mask_secrets=False)

        effective = asyncio.run(_setup())
        assert effective["limits.workflows_runs_per_day"] == {"value": 50, "source": "team"}
        assert asyncio.run(quota_resolver.user_quota(user_id, "limits.workflows_runs_per_day")) == 50
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/UserProfile/test_user_profile_limits.py -q`
Expected: FAIL on the missing catalog keys, `editable_by`, `unsupported_key`, the `null_not_allowed` skip, and the lowest-id team value of 10.

- [ ] **Step 3: Catalog**

In `user_profile_catalog.yaml`, set `editable_by: [platform_admin]` on every existing `limits.*` entry. Then add these entries after `limits.prompt_studio_submits_per_min`, in the same shape:

```yaml
  - key: limits.transcription_minutes_per_month
    label: Transcription Minutes Per Month
    description: Transcription minutes per calendar month (UTC). Unset means unlimited.
    type: integer
    minimum: 0
    default: null
    editable_by: [platform_admin]
    sensitivity: internal
    ui: number
```

Repeat for each of the following, with the `label` and `description` shown and `type: integer` throughout:
- `limits.llm_tokens_per_month`: "LLM Tokens Per Month" / "LLM tokens per calendar month (UTC), counted from llm_usage_log; enforced on /chat/completions."
- `limits.rag_queries_per_day`: "RAG Queries Per Day" / "RAG queries per UTC day, including Text2SQL and MCP RAG."
- `limits.media_ingest_mb_per_day`: "Media Ingest MB Per Day" / "Uploaded media megabytes per UTC day."
- `limits.workflows_runs_per_day`: "Workflow Runs Per Day" / "Workflow runs per UTC day, API-started and scheduled."
- `limits.evaluation_tokens_per_day`: "Evaluation Tokens Per Day" / "Evaluation tokens per UTC day."
- `limits.chatbooks_exports_per_day`: "Chatbook Exports Per Day" / "Chatbook exports per UTC day."
- `limits.chatbooks_imports_per_day`: "Chatbook Imports Per Day" / "Chatbook imports per UTC day."
- `limits.chatbooks_concurrent_jobs`: "Chatbook Concurrent Jobs" / "Chatbook export/import jobs running at once."

- [ ] **Step 4: The update service**

In `update_service.py`, add the import `from tldw_Server_API.app.core.Usage.quota_resolver import invalidate_user`.

In the null branch, replace `if key.startswith("preferences."):` with:

```python
                if key.startswith("preferences.") or (
                    key.startswith("limits.") and key != "limits.storage_quota_mb"
                ):
```

and, right after `await repo.delete_override(user_id=user_id, key=key, db_conn=db_conn)`, add `invalidate_user(user_id)`.

In `_apply_key_update`, delete the `if key in {"limits.audio_daily_minutes", "limits.audio_concurrent_jobs"}:` branch and the whole `if key in {"limits.evaluations_per_minute", "limits.evaluations_per_day"}:` branch, including its `upgrade_user_tier` call. Put this single branch in their place, after the `limits.storage_quota_mb` branch:

```python
        if key.startswith("limits."):
            # Every limits.* key except storage (PR C) is a plain user override the
            # quota resolver reads (spec 2 §3).
            if not dry_run:
                repo = repo_holder.get("repo")
                if repo is None:
                    repo = UserProfileOverridesRepo(self._db_pool)
                    await repo.ensure_tables(db_conn=db_conn)
                    repo_holder["repo"] = repo
                await anchor.capture()
                await repo.upsert_override(
                    user_id=user_id,
                    key=key,
                    value=value,
                    updated_by=updated_by,
                    db_conn=db_conn,
                )
                anchor.mark_changed()
                invalidate_user(user_id)
            return True
```

- [ ] **Step 5: The profile view**

In `service.py`, import `from tldw_Server_API.app.core.UserProfiles.limits_precedence import LIMITS_PREFIX, most_generous_values`.

In `_build_effective_config`:
- declare `org_rows: list[dict[str, Any]] = []` and `team_rows: list[dict[str, Any]] = []` before the `try:`, and keep assigning them inside it as today;
- after the `try/except`, add `team_limits = most_generous_values(team_rows)` and `org_limits = most_generous_values(org_rows)`;
- in the loop, insert these two branches between the user-override branch and `elif key in team_overrides:`:

```python
            elif key.startswith(LIMITS_PREFIX) and key in team_limits:
                value = team_limits[key]
                source = "team"
            elif key.startswith(LIMITS_PREFIX) and key in org_limits:
                value = org_limits[key]
                source = "org"
```

- [ ] **Step 6: Run the new tests and the UserProfile suite**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Usage`
Expected: the new tests pass.
- An existing test that relied on `limits.evaluations_*` flipping the CUSTOM tier, or on an org admin editing a `limits.*` key, now fails. Rewrite it to the new contract (plain override; platform admin only) and say so in your report.
- `test_admin_profile_update_audio_limits` must still pass. It reads the profile's audio quotas section, which Task 4 moves onto the resolver. If it fails here only on the section's value, note it and continue: Task 4 owns that read.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/Config_Files/user_profile_catalog.yaml tldw_Server_API/app/core/UserProfiles tldw_Server_API/tests/UserProfile/test_user_profile_limits.py
git commit -m "feat(quotas): limits.* keys are platform-admin plain overrides; null deletes; most-generous profile view

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 3: Team and org override routes

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/user_profile_schemas.py`: add two models.
- Modify: `tldw_Server_API/app/services/admin_profiles_service.py`: add `set_group_limit_override`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_profiles.py`: add four routes.
- Create: `tldw_Server_API/tests/UserProfile/test_group_limit_overrides.py`.

**Interfaces:**
- Consumes: `update_service._validate_value`, `quota_resolver.invalidate_all` and `user_quota`.
- Produces, the routes:
  - `PUT /api/v1/admin/orgs/{org_id}/profile/overrides/{key}`, body `{"value": number | null}`;
  - `DELETE /api/v1/admin/orgs/{org_id}/profile/overrides/{key}`;
  - the same two for `/api/v1/admin/teams/{team_id}/...`.

  Each responds `{"scope", "id", "key", "value"}`.

- [ ] **Step 1: Write the failing tests**

```python
"""Team/org limits.* override routes: platform admin only, validated, null/DELETE removes (spec 2 §3)."""

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.orgs_teams import add_org_member, add_team_member, create_organization, create_team
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.main import app
from tldw_Server_API.app.services import admin_profiles_service

pytestmark = pytest.mark.unit

KEY = "limits.rag_queries_per_day"


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, resolver cache empty."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()


def _setup_org_and_team(user_id: int) -> tuple[int, int]:
    """An org and a team inside it, with the user a member of both."""
    suffix = uuid.uuid4().hex[:8]

    async def _go() -> tuple[int, int]:
        """Create and join."""
        org = await create_organization(name=f"Quota Org {suffix}", owner_user_id=None)
        team = await create_team(org_id=int(org["id"]), name=f"Quota Team {suffix}")
        await add_org_member(org_id=int(org["id"]), user_id=user_id)
        await add_team_member(team_id=int(team["id"]), user_id=user_id)
        return int(org["id"]), int(team["id"])

    return asyncio.run(_go())


def _resolve(user_id: int) -> object:
    """The user's effective rag-queries limit."""
    return asyncio.run(quota_resolver.user_quota(user_id, KEY))


def test_org_override_applies_to_members_and_delete_removes_it(auth_headers: dict) -> None:
    """Setting an org value gives every member that allowance; DELETE returns them to unlimited."""
    with TestClient(app) as client:
        user_id = int(client.get("/api/v1/users/me/profile", headers=auth_headers).json()["user"]["id"])
        org_id, _team_id = _setup_org_and_team(user_id)
        resp = client.put(f"/api/v1/admin/orgs/{org_id}/profile/overrides/{KEY}", headers=auth_headers, json={"value": 30})
        assert resp.status_code == 200
        assert resp.json() == {"scope": "org", "id": org_id, "key": KEY, "value": 30}
        assert _resolve(user_id) == 30
        resp = client.delete(f"/api/v1/admin/orgs/{org_id}/profile/overrides/{KEY}", headers=auth_headers)
        assert resp.status_code == 200 and resp.json()["value"] is None
        assert _resolve(user_id) is None


def test_deleting_team_override_falls_back_to_org(auth_headers: dict) -> None:
    """A team value outranks the org's; removing it drops members back to the org value."""
    with TestClient(app) as client:
        user_id = int(client.get("/api/v1/users/me/profile", headers=auth_headers).json()["user"]["id"])
        org_id, team_id = _setup_org_and_team(user_id)
        assert client.put(f"/api/v1/admin/orgs/{org_id}/profile/overrides/{KEY}", headers=auth_headers, json={"value": 100}).status_code == 200
        assert client.put(f"/api/v1/admin/teams/{team_id}/profile/overrides/{KEY}", headers=auth_headers, json={"value": 20}).status_code == 200
        assert _resolve(user_id) == 20
        assert client.put(f"/api/v1/admin/teams/{team_id}/profile/overrides/{KEY}", headers=auth_headers, json={"value": None}).status_code == 200
        assert _resolve(user_id) == 100


def test_group_override_rejects_bad_input(auth_headers: dict) -> None:
    """Unknown keys, storage, non-limits keys, invalid values and missing groups are refused."""
    with TestClient(app) as client:
        user_id = int(client.get("/api/v1/users/me/profile", headers=auth_headers).json()["user"]["id"])
        org_id, _ = _setup_org_and_team(user_id)
        base = f"/api/v1/admin/orgs/{org_id}/profile/overrides"
        assert client.put(f"{base}/limits.no_such_key", headers=auth_headers, json={"value": 1}).status_code == 400
        assert client.put(f"{base}/limits.storage_quota_mb", headers=auth_headers, json={"value": 1}).status_code == 400
        assert client.put(f"{base}/preferences.ui.theme", headers=auth_headers, json={"value": 1}).status_code == 400
        assert client.put(f"{base}/{KEY}", headers=auth_headers, json={"value": -1}).status_code == 400
        assert client.put(f"/api/v1/admin/orgs/987654321/profile/overrides/{KEY}", headers=auth_headers, json={"value": 1}).status_code == 404


async def test_group_override_requires_platform_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A principal that is neither single-user nor platform admin gets 403 before any write."""
    monkeypatch.setattr(admin_profiles_service, "is_single_user_principal", lambda _p: False)
    monkeypatch.setattr(admin_profiles_service.admin_scope_service, "is_platform_admin", lambda _p: False)
    principal = AuthPrincipal(kind="user", user_id=5, is_admin=False)
    with pytest.raises(HTTPException) as exc:
        await admin_profiles_service.set_group_limit_override(scope="org", group_id=1, key=KEY, value=3, principal=principal)
    assert exc.value.status_code == 403
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/UserProfile/test_group_limit_overrides.py -q`
Expected: FAIL with 404/405 from the missing routes, and `AttributeError` for `set_group_limit_override`.

- [ ] **Step 3: Schemas**

Append to `user_profile_schemas.py`:

```python
class GroupLimitOverrideRequest(BaseModel):
    """Body for setting a team/org ``limits.*`` override; ``value`` null removes it."""

    value: int | float | None = None


class GroupLimitOverrideResponse(BaseModel):
    """The team/org override after the write; ``value`` null means removed."""

    scope: Literal["org", "team"]
    id: int
    key: str
    value: int | float | None
```

Add `Literal` to the module's `typing` import if it isn't there.

- [ ] **Step 4: The service function**

Append to `admin_profiles_service.py`, adding the imports shown at the top of the module:

```python
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.Usage.quota_resolver import invalidate_all as invalidate_all_quotas
from tldw_Server_API.app.core.UserProfiles.overrides_repo import OrgProfileOverridesRepo, TeamProfileOverridesRepo
from tldw_Server_API.app.core.UserProfiles.update_service import _validate_value


async def set_group_limit_override(
    *,
    scope: str,
    group_id: int,
    key: str,
    value: Any,
    principal: AuthPrincipal,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Set (or, with value None, remove) a team/org ``limits.*`` override (spec 2 §3).

    Platform admins only: a customer's org admin must not lift their own members.
    The value is each member's allowance; storage stays per-user until PR C.
    """
    if not (is_single_user_principal(principal) or admin_scope_service.is_platform_admin(principal)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Platform admin required")
    entry = {e.key: e for e in load_user_profile_catalog().entries}.get(key)
    if entry is None or not key.startswith("limits.") or key == "limits.storage_quota_mb":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": "unsupported_key", "key": key})
    normalized = None
    if value is not None:
        ok, normalized, err = _validate_value(entry, value)
        if not ok:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": err or "invalid_value", "key": key})

    pool = await get_db_pool()
    memberships = AuthnzOrgsTeamsRepo(db_pool=pool)
    group = await (memberships.get_organization_metadata(group_id) if scope == "org" else memberships.get_team(group_id))
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{scope} {group_id} not found")

    if scope == "org":
        repo = OrgProfileOverridesRepo(pool)
        await repo.ensure_tables()
        if value is None:
            await repo.delete_override(org_id=group_id, key=key)
        else:
            await repo.upsert_override(org_id=group_id, key=key, value=normalized, updated_by=principal.user_id)
    else:
        repo = TeamProfileOverridesRepo(pool)
        await repo.ensure_tables()
        if value is None:
            await repo.delete_override(team_id=group_id, key=key)
        else:
            await repo.upsert_override(team_id=group_id, key=key, value=normalized, updated_by=principal.user_id)
    invalidate_all_quotas()

    response = {"scope": scope, "id": group_id, "key": key, "value": normalized}
    audit_info = {
        "event_type": "data.update",
        "category": "data_modification",
        "resource_type": f"{scope}_profile_override",
        "resource_id": f"{group_id}:{key}",
        "action": f"{scope}_profile_override.{'delete' if value is None else 'set'}",
        "metadata": {"key": key, "value": normalized},
    }
    return response, audit_info
```

Confirm that `TeamProfileOverridesRepo.upsert_override` takes `team_id=`, and that `get_organization_metadata` returns `None` for a missing org. If it returns `{}` instead, the `if not group` check still holds.

- [ ] **Step 5: The routes**

Append to `admin_profiles.py`, importing the two schema models:

```python
async def _group_override(
    scope: str, group_id: int, key: str, value: Any, http_request: Request, principal: AuthPrincipal
) -> GroupLimitOverrideResponse:
    """Apply a team/org limits.* override and emit its admin audit event."""
    response, audit_info = await admin_profiles_service.set_group_limit_override(
        scope=scope, group_id=group_id, key=key, value=value, principal=principal
    )
    try:
        await _get_emit_admin_audit_event()(http_request, principal, **audit_info)
    except Exception:
        logger.warning("Admin audit emission failed")
    return GroupLimitOverrideResponse(**response)


@router.put("/orgs/{org_id}/profile/overrides/{key}", response_model=GroupLimitOverrideResponse)
async def admin_set_org_limit_override(
    org_id: int, key: str, payload: GroupLimitOverrideRequest, http_request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> GroupLimitOverrideResponse:
    """Set (value null removes) an org's limits.* override; each member's allowance (platform admin)."""
    return await _group_override("org", org_id, key, payload.value, http_request, principal)


@router.delete("/orgs/{org_id}/profile/overrides/{key}", response_model=GroupLimitOverrideResponse)
async def admin_delete_org_limit_override(
    org_id: int, key: str, http_request: Request, principal: AuthPrincipal = Depends(get_auth_principal),
) -> GroupLimitOverrideResponse:
    """Remove an org's limits.* override (platform admin)."""
    return await _group_override("org", org_id, key, None, http_request, principal)


@router.put("/teams/{team_id}/profile/overrides/{key}", response_model=GroupLimitOverrideResponse)
async def admin_set_team_limit_override(
    team_id: int, key: str, payload: GroupLimitOverrideRequest, http_request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> GroupLimitOverrideResponse:
    """Set (value null removes) a team's limits.* override; each member's allowance (platform admin)."""
    return await _group_override("team", team_id, key, payload.value, http_request, principal)


@router.delete("/teams/{team_id}/profile/overrides/{key}", response_model=GroupLimitOverrideResponse)
async def admin_delete_team_limit_override(
    team_id: int, key: str, http_request: Request, principal: AuthPrincipal = Depends(get_auth_principal),
) -> GroupLimitOverrideResponse:
    """Remove a team's limits.* override (platform admin)."""
    return await _group_override("team", team_id, key, None, http_request, principal)
```

- [ ] **Step 6: Run the tests, the UserProfile suite and the route lints**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/UserProfile tldw_Server_API/tests/lint/test_route_auth_ratchet.py tldw_Server_API/tests/lint/test_rg_route_map_lint.py`
Expected: all pass. Every new route authenticates through `get_auth_principal`; if the route-auth ratchet flags one anyway, fix the dependency rather than the baseline.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/user_profile_schemas.py tldw_Server_API/app/services/admin_profiles_service.py tldw_Server_API/app/api/v1/endpoints/admin/admin_profiles.py tldw_Server_API/tests/UserProfile/test_group_limit_overrides.py
git commit -m "feat(quotas): platform-admin team/org limits.* override routes (spec 2 §3)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 4: Audio reads the resolver

**Files:**
- Modify: `tldw_Server_API/app/core/Usage/audio_quota.py`: `_UNLIMITED_AUDIO_LIMITS`, `get_limits_for_user`, the new `_monthly_minutes_exhausted`, `consume_daily_minutes`, `check_daily_minutes_allow`, `can_start_job` and `can_start_stream`. Delete `_get_user_override_limits`.
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py`: the two fallback dicts.
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`: the WebSocket billing gate (near line 1368).
- Modify: `tldw_Server_API/tests/Usage/test_usage_quotas_audio.py` (the PR A expectations) and create `tldw_Server_API/tests/Usage/test_audio_quota_per_user.py`.

**Interfaces:**
- Consumes: `quota_resolver.user_quota`, `quota_checks.ledger_used_this_month`.
- Produces: `get_limits_for_user` returns the keys `daily_minutes`, `monthly_minutes`, `concurrent_jobs`, `concurrent_streams` and `max_file_size_mb`. The last two are always `None`, because synchronous concurrency and per-tier file sizes are retired.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_audio_quota_per_user.py`:

```python
"""Audio quotas come from the user's limits.* values (spec 2 §4)."""

import pytest

from tldw_Server_API.app.core.Usage import audio_quota

pytestmark = pytest.mark.unit


@pytest.fixture()
def limits(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Quotas on, with per-key limits the test fills in."""
    values: dict = {}

    async def _user_quota(_user_id: int, key: str) -> object:
        """Serve the test's limit for this key."""
        return values.get(key)

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.setattr(audio_quota, "user_quota", _user_quota)
    return values


async def test_limits_come_from_the_resolver(limits: dict) -> None:
    """Daily/monthly minutes and queued-job concurrency are the user's limits.* values."""
    limits.update({"limits.audio_daily_minutes": 45, "limits.transcription_minutes_per_month": 600, "limits.audio_concurrent_jobs": 3})
    assert await audio_quota.get_limits_for_user(1) == {
        "daily_minutes": 45, "monthly_minutes": 600, "concurrent_jobs": 3,
        "concurrent_streams": None, "max_file_size_mb": None,
    }


async def test_no_limits_set_means_unlimited(limits: dict) -> None:
    """A user with no limits.* values is unlimited even with quotas on."""
    allowed, remaining = await audio_quota.check_daily_minutes_allow(1, 500.0)
    assert allowed is True and remaining is None


async def test_monthly_minutes_block_when_exhausted(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A month already at its limit refuses more minutes, checked before the daily ledger."""
    limits["limits.transcription_minutes_per_month"] = 60

    async def _sixty_minutes_used(_uid: str, _category: str) -> float:
        """3600 ledger seconds this month."""
        return 3600.0

    monkeypatch.setattr(audio_quota, "ledger_used_this_month", _sixty_minutes_used)
    assert await audio_quota.check_daily_minutes_allow(1, 1.0) == (False, 0.0)
    assert await audio_quota.consume_daily_minutes(1, 1.0) == (False, 0.0)


async def test_synchronous_concurrency_is_unlimited(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """can_start_job/can_start_stream no longer reserve RG leases (per-user sync concurrency is deferred)."""
    fetched: list[str] = []

    async def _recording_governor() -> None:
        """Record any governor fetch."""
        fetched.append("governor")

    monkeypatch.setattr(audio_quota, "_get_audio_rg_governor", _recording_governor)
    assert await audio_quota.can_start_job(1) == (True, "OK")
    assert await audio_quota.can_start_stream(1) == (True, "OK")
    assert fetched == []
```

In `tldw_Server_API/tests/Usage/test_usage_quotas_audio.py`, add `"monthly_minutes": None` to the dict that `test_limits_are_unlimited_when_quotas_off` expects.

- [ ] **Step 2: Run them and confirm they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_audio_quota_per_user.py tldw_Server_API/tests/Usage/test_usage_quotas_audio.py -q`
Expected: FAIL. `user_quota` and `ledger_used_this_month` aren't attributes of `audio_quota` yet, and the dict has no `monthly_minutes`.

- [ ] **Step 3: Implement**

In `audio_quota.py`:
- Add these top-level imports beside `usage_quotas_enabled`:

```python
from tldw_Server_API.app.core.Usage.quota_checks import ledger_used_this_month
from tldw_Server_API.app.core.Usage.quota_resolver import user_quota
```

- Add `"monthly_minutes": None,` to `_UNLIMITED_AUDIO_LIMITS`.
- Replace `get_limits_for_user` and delete `_get_user_override_limits`:

```python
async def get_limits_for_user(user_id: int) -> dict[str, float | None]:
    """The user's audio limits (spec 2): their limits.* values; None is unlimited."""
    limits = dict(_UNLIMITED_AUDIO_LIMITS)
    if not usage_quotas_enabled():
        return limits
    uid = int(user_id)
    limits["daily_minutes"] = await user_quota(uid, "limits.audio_daily_minutes")
    limits["monthly_minutes"] = await user_quota(uid, "limits.transcription_minutes_per_month")
    limits["concurrent_jobs"] = await user_quota(uid, "limits.audio_concurrent_jobs")
    return limits


async def _monthly_minutes_exhausted(user_id: int, monthly_limit: float | None, minutes_requested: float) -> bool:
    """True when this request would take the user past their calendar-month (UTC) minutes."""
    if monthly_limit is None:
        return False
    used_seconds = await ledger_used_this_month(str(int(user_id)), "minutes")
    return used_seconds + _audio_minutes_units(minutes_requested) > _audio_minutes_units(float(monthly_limit))
```

- In `consume_daily_minutes`, add this right after the second `limits = await get_limits_for_user(user_id)`, the one on the `units > 0` path:

```python
    if await _monthly_minutes_exhausted(user_id, limits.get("monthly_minutes"), minutes_requested):
        _metrics_increment("audio_quota_violations_total", {"type": "monthly_minutes"})
        return False, 0.0
```

- In `check_daily_minutes_allow`, add the same three lines right after `limits = await get_limits_for_user(user_id)`.
- Replace the bodies of `can_start_job` and `can_start_stream` with `return True, "OK"`. Keep their docstrings, rewritten to say: "Per-user synchronous concurrency is deferred (spec 2 Non-goals); always admits." `finish_job`/`finish_stream` stay; with no handles they are no-ops.

In `_apply_tier_overrides_from_config` (spec §7, deprecating the tier knobs): when it finds any override in `AUDIO_TIER_LIMITS_JSON` or a `[Audio-Quota] {tier}_*` key, log this warning once:

```python
        logger.warning(
            "AUDIO_TIER_LIMITS_JSON / [Audio-Quota] {tier}_* no longer set limits; "
            "use the limits.audio_* UserProfiles values (spec 2)"
        )
```

Keep building `TIER_LIMITS`, because the deprecated tier admin API validates tier names against it.

In `audio_jobs_worker.py`, replace both fallback dicts (`limits = {"daily_minutes": 30.0, ...}` and `limits_owner = {...}`) with `{}`, so a lookup failure means unlimited. Update the adjacent log text to say "assuming unlimited".

In `audio_streaming.py`, near line 1368, the WebSocket billing block opens with `if enforcement_enabled():`. Change it to `if await billing_checks_active():`, importing `billing_checks_active` from `tldw_Server_API.app.core.Billing.enforcement`. This clears a deferred minor from PR A.

- [ ] **Step 4: Run the audio suites**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Usage tldw_Server_API/tests/Audio tldw_Server_API/tests/AudioJobs tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Resource_Governance/test_rg_cutover_audio_quota.py`
Expected: the new tests pass.
- Some existing tests asserted tier values (free = 30 min/day, `concurrent_jobs` = 1) or RG lease concurrency. Rewrite them to the resolver contract: set the value through a patched `user_quota` or a profile override.
- Delete the tests that only pinned RG audio lease reservation, and list them in your report.
- A failure that is just the known missing `pyaudio` import in `test_nemo_streaming_error_sentinel.py` is environmental.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Usage/audio_quota.py tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py tldw_Server_API/tests
git commit -m "feat(quotas): audio minutes (daily + monthly) and queued-job concurrency from limits.* (spec 2 §4)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 5: LLM tokens per month and RAG queries per day

**Files:**
- Create: `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`: after the billing `LimitEnforcer` block (near line 4244).
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`: lines near 1390, 1652, 1831 and 2202.
- Modify: `tldw_Server_API/app/api/v1/endpoints/text2sql.py`: the dependencies and a usage log.
- Modify: `tldw_Server_API/app/core/RAG/rag_service/transport.py`: `enforce_rag_query_limit_for_org_context` and `log_rag_queries_for_org_context`.
- Create: `tldw_Server_API/tests/Usage/test_quota_llm_rag.py`.

**Interfaces:**
- Consumes: `check_usage`, `llm_tokens_this_month`, `rag_queries_decision`, `seconds_until_utc_midnight`, `as_quota_user_id` and `RAG_QUERIES_CATEGORY` (Task 1).
- Produces: `usage_quota_deps.require_rag_query_quota(units: int = 1)` (a FastAPI dependency factory), and `usage_quota_deps.enforce_rag_query_quota(user_id, units) -> None`, which raises a 402 `HTTPException`.

- [ ] **Step 1: Write the failing tests**

```python
"""Per-user LLM-token and RAG-query quotas (spec 2 §4)."""

from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.RAG.rag_service import transport
from tldw_Server_API.app.core.Usage import quota_checks

pytestmark = pytest.mark.unit


@pytest.fixture()
def rag_limit(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A rag-queries limit and today's usage, both set by the test."""
    state = {"limit": None, "used": 0.0}

    async def _user_quota(_uid: int, _key: str) -> object:
        """The test's limit."""
        return state["limit"]

    async def _used(_uid: str, _category: str) -> float:
        """The test's usage."""
        return state["used"]

    monkeypatch.setattr(quota_checks, "user_quota", _user_quota)
    monkeypatch.setattr(quota_checks, "ledger_used_today", _used)
    return state


def _app() -> FastAPI:
    """A tiny app whose route carries the RAG quota dependency."""
    app = FastAPI()

    @app.get("/q", dependencies=[Depends(usage_quota_deps.require_rag_query_quota(1))])
    async def _q() -> dict:
        """The guarded route."""
        return {"ok": True}

    app.dependency_overrides[get_request_user] = lambda: User(id=7, username="u", email="u@x.test", is_active=True, is_admin=False)
    return app


def test_rag_dependency_402_when_daily_allowance_spent(rag_limit: dict) -> None:
    """Within the allowance passes; at the allowance the route returns 402 limit_exceeded with Retry-After."""
    client = TestClient(_app())
    rag_limit.update(limit=3, used=2.0)
    assert client.get("/q").status_code == 200
    rag_limit["used"] = 3.0
    resp = client.get("/q")
    assert resp.status_code == 402
    assert resp.json()["detail"]["category"] == "rag_queries_day"
    assert int(resp.headers["Retry-After"]) >= 1


async def test_transport_check_refuses_over_allowance(rag_limit: dict) -> None:
    """The MCP path (transport) raises PermissionError when the user's allowance is spent."""
    rag_limit.update(limit=1, used=1.0)
    with pytest.raises(PermissionError):
        await transport.enforce_rag_query_limit_for_org_context(current_user=SimpleNamespace(id=7), units=1)


async def test_transport_logs_a_per_user_row_without_an_org(monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage is recorded per user even when no org resolves (gate the check, never the record)."""
    added: list = []

    class _Ledger:
        """Records ledger writes."""

        async def initialize(self) -> None:
            """No-op init."""

        async def add(self, entry: object) -> bool:
            """Record the entry."""
            added.append(entry)
            return True

    async def _no_org(**_kw: object) -> None:
        """No org context."""
        return None

    monkeypatch.setattr("tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger.ResourceDailyLedger", _Ledger)
    monkeypatch.setattr(transport, "resolve_org_id_for_rag_context", _no_org)
    await transport.log_rag_queries_for_org_context(current_user=SimpleNamespace(id=7), units=2)
    assert [(e.entity_scope, e.entity_value, e.category, e.units) for e in added] == [("user", "7", "rag_queries", 2)]


async def test_llm_tokens_decision_counts_the_month(monkeypatch: pytest.MonkeyPatch) -> None:
    """A month at its token allowance refuses an estimated request that would exceed it."""

    async def _limit(_uid: int, _key: str) -> int:
        """1000-token allowance."""
        return 1000

    async def _used(_uid: int) -> float:
        """900 tokens used."""
        return 900.0

    monkeypatch.setattr(quota_checks, "user_quota", _limit)
    monkeypatch.setattr(quota_checks, "llm_tokens_this_month", _used)
    allowed = await quota_checks.check_usage(7, "limits.llm_tokens_per_month", 100, lambda: quota_checks.llm_tokens_this_month(7))
    refused = await quota_checks.check_usage(7, "limits.llm_tokens_per_month", 101, lambda: quota_checks.llm_tokens_this_month(7))
    assert allowed.allowed is True and refused.allowed is False
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_quota_llm_rag.py -q`
Expected: FAIL. `usage_quota_deps` doesn't exist, and the transport writes no user row.

- [ ] **Step 3: The dependency module**

`tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py`:

```python
"""FastAPI dependencies for per-user usage quotas (spec 2 §4)."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, status

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Usage.quota_checks import rag_queries_decision, seconds_until_utc_midnight


async def enforce_rag_query_quota(user_id: Any, units: int) -> None:
    """402 limit_exceeded when the user's daily RAG-query allowance would be exceeded."""
    decision = await rag_queries_decision(user_id, units)
    if decision.allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail={
            "error": "limit_exceeded",
            "category": "rag_queries_day",
            "current": int(decision.used),
            "limit": decision.limit,
            "message": "Daily RAG query limit reached",
        },
        headers={"Retry-After": str(seconds_until_utc_midnight())},
    )


def require_rag_query_quota(units: int = 1):
    """Dependency factory enforcing ``limits.rag_queries_per_day`` for the request user."""

    async def _check(current_user: User = Depends(get_request_user)) -> None:
        await enforce_rag_query_quota(getattr(current_user, "id", None), units)

    return _check
```

- [ ] **Step 4: Wire the sites**

- **`rag_unified.py`:** at lines ~1390, ~1831 and ~2202, add `Depends(require_rag_query_quota(1)),` directly after each `Depends(require_within_limit(LimitCategory.RAG_QUERIES_DAY, 1)),`. At the inline site (~1652), add this right after the `await limit_checker(...)` call:

```python
        await enforce_rag_query_quota(getattr(current_user, "id", None), requested_units)
```

  Import both names from `usage_quota_deps`. Confirm that `current_user` is the parameter name in that function; if it isn't, use the function's request-user parameter.
- **`text2sql.py`:**
  - Add `Depends(require_rag_query_quota(1)),` after its `require_within_limit` line.
  - After `generate_and_execute` succeeds, before the response is built, add `await rag_transport.log_rag_queries_for_org_context(current_user=current_user, units=1)`, importing `from tldw_Server_API.app.core.RAG.rag_service import transport as rag_transport`.
- **`transport.py`:**
  - In `enforce_rag_query_limit_for_org_context`, right after `if units <= 0: return`, add:

```python
    from tldw_Server_API.app.core.Usage.quota_checks import rag_queries_decision

    if not (await rag_queries_decision(getattr(current_user, "id", None), units)).allowed:
        raise PermissionError("Daily RAG query limit reached")
```

  - In `log_rag_queries_for_org_context`, restructure the body inside its `try:` so the user row is always written, and the org row is written only when an org resolves:

```python
    try:
        from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (
            LedgerEntry,
            ResourceDailyLedger,
        )

        ledger = ResourceDailyLedger()
        await ledger.initialize()
        now = datetime.now(timezone.utc)
        user_id = getattr(current_user, "id", None)
        if user_id is not None:
            await ledger.add(
                LedgerEntry(  # type: ignore[call-arg]
                    entity_scope="user",
                    entity_value=str(user_id),
                    category="rag_queries",
                    units=int(units),
                    op_id=f"rag:user:{user_id}:{uuid4()}",
                    occurred_at=now,
                )
            )
        org_id = await resolve_org_id_for_rag_context(request_like=request_like, current_user=current_user)
        if org_id is not None:
            await ledger.add(
                LedgerEntry(  # type: ignore[call-arg]
                    entity_scope="org",
                    entity_value=str(org_id),
                    category="rag_queries",
                    units=int(units),
                    op_id=f"rag:{org_id}:{uuid4()}",
                    occurred_at=now,
                )
            )
    except Exception:  # noqa: BLE001 - ledger failures must not impact callers.
        logger.debug("RAG query logging failed; continuing without usage record", exc_info=True)
```

- **`chat.py`:** directly after the billing block that ends `_billing_enforcer = None` (near line 4258), add:

```python
            # Usage quotas (spec 2 §4): the user's monthly LLM token allowance.
            _quota_uid = as_quota_user_id(getattr(current_user, "id", None))
            try:
                _quota_tokens = (
                    estimate_tokens_from_json(_sanitize_json_for_rate_limit(request_json)) if request_json else 1000
                )
            except _CHAT_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                _quota_tokens = 1000
            _llm_decision = await check_usage(
                _quota_uid,
                "limits.llm_tokens_per_month",
                max(1, _quota_tokens),
                lambda: llm_tokens_this_month(_quota_uid),
            )
            if not _llm_decision.allowed:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail={
                        "error": "limit_exceeded",
                        "category": "llm_tokens_month",
                        "current": int(_llm_decision.used),
                        "limit": _llm_decision.limit,
                        "message": "Monthly LLM token limit reached",
                    },
                )
```

  Import `as_quota_user_id`, `check_usage` and `llm_tokens_this_month` from `tldw_Server_API.app.core.Usage.quota_checks`. Confirm that `status` is in chat.py's FastAPI import.

- [ ] **Step 5: Run the suites**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Usage tldw_Server_API/tests/RAG_NEW/unit tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py tldw_Server_API/tests/Text2SQL`
Expected: all pass. If `tests/Text2SQL` doesn't exist, run whatever `git grep -l text2sql -- tldw_Server_API/tests` finds.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/app/api/v1/endpoints/text2sql.py tldw_Server_API/app/core/RAG/rag_service/transport.py tldw_Server_API/tests/Usage/test_quota_llm_rag.py
git commit -m "feat(quotas): per-user LLM tokens/month and RAG queries/day; RAG usage counted per user (spec 2 §4)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 6: Media ingest bytes per day, and workflows

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`.
  - Delete `_resolve_media_budget_context` and its call, the `# --- 1b. Resource Governor per-user concurrency budget ---` block, and the `rg_media_handle_id` release in the `finally:`.
  - Replace the `# --- Resource Governor per-user upload-bytes budget ---` block with a call to the new `_enforce_and_record_media_bytes`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/workflows.py`: `_enforce_workflows_daily_cap` and `_record_workflow_run_usage`.
- Modify: `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`: `workflow_run`, before a new run is created.
- Create: `tldw_Server_API/tests/Usage/test_quota_media_workflows.py`.

**Interfaces:**
- Consumes: `check_usage`, `ledger_used_today`, `seconds_until_utc_midnight` and `workflows_runs_decision` (Task 1).
- Produces: `persistence._enforce_and_record_media_bytes(user_id: int | None, total_uploaded_bytes: int) -> None`, which raises a 429 `HTTPException`.

- [ ] **Step 1: Write the failing tests**

```python
"""Per-user media-ingest bytes/day and workflow runs/day (spec 2 §4)."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import workflows as workflows_ep
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence
from tldw_Server_API.app.core.Usage import quota_checks

pytestmark = pytest.mark.unit
MB = 1024 * 1024


@pytest.fixture()
def quota(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A per-key limit and today's ledger usage, set by the test; ledger writes recorded."""
    state: dict = {"limits": {}, "used": 0.0, "recorded": []}

    async def _user_quota(_uid: int, key: str) -> object:
        """The test's limit for the key."""
        return state["limits"].get(key)

    async def _used(_uid: str, _category: str) -> float:
        """The test's usage."""
        return state["used"]

    async def _record(**kwargs: object) -> bool:
        """Record a media-bytes ledger write."""
        state["recorded"].append(kwargs)
        return True

    monkeypatch.setattr(quota_checks, "user_quota", _user_quota)
    monkeypatch.setattr(quota_checks, "ledger_used_today", _used)
    monkeypatch.setattr(persistence, "_record_media_ingestion_bytes_ledger_entry", _record)
    return state


async def test_media_bytes_recorded_even_without_a_limit(quota: dict) -> None:
    """Uploads are always counted (gate the check, never the record)."""
    await persistence._enforce_and_record_media_bytes(7, 3 * MB)
    assert [(r["entity_scope"], r["entity_value"], r["units"]) for r in quota["recorded"]] == [("user", "7", 3 * MB)]


async def test_media_bytes_429_when_daily_mb_spent(quota: dict) -> None:
    """An upload that would pass the user's daily MB is refused with 429 and not recorded."""
    quota["limits"]["limits.media_ingest_mb_per_day"] = 10
    quota["used"] = 8 * MB
    with pytest.raises(HTTPException) as exc:
        await persistence._enforce_and_record_media_bytes(7, 3 * MB)
    assert exc.value.status_code == 429 and int(exc.value.headers["Retry-After"]) >= 1
    assert quota["recorded"] == []
    await persistence._enforce_and_record_media_bytes(7, 2 * MB)
    assert len(quota["recorded"]) == 1


async def test_workflows_cap_429_at_allowance(quota: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """At the user's daily run allowance the endpoint refuses with 429 and rate-limit headers."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.delenv("WORKFLOWS_DISABLE_QUOTAS", raising=False)
    quota["limits"]["limits.workflows_runs_per_day"] = 2
    quota["used"] = 1.0
    await workflows_ep._enforce_workflows_daily_cap(request=SimpleNamespace(), current_user=SimpleNamespace(id=7), db=None)
    quota["used"] = 2.0
    with pytest.raises(HTTPException) as exc:
        await workflows_ep._enforce_workflows_daily_cap(request=SimpleNamespace(), current_user=SimpleNamespace(id=7), db=None)
    assert exc.value.status_code == 429


async def test_scheduler_refuses_a_run_past_the_allowance(quota: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduled runs count against the same daily allowance and are refused before the run is created."""
    from tldw_Server_API.app.core.Scheduler.handlers import workflows as sched

    created: list[str] = []
    monkeypatch.setattr(sched, "_get_wf_db", lambda: SimpleNamespace(create_run=lambda **kw: created.append("run")))
    quota["limits"]["limits.workflows_runs_per_day"] = 1
    quota["used"] = 1.0
    with pytest.raises(RuntimeError, match="Daily workflow run quota"):
        await sched.workflow_run({"user_id": 7, "workflow_id": 1, "definition_snapshot": {}})
    assert created == []
```

The scheduler payload above is a minimal guess. Read `workflow_run` and `_resolve_payload_user_id` first and use the smallest valid payload, keeping the assertions unchanged.

- [ ] **Step 2: Run them and confirm they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_quota_media_workflows.py -q`
Expected: FAIL. `_enforce_and_record_media_bytes` doesn't exist, the workflows cap still reads the RG policy, and the scheduler doesn't check.

- [ ] **Step 3: Media**

In `persistence.py`, import the module (`from tldw_Server_API.app.core.Usage import quota_checks`), not its names. Tests patch `quota_checks.ledger_used_today`, and a name imported into `persistence` wouldn't see the patch. Then add this module-level helper next to `_record_media_ingestion_bytes_ledger_entry`:

```python
async def _enforce_and_record_media_bytes(user_id: int | None, total_uploaded_bytes: int) -> None:
    """Refuse an upload past the user's daily MB (429); otherwise count it (spec 2 §4)."""
    if user_id is None or total_uploaded_bytes <= 0:
        return
    uid = str(int(user_id))

    async def _used_mb() -> float:
        return (await quota_checks.ledger_used_today(uid, _MEDIA_INGESTION_BYTES_CATEGORY)) / (1024 * 1024)

    decision = await quota_checks.check_usage(
        int(uid), "limits.media_ingest_mb_per_day", total_uploaded_bytes / (1024 * 1024), _used_mb
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily ingestion size budget exceeded.",
            headers={"Retry-After": str(quota_checks.seconds_until_utc_midnight())},
        )
    await _record_media_ingestion_bytes_ledger_entry(
        entity_scope="user",
        entity_value=uid,
        units=int(total_uploaded_bytes),
        op_id=f"media-ingestion-bytes:user:{uid}:{uuid4().hex}",
    )
```

Replace the whole `# --- Resource Governor per-user upload-bytes budget ---` block (the `if rg_governor is not None ... and total_uploaded_bytes > 0:` try/except, through its `except` logging) with:

```python
            # --- Usage quota: per-user upload bytes per day (spec 2 §4) ---
            await _enforce_and_record_media_bytes(
                getattr(current_user, "id", None) if current_user is not None else None,
                total_uploaded_bytes,
            )
```

Then delete:
- `_resolve_media_budget_context`;
- its call and the `rg_governor, rg_policy_id, rg_policy, rg_entity = ...` assignment;
- `rg_jobs_limit`, `rg_daily_bytes_cap`, `rg_media_handle_id` and their uses: the `# --- 1b. Resource Governor per-user concurrency budget ---` block and the `finally:` release block.

Remove imports that become unused, such as `RGRequest` and `_build_ingestion_budget_headers` if nothing else uses them; check with `git grep`. Ruff must not report new F401/F841.

- [ ] **Step 4: Workflows**

In `workflows.py`, import `as_quota_user_id`, `seconds_until_utc_midnight` and `workflows_runs_decision` from `quota_checks`, then replace `_enforce_workflows_daily_cap` with:

```python
async def _enforce_workflows_daily_cap(
    *,
    request: Request,
    current_user: User,
    db: WorkflowsDatabase,
) -> None:
    """Enforce the user's daily workflow-run allowance (spec 2 §4); 429 when spent."""
    if not usage_quotas_enabled():
        return
    try:
        if env_flag_enabled("WORKFLOWS_DISABLE_QUOTAS"):
            return
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Workflows quota: WORKFLOWS_DISABLE_QUOTAS check failed: {}", exc)
    decision = await workflows_runs_decision(as_quota_user_id(getattr(current_user, "id", None)))
    if decision.allowed:
        return
    retry_after = seconds_until_utc_midnight()
    limit = int(decision.limit or 0)
    headers = _build_rate_limit_headers(limit, max(0, limit - int(decision.used)), int(time.time()) + retry_after)
    raise HTTPException(status_code=429, detail="Daily quota exceeded", headers=headers)
```

In `_record_workflow_run_usage`, always key by user: replace the `if request is not None: entity = derive_entity_key(request) else: ...` lines with `entity = f"user:{resolve_user_id_for_request(current_user, error_status=500)}"`.

Remove imports that become unused, such as `derive_entity_key`, `RGRequest`, `_log_workflows_quota_rg_fallback_once`, `backfill_legacy_runs_to_ledger`, `_WORKFLOWS_BACKFILL_CACHE` and `_tenant_id_for_user`; delete only names with no remaining use.

- [ ] **Step 5: Scheduler**

In `Scheduler/handlers/workflows.py` `workflow_run`, find the branch where a new run is created (`resume_run_id` is falsy, just before `db.create_run(`). Add this as its first statement:

```python
        from tldw_Server_API.app.core.Usage.quota_checks import workflows_runs_decision

        if not (await workflows_runs_decision(user_id)).allowed:
            raise RuntimeError("Daily workflow run quota exceeded")
```

- [ ] **Step 6: Run the suites**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Usage tldw_Server_API/tests/MediaIngestion_NEW/unit tldw_Server_API/tests/Workflows tldw_Server_API/tests/Resource_Governance tldw_Server_API/tests/Scheduler`
Expected: all pass.
- Tests that pinned the RG media budget (`test_resource_governor_endpoint.py` media cases, `test_persistence_original_storage.py` RG parts) or the RG/backfill workflows cap (`test_workflows_runs_daily_cap.py`, `test_e2e_workflows_daily_cap.py`, `test_rg_cutover_workflows_quota.py`) must be rewritten to the resolver contract.
- Delete a case only when it pinned the removed RG mechanism itself, and list every deletion in your report.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py tldw_Server_API/app/api/v1/endpoints/workflows.py tldw_Server_API/app/core/Scheduler/handlers/workflows.py tldw_Server_API/tests
git commit -m "feat(quotas): per-user media MB/day (always counted) and workflow runs/day incl. scheduled (spec 2 §4)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 7: Evaluations and chatbooks

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`: `check_rate_limit`, `_reserve_request_usage` and `_reserve_request_usage_sync`.
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`: `ChatbookJobLimits`, `_resolve_job_limits`, the admission helpers and their two callers.
- Create: `tldw_Server_API/tests/Usage/test_quota_evals_chatbooks.py`.

**Interfaces:**
- Consumes: `quota_resolver.user_quota` and `quota_checks.as_quota_user_id`.
- Produces:
  - `_reserve_request_usage(..., max_evaluations_per_day: float | None = None, max_tokens_per_day: float | None = None)`;
  - `ChatbookJobLimits(exports_per_day, imports_per_day, concurrent_jobs)`, all `float | None`;
  - `ChatbookService._resolve_job_limits() -> ChatbookJobLimits`;
  - admission helpers that take `limits: ChatbookJobLimits`.

- [ ] **Step 1: Write the failing tests**

```python
"""Per-user evaluation caps and chatbook job allowances (spec 2 §4)."""

import pytest

from tldw_Server_API.app.core.Chatbooks import chatbook_service
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookJobLimits
from tldw_Server_API.app.core.Chatbooks.quota_manager import QuotaExceededError
from tldw_Server_API.app.core.Evaluations.user_rate_limiter import UserRateLimiter

pytestmark = pytest.mark.unit


def _limiter(tmp_path) -> UserRateLimiter:
    """A limiter on a throwaway SQLite DB with the schema in place."""
    return UserRateLimiter(db_path=str(tmp_path / "evals.db"))


async def test_daily_evaluation_cap_is_atomic_with_the_reservation(tmp_path) -> None:
    """The Nth+1 evaluation of the day is refused inside the same transaction that counts it."""
    limiter = _limiter(tmp_path)
    config = await limiter._get_user_config("7")
    for _ in range(2):
        ok, _meta = await limiter._reserve_request_usage("7", "/e", 0, 0.0, config, max_evaluations_per_day=2)
        assert ok is True
    ok, meta = await limiter._reserve_request_usage("7", "/e", 0, 0.0, config, max_evaluations_per_day=2)
    assert ok is False and meta["error"] == "Daily evaluation limit exceeded" and meta["limit"] == 2


async def test_daily_evaluation_token_cap(tmp_path) -> None:
    """A request whose tokens would pass the daily token cap is refused."""
    limiter = _limiter(tmp_path)
    config = await limiter._get_user_config("7")
    assert (await limiter._reserve_request_usage("7", "/e", 900, 0.0, config, max_tokens_per_day=1000))[0] is True
    ok, meta = await limiter._reserve_request_usage("7", "/e", 101, 0.0, config, max_tokens_per_day=1000)
    assert ok is False and meta["error"] == "Daily evaluation token limit exceeded"


class _Service(chatbook_service.ChatbookService):
    """A ChatbookService with the DB counters stubbed."""

    def __init__(self, exports: int, active: int) -> None:
        """Skip the real constructor; set only what admission reads."""
        self.user_id = "7"
        self.user_tier = "free"
        self.db = None
        self._exports = exports
        self._active = active

    def _count_operations_today_for_quota(self, operation_type: str) -> int:
        """Today's exports."""
        return self._exports

    def _count_active_jobs_for_quota(self) -> int:
        """Active jobs."""
        return self._active


def _quotas_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the test-mode flags that auto-disable chatbook quotas; switch quotas on."""
    for name in ("CHATBOOKS_DISABLE_QUOTAS", "TEST_MODE", "TESTING", "PYTEST_CURRENT_TEST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")


def test_chatbook_admission_uses_the_resolved_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exports/day and concurrency come from the passed limits; None is unlimited."""
    _quotas_live(monkeypatch)
    _Service(exports=100, active=100)._check_chatbook_job_admission("export", ChatbookJobLimits())
    with pytest.raises(QuotaExceededError):
        _Service(exports=3, active=0)._check_chatbook_job_admission("export", ChatbookJobLimits(exports_per_day=3))
    with pytest.raises(QuotaExceededError):
        _Service(exports=0, active=2)._check_chatbook_job_admission("export", ChatbookJobLimits(concurrent_jobs=2))


async def test_resolve_job_limits_reads_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service resolves its three chatbook keys for its user."""
    values = {"limits.chatbooks_exports_per_day": 4, "limits.chatbooks_concurrent_jobs": 1}

    async def _user_quota(_uid: int, key: str) -> object:
        """The test's values."""
        return values.get(key)

    monkeypatch.setattr(chatbook_service, "user_quota", _user_quota)
    service = _Service(exports=0, active=0)
    service.user_id_int = 7
    assert await service._resolve_job_limits() == ChatbookJobLimits(exports_per_day=4, imports_per_day=None, concurrent_jobs=1)
```

`UserRateLimiter(db_path=...)` creates its schema in the constructor. `QuotaExceededError` is the one `chatbook_service` raises; import it from wherever that module imports it.

- [ ] **Step 2: Run them and confirm they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_quota_evals_chatbooks.py -q`
Expected: FAIL with an unexpected keyword `max_evaluations_per_day`, and `ChatbookJobLimits` missing.

- [ ] **Step 3: Evaluations**

In `user_rate_limiter.py`:
- Add the keyword-only arguments `max_evaluations_per_day: float | None = None, max_tokens_per_day: float | None = None` to `_reserve_request_usage` and `_reserve_request_usage_sync`, and pass them through.
- In `_reserve_request_usage_sync`, right after `conn.execute("BEGIN IMMEDIATE")` and the `try:`, before the cost block, add:

```python
                if max_evaluations_per_day is not None or max_tokens_per_day is not None:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT total_evaluations, total_tokens FROM daily_usage WHERE user_id = ? AND date = ?",
                        (user_id, str(today)),
                    )
                    row = cursor.fetchone()
                    used_evaluations = int(row[0] or 0) if row else 0
                    used_tokens = int(row[1] or 0) if row else 0
                    breach = None
                    if max_evaluations_per_day is not None and used_evaluations + 1 > max_evaluations_per_day:
                        breach = ("Daily evaluation limit exceeded", max_evaluations_per_day, used_evaluations)
                    elif max_tokens_per_day is not None and used_tokens + normalized_tokens > max_tokens_per_day:
                        breach = ("Daily evaluation token limit exceeded", max_tokens_per_day, used_tokens)
                    if breach is not None:
                        conn.rollback()
                        reset_at = datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
                        return False, {
                            "error": breach[0],
                            "limit": breach[1],
                            "used": breach[2],
                            "retry_after": max(1, int((reset_at - datetime.now(timezone.utc)).total_seconds())),
                            "resets_at": reset_at.isoformat(),
                        }, now, normalized_tokens
```

- In `check_rate_limit`, right after `config = await self._get_user_config(user_id)`, add:

```python
        from tldw_Server_API.app.core.Usage.quota_checks import as_quota_user_id
        from tldw_Server_API.app.core.Usage.quota_resolver import user_quota

        quota_uid = as_quota_user_id(user_id)
        daily_caps = {
            "max_evaluations_per_day": await user_quota(quota_uid, "limits.evaluations_per_day"),
            "max_tokens_per_day": await user_quota(quota_uid, "limits.evaluation_tokens_per_day"),
        }
```

  Then pass `**daily_caps` to each of the three `self._reserve_request_usage(...)` calls in `check_rate_limit`.

- [ ] **Step 4: Chatbooks**

In `chatbook_service.py`:
- Add `from dataclasses import dataclass` (if missing) and `from tldw_Server_API.app.core.Usage.quota_resolver import user_quota`.
- Add at module level:

```python
@dataclass(frozen=True)
class ChatbookJobLimits:
    """A user's chatbook job allowances (spec 2 §4); None means unlimited."""

    exports_per_day: float | None = None
    imports_per_day: float | None = None
    concurrent_jobs: float | None = None
```

- Add this method to `ChatbookService`:

```python
    async def _resolve_job_limits(self) -> ChatbookJobLimits:
        """This user's chatbook allowances from their limits.* values."""
        return ChatbookJobLimits(
            exports_per_day=await user_quota(self.user_id_int, "limits.chatbooks_exports_per_day"),
            imports_per_day=await user_quota(self.user_id_int, "limits.chatbooks_imports_per_day"),
            concurrent_jobs=await user_quota(self.user_id_int, "limits.chatbooks_concurrent_jobs"),
        )
```

- Change `_check_chatbook_job_admission(self, operation_type: str, limits: ChatbookJobLimits)`.
  - Keep the `QuotaManager(...)._quotas_disabled` early return; that is the env and test off-switch.
  - Replace the three `quota_manager.quotas[...]` reads with `limits.exports_per_day` / `limits.imports_per_day` / `limits.concurrent_jobs`, treating `None` as unlimited (skip the count).
  - Format the limits in the messages with `int(...)`.
- Add a `limits: ChatbookJobLimits` parameter to `_check_chatbook_job_admission_with_lock`, `_save_export_job_with_quota` and `_save_import_job_with_quota`, and pass it through.
- In `create_chatbook` and `import_chatbook`, compute `job_limits = await self._resolve_job_limits()` once, before the first admission call. Pass it to both the sync and async admission paths (`_check_chatbook_job_admission_with_lock("export", job_limits)`, `_save_export_job_with_quota(job, job_limits)`, and the import equivalents).

- [ ] **Step 5: Run the suites**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Usage tldw_Server_API/tests/Evaluations tldw_Server_API/tests/Chatbooks`
Expected: all pass.
- Tests that call the admission helpers with the old signature pass `ChatbookJobLimits(...)`.
- Tests that pinned the tier-table daily caps assert the passed limits instead.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Evaluations/user_rate_limiter.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/tests
git commit -m "feat(quotas): per-user evaluation daily caps (atomic) and chatbook job allowances (spec 2 §4)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 8: RG policy cleanup, docs and follow-up tasks

**Files:**
- Modify: `tldw_Server_API/Config_Files/resource_governor_policies.yaml`: `audio.default` (lines ~113-121), `workflows.default` (~198-201), `media.default` (~225-229).
- Modify: `tldw_Server_API/tests/Usage/test_usage_quotas_policy_defaults.py`.
- Modify: `Docs/Operations/Env_Vars.md` (the Usage Quotas section), `Docs/Operations/Rate_Limits_Troubleshooting.md`, and `Docs/Published/Env_Vars.md` via the refresh script.

- [ ] **Step 1: Extend the policy test**

Add to `test_usage_quotas_policy_defaults.py`:

```python
def test_stock_policies_carry_no_usage_quota_categories() -> None:
    """Media, audio and workflow quotas moved to limits.*; the stock RG policy keeps only rates."""
    policies = yaml.safe_load(_POLICIES.read_text())["policies"]
    assert not {"jobs", "ingestion_bytes"} & set(policies["media.default"])
    assert not {"streams", "jobs", "minutes"} & set(policies["audio.default"])
    assert "workflows_runs" not in policies["workflows.default"]
```

Run it; it must FAIL.

- [ ] **Step 2: Edit the YAML**

- Delete these lines:
  - `streams`, `jobs` and `minutes` from `audio.default`;
  - `workflows_runs` from `workflows.default`;
  - `jobs` and `ingestion_bytes` from `media.default`.
- Keep every `requests`, `scopes` and `fail_mode` line.
- Change the `audio.default` comment to: "Per-user audio minutes and queued-job concurrency are UserProfiles limits.* values (spec 2)."

Run the policy test and the RG suite; both must pass.

- [ ] **Step 3: Docs**

Replace the body of `## Usage Quotas` in `Env_Vars.md` with:

```markdown
## Usage Quotas

Usage quotas are per-user budgets, **off by default**: a stock install, single-user or multi-user, applies none. Request rate limits are separate (Resource Governor, below). Design: `Docs/Design/2026-10-02-usage-quota-posture-design.md`.

- `USAGE_QUOTAS_ENABLED`: master switch (`true|1|false|0`). Resolution: this env var > `LIMIT_ENFORCEMENT_ENABLED` (legacy, when set; warns once) > `config.txt` `[Usage-Quotas] enabled` > default `false`. Usage is recorded whether or not it is on.
- With the switch on, a quota applies only where a platform admin set a value. Values are UserProfiles `limits.*` keys, set per user (`PATCH /api/v1/admin/users/{id}/profile`) or per team/org (`PUT`/`DELETE /api/v1/admin/{orgs|teams}/{id}/profile/overrides/{key}`, body `{"value": n}`; `null` removes). A team or org value is each member's own allowance. The user's own value wins; otherwise the most generous value among their teams that set the key; otherwise the most generous among their orgs. `0` blocks; no value anywhere means unlimited. Changes reach every worker within 60 s.
- Keys: `limits.audio_daily_minutes`, `limits.transcription_minutes_per_month`, `limits.audio_concurrent_jobs` (queued jobs), `limits.llm_tokens_per_month` (enforced on `/chat/completions`), `limits.rag_queries_per_day` (RAG, Text2SQL, MCP), `limits.media_ingest_mb_per_day`, `limits.workflows_runs_per_day` (API-started and scheduled), `limits.evaluations_per_day`, `limits.evaluation_tokens_per_day`, `limits.chatbooks_exports_per_day`, `limits.chatbooks_imports_per_day`, `limits.chatbooks_concurrent_jobs`, `limits.storage_quota_mb` (per user only for now). Days and months are UTC.
- Not quotas (unchanged): per-file upload size caps, character-chat count caps, per-minute rates. Synchronous concurrency (media ingest requests, audio streams, direct transcription) is not limited per user.
- Billing-plan limits additionally need a billing repository (hosted product only); without one, billing checks never run. A wired repository with the switch off logs a warning once, at startup.
- Operators on `RG_POLICY_STORE=db` whose stored `evals.*` policies carry a `daily_cap` keep that cap until they remove it from the stored policy; evaluation daily caps now come from `limits.evaluations_per_day` / `limits.evaluation_tokens_per_day`.
```

In `Rate_Limits_Troubleshooting.md`:
- **Media-concurrency 429 row:** it no longer occurs (synchronous concurrency isn't limited). Replace it with a "Daily ingestion size budget exceeded" 429 row, fixed by "raise or remove the user's `limits.media_ingest_mb_per_day`".
- **Audio 402 row:** the fix becomes "raise or remove the user's `limits.audio_daily_minutes` / `limits.transcription_minutes_per_month`". Drop `AUDIO_TIER_LIMITS_JSON`.
- **New rows:** RAG `rag_queries_day` 402, chat `llm_tokens_month` 402, workflows "Daily quota exceeded" 429, chatbooks daily/concurrency 429. Each is fixed by "raise or remove the user's `limits.*` value" and applies "only when `USAGE_QUOTAS_ENABLED` is on".

Then run:

```bash
bash Helper_Scripts/refresh_docs_published.sh
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Docs
```

- [ ] **Step 4: File the follow-up tasks**

```bash
backlog task create "Per-user synchronous concurrency quotas (media ingest, audio streams, direct transcription)" --labels quotas,backend -d "Deferred from spec 2 (Docs/Design/2026-10-02-usage-quota-posture-design.md, review ruling R1). RGRequest has no per-request max_concurrent, and PR B stopped reserving the media/audio jobs/streams leases, so these paths are unlimited. Approach: an optional RGRequest.max_concurrent honored by the memory and Redis governors, with limits.media_concurrent_jobs / limits.audio_concurrent_streams resolved per user." --ac "A per-user concurrency limit on media ingest and audio streams is enforced on both RG backends" --ac "Unset means unlimited"
backlog task create "Evaluation cost caps per user" --labels quotas,evaluations -d "Deferred from spec 2. Cost caps never fired: every caller passes estimated_cost=0.0 and the monthly cap is never compared. Needs real cost estimation, then limits.evaluation_cost_per_day_usd / _per_month_usd." --ac "A per-user daily and monthly evaluation cost cap blocks once recorded cost reaches it"
backlog task create "Gate LLM tokens/month at every LLM entry point" --labels quotas,llm -d "Deferred from spec 2. limits.llm_tokens_per_month is checked only on /chat/completions; character chat, RAG generation, workflows and persona LLM calls are counted in llm_usage_log but not gated." --ac "Every user-initiated LLM call checks limits.llm_tokens_per_month before dispatch"
```

Check each new ID against open PRs, using the same check as PR A's Task 0, and renumber on a collision.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/Config_Files/resource_governor_policies.yaml tldw_Server_API/tests/Usage/test_usage_quotas_policy_defaults.py Docs/Operations/Env_Vars.md Docs/Operations/Rate_Limits_Troubleshooting.md Docs/Published backlog/tasks
git commit -m "docs(quotas): limits.* keys and group routes; RG policy keeps only rates; file deferred follow-ups

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 9: Ship PR B

- [ ] **Step 1: Sweep every test that touches a changed contract**

List test files only. The PR A sweep broke on conftest and JSON paths, so filter to `test_*.py`.

```bash
git grep -l -e get_limits_for_user -e can_start_job -e can_start_stream -e _enforce_workflows_daily_cap -e _record_workflow_run_usage \
  -e _resolve_media_budget_context -e _reserve_request_usage -e _check_chatbook_job_admission -e upgrade_user_tier \
  -e "limits\." -e log_rag_queries_for_org_context -e enforce_rag_query_limit_for_org_context -e "evals\." -e "media.default" \
  -e "audio.default" -e "workflows.default" -- 'tldw_Server_API/tests/**/test_*.py' | sort > /tmp/prb_tests.txt
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 -p no:cacheprovider $(cat /tmp/prb_tests.txt) tldw_Server_API/tests/Usage tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Docs
```

Expected: everything passes, or fails identically on `origin/dev`. Prove any such match by running the failing IDs on a detached `origin/dev` checkout. Fix every new failure.

- [ ] **Step 2: Ruff and Bandit against the merge base**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check $(git diff --name-only origin/dev...HEAD -- 'tldw_Server_API/app/*.py')
uvx bandit -q -ll $(git diff --name-only origin/dev...HEAD -- 'tldw_Server_API/app/*.py')
```

Expected: Bandit finds nothing at medium or above. Ruff findings must be identical to the merge base for the touched files; compare per-rule tallies.

- [ ] **Step 3: Coordinate, open the PR, run the Qodo loop, merge**

1. **Coordinate:** message `tldw-server-03` for a merge slot, listing the touched areas: UserProfiles, the admin routes, audio quota, chat, RAG, text2sql, persistence, workflows and the scheduler, evaluations, chatbooks, the RG YAML, and Env_Vars with its mirror. Push and run CI meanwhile.
2. **Open the PR:**

```bash
git push -u origin fix/usage-quotas-per-user
gh pr create --base dev --title "feat(quotas): per-user/team/org usage quota values and enforcement (spec 2, PR B)" --body-file <body>
```

   The body covers:
   - the resolver and precedence;
   - the write paths and group routes;
   - each site's key and counter;
   - what became unlimited (synchronous concurrency);
   - the DB policy-store note;
   - the follow-up task IDs;
   - the verification counts.

   It ends with the waiver line and the Claude Code footer.
3. **Qodo:** address every finding, fixing it or declining it with a posted rationale. Add the accepted titles to the scratchpad `qodo_open.py`.
4. **Merge:** only on the peer's go-ahead, and by running the merge queue for this PR alone. Afterwards, message the peer and run `backlog task edit 13434 --check-ac 2 --append-notes "PR B #<n> merged <sha> ..."`.
