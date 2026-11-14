# Usage Quotas PR D (Reporting, Removals and Docs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every endpoint that reports a usage quota shows the enforced per-user value (and `null` for unlimited) with real usage figures. The last tier-based quota code is removed, including a chatbooks double enforcement that still caps users at 10 exports a day. The quota model is documented in an ADR and an operator page.

**Architecture:** Reporting reads the same sources enforcement uses: `quota_resolver.user_quota` for limits, and the `ResourceDailyLedger` (through `core/Usage/quota_checks.py`) or the evaluations `daily_usage` table for usage. Tier tables stay only where an admin API still stores a tier, which now affects nothing. Docs describe what the code does today.

**Tech Stack:** FastAPI, Pydantic v2, SQLite and PostgreSQL, loguru, pytest. The frontend is TypeScript and React in `apps/packages/ui`.

**Spec:** `Docs/Design/2026-10-02-usage-quota-posture-design.md`, sections 7 and 9, plus Delivery "PR D". PRs A (#3098), B (#3144) and C (#3199, merged 5775d3fbbe) are on `dev`.

## Global Constraints

- **Branches and PRs:** all PRs target `dev`, never `main`. Never use `git stash`; never pass `--no-verify`. The branch is `fix/usage-quotas-reporting`; it already carries the TASK-13434 AC3 commit.
- **Commit and PR text:**
  - Commits end with exactly `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. That trailer is the owner's rule.
  - The PR body has a `## Change summary` section recording the ADR-004 waiver: "Waived by the repository owner (@rmusser01) on 2026-09-22 … *"I authorize waiving the change summary for each."* On 2026-10-03 … confirmed that this waiver covers spec 2 PRs B, C and D." It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Backlog:** use backlog-py only (`PYTHONPATH=tools/backlog-py/src <venv python> -m backlog_py --cwd . task ...`, ids written as `TASK-N`). Never use the Node CLI, never hand-edit, and stage with `git add -A backlog/tasks`.
- **Code conventions:** loguru only; no new dependencies.
  - New tests carry `pytestmark = pytest.mark.unit` (integration tests carry `pytest.mark.integration`), plus a one-line docstring per test and helper.
  - No review-process tags in code comments.
- **Running tests:**
  - Use `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (there is no bare `python`), with `TLDW_TEST_NO_DOCKER=1` for suites.
  - The host is shared and heavily loaded: use `-n 4`, add `--timeout=600`, and split runs that stall.
  - A failure counts as environmental only if it also fails on `origin/dev` (a detached checkout) or its import graph excludes the changed files.
- **Quotas are ON in the test session.** The root conftest sets `LIMIT_ENFORCEMENT_ENABLED=true` and clears the resolver cache per test. Quotas-off tests clear both `USAGE_QUOTAS_ENABLED` and `LIMIT_ENFORCEMENT_ENABLED`.
- **Fakes that can't fail.** Never assert inside a fake that raises within a module that swallows `AssertionError`, or within a broad `except`. Use call-recording fakes.
- **Shared test DBs** are session-scoped per xdist worker. Every test removes the overrides it writes, in `finally`.
- **Semantics:** quota values are UserProfiles `limits.*` keys resolved per user (the user's own value, then the most generous team value, then the most generous org value). `0` blocks; `null` (None) everywhere means unlimited. Every quota is off unless `USAGE_QUOTAS_ENABLED` is on, and usage is recorded either way.
- **API contract:** schema changes alter the OpenAPI contract. Refresh the fingerprint with `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/export_openapi_schema.py --fingerprint apps/tldw-frontend/lib/api/openapi.fingerprint.json`, then run the same script with `--check apps/tldw-frontend/lib/api/openapi.fingerprint.json`. Commit the fingerprint in the task that changes the schema.
- **Line-numbered lint baselines.** `tests/lint` holds ratchets keyed by file and line (for example `Helper_Scripts/ci/scope_predicate_baseline.txt`). If an edit moves a recorded line, regenerate that baseline with its script's `--write-baseline` and commit it. Never add a new finding.
- **Mirrored docs:** any edit under a tree that `Helper_Scripts/refresh_docs_published.sh` mirrors needs that script run, with the `Docs/Published` diff in the same commit, then `tests/Docs`.
- **Merging:** merge into `dev` only on the go-ahead of peer session `tldw-server-77`, and message it when the merge lands. `gh variable get MERGE_QUEUE` is unset, so rebase only the PR about to merge, wait for the seven required statuses, then merge.

## Rulings carried into this plan

- **ADR number.** Spec Delivery says "ADR-058", but 058 is taken (`058-jobs-completion-row-identity.md`). The highest is 063, and 060 to 062 are absent from dev. This PR uses **ADR-064**. Before opening the PR, re-check that no open PR adds an `Docs/ADR/064-*` file; if one does, take the next free number.
- **There is no chatbooks usage summary endpoint.** Spec §9 lists "the chatbooks usage summary". `QuotaManager.get_usage_summary` has no route and no callers, and its numbers come from the tier tables. It is deleted. A user's chatbooks limits are already visible as `limits.chatbooks_*` in the profile config. That is a spec erratum, recorded in the spec and the PR body.
- **Concurrency hooks stay as no-ops.** `can_start_job`, `can_start_stream`, `finish_job`, `finish_stream`, `heartbeat_stream` and `heartbeat_jobs` in `audio_quota.py` keep their signatures and callers as documented no-op hooks for TASK-13435 (per-user synchronous concurrency). Only the dead Resource Governor handle machinery behind them is deleted. Tests that pin an endpoint's response to a denial (for example `test_http_concurrent_jobs_cap`) stay: they test the endpoint contract that TASK-13435 will use.
- **Evaluations tier fields stay in storage.** `RateLimitConfig.evaluations_per_day` / `total_tokens_per_day`, the yaml tiers and the `user_rate_limits` columns are no longer enforced (PR B) and are no longer displayed (this PR). Dropping the columns would need a migration for no behavior change, so they stay inert.
- **`audio_usage_daily` stays.** `increment_jobs_started` still writes `jobs_started` and the one-time ledger backfill still reads old rows. Nothing reads them for limits or reporting after this PR. Removing the table is outside this spec.
- **`DEFAULT_STORAGE_QUOTA_MB` stays as a setting.** The PR C migration reads it to choose which old values to skip. It gets a one-time deprecation warning when set explicitly, and its `ge=100` validator is relaxed to `ge=0`, so that a deprecated, ignored value can't fail startup.
- **Audit events on the two storage quota admin endpoints** are a follow-up. Task 5 files it as a backlog task.

## Review Focus

1. **Chatbooks with quotas on and nothing set** must allow an 11th export in a day and a 3rd concurrent job. Today the endpoint's tier pre-check refuses both, which is the regression this PR removes. Covered by the Task 1 test `test_export_endpoint_has_no_tier_cap`.
2. **Unlimited is reported as `null`, never `0`,** on `GET /audio/jobs/admin/owner/{id}/processing`, `GET /audio/stream/limits` (including its error fallback) and `GET /evaluations/rate-limits`. Covered by the Task 2 and Task 3 tests.
3. **Audio "used" minutes reflect real consumption.** After minutes are recorded in the ledger, both `/audio/stream/limits` and the profile `quotas.audio` show them, daily and monthly. Today both always say 0. Covered by the Task 2 test `test_daily_and_monthly_minutes_come_from_the_ledger`.
4. **A monthly audio breach says "monthly", not "daily".** Covered by the Task 2 test `test_monthly_breach_message_names_the_month`.
5. **The evaluations widget shows "Unlimited"** rather than "5/null" or "5/0" when no daily limit applies, and a storage quota of 0 shows as full in the Research Workspace header rather than hidden. Covered by the Task 3 and Task 4 frontend checks.

---

### Task 1: Chatbooks — remove the tier tables and the endpoint double enforcement

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`:
  - delete the export pre-checks (~558-571: the `QuotaManager(...)` construction, `check_export_quota` and `check_concurrent_jobs`);
  - delete the import pre-checks (~862-875: the same three, but keep the `QuotaManager` construction if the import path's later `check_file_size` (~913) uses that variable);
  - in the preview handler (~1145-1158), drop the duplicate hard-coded `100 * 1024 * 1024` check.
- Modify: `tldw_Server_API/app/core/Chatbooks/quota_manager.py`:
  - delete `UserTier`, `DEFAULT_QUOTAS`, `PREMIUM_QUOTAS`, `VALID_TIERS`, `UNLIMITED_QUOTA`, `_get_quotas_for_tier`, `check_storage_quota`, `check_export_quota`, `check_import_quota`, `check_concurrent_jobs`, `get_usage_summary` and its private counters (`_get_current_storage_usage`, `_get_operations_count_today`, `_get_active_jobs_count`), `usage_cache` and `get_quota_manager`;
  - delete `record_operation` too, if `git grep -n record_operation -- tldw_Server_API` shows no callers outside this file;
  - keep `QuotaManager.__init__(user_id, user_tier='free', db=None)` (the tier argument is now ignored), `_quotas_disabled`, and `check_file_size`.
- Modify: `tldw_Server_API/tests/lint/test_private_coercion_ratchet.py`, the `"app/core/Chatbooks/quota_manager.py"` entry, if the count changes.
- Test: create `tldw_Server_API/tests/Chatbooks/test_chatbooks_no_tier_quotas.py`. Update `tests/Chatbooks/test_chatbooks_full_account_export_contract.py` (its `_PassingQuotaManager` defines methods the endpoint no longer calls; keep the class if the endpoint still constructs `QuotaManager`, and drop the dead methods).

**Interfaces:**
- Produces: `quota_manager.MAX_CHATBOOK_FILE_SIZE_MB = 100` (module constant), and `QuotaManager.check_file_size(file_size_bytes) -> tuple[bool, str]`, which uses that constant.
- Unchanged and relied on: `QuotaManager(...)._quotas_disabled`, which `chatbook_service._check_chatbook_job_admission` reads. Admission against `limits.chatbooks_exports_per_day`, `limits.chatbooks_imports_per_day` and `limits.chatbooks_concurrent_jobs` lives in `chatbook_service.py` (`_resolve_job_limits` ~7968, `_check_chatbook_job_admission` ~7974) and is not changed here.

- [ ] **Step 1: Write the failing tests**

```python
"""Chatbooks quotas come only from limits.chatbooks_* (spec 2 §7): no tier caps remain."""

import pytest

from tldw_Server_API.app.core.Chatbooks import quota_manager as qm

pytestmark = pytest.mark.unit


def test_quota_manager_has_no_tier_tables() -> None:
    """The free/premium/enterprise tables and their checks are gone."""
    for name in ("DEFAULT_QUOTAS", "PREMIUM_QUOTAS", "UserTier", "get_quota_manager"):
        assert not hasattr(qm, name), name
    for method in ("check_export_quota", "check_import_quota", "check_concurrent_jobs",
                   "check_storage_quota", "get_usage_summary"):
        assert not hasattr(qm.QuotaManager, method), method


async def test_file_size_cap_is_the_constant_for_every_tier() -> None:
    """check_file_size uses MAX_CHATBOOK_FILE_SIZE_MB whatever tier string is passed."""
    assert qm.MAX_CHATBOOK_FILE_SIZE_MB == 100
    limit = qm.MAX_CHATBOOK_FILE_SIZE_MB * 1024 * 1024
    for tier in ("free", "premium", "enterprise", "nonsense"):
        manager = qm.QuotaManager("7", tier)
        assert (await manager.check_file_size(limit))[0] is True
        allowed, message = await manager.check_file_size(limit + 1)
        assert allowed is False and "100MB" in message
```

Add `test_export_endpoint_has_no_tier_cap`, with the docstring "With quotas on and no limits.chatbooks_* set, an 11th export in a day is not refused by a tier cap." Drive the real export route the way `tests/Chatbooks/test_chatbooks_full_account_export_contract.py` does (same client and service fixtures), and make sure the endpoint really reaches `QuotaManager`:
- Stop the test-mode auto-disable (`_quotas_disabled` is True under `PYTEST_CURRENT_TEST`). Do it by patching `QuotaManager._quotas_disabled` to False through a small subclass, or by `monkeypatch.delenv` of `PYTEST_CURRENT_TEST`, `TEST_MODE` and `TESTING` around the request. Read how `tests/Usage/test_quota_evals_chatbooks.py` gets admission to run, and mirror that.
- Make the service's daily export count report 10 (the old free-tier limit), with `limits.chatbooks_exports_per_day` unset.
- Assert the response is not a 429. Today it is a 429 from `check_export_quota`.

- [ ] **Step 2: Run them and confirm they FAIL**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_no_tier_quotas.py -q -p no:xdist`

Expected: FAIL. The tier tables exist, `MAX_CHATBOOK_FILE_SIZE_MB` is undefined, and the export returns 429.

- [ ] **Step 3: Implement**

`quota_manager.py` becomes the module docstring, the imports it still needs, `_CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS` (only if still referenced), `_env_flag`, and:

```python
# Fixed per-request guardrail (spec 2 §7 keeps it; it is not a usage quota).
MAX_CHATBOOK_FILE_SIZE_MB = 100


class QuotaManager:
    """Chatbook quota helpers.

    Per-user daily export/import limits and concurrent-job limits are UserProfiles
    ``limits.chatbooks_*`` values, enforced in ``chatbook_service`` job admission.
    This class keeps only the off switch that admission reads and the fixed file-size cap.
    """

    def __init__(self, user_id: str, user_tier: str = "free", db: Optional[Any] = None):
        """Bind to a user; ``user_tier`` is accepted for compatibility and ignored."""
        self.user_id = user_id
        self.db = db
        self._quotas_disabled = (
            not usage_quotas_enabled()
            or _env_flag("CHATBOOKS_DISABLE_QUOTAS")
            or _env_flag("TEST_MODE")
            or _env_flag("TESTING")
            or bool(os.getenv("PYTEST_CURRENT_TEST"))
        )

    async def check_file_size(self, file_size_bytes: int) -> tuple[bool, str]:
        """Refuse a file larger than MAX_CHATBOOK_FILE_SIZE_MB."""
        if file_size_bytes > MAX_CHATBOOK_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File too large. Maximum size is {MAX_CHATBOOK_FILE_SIZE_MB}MB"
        return True, "File size OK"
```

Drop any import that becomes unused (`sqlite3`, `datetime`, `Enum`, `DatabasePaths`, `get_metrics_registry`, and so on); ruff F401 will tell you which.

In `chatbooks.py`:
- Delete the export pre-check block and the import pre-check block (the `QuotaManager` construction plus both checks and their 429s). If the import path uses `quota_manager` later for `check_file_size`, keep its construction there.
- In the preview handler, delete the `if file_size > 100 * 1024 * 1024:` duplicate; `check_file_size` already enforces the same cap.
- Leave the endpoint's `QuotaManager` import if `check_file_size` still uses it.

- [ ] **Step 4: Run the tests and confirm they PASS; fix the pinning tests**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 --timeout=600 \
  tldw_Server_API/tests/Chatbooks tldw_Server_API/tests/Usage tldw_Server_API/tests/lint
```

- In `test_chatbooks_full_account_export_contract.py`, `_PassingQuotaManager` may define `check_export_quota` and `check_concurrent_jobs`; the endpoint no longer calls them, so delete those methods and keep any assertion about the export contract.
- `tests/Usage/test_usage_quotas_sites.py::test_chatbooks_quotas_follow_the_switch` reads `_quotas_disabled`; it should keep passing.
- If the private-coercion ratchet count for `quota_manager.py` drops, lower the baseline entry in the same commit.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Chatbooks/quota_manager.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_no_tier_quotas.py <each updated test or baseline file>
git commit -m "fix(quotas): chatbooks drop the tier quota tables and the endpoint pre-checks that still capped exports at 10/day (spec 2 §7)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Audio reporting from the ledger; the dead RG handle machinery goes

**Files:**
- Modify `tldw_Server_API/app/core/Usage/audio_quota.py`:
  - `get_daily_minutes_used` (~607): read the ledger;
  - add `get_monthly_minutes_used` and `monthly_minutes_exhausted`;
  - delete the dead RG handle machinery: `_rg_audio_enabled` (133), `_rg_job_handles` / `_rg_stream_handles` (153-154), `_rg_audio_context` (175), `_log_rg_audio_init_failure` (217), `_log_rg_audio_fallback` (234), `_get_audio_rg_governor` (266), `_get_job_handle_lock` (323), `_cleanup_job_handle_lock` (334), `_cleanup_stream_handles` (366), `_rg_audio_governor` / `_loader` / `_lock`, `_rg_job_handle_locks`, and whatever `_reset_in_process_counters_for_tests` (253) only resets for them;
  - make `finish_job`, `finish_stream`, `heartbeat_stream` and `heartbeat_jobs` documented no-ops with unchanged signatures;
  - delete `active_streams_count` (1238).
- Modify `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`: `streaming_limits` (4505-4592), its `_active_streams_count` shim (~1129) and re-exports, and the WS quota message (~1809).
- Modify `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`: the 402 message (~928).
- Modify `tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py`:
  - `owner_processing_summary` (~637-693): None stays None;
  - the tier admin routes (~705, ~722): `deprecated=True`, plus a summary saying the tier no longer affects any limit.
- Modify `tldw_Server_API/app/api/v1/schemas/audio_schemas.py`, `StreamingLimitsResponse` (~318-345): `active_streams: Optional[int] = None`, and add `used_month_minutes: Optional[float] = None` and `remaining_month_minutes: Optional[float] = None`.
- Modify `tldw_Server_API/app/core/UserProfiles/service.py`, the `quotas["audio"]` block (~464-500).
- Modify the re-export shims that name `active_streams_count`: `endpoints/audio/__init__.py`, `endpoints/audio.py`, `audio_streaming.py`. Run `git grep -n active_streams_count -- tldw_Server_API/app`.
- Test: create `tldw_Server_API/tests/Usage/test_audio_reporting.py`. Update or delete the tests that pin the dead machinery:
  - `tests/Audio/test_audio_quota_unit.py:~158` (`get_daily_minutes_used` SQL path) and `:~219-407` (RG release and heartbeat internals);
  - `tests/Usage/test_audio_rg_minutes_and_heartbeat.py:~25` and `~77-88`;
  - `tests/Audio/test_ws_concurrent_streams.py:~43-45` and `tests/Audio/test_failopen_cap_minutes.py:~435` (`active_streams_count` references);
  - `tests/Audio/test_stream_limits_endpoint.py` (shape);
  - `tests/AudioJobs/test_audio_jobs_admin.py:~65` (owner processing).

**Interfaces:**
- Consumes: `quota_checks.ledger_used_today(entity_value: str, category: str) -> float` and `quota_checks.ledger_used_this_month(entity_value: str, category: str) -> float`. The audio `"minutes"` category is recorded in **seconds** (`_audio_minutes_units` is `minutes * 60`), so divide by 60. Also `audio_quota.get_limits_for_user(user_id) -> dict` (`daily_minutes`, `monthly_minutes`, `concurrent_streams`, `concurrent_jobs`, `max_file_size_mb`; None means unlimited).
- Produces:
  - `audio_quota.get_daily_minutes_used(user_id) -> float`, today's minutes from the ledger;
  - `audio_quota.get_monthly_minutes_used(user_id) -> float`;
  - `audio_quota.monthly_minutes_exhausted(user_id) -> bool`, true only when a monthly limit is set and already used up.

- [ ] **Step 1: Write the failing tests**

```python
"""Audio quota reporting reads the ledger and reports None as unlimited (spec 2 §9)."""

import pytest

from tldw_Server_API.app.core.Usage import audio_quota

pytestmark = pytest.mark.unit


@pytest.fixture()
def ledger(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Ledger seconds the test controls, per (period, user)."""
    used = {"today": 0.0, "month": 0.0}

    async def _today(entity_value: str, category: str) -> float:
        """Today's recorded seconds."""
        return used["today"] if category == "minutes" else 0.0

    async def _month(entity_value: str, category: str) -> float:
        """This month's recorded seconds."""
        return used["month"] if category == "minutes" else 0.0

    monkeypatch.setattr(audio_quota, "ledger_used_today", _today, raising=False)
    monkeypatch.setattr(audio_quota, "ledger_used_this_month", _month, raising=False)
    return used


async def test_daily_and_monthly_minutes_come_from_the_ledger(ledger: dict) -> None:
    """Recorded seconds show up as minutes used today and this month."""
    ledger["today"], ledger["month"] = 600.0, 5400.0
    assert await audio_quota.get_daily_minutes_used(7) == 10.0
    assert await audio_quota.get_monthly_minutes_used(7) == 90.0


async def test_monthly_minutes_exhausted_only_with_a_limit(ledger: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """No monthly limit is never exhausted; a spent one is."""
    limits = {"monthly_minutes": None}

    async def _limits(user_id: int) -> dict:
        """The test's limits."""
        return dict(limits)

    monkeypatch.setattr(audio_quota, "get_limits_for_user", _limits)
    ledger["month"] = 3600.0
    assert await audio_quota.monthly_minutes_exhausted(7) is False
    limits["monthly_minutes"] = 60
    assert await audio_quota.monthly_minutes_exhausted(7) is True
    limits["monthly_minutes"] = 61
    assert await audio_quota.monthly_minutes_exhausted(7) is False


def test_dead_rg_handle_machinery_is_gone() -> None:
    """Nothing tracks per-process audio RG handles any more."""
    for name in ("_rg_job_handles", "_rg_stream_handles", "_get_audio_rg_governor", "active_streams_count"):
        assert not hasattr(audio_quota, name), name
```

If `audio_quota` imports the ledger helpers under different names (for example it imports the `quota_checks` module), patch whatever name the module really calls, and drop `raising=False`. Read the module's imports first.

Endpoint tests in the same file:
- `test_stream_limits_report_ledger_minutes_and_null_when_unlimited`: with quotas on and no `limits.audio_*` set, `GET /api/v1/audio/stream/limits` returns `remaining_minutes is None`, `active_streams is None`, and `used_today_minutes` equal to the ledger figure (seed it with the `ledger` fixture, 600 s gives 10.0). Use the client setup `tests/Audio/test_stream_limits_endpoint.py` uses.
- `test_stream_limits_error_fallback_is_unlimited`: make `get_limits_for_user` raise one of the endpoint's `EXPECTED_DB_EXC` (read the tuple) and assert the response has every limit `None`, not 30 or 25.
- `test_owner_processing_limit_none_when_unlimited`: with quotas on and no `limits.audio_concurrent_jobs` set, `GET /api/v1/audio/jobs/admin/owner/{id}/processing` returns `limit is None`. Use the admin client from `tests/AudioJobs/test_audio_jobs_admin.py`.
- `test_monthly_breach_message_names_the_month`: drive the transcription path to a minutes denial with `monthly_minutes_exhausted` returning True, then assert the 402 message is "Transcription quota exceeded (monthly minutes)". With it returning False, the message is "(daily minutes)". Reuse the harness of whichever existing test already reaches that 402: `git grep -n "Transcription quota exceeded" -- tldw_Server_API/tests`.

- [ ] **Step 2: Run them and confirm they FAIL**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Usage/test_audio_reporting.py -q -p no:xdist`

- [ ] **Step 3: Implement**

`audio_quota.py`:

```python
async def get_daily_minutes_used(user_id: int) -> float:
    """Audio minutes used today (UTC), from the resource ledger; 0.0 if it can't be read."""
    try:
        return float(await ledger_used_today(str(int(user_id)), "minutes")) / 60.0
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("get_daily_minutes_used failed")
        return 0.0


async def get_monthly_minutes_used(user_id: int) -> float:
    """Audio minutes used this calendar month (UTC), from the resource ledger; 0.0 if it can't be read."""
    try:
        return float(await ledger_used_this_month(str(int(user_id)), "minutes")) / 60.0
    except _AUDIO_QUOTA_NONCRITICAL_EXCEPTIONS:
        logger.debug("get_monthly_minutes_used failed")
        return 0.0


async def monthly_minutes_exhausted(user_id: int) -> bool:
    """True when the user has a monthly minutes limit and has already used it up."""
    limit = (await get_limits_for_user(user_id)).get("monthly_minutes")
    if limit is None:
        return False
    return await get_monthly_minutes_used(user_id) >= float(limit)
```

Import `ledger_used_today` the same way the module already imports `ledger_used_this_month` (it is used at ~598).

The no-op hooks keep their exact current signatures; read each one. For example:

```python
async def finish_job(user_id: int, job_id: Any = None) -> None:
    """No-op hook: per-user job concurrency isn't tracked (see TASK-13435)."""
    return None
```

`streaming_limits`:
- The exception fallback becomes `limits = {"daily_minutes": None, "monthly_minutes": None, "concurrent_streams": None, "concurrent_jobs": None, "max_file_size_mb": None}`. Lookups fail open, so the report says unlimited.
- Drop the `_active_streams_count` call and set `active_streams = None`.
- `can_start = True`. Stream concurrency isn't limited per user (spec non-goal).
- Add `used_month = await _get_monthly_minutes_used(current_user.id)`, a new shim next to `_get_daily_minutes_used` that follows the same pattern. Compute `remaining_month = None if limits.get("monthly_minutes") is None else max(0.0, float(limits["monthly_minutes"]) - used_month)`.
- Return `used_month_minutes` and `remaining_month_minutes`.
- Keep `tier` (still stored by the deprecated tier API).

`owner_processing_summary`: `raw = limits.get("concurrent_jobs")`, then `limit = None if raw is None else int(raw)`.

Tier admin routes: add `deprecated=True` to both decorators, and set each `summary` to say "(deprecated: the tier no longer affects any limit; set limits.audio_* instead)".

The two "daily minutes" messages: at each denial, pick the period first:

```python
period = "monthly" if await audio_quota.monthly_minutes_exhausted(user_id) else "daily"
message = f"Transcription quota exceeded ({period} minutes)"
```

Call it through the module or shim the endpoint already uses for quota calls; read how `audio_transcriptions.py` reaches `audio_quota`. Do the same for the streaming WS message: "Streaming transcription quota exceeded ({period} minutes)". This runs only on the deny path.

Profile `quotas["audio"]`:
- Drop the `active_streams_count` import.
- Add `monthly_minutes_limit`, `monthly_minutes_used` (from `get_monthly_minutes_used`) and `monthly_minutes_remaining` (None when the limit is None).
- Set `concurrent_streams_active` to `None`.

- [ ] **Step 4: Run the tests and confirm they PASS; fix the pinning tests**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 --timeout=600 \
  tldw_Server_API/tests/Usage tldw_Server_API/tests/Audio tldw_Server_API/tests/AudioJobs \
  tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Resource_Governance tldw_Server_API/tests/lint
```

- Delete the tests that only pin the deleted RG machinery: `test_audio_quota_unit.py` ~219-407 and the `_rg_job_handles` parts of `test_audio_rg_minutes_and_heartbeat.py`. List each one in the report.
- Rewrite `test_audio_quota_unit.py:~158` to the ledger path.
- Tests that stub `can_start_job` / `can_start_stream` keep working; the hooks still exist.
- Refresh the OpenAPI fingerprint (the `StreamingLimitsResponse` change) and run `--check`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Usage/audio_quota.py tldw_Server_API/app/api/v1/endpoints/audio/ \
  tldw_Server_API/app/api/v1/endpoints/audio.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py \
  tldw_Server_API/app/core/UserProfiles/service.py tldw_Server_API/tests/Usage/test_audio_reporting.py \
  apps/tldw-frontend/lib/api/openapi.fingerprint.json <each updated or deleted test file>
git commit -m "feat(quotas): audio limits report ledger minutes (daily and monthly) and null for unlimited; drop the dead RG handle machinery (spec 2 §9)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The evaluations limits view reports the enforced daily caps

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`, `get_usage_summary` (~1075-1150).
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`, `get_rate_limit_status` (~838-880).
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`, `_apply_rate_limit_headers` (~347-372).
- Modify: `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py`, `RateLimitStatusResponse` (~974).
- Modify: `apps/packages/ui/src/components/Option/Evaluations/components/RateLimitsWidget.tsx` (~118-210), and its type in `apps/packages/ui/src/services/evaluations.ts`. Find the rate-limits response type there and make the two daily limits and remainings `number | null`.
- Test: create `tldw_Server_API/tests/Evaluations/test_rate_limits_view_resolver.py`. Update `tests/Evaluations/integration/test_rate_limits_endpoint.py` (it asserts `isinstance(lim[k], int)`), plus `tests/Usage/test_quota_evals_chatbooks.py:~64,78` if they assert the old daily numbers, and any test asserting `X-RateLimit-Daily-Limit` (`Resource_Governance/test_e2e_evals_authnz_character_headers.py`).

**Interfaces:**
- Consumes: `quota_resolver.user_quota(user_id: int, key) -> float | None` and `quota_checks.as_quota_user_id(value) -> int | None`. `check_rate_limit` already resolves the same two keys (~370-372: `limits.evaluations_per_day`, `limits.evaluation_tokens_per_day`); reuse the same calls.
- Produces: in `get_usage_summary(user_id)`, `limits.daily.evaluations` and `limits.daily.tokens` are the resolved values (None means unlimited), and `remaining.daily_evaluations` / `remaining.daily_tokens` are `max(0, cap - used)`, or None when unlimited. Per-minute and cost figures stay as they are: per-minute is RG-enforced by tier, and the spec keeps the cost caps.

- [ ] **Step 1: Write the failing tests**

```python
"""The evaluations limits view shows the enforced limits.* daily caps (spec 2 §9)."""

import pytest

from tldw_Server_API.app.core.Evaluations import user_rate_limiter as url_mod
from tldw_Server_API.app.core.Evaluations.user_rate_limiter import UserRateLimiter

pytestmark = pytest.mark.unit


@pytest.fixture()
def caps(monkeypatch: pytest.MonkeyPatch) -> dict:
    """limits.* daily caps the test controls."""
    values: dict = {}

    async def _user_quota(user_id: int, key: str):
        """The test's cap for the key."""
        return values.get(key)

    monkeypatch.setattr(url_mod, "user_quota", _user_quota)
    return values


async def test_summary_unlimited_when_no_cap_set(caps: dict, tmp_path) -> None:
    """No limits.* daily caps: limits and remaining are None, not the tier's 100/day."""
    limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
    summary = await limiter.get_usage_summary("7")
    assert summary["limits"]["daily"]["evaluations"] is None
    assert summary["limits"]["daily"]["tokens"] is None
    assert summary["remaining"]["daily_evaluations"] is None
    assert summary["remaining"]["daily_tokens"] is None


async def test_summary_reports_the_resolved_cap(caps: dict, tmp_path) -> None:
    """A limits.evaluations_per_day of 3 is reported with its remaining count."""
    caps["limits.evaluations_per_day"] = 3
    caps["limits.evaluation_tokens_per_day"] = 0
    limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
    summary = await limiter.get_usage_summary("7")
    assert summary["limits"]["daily"]["evaluations"] == 3
    assert summary["remaining"]["daily_evaluations"] == 3
    assert summary["limits"]["daily"]["tokens"] == 0
    assert summary["remaining"]["daily_tokens"] == 0
```

If `user_rate_limiter` reaches the resolver another way (for example through the `quota_resolver` module), patch that name; read the module's imports first.

Also add an endpoint test: with the same `caps` fixture empty, `GET /api/v1/evaluations/rate-limits` returns `limits.evaluations_per_day is None` and `remaining.daily_evaluations is None`. Use the client setup from `tests/Evaluations/integration/test_rate_limits_endpoint.py`, and mark this test `integration`.

- [ ] **Step 2: Run them and confirm they FAIL** (the summary reports the tier's 100 and 100000).

- [ ] **Step 3: Implement**

In `get_usage_summary`, after reading usage:

```python
        quota_uid = as_quota_user_id(user_id)
        if quota_uid is None:
            evals_cap = tokens_cap = None
        else:
            evals_cap = await user_quota(quota_uid, "limits.evaluations_per_day")
            tokens_cap = await user_quota(quota_uid, "limits.evaluation_tokens_per_day")
        evals_cap = None if evals_cap is None else int(evals_cap)
        tokens_cap = None if tokens_cap is None else int(tokens_cap)
```

Then set `"daily": {"evaluations": evals_cap, "tokens": tokens_cap, "cost": config.max_cost_per_day}`, `"daily_evaluations": None if evals_cap is None else max(0, evals_cap - total_evaluations)`, and the same for tokens. Leave per-minute, cost and `tier` unchanged.

`RateLimitStatusResponse`: change `limits` and `remaining` to `dict[str, Optional[int]]`.

`get_rate_limit_status`: pass the two daily values through as-is (no `, 0)` default that turns None into 0). Keep the int conversion for the cost fields.

`_apply_rate_limit_headers`: set `X-RateLimit-Daily-Limit`, `X-RateLimit-Daily-Remaining` and `X-RateLimit-Tokens-Remaining` only when the value is not None, and leave the header out otherwise.

`RateLimitsWidget.tsx`: when `dailyLimit` is `null`, render the used count with "Unlimited" in place of the denominator and no progress bar; do the same for tokens. Use the widget's existing `t(...)` pattern with a `defaultValue` ("Unlimited"). A limit of 0 shows `used/0` at 100%. Typecheck with `cd apps/tldw-frontend && npm run typecheck`, which covers `@tldw/ui` through its path mapping. If `apps/packages/ui` has vitest tests for this widget (`git grep -ln RateLimitsWidget -- 'apps/packages/ui/src/**/__tests__'`), add a null-limit case there.

- [ ] **Step 4: Run the tests and confirm they PASS; fix the pinning tests**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 --timeout=600 \
  tldw_Server_API/tests/Evaluations tldw_Server_API/tests/Usage tldw_Server_API/tests/UserProfile \
  tldw_Server_API/tests/Resource_Governance/test_e2e_evals_authnz_character_headers.py \
  tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py
```

- In `test_rate_limits_endpoint.py`, let the two daily keys be `int` or `None`, and keep the other keys asserted as int.
- Refresh the OpenAPI fingerprint and run `--check`.

- [ ] **Step 5: Commit** with message `feat(quotas): evaluations limits view reports the enforced limits.* daily caps; null is unlimited (spec 2 §9)`, ending with the trailer. Include the fingerprint and the frontend files.

---

### Task 4: Media 429 headers, the `DEFAULT_STORAGE_QUOTA_MB` deprecation, and the storage quota 0 display

**Files:**
- Modify `tldw_Server_API/app/core/Usage/quota_checks.py`: add `rate_limit_headers`.
- Modify `tldw_Server_API/app/api/v1/endpoints/workflows.py`: delete `_build_rate_limit_headers` (~1209-1227) and call the shared helper in `_enforce_workflows_daily_cap` (~1254).
- Modify `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`, `_enforce_and_record_media_bytes` (~278-334).
- Modify `tldw_Server_API/app/core/AuthNZ/settings.py:~324-328`.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx:~2153-2185` and `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx:~455-480`.
- Test: extend `tldw_Server_API/tests/Usage/test_quota_media_workflows.py`, and create `tldw_Server_API/tests/AuthNZ/unit/test_default_storage_quota_deprecation.py`. Update any test that imports `_build_rate_limit_headers` (`git grep -n _build_rate_limit_headers -- tldw_Server_API/tests`).

**Interfaces:**
- Produces: `quota_checks.rate_limit_headers(limit: int, remaining: int, reset_seconds: int) -> dict[str, str]`, returning `RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset` (delta seconds), `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset` (epoch seconds), the same set workflows sends today.

- [ ] **Step 1: Write the failing tests**

In `test_quota_media_workflows.py`:

```python
def test_rate_limit_headers_shape() -> None:
    """The shared helper returns the RFC-style and legacy headers, clamped at 0 remaining."""
    import time

    from tldw_Server_API.app.core.Usage import quota_checks

    headers = quota_checks.rate_limit_headers(limit=5, remaining=-2, reset_seconds=30)
    assert headers["RateLimit-Limit"] == headers["X-RateLimit-Limit"] == "5"
    assert headers["RateLimit-Remaining"] == headers["X-RateLimit-Remaining"] == "0"
    assert headers["RateLimit-Reset"] == headers["Retry-After"] == "30"
    assert abs(int(headers["X-RateLimit-Reset"]) - (int(time.time()) + 30)) <= 2
```

Also add `test_media_bytes_429_carries_rate_limit_headers`, with the docstring "A refused upload sends X-RateLimit-* in MB alongside Retry-After." Extend the file's existing 429 test the same way it seeds the ledger and the limit. Assert that `X-RateLimit-Limit` equals the MB limit as a string and `X-RateLimit-Remaining` is present.

`test_default_storage_quota_deprecation.py`:

```python
"""DEFAULT_STORAGE_QUOTA_MB is deprecated: values below 100 are accepted and an explicit value warns (spec 2 §7)."""

import pytest

from tldw_Server_API.app.core.AuthNZ.settings import Settings

pytestmark = pytest.mark.unit


def test_values_below_100_no_longer_fail_validation() -> None:
    """A deprecated, ignored setting can't stop startup."""
    assert Settings(DEFAULT_STORAGE_QUOTA_MB=0).DEFAULT_STORAGE_QUOTA_MB == 0
```

Construct `Settings` the way `tests/AuthNZ/unit/test_settings_guardrails.py` does. If it needs other required kwargs or env, copy that setup.

Add a second test: when the `DEFAULT_STORAGE_QUOTA_MB` env var is set, loading settings logs a single deprecation warning. Capture loguru the way other tests in `tests/AuthNZ/unit` do: `git grep -n "logger.add" -- tldw_Server_API/tests/AuthNZ/unit | head`.

- [ ] **Step 2: Run them and confirm they FAIL** (the helper doesn't exist, the media 429 has only `Retry-After`, and `ge=100` rejects 0).

- [ ] **Step 3: Implement**

`quota_checks.py`:

```python
def rate_limit_headers(limit: int, remaining: int, reset_seconds: int) -> dict[str, str]:
    """RFC-style RateLimit-* (reset as delta seconds) plus legacy X-RateLimit-* (reset as epoch)."""
    delta = max(0, int(reset_seconds))
    left = str(max(0, int(remaining)))
    return {
        "RateLimit-Limit": str(int(limit)),
        "RateLimit-Remaining": left,
        "RateLimit-Reset": str(delta),
        "Retry-After": str(delta),
        "X-RateLimit-Limit": str(int(limit)),
        "X-RateLimit-Remaining": left,
        "X-RateLimit-Reset": str(int(time.time()) + delta),
    }
```

Import `time` if it isn't already imported.

In workflows, delete `_build_rate_limit_headers` and call `rate_limit_headers(limit, max(0, limit - int(decision.used)), retry_after)`.

In persistence, keep the second return value of `add_if_within_daily_cap` (rename `_remaining` to `remaining_bytes`), and on refusal:

```python
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily ingestion size budget exceeded.",
            headers=quota_checks.rate_limit_headers(
                limit=int(limit_mb),
                remaining=int(max(0, remaining_bytes or 0) // (1024 * 1024)),
                reset_seconds=quota_checks.seconds_until_utc_midnight(),
            ),
        )
```

Check `add_if_within_daily_cap`'s docstring for what its second value means (remaining units after the add, or before it). If it is not "remaining bytes", compute remaining as `daily_cap_bytes - used_bytes` from whatever it returns, and say which in the report.

`settings.py`:
- Change the field to `Field(default=5120, ge=0, description="Deprecated (spec 2): no longer sets any user's storage quota; only the one-time storage quota migration reads it to skip the old default.")`.
- Emit a one-time `logger.warning` when the `DEFAULT_STORAGE_QUOTA_MB` environment variable is set. Put it in the settings construction path the module already uses for one-time warnings: `git grep -n "_warned\|warning(" tldw_Server_API/app/core/AuthNZ/settings.py | head`.

Frontend, `index.tsx` (both parse sites): replace the `quotaMb > 0` parse with a raw read, so null stays null and 0 stays 0:

```ts
const rawQuota = response?.storage_quota_mb
const quotaMb = typeof rawQuota === "number" ? rawQuota : null
const accountQuotaBytes =
  quotaMb !== null && Number.isFinite(quotaMb) && quotaMb >= 0 ? quotaMb * 1024 * 1024 : null
```

Use `quotas?.storage_quota_mb` at the profile fallback site.

`WorkspaceHeader.tsx`: in `hasAccountUsage`, accept `storageAccountQuotaBytes >= 0`. Compute `accountRatio = storageAccountQuotaBytes > 0 ? Math.max(0, Math.min(1, used / quota)) : 1`, so a 0 quota reads as full. The label shows `x/0 MB`.

Typecheck with `cd apps/tldw-frontend && npm run typecheck`. If `WorkspaceHeader` has vitest tests (`git grep -ln WorkspaceHeader -- 'apps/packages/ui/src/**/__tests__'`), add a 0-quota case.

- [ ] **Step 4: Run the tests and confirm they PASS**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 --timeout=600 \
  tldw_Server_API/tests/Usage tldw_Server_API/tests/Workflows tldw_Server_API/tests/MediaIngestion_NEW \
  tldw_Server_API/tests/AuthNZ/unit tldw_Server_API/tests/lint
```

- [ ] **Step 5: Commit** with message `fix(quotas): media 429 sends RateLimit headers via a shared helper; DEFAULT_STORAGE_QUOTA_MB deprecated; a 0 storage quota shows as full`, ending with the trailer.

---

### Task 5: ADR-064, the operator quotas page, and the docs that misstate the model

**Files:**
- Create: `Docs/ADR/064-usage-quotas-per-user-limits.md`, following `Docs/ADR/000-template.md`. Add its row to the index table in `Docs/ADR/README.md` (after ADR-063).
- Create: `Docs/Operations/Usage_Quotas.md`.
- Modify: `Docs/Operations/Env_Vars.md` (Usage Quotas section, ~316-327).
- Modify: `Docs/User_Guides/Server/Organization_Administration.md`: the billing rows (~298-299), `## Billing and Subscriptions` through `## Limit Enforcement` (~311-460), "Overriding Plan Limits" (~481-497), "Billing Management" (~513-517), and the link (~522).
- Modify: `Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md` (~17, ~24, ~94-96) and `Docs/Code_Documentation/Chatbook_Developer_Guide.md` (the QuotaManager tier examples at ~31, ~72, ~89, ~527-550, ~1041-1068).
- Modify: `Docs/API-related/Storage_API_Documentation.md`:
  - re-verify every `file:line` anchor; the user, team and org admin quota handlers moved to `app/api/v1/endpoints/storage_admin_quotas.py`;
  - document `SetUserQuotaRequest` (`quota_mb` int or null, `ge=0`, null removes the user's own value, 0 blocks) versus the team/org `SetQuotaRequest` (minimum 100);
  - say that `quota_mb` / `storage_quota_mb` are nullable (null means unlimited);
  - add the `GET|PUT /api/v1/admin/storage-quotas/users/{user_id}` routes.
- Modify the docs that tell operators to set `DEFAULT_STORAGE_QUOTA_MB`:
  - `Docs/User_Guides/Server/Multi-User_Deployment_Guide.md:~272`
  - `Docs/Deployment/Long_Term_Admin_Guide.md:~168`
  - `Docs/API-related/User_Registration_API_Documentation.md:~876`
  - `Docs/Development/POSTGRES_SETUP_GUIDE.md:~136`
  - `Docs/API-related/AuthNZ-API-Guide.md:~185`

  Mark it deprecated and point to `limits.storage_quota_mb`. Response examples showing `"storage_quota_mb": 5120` can stay as examples of a set quota, but add one line saying `null` means unlimited.
- Modify: `Docs/Design/2026-10-02-usage-quota-posture-design.md`. Set Status to "Implemented (PRs #3098, #3144, #3199, and this PR)", and add a short `## Errata` section:
  - the ADR is 064, not 058;
  - the registration-code writer in §5 and Testing 9 was dead code and was deleted in PR C;
  - there is no chatbooks usage summary endpoint (§9); limits show in the profile config.
- Backlog:
  - Fix TASK-13434's AC4 text ("ADR-058" becomes "ADR-064") with backlog-py (`task edit TASK-13434` with the remove-AC and add-AC flags; run `... task edit --help`).
  - Create the follow-up task "Audit events on the storage quota admin endpoints", with an explicit id. First compute the max id across `origin/dev` and every open PR branch (the peer holds TASK-13501, so expect 13502 or higher). Description: "`PUT /api/v1/storage/admin/quotas/user/{id}` and `PUT /api/v1/admin/storage-quotas/users/{id}` change a user's storage quota without an audit event; `PUT /admin/users/{id}` and the profile path do emit one. Parent: TASK-13434." AC: "Both endpoints emit the same audit event as PUT /admin/users/{id} when the quota changes".
- Run `bash Helper_Scripts/refresh_docs_published.sh`, then `tests/Docs`. The Published mirror covers ADR, API-related, Code_Documentation, User_Guides, Deployment and `Operations/Env_Vars.md`. The new `Operations/Usage_Quotas.md` is not mirrored.

**What each doc must say**, all checked against the code (state nothing the code doesn't do):

`ADR-064`:
- **Decision:** "Usage quotas are per-user UserProfiles `limits.*` values, off by default (`USAGE_QUOTAS_ENABLED`), resolved per user from the user's own value, then the most generous team value, then the most generous org value; none is set by default."
- **Context:** self-hosters were limited by tier defaults they never chose (spec 2 Problem); the owner's commercial offering needs per-user and per-group values.
- **Alternatives considered:** keep tiers; a shared group pool; the billing plan only.
- **Consequences:** the precedence rule; 0 blocks; usage is recorded whether quotas are on or off; the deprecated tier APIs; the column kept but unread; a link to `Docs/Operations/Usage_Quotas.md`.
- Header lines: Status "Accepted", Date today, Decision owner "Repository owner (@rmusser01)", Related task TASK-13434, and Related spec/plan pointing to the spec.

`Usage_Quotas.md` (operator page):
- the master switch and its precedence;
- how to set values: user (`PATCH /api/v1/admin/users/{id}/profile`), team and org (`PUT`/`DELETE /api/v1/admin/{orgs|teams}/{id}/profile/overrides/{key}`, body `{"value": n}`), and the storage endpoints;
- a table of every `limits.*` key, giving what it limits, the counter it reads, and the refusal (status and message), taken from `Rate_Limits_Troubleshooting.md`;
- what is not a quota (per-file caps, per-minute rates, synchronous concurrency);
- upgrade notes (the storage copy and its skip values);
- where users can see their limits: `/users/storage`, `/audio/stream/limits`, `/evaluations/rate-limits`, and the profile `quotas` section and `limits.*` config;
- the deprecated settings and APIs (`AUDIO_TIER_LIMITS_JSON`, `[Audio-Quota] {tier}_*`, the audio tier admin API, `DEFAULT_STORAGE_QUOTA_MB`).

`Env_Vars.md`: add bullets for the deprecated `AUDIO_TIER_LIMITS_JSON` / `[Audio-Quota] {tier}_*` (they log a warning and change nothing) and `DEFAULT_STORAGE_QUOTA_MB`, plus a link to `Usage_Quotas.md`.

`Organization_Administration.md`: replace the billing and limit-enforcement text with what is true.
- The OSS server ships no billing: `BILLING_ENABLED` is read nowhere, and `is_billing_enabled()` returns False. The only billing route is the admin `GET /api/v1/billing/subscriptions`.
- Billing plan limits run only when a billing repository is wired (the hosted product) and `USAGE_QUOTAS_ENABLED` is on.
- Per-user, team and org quotas are `limits.*` values; link `Docs/Operations/Usage_Quotas.md` (a relative path from that file).
- Remove the claims about checkout, portal, usage and cancel routes, the soft-limit `X-Billing-Warning` header and grace behavior, unless `git grep` finds them in `tldw_Server_API/app`.

The chatbooks guides: daily export/import limits and concurrent jobs are `limits.chatbooks_*` values enforced in `chatbook_service` job admission; `QuotaManager` keeps only the off switch and the 100 MB file-size cap; there are no tiers.

- [ ] **Step 1: Write the docs, run the refresh script and `tests/Docs`**

```bash
bash Helper_Scripts/refresh_docs_published.sh
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Docs
```

- [ ] **Step 2: Run the backlog edits and check the format**

Check with `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task normalize --check backlog/tasks/task-13434*.md <new task file>`.

- [ ] **Step 3: Commit**

```bash
git add Docs/ADR/064-usage-quotas-per-user-limits.md Docs/ADR/README.md Docs/Operations/Usage_Quotas.md \
  Docs/Operations/Env_Vars.md Docs/User_Guides Docs/Code_Documentation Docs/API-related Docs/Deployment \
  Docs/Development/POSTGRES_SETUP_GUIDE.md Docs/Design/2026-10-02-usage-quota-posture-design.md Docs/Published
git add -A backlog/tasks
git commit -m "docs(quotas): ADR-064, an operator quotas page, and the docs that misstated billing, chatbooks and storage quotas

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Ship PR D

- [ ] **Step 1: Rebase and run the post-rebase checks.** Rebase onto `origin/dev`.
  - Run `export_openapi_schema.py --check apps/tldw-frontend/lib/api/openapi.fingerprint.json`; if it fails, refresh and commit.
  - Run the line-numbered lint ratchets in `tests/lint`, and regenerate any baseline whose recorded lines moved.
  - Run `task normalize --check` on the touched task files.
  - Confirm no open PR adds `Docs/ADR/064-*`.
- [ ] **Step 2: Sweep.** List every test file that references a changed symbol:

  ```bash
  git grep -l -e QuotaManager -e get_usage_summary -e get_daily_minutes_used -e active_streams_count \
    -e streaming_limits -e owner_processing -e _build_rate_limit_headers -e rate_limit_headers \
    -e DEFAULT_STORAGE_QUOTA_MB -e _rg_job_handles -e _rg_stream_handles -e RateLimitStatusResponse \
    -- 'tldw_Server_API/tests/**/test_*.py'
  ```

  Run that list plus `tests/Usage tests/Chatbooks tests/Audio tests/AudioJobs tests/Evaluations tests/UserProfile tests/Workflows tests/MediaIngestion_NEW tests/AuthNZ/unit tests/Docs tests/lint`, through a scratchpad script, with `-n 4 --timeout=600`. Triage every failure: re-run it alone, then prove it on a detached `origin/dev` checkout or fix it.
- [ ] **Step 3: Static checks.** Run Bandit `-ll` on the changed app files; it must find nothing at medium or above. Ruff per-rule tallies must be identical to the merge base, compared via `--stdin-filename`. Run the frontend typecheck.
- [ ] **Step 4: Open the PR.** Message `tldw-server-77` for the merge slot, listing the touched areas. Push, then `gh pr create --base dev`. The body covers:
  - the chatbooks double-enforcement fix, first, since it is a live cap on users;
  - each reporting endpoint and what it now shows;
  - the deleted RG machinery and the kept hooks;
  - the evaluations view;
  - the media headers;
  - the deprecations;
  - the frontend fixes;
  - the docs and ADR-064;
  - the spec errata;
  - the follow-up task id;
  - the verification counts;
  - the `## Change summary` waiver section and the Claude Code footer.
- [ ] **Step 5: Review and merge.**
  - Qodo is paused (out of credits). Before queuing, dispatch one focused fable review of the final diff in its place, fix or decline every finding it reports, and note in the PR that it stood in for Qodo.
  - Merge only on the peer's go-ahead, running the merge queue for this PR alone.
  - After the queue rebases, if any required check shows CANCELLED, cancel the stale old-head runs and re-run the cancelled current-head runs. Do that before treating it as a failure (it happened on #3199).
  - After the merge, message the peer, then run backlog-py: `task edit TASK-13434 --check-ac 4 --append-notes "PR D merged as #<n> (<sha>) ..."` and set the status to Done if every AC is checked.
