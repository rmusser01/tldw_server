# LLM Usage Accounting PR 2a (the keys, the counters and the boundary check) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With usage quotas on, a user over a monthly LLM token limit, or a new daily or monthly LLM cost limit, is refused with HTTP 402 before the main LLM entry points start any work; local providers stop counting as spend; operators are warned about settings that defeat the limits; the profile reports LLM usage.

**Architecture:** One decision function (`llm_quota_refusal`) reads the user's three limits and only the counters whose limit is set. One error, `LLMQuotaExceededError`, is an `HTTPException` 402 carrying today's `/chat/completions` body. Each entry point in the spec's §4a table calls `enforce_llm_quota` (or depends on `require_llm_quota`) before any LLM work and before any streaming response, and a passing check marks the request's task `prechecked`. Nothing in this PR checks at the LLM call itself; that is PR 2b.

**Tech Stack:** Python 3.12, FastAPI, asyncio, `contextvars`, SQLite and PostgreSQL through the AuthNZ pool, pytest (asyncio mode auto).

**Spec:** `Docs/Design/2026-10-09-llm-usage-accounting-design.md`: §4 (the decision, the error, §4a), §6, §7, §9, and Delivery "PR 2a". Read it, and the owner decisions at its top, before any task. Spec 2, `Docs/Design/2026-10-02-usage-quota-posture-design.md`, built the quota machinery this extends.

## Global Constraints

- **Quotas are off unless an admin turns them on.** Nothing is refused unless `USAGE_QUOTAS_ENABLED` is on and the user has a limit. `0` blocks, an unset limit is unlimited, and a lookup or counter failure admits, with a warning.
- **The public refusal is exactly:** HTTP 402, a `Retry-After` header, and the body `{"detail": {"error": "limit_exceeded", "category": ..., "current": ..., "limit": ..., "message": ...}}`. Categories `llm_tokens_month`, `llm_cost_day`, `llm_cost_month`; messages "Monthly LLM token limit reached", "Daily LLM cost limit reached", "Monthly LLM cost limit reached". `Retry-After` is the next UTC midnight for the day cap and the start of the next UTC month for the monthly ones.
- **The token key refuses when this month's tokens plus the requested estimate pass the limit.** `/chat/completions` passes its prompt estimate; every other caller passes 1. **A cost key refuses once spend has reached the limit.**
- **`LLMQuotaExceededError` is an `HTTPException`, never a `ChatAPIError`,** and never both.
- **The boundary check comes before any LLM work, any long work (download, retrieval, transcription) and any `StreamingResponse`,** and where a `try` surrounds it, that `try` lets `HTTPException` through.
- **The `prechecked` mark is a plain `ContextVar.set`** (`mark_llm_quota_prechecked`), never a `with llm_call_context(...)` block.
- **No check at the call in this PR.** Do not change `core/LLM_Calls/usage_accounting.py`, `perform_chat_api_call` or `perform_chat_api_call_async`.
- **Embeddings never count** toward any LLM limit (the `_LLM_GENERATION_PREDICATE` from PR 1a).
- **Test-first.** Each task's tests are written and seen failing before its code, except tests that guard behaviour which already holds; the plan says which.
- **Every task was dry-run** by an independent reviewer in a scratch checkout of this branch, in order, and the test code in this plan is the code that ran there. If an edit does not match the file, or a test does not behave as a step says, stop and report it. Do not improvise.
- **Import placement.** Where a step says where an import goes, put it there: ruff's isort rules decide, and the wrong place adds an `I001` finding. Never run `ruff --fix` on a whole file: several files have unrelated findings that a fix would also move.
- **New test files go only where CI runs the directory whole:** `tldw_Server_API/tests/Usage` and `tldw_Server_API/tests/Media`. Elsewhere, add tests to existing files: `.github/workflows/ci.yml` lists many files one by one, and a new file there would not run.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit ids, `git add -A backlog/tasks`.
- **Commits** end with exactly `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `git stash`. Never `--no-verify`.
- **Python:** `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (written `$PY` below). Tests run with `TLDW_TEST_NO_DOCKER=1`; the Evaluations tests also need `RUN_EVALUATIONS=1`.
- **Known failures on `dev` in this environment:** `tests/Services/test_router_groups_contract.py` (10 tests), `tests/Audio/test_nemo_streaming_error_sentinel.py` (no `pyaudio`). **Load flakes under `-n`:** `tests/Admin/test_admin_llm_provider_test_runtime.py`, `tests/LLM_Calls/test_mlx_provider.py`, `tests/Web_Scraping/test_phase4_safe_regex.py`, `tests/Workflows/test_workflows_extras.py::test_rag_search_workflow`. Re-run one alone, and at `origin/dev`, before reporting it.

## Review Focus

Inputs a person would hit that the spec implies. Each has a test in the task named.

1. **The common case: quotas off, or no LLM limit set.** Expected: nothing is refused and no counter is read. (Task 4)
2. **An over-quota user who asks for a stream.** Expected: a real 402, not a 200 with an error frame. (Tasks 4, 5, 6)
3. **A request that does no LLM work:** RAG without generation, media without analysis, an evaluation batch of type `propositions`. Expected: never refused. (Tasks 6, 7)
4. **A user who runs local models and has a USD limit.** Expected: local calls cost nothing and do not use up the limit. (Task 3)
5. **An admin who sets a cost limit of 0, or of $2.50.** Expected: 0 blocks, and a decimal value is accepted and compared as dollars. (Tasks 2, 4)
6. **A counter that fails.** Expected: the request is admitted and a warning logged. (Task 4)

## File Structure

| File | Responsibility |
|---|---|
| `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py` | One summing helper; the new cost sum. |
| `tldw_Server_API/app/core/Usage/quota_checks.py` | The counters, the decision, the startup settings check, the profile summary. |
| `tldw_Server_API/app/core/exceptions.py` | `LLMQuotaExceededError`. |
| `tldw_Server_API/app/core/Usage/llm_call_context.py` | `mark_llm_quota_prechecked`. |
| `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py` | `enforce_llm_quota`, `require_llm_quota`. |
| `tldw_Server_API/app/core/Usage/pricing_catalog.py`, `Config_Files/model_pricing.json` | Canonical provider names, provider-wide defaults, zero prices for local providers. |
| `Config_Files/user_profile_catalog.yaml` | The two cost keys. |
| The endpoint files in Tasks 4 to 7 | One boundary check each. |
| `tldw_Server_API/app/services/startup_auth_runtime.py` | Calls the settings check at startup. |
| `tldw_Server_API/app/core/UserProfiles/service.py`, `schemas/user_profile_schemas.py` | The profile's `quotas.llm` block. |

---

### Task 1: The cost counters

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py` (`sum_user_llm_tokens_since`, near line 255)
- Modify: `tldw_Server_API/app/core/Usage/quota_checks.py` (`llm_tokens_this_month`, near line 144)
- Test: `tldw_Server_API/tests/Usage/test_llm_cost_counters.py`

**Interfaces:**
- Produces:
  - `AuthnzUsageRepo.sum_user_llm_cost_since(*, user_id: int, since: datetime) -> float`
  - `quota_checks.llm_cost_today(user_id: int) -> float` and `quota_checks.llm_cost_this_month(user_id: int) -> float`, in USD
  - `quota_checks.llm_tokens_this_month(user_id: int) -> float`, unchanged in behaviour

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_llm_cost_counters.py`:

```python
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
```

The SQLite test inserts explicit `ts` strings, so it does not depend on the day of the month. The parametrized test does not touch a database.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_cost_counters.py`
Expected: collection fails with `AttributeError: module ... quota_checks has no attribute 'llm_cost_today'`.

- [ ] **Step 3: One summing helper in the repo**

In `usage_repo.py`, replace the whole `sum_user_llm_tokens_since` method with:

```python
    async def _sum_llm_column_since(self, column: str, *, user_id: int, since: datetime) -> float:
        """Sum one ``llm_usage_log`` column for a user from ``since`` (inclusive), embeddings excluded.

        ``since`` is bound naive on Postgres: ``llm_usage_log.ts`` is TIMESTAMP
        without time zone there, so an aware bound would never match.
        ``column`` is a constant chosen by the callers below, never user input.
        """
        bound: Any = _strip_tzinfo(since) if self._is_postgres_backend() else since.strftime("%Y-%m-%d %H:%M:%S")
        value = await self.db_pool.fetchval(
            f"SELECT COALESCE(SUM({column}), 0) FROM llm_usage_log "  # nosec B608
            f"WHERE user_id = ? AND ts >= ? AND {_LLM_GENERATION_PREDICATE}",
            int(user_id),
            bound,
        )
        return float(value or 0)

    async def sum_user_llm_tokens_since(self, *, user_id: int, since: datetime) -> float:
        """Sum ``llm_usage_log.total_tokens`` for a user from ``since`` onward (LLM usage accounting spec §6)."""
        return await self._sum_llm_column_since("total_tokens", user_id=user_id, since=since)

    async def sum_user_llm_cost_since(self, *, user_id: int, since: datetime) -> float:
        """Sum ``llm_usage_log.total_cost_usd`` (USD) for a user from ``since`` onward (LLM usage accounting spec §6)."""
        return await self._sum_llm_column_since("total_cost_usd", user_id=user_id, since=since)
```

`SUM` skips rows whose cost is NULL; rows `log_llm_usage` writes always carry a cost.

- [ ] **Step 4: The counters in `quota_checks.py`**

Replace the whole `llm_tokens_this_month` function with:

```python
def _utc_day_start() -> datetime:
    """Midnight UTC today."""
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


async def _usage_repo() -> Any:
    """The AuthNZ usage repository on the shared pool."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo

    return AuthnzUsageRepo(await get_db_pool())


async def llm_tokens_this_month(user_id: int) -> float:
    """This calendar month's (UTC) ``llm_usage_log`` token total for the user."""
    repo = await _usage_repo()
    return await repo.sum_user_llm_tokens_since(user_id=user_id, since=_utc_day_start().replace(day=1))


async def llm_cost_today(user_id: int) -> float:
    """Today's (UTC) LLM spend in USD for the user."""
    repo = await _usage_repo()
    return await repo.sum_user_llm_cost_since(user_id=user_id, since=_utc_day_start())


async def llm_cost_this_month(user_id: int) -> float:
    """This calendar month's (UTC) LLM spend in USD for the user."""
    repo = await _usage_repo()
    return await repo.sum_user_llm_cost_since(user_id=user_id, since=_utc_day_start().replace(day=1))
```

The import of `get_db_pool` stays inside the function, so the existing test that replaces `tldw_Server_API.app.core.AuthNZ.database.get_db_pool` keeps working.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_cost_counters.py tldw_Server_API/tests/Usage/test_llm_token_counter_excludes_embeddings.py tldw_Server_API/tests/Usage/test_quota_resolver.py`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py tldw_Server_API/app/core/Usage/quota_checks.py \
  tldw_Server_API/tests/Usage/test_llm_cost_counters.py
git commit -m "feat(usage): daily and monthly LLM cost counters, embeddings excluded (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The two cost keys

**Files:**
- Modify: `tldw_Server_API/Config_Files/user_profile_catalog.yaml` (the `limits.llm_tokens_per_month` entry, near line 278)
- Test: `tldw_Server_API/tests/UserProfile/test_user_profile_limits.py` (`NEW_KEYS`, near line 21) and one new test in the same file

**Interfaces:**
- Produces: the keys `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd`, `type: number`, readable with `user_quota(user_id, key)`.

- [ ] **Step 1: Write the failing test**

In `tests/UserProfile/test_user_profile_limits.py`, add the two keys to the `NEW_KEYS` set:

```python
    "limits.llm_cost_per_day_usd",
    "limits.llm_cost_per_month_usd",
```

and append this test:

```python
def test_llm_cost_keys_take_decimal_dollars() -> None:
    """The LLM cost keys are numbers with a floor of zero, so an admin can set $2.50."""
    entries = {entry.key: entry for entry in load_user_profile_catalog().entries}
    for key in ("limits.llm_cost_per_day_usd", "limits.llm_cost_per_month_usd"):
        assert entries[key].type == "number"
        assert entries[key].minimum == 0
    assert "/chat/completions" not in entries["limits.llm_tokens_per_month"].description
```

Use the catalog loader the file already imports. If its name or the attribute that lists entries differs from `load_user_profile_catalog().entries`, use what `test_catalog_has_every_limit_key_platform_admin_only` uses, and keep the assertions.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/UserProfile/test_user_profile_limits.py`
Expected: `test_catalog_has_every_limit_key_platform_admin_only` and the new test fail with a `KeyError` or a failed subset assertion for the two keys.

- [ ] **Step 3: Add the keys**

In `user_profile_catalog.yaml`, replace the `limits.llm_tokens_per_month` entry's `description:` line with:

```yaml
    description: LLM tokens per calendar month (UTC), counted from llm_usage_log. Checked before LLM work; embeddings do not count.
```

and add, directly after that entry (keep the blank line between entries):

```yaml
  - key: limits.llm_cost_per_day_usd
    label: LLM Cost Per Day (USD)
    description: LLM spend per UTC day in US dollars, summed from llm_usage_log. Checked before LLM work; embeddings do not count and local providers cost nothing by default.
    type: number
    minimum: 0
    default: null
    editable_by: [platform_admin]
    sensitivity: internal
    ui: number

  - key: limits.llm_cost_per_month_usd
    label: LLM Cost Per Month (USD)
    description: LLM spend per calendar month (UTC) in US dollars, summed from llm_usage_log. Checked before LLM work; embeddings do not count and local providers cost nothing by default.
    type: number
    minimum: 0
    default: null
    editable_by: [platform_admin]
    sensitivity: internal
    ui: number
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/UserProfile`
Expected: no failures.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/Config_Files/user_profile_catalog.yaml tldw_Server_API/tests/UserProfile/test_user_profile_limits.py
git commit -m "feat(usage): limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Local providers cost nothing by default

**Files:**
- Modify: `tldw_Server_API/app/core/Usage/pricing_catalog.py` (`DEFAULT_PRICING`, near lines 54-167; `get_rate_details`, near lines 252-275; `list_provider_models`, near line 382)
- Modify: `tldw_Server_API/Config_Files/model_pricing.json` (the end of the file)
- Test: `tldw_Server_API/tests/Usage/test_pricing_local_providers.py`

**Interfaces:**
- Produces: `PricingCatalog.get_rate_details(provider, model)` resolves the provider's canonical name, then the exact model, then a substring match, then the provider's `default` entry, then the unknown-model rate. `list_provider_models(provider)` never lists `default`: a provider-wide price is not a model.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_pricing_local_providers.py`:

```python
"""Local providers cost nothing by default, as data an operator can change (LLM usage accounting spec §6)."""

import json

import pytest

from tldw_Server_API.app.core.Usage.pricing_catalog import UNKNOWN_BILLABLE_RATE, PricingCatalog
from tldw_Server_API.app.core.Usage.usage_tracker import compute_costs

pytestmark = pytest.mark.unit

LOCAL = ("llama.cpp", "kobold", "ooba", "tabbyapi", "vllm", "local-llm", "ollama", "aphrodite", "mlx")


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts from the shipped catalog."""
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)


@pytest.mark.parametrize("provider", LOCAL)
def test_each_local_provider_is_free_and_not_estimated(provider: str) -> None:
    """Any model of a local provider is priced at zero, and the price is not an estimate."""
    rates, estimated = PricingCatalog().get_rate_details(provider, "some-local-model-q4")
    assert (rates["prompt"], rates["completion"], estimated) == (0.0, 0.0, False)


@pytest.mark.parametrize("alias", ["llamacpp", "LLaMA.cpp", "oobabooga", "tabby", "local_llm", "koboldcpp"])
def test_aliases_price_like_their_canonical_provider(alias: str) -> None:
    """The writers pass different spellings; an alias is priced as its canonical provider."""
    rates, estimated = PricingCatalog().get_rate_details(alias, "m")
    assert (rates["prompt"], rates["completion"], estimated) == (0.0, 0.0, False)


def test_a_local_call_records_zero_cost() -> None:
    """compute_costs, which log_llm_usage calls, stores zero for a local call."""
    assert compute_costs("ollama", "llama3.1:8b", 10_000, 1_000) == (0.0, 0.0, 0.0, False)


def test_unknown_providers_and_custom_endpoints_keep_the_unknown_rate() -> None:
    """A provider with no entry, including a custom OpenAI-compatible endpoint, is still estimated at the unknown rate."""
    for provider in ("custom-openai-api", "some-new-cloud"):
        rates, estimated = PricingCatalog().get_rate_details(provider, "m")
        assert (rates["prompt"], rates["completion"], estimated) == (
            UNKNOWN_BILLABLE_RATE["prompt"],
            UNKNOWN_BILLABLE_RATE["completion"],
            True,
        )


def test_the_huggingface_default_is_now_reachable() -> None:
    """The provider-wide default the file already had is used, and stays marked as an estimate."""
    rates, estimated = PricingCatalog().get_rate_details("huggingface", "some/model")
    assert (rates["prompt"], estimated) == (pytest.approx(0.05e-3), True)


def test_an_operator_can_price_a_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local provider pointed at a paid service can be given a price."""
    monkeypatch.setenv("PRICING_OVERRIDES", json.dumps({"vllm": {"default": {"prompt": 0.002, "completion": 0.004}}}))
    rates, _ = PricingCatalog().get_rate_details("vllm", "anything")
    assert (rates["prompt"], rates["completion"]) == (0.002, 0.004)


def test_a_model_match_beats_the_provider_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model entry wins over the provider-wide default, even when the model name contains 'default'."""
    monkeypatch.setenv("PRICING_OVERRIDES", json.dumps({"vllm": {"llama": {"prompt": 0.001, "completion": 0.002}}}))
    rates, estimated = PricingCatalog().get_rate_details("vllm", "llama-default-q4")
    assert (rates["prompt"], rates["completion"], estimated) == (0.001, 0.002, True)


def test_the_provider_default_is_not_advertised_as_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider-wide price is not a model, so model validation for a local provider stays unarmed."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Usage.pricing_catalog import list_provider_models

    monkeypatch.setattr(chat_service, "_configured_models_for_provider", lambda _provider: ())
    assert "default" not in list_provider_models("vllm")
    assert chat_service.is_model_known_for_provider("vllm", "llama3.1:8b") is None
```

The last test guards a regression the dry run found. The chat endpoints merge `list_provider_models` into the set of known models, and with strict model selection (the default outside test mode) they refuse a model that is not in a non-empty set. If `default` were listed, every local model not named in `config.txt` would be refused with `model_not_available`. The test passes before this task; it fails if Step 4 is applied without the `list_provider_models` change in Step 3.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_pricing_local_providers.py`
Expected: the local-provider and alias tests fail with the unknown rate where zero is expected; the huggingface test fails with the unknown rate.

- [ ] **Step 3: Resolve the canonical name and the provider-wide default**

In `pricing_catalog.py`, add next to the other imports:

```python
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
```

In `get_rate_details`, change:

```python
        prov = (provider or "").lower()
```

to:

```python
        prov = canonical_provider_name(provider or "")
```

and change the substring loop and the final return:

```python
        for mk, rates in prov_map.items():
            if mdl and mk and (mk in mdl or mdl in mk):
```

to:

```python
        for mk, rates in prov_map.items():
            if mk == "default":
                continue
            if mdl and mk and (mk in mdl or mdl in mk):
```

```python
        return _unknown_billable_rate(), True
```

to:

```python
        if "default" in prov_map:
            # A provider-wide price (LLM usage accounting spec §6): local providers ship one at zero.
            return self._entry_rate_details(prov_map["default"], estimated=True)
        return _unknown_billable_rate(), True
```

`canonical_provider_name` lowercases and maps aliases (`llamacpp` to `llama.cpp`), and returns any other name lowercased, so known providers are looked up as before. No provider in the catalog today is an alias of another, so no existing price changes; aliases such as `oai` are now priced as their provider.

In `list_provider_models`, change:

```python
                for name, meta in models_map.items()
                if not (isinstance(meta, dict) and meta.get("placeholder"))
```

to:

```python
                for name, meta in models_map.items()
                # "default" is a provider-wide price, not a model (LLM usage accounting spec §6).
                if name != "default" and not (isinstance(meta, dict) and meta.get("placeholder"))
```

This also stops `huggingface`, whose `default` entry already existed, from listing `default` as a model.

- [ ] **Step 4: Ship the zero defaults**

In `DEFAULT_PRICING`, directly after the `"huggingface"` entry, add:

```python
    # Local providers cost nothing by default (LLM usage accounting spec §6). Change them in model_pricing.json.
    **{
        provider: {"default": {"prompt": 0.0, "completion": 0.0, "estimated": False}}
        for provider in ("llama.cpp", "kobold", "ooba", "tabbyapi", "vllm", "local-llm", "ollama", "aphrodite", "mlx")
    },
```

In `Config_Files/model_pricing.json`, after the `"huggingface"` object (add a comma after its closing brace), add:

```json
  "_local_providers_note": "Local providers cost nothing by default. Give one a price if it points at a paid service.",
  "llama.cpp": { "default": { "prompt": 0, "completion": 0, "estimated": false } },
  "kobold": { "default": { "prompt": 0, "completion": 0, "estimated": false } },
  "ooba": { "default": { "prompt": 0, "completion": 0, "estimated": false } },
  "tabbyapi": { "default": { "prompt": 0, "completion": 0, "estimated": false } },
  "vllm": { "default": { "prompt": 0, "completion": 0, "estimated": false } },
  "local-llm": { "default": { "prompt": 0, "completion": 0, "estimated": false } },
  "ollama": { "default": { "prompt": 0, "completion": 0, "estimated": false } },
  "aphrodite": { "default": { "prompt": 0, "completion": 0, "estimated": false } },
  "mlx": { "default": { "prompt": 0, "completion": 0, "estimated": false } }
```

The merge skips a top-level value that is not an object, so the note string is ignored the way `_catalog_note` is. Check that the file is valid JSON: `$PY -c "import json; json.load(open('tldw_Server_API/Config_Files/model_pricing.json'))"`.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_pricing_local_providers.py tldw_Server_API/tests/Usage/test_pricing_catalog_overrides.py tldw_Server_API/tests/Usage/test_pricing_catalog_path.py tldw_Server_API/tests/Usage/test_pricing_catalog.py`
Expected: all pass. Then the whole Usage suite once: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider tldw_Server_API/tests/Usage` (no new failures).

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Usage/pricing_catalog.py tldw_Server_API/Config_Files/model_pricing.json \
  tldw_Server_API/tests/Usage/test_pricing_local_providers.py
git commit -m "feat(usage): local providers cost nothing by default, as catalog data (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The decision, the error, the boundary helpers, and `/chat/completions`

**Files:**
- Modify: `tldw_Server_API/app/core/exceptions.py` (append one class at the end of the file)
- Modify: `tldw_Server_API/app/core/Usage/quota_checks.py` (add the decision)
- Modify: `tldw_Server_API/app/core/Usage/llm_call_context.py` (add one function at the end)
- Modify: `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py` (add two functions)
- Modify: `tldw_Server_API/app/core/Metrics/metrics_manager.py` (register one counter) and `tldw_Server_API/app/core/Metrics/README.md` (list it, next to `llm_cost_dollars`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (the quota imports near line 363; the check near lines 4292-4311)
- Test: `tldw_Server_API/tests/Usage/test_llm_quota_decision.py`; `tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py`

**Interfaces:**
- Consumes: from Task 1, `quota_checks.llm_tokens_this_month`, `llm_cost_today`, `llm_cost_this_month`; from Task 2, the two keys.
- Produces:
  - `LLMQuotaExceededError(*, category, current, limit, message, retry_after)` in `core/exceptions.py`, an `HTTPException` with status 402.
  - `quota_checks.LLMQuotaRefusal` (frozen dataclass: `category`, `current`, `limit`, `message`, `retry_after`) and `async quota_checks.llm_quota_refusal(user_id, requested_tokens=1) -> LLMQuotaRefusal | None`.
  - `quota_checks.LLM_TOKENS_PER_MONTH_KEY`, `LLM_COST_PER_DAY_KEY`, `LLM_COST_PER_MONTH_KEY`.
  - `llm_call_context.mark_llm_quota_prechecked() -> None`.
  - `usage_quota_deps.enforce_llm_quota(user_id, estimated_tokens=1)` (async) and the dependency `usage_quota_deps.require_llm_quota`.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_llm_quota_decision.py`:

```python
"""The LLM quota decision, the 402 error and the boundary helpers (LLM usage accounting spec §4, §4a)."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage import quota_checks
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context

pytestmark = pytest.mark.unit


def _set_limits(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokens: float | None = None,
    cost_day: float | None = None,
    cost_month: float | None = None,
    used_tokens: float = 0.0,
    spent_today: float = 0.0,
    spent_month: float = 0.0,
) -> list[str]:
    """Give the user these limits and this usage; return the list of counters that ran."""
    limits = {
        "limits.llm_tokens_per_month": tokens,
        "limits.llm_cost_per_day_usd": cost_day,
        "limits.llm_cost_per_month_usd": cost_month,
    }
    ran: list[str] = []

    async def _user_quota(_uid: object, key: str) -> float | None:
        """The configured limit for ``key``."""
        return limits.get(key)

    def _counter(name: str, value: float):
        """A counter that records that it ran."""

        async def _read(_uid: object) -> float:
            ran.append(name)
            return value

        return _read

    monkeypatch.setattr(quota_checks, "user_quota", _user_quota)
    monkeypatch.setattr(quota_checks, "llm_tokens_this_month", _counter("tokens", used_tokens))
    monkeypatch.setattr(quota_checks, "llm_cost_today", _counter("cost_today", spent_today))
    monkeypatch.setattr(quota_checks, "llm_cost_this_month", _counter("cost_month", spent_month))
    return ran


async def test_no_limits_admit_without_reading_any_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset limits cost no query."""
    ran = _set_limits(monkeypatch)
    assert await quota_checks.llm_quota_refusal(7, 500) is None
    assert ran == []


async def test_tokens_refuse_when_this_request_would_pass_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Used plus requested above the limit refuses; at the limit admits."""
    _set_limits(monkeypatch, tokens=100, used_tokens=99)
    assert await quota_checks.llm_quota_refusal(7, 1) is None
    refusal = await quota_checks.llm_quota_refusal(7, 2)
    assert refusal is not None
    assert (refusal.category, refusal.current, refusal.limit, refusal.message) == (
        "llm_tokens_month", 99, 100, "Monthly LLM token limit reached",
    )
    assert 1 <= refusal.retry_after <= 31 * 86400


@pytest.mark.parametrize(
    ("limits", "category", "message", "max_retry"),
    [
        ({"cost_day": 0.5, "spent_today": 0.5}, "llm_cost_day", "Daily LLM cost limit reached", 86400),
        ({"cost_month": 2.0, "spent_month": 2.5}, "llm_cost_month", "Monthly LLM cost limit reached", 31 * 86400),
    ],
)
async def test_cost_refuses_once_spend_reaches_the_cap(
    monkeypatch: pytest.MonkeyPatch, limits: dict, category: str, message: str, max_retry: int
) -> None:
    """Spend at or over a cost cap refuses with that cap's category, message and reset time."""
    _set_limits(monkeypatch, **limits)
    refusal = await quota_checks.llm_quota_refusal(7)
    assert refusal is not None
    assert (refusal.category, refusal.message) == (category, message)
    assert refusal.current == max(limits.get("spent_today", 0), limits.get("spent_month", 0))
    assert 1 <= refusal.retry_after <= max_retry


async def test_spend_under_the_cap_admits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cent under the daily cap is admitted."""
    _set_limits(monkeypatch, cost_day=0.5, spent_today=0.49)
    assert await quota_checks.llm_quota_refusal(7) is None


@pytest.mark.parametrize("limits", [{"tokens": 0}, {"cost_day": 0}, {"cost_month": 0}])
async def test_zero_blocks(monkeypatch: pytest.MonkeyPatch, limits: dict) -> None:
    """A limit of 0 refuses a user with no usage at all."""
    _set_limits(monkeypatch, **limits)
    assert await quota_checks.llm_quota_refusal(7) is not None


async def test_tokens_are_checked_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over both the token and the daily cost limit, the token refusal is reported and no cost counter runs."""
    ran = _set_limits(monkeypatch, tokens=10, used_tokens=50, cost_day=1.0, spent_today=5.0)
    refusal = await quota_checks.llm_quota_refusal(7)
    assert refusal is not None and refusal.category == "llm_tokens_month"
    assert ran == ["tokens"]


async def test_a_counter_failure_admits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cost counter that fails lets the request through, as every quota counter does."""
    _set_limits(monkeypatch, cost_day=0.5)

    async def _broken(_uid: object) -> float:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(quota_checks, "llm_cost_today", _broken)
    assert await quota_checks.llm_quota_refusal(7) is None


def test_the_error_is_the_public_402() -> None:
    """The error carries the body /chat/completions returns today, and Retry-After."""
    error = LLMQuotaExceededError(
        category="llm_cost_day", current=0.5, limit=0.5, message="Daily LLM cost limit reached", retry_after=60
    )
    assert error.status_code == 402
    assert error.detail == {
        "error": "limit_exceeded",
        "category": "llm_cost_day",
        "current": 0.5,
        "limit": 0.5,
        "message": "Daily LLM cost limit reached",
    }
    assert error.headers == {"Retry-After": "60"}


async def test_enforce_marks_the_task_only_when_it_admits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passing check marks the rest of the task prechecked; a refusal raises and marks nothing."""

    async def _admitted() -> bool:
        await usage_quota_deps.enforce_llm_quota(7)
        return current_llm_call_context().prechecked

    async def _refused() -> bool:
        with pytest.raises(LLMQuotaExceededError) as caught:
            await usage_quota_deps.enforce_llm_quota(7)
        assert caught.value.detail["category"] == "llm_tokens_month"
        return current_llm_call_context().prechecked

    _set_limits(monkeypatch)
    assert await asyncio.create_task(_admitted()) is True
    _set_limits(monkeypatch, tokens=0)
    assert await asyncio.create_task(_refused()) is False
    assert current_llm_call_context().prechecked is False


def _app_with_a_quota_dependency() -> FastAPI:
    """A one-route app whose route depends on require_llm_quota and streams what its body sees."""
    app = FastAPI()
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=7)

    @app.get("/stream", dependencies=[Depends(usage_quota_deps.require_llm_quota)])
    async def _stream() -> StreamingResponse:
        async def _body():
            yield f"prechecked={current_llm_call_context().prechecked}"

        return StreamingResponse(_body(), media_type="text/plain")

    return app


def test_the_dependency_mark_reaches_a_streaming_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mark set by require_llm_quota is still in effect when the streaming response body runs."""
    _set_limits(monkeypatch)
    with TestClient(_app_with_a_quota_dependency()) as client:
        response = client.get("/stream")
    assert (response.status_code, response.text) == (200, "prechecked=True")


def test_the_dependency_refuses_with_402_before_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """An over-quota user gets the 402 body and Retry-After, and the route never runs."""
    _set_limits(monkeypatch, cost_month=1.0, spent_month=3.0)
    with TestClient(_app_with_a_quota_dependency()) as client:
        response = client.get("/stream")
    assert response.status_code == 402
    assert response.json()["detail"]["category"] == "llm_cost_month"
    assert int(response.headers["Retry-After"]) >= 1
```

In `tests/Chat/integration/test_chat_llm_tokens_quota_402.py`:
- change `monkeypatch.setattr(chat_ep, "llm_tokens_this_month", _month_already_spent)` to `monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.llm_tokens_this_month", _month_already_spent)`, because `chat.py` no longer imports the counter;
- replace `assert response.json()["detail"]["category"] == "llm_tokens_month"` with the whole body:

```python
    assert response.json()["detail"] == {
        "error": "limit_exceeded",
        "category": "llm_tokens_month",
        "current": 1_000_000,
        "limit": 1,
        "message": "Monthly LLM token limit reached",
    }
```

and append a test for the new keys, streaming and not:

```python
async def _daily_cost_limit(_uid, key):
    """Only limits.llm_cost_per_day_usd is set, at 50 cents."""
    return 0.5 if key == "limits.llm_cost_per_day_usd" else None


async def _spent_fifty_cents(_uid):
    """The user has spent 50 cents today."""
    return 0.5


@pytest.mark.parametrize("stream", [False, True])
def test_chat_completions_402_on_daily_llm_cost_quota(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch, stream
):
    """At the daily LLM cost cap, /chat/completions returns 402 before dispatch, streaming or not."""
    dispatch_calls: list[object] = []

    async def _recording_dispatch(*args: object, **kwargs: object) -> None:
        """Record a dispatch call; the quota check must short-circuit before this runs."""
        dispatch_calls.append((args, kwargs))
        raise AssertionError("LLM dispatch must not run once the daily cost quota is spent")

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", _daily_cost_limit)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.llm_cost_today", _spent_fifty_cents)
    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _recording_dispatch)
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_orchestrator.chat_api_call", _recording_dispatch)
    monkeypatch.setitem(chat_ep.API_KEYS, "local-llm", "dummy-key")

    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
        api_provider="local-llm",
        stream=stream,
    )
    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert response.json()["detail"]["category"] == "llm_cost_day"
    assert 1 <= int(response.headers["Retry-After"]) <= 86400
    assert dispatch_calls == []
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_quota_decision.py tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py`
Expected: `test_llm_quota_decision.py` fails to collect (`ImportError: cannot import name 'LLMQuotaExceededError'`). In the chat file, all three tests fail with 502 where 402 is expected: the recording dispatch raises, and the endpoint reports that as a provider error.

- [ ] **Step 3: The error**

Append to `tldw_Server_API/app/core/exceptions.py`:

```python


class LLMQuotaExceededError(HTTPException):
    """The user is over an LLM quota: HTTP 402 with a ``limit_exceeded`` body (LLM usage accounting spec §4).

    An ``HTTPException`` and not a ``ChatAPIError``: endpoints report a
    ``ChatAPIError`` as a provider failure, and most let an ``HTTPException``
    through unchanged.
    """

    def __init__(self, *, category: str, current: float, limit: int | float, message: str, retry_after: int) -> None:
        super().__init__(
            status_code=402,
            detail={
                "error": "limit_exceeded",
                "category": category,
                "current": current,
                "limit": limit,
                "message": message,
            },
            headers={"Retry-After": str(int(retry_after))},
        )
        self.category = category
        self.current = current
        self.limit = limit
        self.retry_after = int(retry_after)
```

If `core/exceptions.py` has an `__all__` list, add `"LLMQuotaExceededError"` to it.

- [ ] **Step 4: The decision**

In `quota_checks.py`, add after the `UNLIMITED = ...` line:

```python

LLM_TOKENS_PER_MONTH_KEY = "limits.llm_tokens_per_month"
LLM_COST_PER_DAY_KEY = "limits.llm_cost_per_day_usd"
LLM_COST_PER_MONTH_KEY = "limits.llm_cost_per_month_usd"


@dataclass(frozen=True)
class LLMQuotaRefusal:
    """Why LLM work for a user is refused (LLM usage accounting spec §4)."""

    category: str
    current: float
    limit: int | float
    message: str
    retry_after: int
```

and add at the end of the file:

```python


def _count_refusal(category: str) -> None:
    """Count one refusal in ``llm_quota_refusals_total``; metrics never decide a request."""
    try:
        from tldw_Server_API.app.core.Metrics import increment_counter

        increment_counter("llm_quota_refusals_total", 1, labels={"category": category})
    except Exception:  # noqa: BLE001 - a metrics failure must not change the decision
        logger.debug("LLM quota refusal metric failed")


async def _cost_refusal(
    uid: int | None,
    key: str,
    counter: Callable[[int], Awaitable[float]],
    category: str,
    message: str,
    retry_after: Callable[[], int],
) -> LLMQuotaRefusal | None:
    """Refuse when the spend so far has reached the cap. An unset cap, or a failed counter, admits."""
    limit = await user_quota(uid, key)
    if limit is None or uid is None:
        return None
    try:
        used = float(await counter(uid))
    except Exception:  # noqa: BLE001 - a counter failure must not block requests (spec 2 §2)
        logger.opt(exception=True).warning(
            "Usage quota counter failed for key={} user_id={}; treating as unlimited", key, uid
        )
        return None
    if used < float(limit):
        return None
    return LLMQuotaRefusal(category, round(used, 4), limit, message, retry_after())


async def llm_quota_refusal(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
    """The first LLM quota the user is over, or None (LLM usage accounting spec §4).

    Tokens refuse when this month's tokens plus ``requested_tokens`` would pass
    the limit; with 1 requested, that is when the user is at or over it. Cost
    refuses when today's or this month's spend has reached its limit. Quotas
    off, unset limits and counter failures admit.
    """
    uid = as_quota_user_id(user_id)
    tokens = await check_usage(
        uid, LLM_TOKENS_PER_MONTH_KEY, max(1, int(requested_tokens)), lambda: llm_tokens_this_month(uid)
    )
    refusal: LLMQuotaRefusal | None = None
    if not tokens.allowed:
        refusal = LLMQuotaRefusal(
            "llm_tokens_month", int(tokens.used), tokens.limit, "Monthly LLM token limit reached",
            seconds_until_utc_month_start(),
        )
    if refusal is None:
        refusal = await _cost_refusal(
            uid, LLM_COST_PER_DAY_KEY, llm_cost_today, "llm_cost_day", "Daily LLM cost limit reached",
            seconds_until_utc_midnight,
        )
    if refusal is None:
        refusal = await _cost_refusal(
            uid, LLM_COST_PER_MONTH_KEY, llm_cost_this_month, "llm_cost_month", "Monthly LLM cost limit reached",
            seconds_until_utc_month_start,
        )
    if refusal is not None:
        _count_refusal(refusal.category)
    return refusal
```

`llm_cost_today` and the other counters are looked up by name when the function runs, so a test that replaces them on the module takes effect.

In `metrics_manager.py`, directly after the `self.register_metric(...)` call whose `name="llm_cost_dollars"`, add:

```python
        self.register_metric(
            MetricDefinition(
                name="llm_quota_refusals_total",
                type=MetricType.COUNTER,
                description="LLM work refused because a user was over an LLM quota",
                labels=["category"],
            )
        )
```

Add one line for `llm_quota_refusals_total` (counter, label `category`, "LLM work refused because a user was over an LLM quota") to the metrics list in `core/Metrics/README.md`, next to `llm_cost_dollars` and in the same format.

- [ ] **Step 5: The mark and the boundary helpers**

Append to `llm_call_context.py`:

```python


def mark_llm_quota_prechecked() -> None:
    """Mark the rest of the current task as admitted by a boundary quota check (LLM usage accounting spec §4a).

    A plain set, not ``llm_call_context``: a ``with`` block is reset before the
    endpoint runs when the check is a dependency, and before a streaming
    response body runs. Tasks and threads started after this inherit the mark.
    """
    _CTX.set(replace(_CTX.get(), prechecked=True))
```

In `usage_quota_deps.py`, change the imports to:

```python
from dataclasses import asdict
from typing import Any

from fastapi import Depends, HTTPException, status

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage.llm_call_context import mark_llm_quota_prechecked
from tldw_Server_API.app.core.Usage.quota_checks import (
    llm_quota_refusal,
    rag_queries_decision,
    seconds_until_utc_midnight,
)
```

and append:

```python


async def enforce_llm_quota(user_id: Any, estimated_tokens: int = 1) -> None:
    """402 limit_exceeded when the user is over an LLM quota; otherwise mark the request admitted (spec §4a)."""
    refusal = await llm_quota_refusal(user_id, estimated_tokens)
    if refusal is not None:
        raise LLMQuotaExceededError(**asdict(refusal))
    mark_llm_quota_prechecked()


async def require_llm_quota(current_user: User = Depends(get_request_user)) -> None:
    """``enforce_llm_quota`` as a dependency, for endpoints that always do LLM work."""
    await enforce_llm_quota(getattr(current_user, "id", None))
```

`require_llm_quota` is used as `Depends(require_llm_quota)`, not called. FastAPI resolves `get_request_user` once per request, so an endpoint that also depends on it authenticates once.

- [ ] **Step 6: `/chat/completions` uses the shared check**

In `chat.py`, replace the import block:

```python
from tldw_Server_API.app.core.Usage.quota_checks import (
    as_quota_user_id,
    check_usage,
    llm_tokens_this_month,
    seconds_until_utc_month_start,
)
```

with:

```python
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota
```

placed directly after the `personalization_deps` import block (the block that ends near line 299), before the `from tldw_Server_API.app.api.v1.schemas.chat_commands_schemas import` line. That is where ruff's isort puts it; left where the removed block was, it adds an `I001`. Then replace:

```python
            # Usage quotas (spec 2 §4): the user's monthly LLM token allowance.
            _quota_uid = as_quota_user_id(getattr(current_user, "id", None))
            _llm_decision = await check_usage(
                _quota_uid,
                "limits.llm_tokens_per_month",
                max(1, _estimated_request_tokens),
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
                    headers={"Retry-After": str(seconds_until_utc_month_start())},
                )
```

with:

```python
            # Usage quotas (LLM usage accounting spec §4a): monthly LLM tokens, and daily and monthly LLM spend.
            await enforce_llm_quota(getattr(current_user, "id", None), max(1, _estimated_request_tokens))
```

The body and header are unchanged for the token category.

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_quota_decision.py tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py tldw_Server_API/tests/Usage`
Expected: no failures. Then `$PY -m ruff check --output-format concise` on every file this task changed: the new test file is clean, and no changed app file gains a finding.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/exceptions.py tldw_Server_API/app/core/Usage/quota_checks.py \
  tldw_Server_API/app/core/Usage/llm_call_context.py tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py \
  tldw_Server_API/app/core/Metrics/metrics_manager.py tldw_Server_API/app/core/Metrics/README.md tldw_Server_API/app/api/v1/endpoints/chat.py \
  tldw_Server_API/tests/Usage/test_llm_quota_decision.py tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py
git commit -m "feat(usage): one LLM quota decision and 402 error; /chat/completions checks the cost keys (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Boundary checks for character chat, document generation, the study assistants and the Messages API

**Files:**
- Create: `tldw_Server_API/tests/helpers/llm_quota.py` (two test helpers shared by Tasks 5 to 7)
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` (`character_chat_completion`, near line 6746)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat_documents.py` (`generate_document`, near line 363)
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py` (the route decorator near line 2461)
- Modify: `tldw_Server_API/app/api/v1/endpoints/quizzes.py` (the route decorator near lines 780-783)
- Modify: `tldw_Server_API/app/api/v1/endpoints/messages.py` (the decorators of `create_messages` and `create_messages_public`, near lines 2217 and 2246)
- Test: one new test in each of `tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py`, `tests/Chat/integration/test_document_generation_endpoints.py`, `tests/Flashcards/test_flashcards_endpoint_integration.py`, `tests/Quizzes/test_quizzes_endpoint_integration.py`, `tests/Chat/integration/test_messages_endpoint.py`

**Interfaces:**
- Consumes: `enforce_llm_quota` and `require_llm_quota` from Task 4.
- Produces: `tldw_Server_API/tests/helpers/llm_quota.py` with `refuse_llm_work(monkeypatch)` and `assert_llm_quota_402(response)`.

- [ ] **Step 1: The test helpers**

`tldw_Server_API/tests/helpers/llm_quota.py`:

```python
"""Test helpers for the LLM quota boundary checks (LLM usage accounting spec §4a)."""

from typing import Any

import pytest


def refuse_llm_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every LLM quota check refuse: the user's monthly LLM token limit is 0 and nothing else is limited."""

    async def _user_quota(_uid: object, key: str) -> int | None:
        """Only the LLM token key is set, at 0."""
        return 0 if key == "limits.llm_tokens_per_month" else None

    async def _no_tokens(_uid: object) -> float:
        """No tokens used this month."""
        return 0.0

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", _user_quota)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.llm_tokens_this_month", _no_tokens)


def assert_llm_quota_402(response: Any) -> None:
    """The response is the public LLM quota refusal: 402, limit_exceeded, Retry-After."""
    assert response.status_code == 402, response.text
    detail = response.json()["detail"]
    assert (detail["error"], detail["category"], detail["limit"]) == ("limit_exceeded", "llm_tokens_month", 0)
    assert int(response.headers["Retry-After"]) >= 1
```

Other quota keys, such as `limits.rag_queries_per_day`, stay unlimited, so spec 2's checks do not refuse first.

- [ ] **Step 2: Write the failing tests**

Each test uses an existing harness in its file, calls `refuse_llm_work(monkeypatch)` before the request, replaces the function that would make the LLM call with a recorder, and asserts `assert_llm_quota_402(response)` and that the recorder was never called. Import both helpers with `from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work`.

Character chat, in `tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py`:

```python
@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
def test_complete_v2_refuses_an_over_quota_user_before_any_llm_call(
    monkeypatch, test_client, auth_headers, populated_character_db, stream
):
    """An over-quota user gets the 402, streaming or not, and the provider is never called."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    characters = test_client.get("/api/v1/characters/", headers=auth_headers)
    assert characters.status_code == 200, characters.text
    created = test_client.post(
        "/api/v1/chats/", headers=auth_headers, json={"character_id": characters.json()[0]["id"]}
    )
    assert created.status_code == 201, created.text
    calls: list[dict] = []

    def _recording_completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "unused"}}]}

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", _recording_completion)
    monkeypatch.setattr(character_chat_sessions, "is_model_known_for_provider", lambda *args: None)
    refuse_llm_work(monkeypatch)
    response = test_client.post(
        f"/api/v1/chats/{created.json()['id']}/complete-v2",
        headers=auth_headers,
        json={"provider": "llama", "model": "garden-model", "append_user_message": "Hi", "save_to_db": False, "stream": stream},
    )
    assert_llm_quota_402(response)
    assert calls == []
```

Document generation, in `tests/Chat/integration/test_document_generation_endpoints.py`, directly after `test_document_generate_bubbles_service_error`:

```python
def test_document_generate_refuses_an_over_quota_user_before_generation(authenticated_client, monkeypatch):
    """An over-quota user gets the 402 for a sync and an async request, and the service never generates."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    class RecordingStubService:
        generate_calls = 0

        def __init__(self, db):
            self._db = db

        def generate_document(self, *args, **kwargs):
            RecordingStubService.generate_calls += 1
            return {"success": True, "content": "unused", "metadata": {}}

        def get_generated_documents(self, *args, **kwargs):
            return []

        def record_streamed_document(self, *args, **kwargs):
            return None

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: RecordingStubService
    RecordingStubService.generate_calls = 0
    refuse_llm_work(monkeypatch)

    for payload in (_make_payload(), {**_make_payload(), "async_generation": True}):
        response = authenticated_client.post("/api/v1/chat/documents/generate", json=payload)
        assert_llm_quota_402(response)
        response.close()
    assert RecordingStubService.generate_calls == 0
```

Flashcards, in `tests/Flashcards/test_flashcards_endpoint_integration.py`, directly after `test_flashcard_assistant_respond_persists_user_and_assistant_messages`:

```python
def test_flashcard_assistant_respond_refuses_an_over_quota_user(
    client_with_flashcards_db: TestClient,
    monkeypatch,
):
    """An over-quota user gets the 402 and the study assistant never calls the LLM."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    created = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "What is the glomerulus?", "back": "It filters blood."},
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    card_uuid = created.json()["uuid"]
    calls: list[dict[str, Any]] = []

    async def recording_generate_reply(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"assistant_text": "unused", "structured_payload": {}, "provider": "openai", "model": "gpt-test"}

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.flashcards.generate_study_assistant_reply",
        recording_generate_reply,
    )
    refuse_llm_work(monkeypatch)

    response = client_with_flashcards_db.post(
        f"/api/v1/flashcards/{card_uuid}/assistant/respond",
        json={"action": "explain"},
        headers=AUTH_HEADERS,
    )

    assert_llm_quota_402(response)
    assert calls == []
```

Quizzes, in `tests/Quizzes/test_quizzes_endpoint_integration.py`, directly after `test_quiz_attempt_question_assistant_respond_persists_user_and_assistant_messages`:

```python
def test_quiz_attempt_question_assistant_respond_refuses_an_over_quota_user(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
    monkeypatch,
):
    """An over-quota user gets the 402 and the study assistant never calls the LLM."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    question_id = question_ids[0]
    calls: list[dict[str, Any]] = []

    async def recording_generate_reply(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"assistant_text": "unused", "structured_payload": {}, "provider": "openai", "model": "gpt-test"}

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.quizzes.generate_study_assistant_reply",
        recording_generate_reply,
    )
    refuse_llm_work(monkeypatch)

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id}/questions/{question_id}/assistant/respond",
        json={"action": "explain"},
        headers=AUTH_HEADERS,
    )

    assert_llm_quota_402(response)
    assert calls == []
```

The Messages API, in `tests/Chat/integration/test_messages_endpoint.py`, after `test_messages_endpoint_streaming`. It covers both paths: `openai/...` goes through the shared call layer, and `llama.cpp/...` through the native path, which posts to the provider directly.

```python
@pytest.mark.integration
@pytest.mark.parametrize("model", ["openai/gpt-4", "llama.cpp/test-model"])
@pytest.mark.parametrize("stream", [False, True])
def test_messages_refuses_an_over_quota_user_before_any_provider_call(client_user_only, monkeypatch, model, stream):
    """An over-quota user gets the 402 on the adapter path and the native path, streaming or not."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[str] = []

    async def _adapter_call(**kwargs):
        calls.append("adapter")
        return _build_openai_response("unused")

    async def _native_post(url, headers, payload, **_kwargs):
        calls.append("native")
        return _build_native_response()

    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", _adapter_call)
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_post)
    refuse_llm_work(monkeypatch)
    payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": stream}
    response = client_user_only.post("/api/v1/messages", json=payload)
    assert_llm_quota_402(response)
    assert calls == []
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider -k "quota or refuses" tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py`
Expected: each new test fails with a 200 (or the endpoint's normal result) where 402 is expected.

- [ ] **Step 4: Add the checks**

Each file gets `from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota` or `require_llm_quota`, at the place ruff's isort puts it:
- `character_chat_sessions.py`: directly after the `llm_routing_deps` import block;
- `chat_documents.py`: directly after the `chat_documents_deps` import;
- `flashcards.py` and `quizzes.py`: directly after the `jobs_deps` import. `flashcards.py` already has an unrelated `I001` on `dev`; place the import by hand and do not run `ruff --fix` there;
- `messages.py`: directly after the `from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, check_rate_limit, get_request_user` line.

Character chat. In `character_chat_completion`, insert directly before the line `        # Prepare rate limiter` (it occurs once in the file):

```python
        # Usage quotas (LLM usage accounting spec §4a): refuse before any LLM call or streaming response.
        await enforce_llm_quota(current_user.id)

```

It sits after the ownership check and the persona check, so a user who does not own the chat still gets 403 or 404. The outer `try` lets `HTTPException` through (`except HTTPException: raise` comes before the broad tuple).

Document generation. In `generate_document`, insert directly before the line `        credential_runtime = ProviderCredentialRuntime(` (it occurs once):

```python
        # Usage quotas (LLM usage accounting spec §4a), before the job branch and any generation.
        await enforce_llm_quota(user_id_int)
```

`user_id_int` was set on the line above by `derive_trusted_credential_scope`. A 402 is not a `ChatAPIError`, so it reaches `except HTTPException: raise`.

Flashcards. Change the decorator:

```python
@router.post("/{card_uuid}/assistant/respond", response_model=StudyAssistantRespondResponse)
```

to:

```python
@router.post(
    "/{card_uuid}/assistant/respond",
    response_model=StudyAssistantRespondResponse,
    dependencies=[Depends(require_llm_quota)],
)
```

Quizzes. In the decorator of `respond_quiz_attempt_question_assistant`, add `    dependencies=[Depends(require_llm_quota)],` after `    response_model=StudyAssistantRespondResponse,` (that line occurs once in the file).

The Messages API. The line `    dependencies=[Depends(check_rate_limit)],` occurs four times in `messages.py`. Change it only in the two decorators whose route is `"/messages"` and `"/v1/messages"` (`summary="Anthropic-compatible Messages API"`), the ones directly above `async def create_messages(` and `async def create_messages_public(`. Leave the two `count_tokens` decorators alone. In those two, change it to:

```python
    dependencies=[Depends(check_rate_limit), Depends(require_llm_quota)],
```

Both routes take `current_user: User = Depends(get_request_user)`, so the dependency adds no second authentication. The `count_tokens` routes are not changed: they count tokens and generate nothing. The dependency covers the native path too, which posts to Anthropic or llama.cpp directly and never reaches a provider adapter; a check at the call (PR 2b) would not see it.

Neither study-assistant route takes a user parameter. Both authenticate through `get_chacha_db_for_user`, which depends on `get_request_user`, so the dependency adds no second authentication.

- [ ] **Step 5: Run the tests and confirm they pass**

Run the Step 3 command. Expected: the new tests pass. Then run the four files whole and confirm no existing test changed result:

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py \
  tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py \
  tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py \
  tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py
```

Then the suites around them with `-n 4`: `tldw_Server_API/tests/Character_Chat_NEW`, `tldw_Server_API/tests/Flashcards`, `tldw_Server_API/tests/Quizzes`. No new failures.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/tests/helpers/llm_quota.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/endpoints/chat_documents.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py tldw_Server_API/app/api/v1/endpoints/quizzes.py \
  tldw_Server_API/app/api/v1/endpoints/messages.py tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py \
  tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py \
  tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py \
  tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py
git commit -m "feat(usage): character chat, document generation, the study assistants and the Messages API refuse over-quota users with 402 (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Boundary checks for RAG and the evaluation endpoints

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` (the `usage_quota_deps` import near line 44; `unified_search_endpoint` near line 1470; `unified_batch_endpoint` near lines 1675 and 1812; `resume_batch_endpoint` near line 2087; `unified_search_stream_endpoint` near line 2230)
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py` (`evaluate_geval` near line 996, `evaluate_rag` near 1211, `evaluate_response_quality` near 1407, `batch_evaluate` near 1696)
- Test: new tests in existing files, which own the harnesses and are already in CI's shard lists (CI lists `RAG_NEW/integration` files one by one, so a new file there would not run): `tests/RAG_NEW/integration/test_rag_agentic_api.py`, `test_rag_stream_parity.py`, `test_rag_batch_checkpoint_api.py`, `test_rag_batch_resume_api.py`, and `tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`

**Interfaces:**
- Consumes: `enforce_llm_quota` (Task 4); `refuse_llm_work` and `assert_llm_quota_402` (Task 5).

The evaluation endpoints do not authenticate with `get_request_user`: they use `get_eval_request_user`, and their tests override that. So they get the inline call, not the dependency.

- [ ] **Step 1: Write the failing tests**

In the RAG files, each test goes in the file that owns its harness.

`tests/RAG_NEW/integration/test_rag_agentic_api.py`, before `test_rag_agentic_search_smoke_api`. The third test is a guard: no profile the request schema accepts (`fast`, `balanced`, `accuracy`) turns generation on, so a profile alone is not refused.

```python
def _recording_pipeline(monkeypatch) -> list[dict]:
    """Replace the unified pipeline with a recorder that returns an empty result."""
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    calls: list[dict] = []

    async def _fake_pipeline(**kwargs):
        calls.append(kwargs)
        return up.UnifiedSearchResult(documents=[], query=kwargs.get("query", ""), errors=[])

    monkeypatch.setattr(rag_ep, "unified_rag_pipeline", _fake_pipeline)
    return calls


def test_rag_search_with_generation_refuses_an_over_quota_user(client_with_agentic_overrides, monkeypatch):
    """A search that generates an answer is refused with the 402 before any retrieval."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    calls = _recording_pipeline(monkeypatch)
    refuse_llm_work(monkeypatch)
    resp = client_with_agentic_overrides.post("/api/v1/rag/search", json={"query": "q", "enable_generation": True})
    assert_llm_quota_402(resp)
    assert calls == []


def test_rag_search_without_generation_is_not_refused_by_the_llm_quota(client_with_agentic_overrides, monkeypatch):
    """A search with generation off does no LLM work at the boundary, so an over-quota user still searches."""
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    calls = _recording_pipeline(monkeypatch)
    refuse_llm_work(monkeypatch)
    resp = client_with_agentic_overrides.post("/api/v1/rag/search", json={"query": "q", "enable_generation": False})
    assert resp.status_code != 402, resp.text
    assert len(calls) == 1


def test_rag_search_profile_without_explicit_generation_is_not_refused_by_the_llm_quota(
    client_with_agentic_overrides, monkeypatch
):
    """The resolved request decides: no shipped profile turns generation on, so a profile alone is not refused."""
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    calls = _recording_pipeline(monkeypatch)
    refuse_llm_work(monkeypatch)
    resp = client_with_agentic_overrides.post("/api/v1/rag/search", json={"query": "q", "rag_profile": "accuracy"})
    assert resp.status_code != 402, resp.text
    assert len(calls) == 1 and calls[0].get("enable_generation") is False
```

`tests/RAG_NEW/integration/test_rag_stream_parity.py`, at the end:

```python
def test_rag_search_stream_refuses_an_over_quota_user_with_a_real_402(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
) -> None:
    """The quota check runs before the streaming response starts, so the client gets a real 402."""
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    calls: list[dict[str, Any]] = []

    async def recording_generate_streaming_response(context: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return context

    monkeypatch.setattr(rag_ep, "generate_streaming_response", recording_generate_streaming_response)
    refuse_llm_work(monkeypatch)

    resp = client_with_stream_overrides.post("/api/v1/rag/search/stream", json={"query": "q", "enable_generation": True})

    assert_llm_quota_402(resp)
    assert calls == []
```

`tests/RAG_NEW/integration/test_rag_batch_checkpoint_api.py`, after `test_rag_batch_creates_checkpoint`. The second test is spec 2's own refusal, which returns 500 on `dev`:

```python
def test_rag_batch_with_generation_refuses_an_over_quota_user(client_with_overrides, monkeypatch, tmp_path):
    """A batch that generates answers is refused with the 402 before any query runs."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    _set_checkpoint_dir(monkeypatch, tmp_path)
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    monkeypatch.setattr(
        rag_ep,
        "_trusted_credential_runtime_scope",
        lambda *args, **kwargs: (1, [7], [11], True),  # noqa: ARG005
    )
    calls: list[str] = []

    async def recording_pipeline(query: str, **kwargs):  # noqa: ARG001
        calls.append(query)
        return up.UnifiedSearchResult(documents=[], query=query, errors=[])

    monkeypatch.setattr(up, "unified_rag_pipeline", recording_pipeline)
    refuse_llm_work(monkeypatch)

    resp = client_with_overrides.post("/api/v1/rag/batch", json={"queries": ["alpha"], "enable_generation": True})

    assert_llm_quota_402(resp)
    assert calls == []


def test_rag_batch_returns_the_rag_query_quota_402_not_a_500(client_with_overrides, monkeypatch, tmp_path):
    """Spec 2's daily RAG query refusal inside the batch handler reaches the client as a 402."""
    _set_checkpoint_dir(monkeypatch, tmp_path)
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    monkeypatch.setattr(
        rag_ep,
        "_trusted_credential_runtime_scope",
        lambda *args, **kwargs: (1, [7], [11], True),  # noqa: ARG005
    )

    async def _rag_queries_blocked(_uid, key):
        return 0 if key == "limits.rag_queries_per_day" else None

    async def _nothing_used_today(_entity, _category):
        return 0.0

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", _rag_queries_blocked)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.ledger_used_today", _nothing_used_today)

    resp = client_with_overrides.post("/api/v1/rag/batch", json={"queries": ["alpha"]})

    assert resp.status_code == 402, resp.text
    assert resp.json()["detail"]["category"] == "rag_queries_day"
    assert int(resp.headers["Retry-After"]) >= 1
```

`tests/RAG_NEW/integration/test_rag_batch_resume_api.py`, before `test_rag_batch_resume_records_errors`:

```python
def test_rag_batch_resume_with_generation_refuses_an_over_quota_owner(client_with_overrides, monkeypatch, tmp_path):
    """A resumed batch whose saved request generates answers is refused with the 402 before any query runs."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    manager = cp_mod.CheckpointManager()

    queries = ["alpha", "beta"]
    checkpoint = manager.create(
        "rag_batch",
        total_items=len(queries),
        config={"queries": queries, "max_concurrent": 2, "enable_generation": True},
    )

    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    calls: list[str] = []

    async def recording_pipeline(query: str, **kwargs):  # noqa: ARG001
        calls.append(query)
        return up.UnifiedSearchResult(documents=[], query=query, errors=[])

    monkeypatch.setattr(up, "unified_rag_pipeline", recording_pipeline)
    refuse_llm_work(monkeypatch)

    resp = client_with_overrides.post(f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}")

    assert_llm_quota_402(resp)
    assert calls == []
```

`tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`, before `test_batch_client_teardown_poison_isolated_before_next_credential_resolution`:

```python
_LLM_EVAL_BODIES = {
    "geval": {
        "source_text": "source text long enough",
        "summary": "summary text long enough",
        "metrics": ["coherence"],
        "api_name": "openai",
        "model": "gpt-eval-test",
    },
    "rag": {
        "query": "What is context precision?",
        "retrieved_contexts": ["Precision measures retrieved relevance."],
        "generated_response": "It measures retrieved relevance.",
        "metrics": ["context_precision"],
        "api_name": "openai",
        "model": "gpt-eval-test",
    },
    "response-quality": {
        "prompt": "What is RAG?",
        "response": "RAG retrieves context.",
        "api_name": "openai",
        "model": "gpt-eval-test",
    },
}


@pytest.mark.parametrize("route", ["geval", "rag", "response-quality"])
def test_sync_evaluation_llm_quota_refuses_before_the_rate_limiter(monkeypatch, route):
    """An over-quota user gets the 402 before the per-user rate limiter reserves anything."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    app = _build_eval_only_app(monkeypatch)

    def _limiter_must_not_run(_uid):
        raise AssertionError("the limiter ran")

    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", _limiter_must_not_run)
    refuse_llm_work(monkeypatch)

    with TestClient(app) as client:
        response = client.post(f"/api/v1/evaluations/{route}", json=_LLM_EVAL_BODIES[route])

    assert_llm_quota_402(response)


def test_batch_evaluation_llm_quota_refuses_llm_types_and_admits_propositions(monkeypatch):
    """A geval batch is refused with the 402; a propositions batch makes no LLM call and is not."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    app = _build_eval_only_app(monkeypatch)

    class _AllowLimiter:
        async def check_rate_limit(self, *_args, **_kwargs):
            return True, {}

    class _PropositionsService:
        async def evaluate_propositions(self, **_kwargs):
            return {"evaluation_id": "eval_props_1", "results": {"score": 1.0}}

    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: _AllowLimiter())
    monkeypatch.setattr(
        eval_unified, "get_unified_evaluation_service_for_user", lambda _uid: _PropositionsService()
    )
    refuse_llm_work(monkeypatch)

    with TestClient(app) as client:
        refused = client.post(
            "/api/v1/evaluations/batch",
            json={"evaluation_type": "geval", "parallel_workers": 1, "items": [_LLM_EVAL_BODIES["geval"]]},
        )
        admitted = client.post(
            "/api/v1/evaluations/batch",
            json={
                "evaluation_type": "propositions",
                "parallel_workers": 1,
                "items": [{"extracted": ["Claim A"], "reference": ["Claim A"], "method": "jaccard", "threshold": 0.5}],
            },
        )

    assert_llm_quota_402(refused)
    assert admitted.status_code != 402, admitted.text
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 RUN_EVALUATIONS=1 $PY -m pytest -q -p no:cacheprovider -k "quota" tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_checkpoint_api.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_resume_api.py tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`

Name every new test with `quota` in it, so `-k quota` selects them. `RUN_EVALUATIONS=1` is needed: the Evaluations tests are skipped without it.
Expected: `10 failed, 1 passed`. The refusal tests fail with the endpoints' normal results where 402 is expected (the stream answers 200 with an error frame; the evaluations answer 500 or 400). The spec 2 batch test fails with `assert 500 == 402`. The generation-off, profile and propositions tests are guards; the generation-off one passes already.

- [ ] **Step 3: RAG**

In `rag_unified.py`, add `enforce_llm_quota` to the existing import from `tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps`.

`unified_search_endpoint`. Directly after the line `        strategy_value = str(standard_bundle.resolved_request.strategy).strip().lower()` (it occurs once), insert:

```python
        # Usage quotas (LLM usage accounting spec §4a): a request that generates an answer is checked.
        # The resolved payload is what the pipeline receives; the agentic branch reads the same value.
        if bool(standard_bundle.resolved_request.payload.get("enable_generation", request.enable_generation)):
            await enforce_llm_quota(getattr(current_user, "id", None))
```

This covers the agentic and the standard branch, before any retrieval. The `try` around it has `except HTTPException: raise`.

`unified_search_stream_endpoint`. Directly before the line `    request = _apply_media_collection_scope(request, collections_db)` that follows the `enable_generation must be true for streaming.` check, insert:

```python
    # Usage quotas (LLM usage accounting spec §4a), before the streaming response starts.
    await enforce_llm_quota(getattr(current_user, "id", None))
```

Check with `git grep -n "request = _apply_media_collection_scope(request, collections_db)" -- tldw_Server_API/app/api/v1/endpoints/rag_unified.py` which occurrence is in this function; edit only that one.

`unified_batch_endpoint`. Directly after the line `        await enforce_rag_query_quota(getattr(current_user, "id", None), requested_units)`, insert:

```python
        if request.enable_generation:
            # Usage quotas (LLM usage accounting spec §4a): one flag turns generation on for every query.
            await enforce_llm_quota(getattr(current_user, "id", None))
```

and in the same function's handlers, directly before the line `    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context`, insert:

```python
    except HTTPException:
        raise
```

Confirm with `git grep -c` that each anchor line occurs once in the file; if the `except Exception` line occurs more than once, edit only the one in `unified_batch_endpoint`, whose next line logs `Batch search error`.

`resume_batch_endpoint`. Directly before these two lines (together they occur once):

```python
        batch_bundle = _build_batch_request_bundle(
            request=resume_request,
```

insert:

```python
        if resume_request.enable_generation:
            # Usage quotas (LLM usage accounting spec §4a): the work runs for the checkpoint's owner.
            await enforce_llm_quota(getattr(execution_user, "id", None))
```

`execution_user` is the checkpoint owner when one is recorded, and otherwise the caller.

- [ ] **Step 4: Evaluations**

In `evaluations_unified.py`, import `enforce_llm_quota` from `tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps`.

In each of `evaluate_geval`, `evaluate_rag` and `evaluate_response_quality`, insert as the first statement inside the `try:`, directly before the line `        # Per-user usage limits`:

```python
        # Usage quotas (LLM usage accounting spec §4a), before the rate limiter reserves anything.
        await enforce_llm_quota(current_user.id)
```

The comment line occurs three times in the file, once in each of these functions. Use each function's docstring last line as the anchor to find the right `try:`: `    G-Eval evaluates summaries on fluency, consistency, relevance, and coherence.`, `    Evaluates relevance, faithfulness, answer similarity, and context precision.` and `    Checks relevance, completeness, accuracy, and format compliance.`

In `batch_evaluate`, directly before the line `        if request.evaluation_type in {"geval", "rag", "response_quality"}:` (it occurs once), insert:

```python
        if request.evaluation_type in {"geval", "rag", "response_quality"}:
            # Usage quotas (LLM usage accounting spec §4a): these types make LLM calls inside the request.
            await enforce_llm_quota(current_user.id)
```

The four `try` blocks let `HTTPException` through: three have `except HTTPException: raise`, and the batch one's broad tuple does not contain it.

- [ ] **Step 5: Run the tests and confirm they pass**

Run the Step 2 command: all pass. Run the five files whole: no existing test changes result. Then the suites around the changed code, with `-n 4`: `tldw_Server_API/tests/RAG_NEW`, `tldw_Server_API/tests/Evaluations` (with `RUN_EVALUATIONS=1`, which the Evaluations suite needs). No new failures.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_checkpoint_api.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_resume_api.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py
git commit -m "feat(usage): RAG with generation and the in-request evaluations refuse over-quota users with 402 (TASK-13437)

The RAG batch endpoint now lets HTTPException through, so spec 2's daily RAG query refusal there is a 402, not a 500,
and the billing limit checker's 429 in the same handler is a 429.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Boundary checks for media analysis

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py` (one function at the end)
- Modify: the seven handlers under `tldw_Server_API/app/api/v1/endpoints/media/`: `process_audios.py`, `process_videos.py`, `process_documents.py`, `process_pdfs.py`, `process_ebooks.py`, `process_emails.py`, `add.py`
- Test: `tldw_Server_API/tests/Media/test_media_llm_quota.py` (CI runs `tests/Media` whole)

**Interfaces:**
- Consumes: `enforce_llm_quota` (Task 4); `refuse_llm_work`, `assert_llm_quota_402` (Task 5).
- Produces: `media_processing_deps.media_analysis_requested(form_data) -> bool`.

None of the seven handlers streams, and none has a `try` around the insertion points. Every decorator already has a `dependencies=[...]` list, but the check depends on the form, so it is an inline call.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Media/test_media_llm_quota.py`:

```python
"""Media processing with analysis refuses an over-quota user before any work (LLM usage accounting spec §4a)."""

from types import SimpleNamespace

import pytest

import tldw_Server_API.app.api.v1.endpoints.media as media_mod
from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import media_analysis_requested
from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work


@pytest.mark.unit
@pytest.mark.parametrize(
    ("perform_analysis", "api_name", "expected"),
    [(True, "openai", True), (True, None, False), (True, "", False), (True, "None", False), (False, "openai", False)],
)
def test_analysis_is_requested_only_with_a_provider(perform_analysis: bool, api_name: str | None, expected: bool) -> None:
    """The same condition the processing libraries use to decide whether to call the LLM."""
    form = SimpleNamespace(perform_analysis=perform_analysis, api_name=api_name)
    assert media_analysis_requested(form) is expected


@pytest.mark.integration
def test_ebook_analysis_is_refused_before_processing(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-quota user asking for analysis gets the 402, and the ebook is never processed."""
    client, _usage_logger = client_with_single_user
    processed: list[object] = []

    def _process_epub(*args: object, **kwargs: object) -> dict:
        processed.append(kwargs)
        return {"status": "Success", "content": "", "metadata": {}}

    monkeypatch.setattr(media_mod.books, "process_epub", _process_epub)
    refuse_llm_work(monkeypatch)
    response = client.post(
        "/api/v1/media/process-ebooks",
        data={"perform_analysis": "true", "api_name": "openai"},
        files=[("files", ("sample.epub", b"fake", "application/epub+zip"))],
    )
    assert_llm_quota_402(response)
    assert processed == []


@pytest.mark.integration
def test_ebook_processing_without_analysis_is_not_refused(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a provider there is no LLM work, so an over-quota user is not refused."""
    client, _usage_logger = client_with_single_user
    monkeypatch.setattr(
        media_mod.books,
        "process_epub",
        lambda *args, **kwargs: {"status": "Success", "content": "", "metadata": {}},
    )
    refuse_llm_work(monkeypatch)
    response = client.post(
        "/api/v1/media/process-ebooks",
        files=[("files", ("sample.epub", b"fake", "application/epub+zip"))],
    )
    assert response.status_code != 402, response.text
```

`client_with_single_user` is in `tldw_Server_API/tests/conftest.py`. Add this fixture, copied from `tests/Media/test_media_usage_events.py`, and the test for the other six handlers, to the same file:

```python
class _StubQuotaService:
    async def check_quota(self, user_id, size_bytes, raise_on_exceed=False):
        """Always allow in tests."""
        return True, {
            "current_usage_mb": 0,
            "new_size_mb": float(size_bytes) / (1024 * 1024),
            "quota_mb": 999999,
            "available_mb": 999999,
        }


@pytest.fixture()
def quota_service_stub(monkeypatch):
    """Patch the storage quota service so no DB is touched (as tests/Media/test_media_usage_events.py does)."""
    import tldw_Server_API.app.services.storage_quota_service as quota_mod

    monkeypatch.setattr(quota_mod, "get_storage_quota_service", lambda: _StubQuotaService())
    yield


_ANALYSIS = {"perform_analysis": "true", "api_name": "openai"}

_OTHER_HANDLERS = [
    pytest.param("/api/v1/media/process-audios", _ANALYSIS, [("files", ("a.mp3", b"fake", "audio/mpeg"))], id="audios"),
    pytest.param("/api/v1/media/process-videos", _ANALYSIS, [("files", ("v.mp4", b"fake", "video/mp4"))], id="videos"),
    pytest.param("/api/v1/media/process-documents", _ANALYSIS, [("files", ("d.txt", b"fake", "text/plain"))], id="documents"),
    pytest.param("/api/v1/media/process-pdfs", _ANALYSIS, [("files", ("d.pdf", b"%PDF-1.4 fake", "application/pdf"))], id="pdfs"),
    pytest.param("/api/v1/media/process-emails", _ANALYSIS, [("files", ("m.eml", b"fake", "message/rfc822"))], id="emails"),
    pytest.param("/api/v1/media/add", {**_ANALYSIS, "media_type": "document"}, [("files", ("d.txt", b"fake", "text/plain"))], id="add"),
]


@pytest.mark.integration
@pytest.mark.parametrize(("path", "data", "files"), _OTHER_HANDLERS)
def test_other_media_handlers_refuse_analysis_for_an_over_quota_user(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch, path: str, data: dict, files: list
) -> None:
    """Each remaining handler answers the 402 before touching its input."""
    client, _usage_logger = client_with_single_user
    refuse_llm_work(monkeypatch)
    response = client.post(path, data=data, files=files)
    assert_llm_quota_402(response)
```

With the checks in place no processing stub is needed for the six: the 402 comes first. Without them, the fake inputs are processed and the handlers answer 200 or 207, which is the failing run.



- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Media/test_media_llm_quota.py`
Expected: collection fails with `ImportError: cannot import name 'media_analysis_requested'`. After Step 3's helper exists, the endpoint refusal tests fail with 200 or 207 where 402 is expected. The no-analysis test passes before and after; it guards that nothing else is refused.

- [ ] **Step 3: The condition**

`media_processing_deps.py` ends with an `__all__` list. Insert this directly before it, and add `"media_analysis_requested"` to the list:

```python


def media_analysis_requested(form_data: Any) -> bool:
    """True when a media request asks for LLM analysis.

    The same condition the audio and video libraries use before they call the
    LLM: analysis on, and a provider named that is not "none".
    """
    api_name = getattr(form_data, "api_name", None)
    return bool(getattr(form_data, "perform_analysis", False) and api_name and str(api_name).lower() != "none")
```

- [ ] **Step 4: The seven checks**

Each handler imports `media_analysis_requested` from `tldw_Server_API.app.api.v1.API_Deps.media_processing_deps` (add the name to the existing import from that module where there is one) and `from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota`, placed directly after the `from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota` line. That is where ruff's isort puts it in all seven files; next to the other `API_Deps` imports it adds an `I001`. Insert the same three lines:

```python
    if media_analysis_requested(form_data):
        # Usage quotas (LLM usage accounting spec §4a), before any download or processing.
        await enforce_llm_quota(current_user.id)

```

at these places, each directly before the named line (every named line occurs once in its file; confirm with `git grep -c`):

| File | Insert directly before |
|---|---|
| `process_audios.py` | `    if (` (the prompt lookup after the input validation `try`) |
| `process_videos.py` | `    final_summary_prompt = await resolve_video_summary_prompts(` |
| `process_documents.py` | `    system_prompt = form_data.system_prompt` |
| `process_pdfs.py` | `    system_prompt = form_data.system_prompt` |
| `process_ebooks.py` | `    system_prompt = form_data.system_prompt` |
| `process_emails.py` | `    system_prompt = form_data.system_prompt` |
| `add.py` | `    if is_research_discovery_handoff(form_data):` |

In `add.py` the check is the first statement of the body, so it covers the research-discovery branch too, which also persists with analysis.

- [ ] **Step 5: Run the tests and confirm they pass**

Run the Step 2 command: all pass. Then `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider tldw_Server_API/tests/Media tldw_Server_API/tests/MediaIngestion_NEW`: no new failures.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py tldw_Server_API/app/api/v1/endpoints/media/process_audios.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_videos.py tldw_Server_API/app/api/v1/endpoints/media/process_documents.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_pdfs.py tldw_Server_API/app/api/v1/endpoints/media/process_ebooks.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_emails.py tldw_Server_API/app/api/v1/endpoints/media/add.py \
  tldw_Server_API/tests/Media/test_media_llm_quota.py
git commit -m "feat(usage): media processing with analysis refuses over-quota users with 402 before any work (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Startup warnings and the profile's LLM block

**Files:**
- Modify: `tldw_Server_API/app/core/Usage/quota_checks.py` (two functions at the end)
- Modify: `tldw_Server_API/app/services/startup_auth_runtime.py` (`initialize_auth_runtime_services`, near line 33; one helper)
- Modify: `tldw_Server_API/app/core/UserProfiles/service.py` (`_build_quotas`, before `return quotas`, near line 539)
- Modify: `tldw_Server_API/app/api/v1/schemas/user_profile_schemas.py` (`UserProfileQuotas`, near line 106)
- Modify: `apps/packages/api-types/openapi.fingerprint.json` (regenerated)
- Test: `tldw_Server_API/tests/Usage/test_llm_quota_settings_and_summary.py`

**Interfaces:**
- Consumes: Task 1's counters; Task 4's key constants.
- Produces: `quota_checks.warn_if_llm_quotas_defeated() -> list[str]`; `async quota_checks.llm_quota_summary(user_id: int) -> dict`; the profile's `quotas.llm` block.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_llm_quota_settings_and_summary.py`:

```python
"""Startup warnings for settings that defeat the LLM quotas, and the profile's LLM block (spec §6, §9)."""

from types import SimpleNamespace

import pytest

from fastapi.encoders import jsonable_encoder

from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import UserProfileQuotas
from tldw_Server_API.app.core.Usage import quota_checks
from tldw_Server_API.app.services import startup_auth_runtime

pytestmark = pytest.mark.unit


@pytest.fixture
def quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, recording on, a 180-day retention, and the warn-once flag cleared."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.setenv("LLM_USAGE_ENABLED", "1")
    monkeypatch.setattr(quota_checks, "_LLM_QUOTA_SETTINGS_WARNED", False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.settings.get_settings",
        lambda: SimpleNamespace(LLM_USAGE_LOG_RETENTION_DAYS=180),
    )


def test_sound_settings_warn_nothing(quotas_on: None) -> None:
    """Quotas on with recording on and a long retention: no warning."""
    assert quota_checks.warn_if_llm_quotas_defeated() == []


def test_recording_off_is_reported_once(quotas_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With LLM_USAGE_ENABLED off the LLM quotas never refuse; the warning names the setting, once per process."""
    monkeypatch.setenv("LLM_USAGE_ENABLED", "0")
    problems = quota_checks.warn_if_llm_quotas_defeated()
    assert len(problems) == 1 and "LLM_USAGE_ENABLED" in problems[0]
    assert quota_checks.warn_if_llm_quotas_defeated() == []


def test_a_short_retention_is_reported(quotas_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under 31 days of retention, the monthly counters lose rows they need."""
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.settings.get_settings",
        lambda: SimpleNamespace(LLM_USAGE_LOG_RETENTION_DAYS=7),
    )
    problems = quota_checks.warn_if_llm_quotas_defeated()
    assert len(problems) == 1 and "LLM_USAGE_LOG_RETENTION_DAYS" in problems[0] and "31" in problems[0]


def test_nothing_is_reported_with_quotas_off(quotas_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With quotas off nothing is enforced, so nothing is defeated."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    monkeypatch.setenv("LLM_USAGE_ENABLED", "0")
    assert quota_checks.warn_if_llm_quotas_defeated() == []


async def test_startup_runs_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The auth runtime startup step calls the check."""
    calls: list[str] = []
    monkeypatch.setattr(quota_checks, "warn_if_llm_quotas_defeated", lambda: calls.append("checked") or [])
    startup_auth_runtime._warn_if_llm_quotas_defeated()
    assert calls == ["checked"]


async def test_the_summary_reports_usage_and_set_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage is always reported; a limit appears only when it is set."""

    async def _user_quota(_uid: object, key: str) -> float | None:
        return 2.5 if key == "limits.llm_cost_per_day_usd" else None

    async def _value(value: float):
        return value

    monkeypatch.setattr(quota_checks, "user_quota", _user_quota)
    monkeypatch.setattr(quota_checks, "llm_tokens_this_month", lambda _uid: _value(1234.0))
    monkeypatch.setattr(quota_checks, "llm_cost_today", lambda _uid: _value(0.123456))
    monkeypatch.setattr(quota_checks, "llm_cost_this_month", lambda _uid: _value(4.5))
    summary = await quota_checks.llm_quota_summary(7)
    assert summary == {
        "tokens_month_used": 1234,
        "tokens_month_limit": None,
        "cost_day_usd_used": 0.1235,
        "cost_day_usd_limit": 2.5,
        "cost_month_usd_used": 4.5,
        "cost_month_usd_limit": None,
    }
    # Both profile endpoints use response_model_exclude_none=True, which FastAPI applies with
    # jsonable_encoder: it drops None inside nested dicts, where model_dump(exclude_none=True) does not.
    dumped = jsonable_encoder(UserProfileQuotas(storage_used_mb=0.0, llm=summary), exclude_none=True)
    assert dumped["llm"] == {
        "tokens_month_used": 1234,
        "cost_day_usd_used": 0.1235,
        "cost_day_usd_limit": 2.5,
        "cost_month_usd_used": 4.5,
    }
```

The last assertion shows what a client receives: both profile endpoints leave out null values, so an unset limit is absent.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_quota_settings_and_summary.py`
Expected: failures with `AttributeError` for `_LLM_QUOTA_SETTINGS_WARNED`, `warn_if_llm_quotas_defeated` and `llm_quota_summary`.

- [ ] **Step 3: The settings check and the summary**

Append to `quota_checks.py`:

```python


_LLM_QUOTA_SETTINGS_WARNED = False


def warn_if_llm_quotas_defeated() -> list[str]:
    """Log, once per process, a setting that silently disables the LLM quotas while quotas are on (spec §6).

    Returns the problems it logged, for tests. Never raises.
    """
    global _LLM_QUOTA_SETTINGS_WARNED
    problems: list[str] = []
    try:
        from tldw_Server_API.app.core.config import usage_quotas_enabled

        if _LLM_QUOTA_SETTINGS_WARNED or not usage_quotas_enabled():
            return problems
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        from tldw_Server_API.app.core.Usage.usage_tracker import _enabled as llm_usage_recording_enabled

        if not llm_usage_recording_enabled():
            problems.append(
                "LLM_USAGE_ENABLED is off, so no LLM usage is recorded and the LLM quotas never refuse anything."
            )
        retention = int(getattr(get_settings(), "LLM_USAGE_LOG_RETENTION_DAYS", 180))
        if retention < 31:
            problems.append(
                f"LLM_USAGE_LOG_RETENTION_DAYS is {retention}, so the monthly LLM quotas count less than a month. "
                "Set it to 31 or more."
            )
        _LLM_QUOTA_SETTINGS_WARNED = True
        for problem in problems:
            logger.warning("Usage quotas are on, but {}", problem)
    except Exception:  # noqa: BLE001 - a startup diagnostic must never stop the server
        logger.opt(exception=True).debug("LLM quota settings check failed")
    return problems


async def llm_quota_summary(user_id: int) -> dict[str, Any]:
    """The user's LLM usage this month and today, and their LLM limits, for the profile (spec §9)."""
    return {
        "tokens_month_used": int(await llm_tokens_this_month(user_id)),
        "tokens_month_limit": await user_quota(user_id, LLM_TOKENS_PER_MONTH_KEY),
        "cost_day_usd_used": round(await llm_cost_today(user_id), 4),
        "cost_day_usd_limit": await user_quota(user_id, LLM_COST_PER_DAY_KEY),
        "cost_month_usd_used": round(await llm_cost_this_month(user_id), 4),
        "cost_month_usd_limit": await user_quota(user_id, LLM_COST_PER_MONTH_KEY),
    }
```

The three `user_quota` reads hit the resolver's per-user cache after the first.

- [ ] **Step 4: Call the check at startup**

In `startup_auth_runtime.py`, change:

```python
        await _warn_if_plan_limits_unenforced()
```

to:

```python
        await _warn_if_plan_limits_unenforced()
        _warn_if_llm_quotas_defeated()
```

and add, directly after the `_warn_if_plan_limits_unenforced` function:

```python


def _warn_if_llm_quotas_defeated() -> None:
    """Log at startup when a setting silently disables the LLM quotas (LLM usage accounting spec §6)."""
    from tldw_Server_API.app.core.Usage import quota_checks

    quota_checks.warn_if_llm_quotas_defeated()
```

`tests/Services/test_startup_auth_runtime.py` asserts the order of the steps it replaces. It does not replace this one, which runs for real there and logs nothing with quotas off.

- [ ] **Step 5: The profile block**

In `UserProfileQuotas`, after the `prompt_studio` field, add:

```python
    llm: Optional[dict[str, Any]] = Field(
        None,
        description="LLM tokens used this month and USD spent today and this month, with the user's LLM limits; "
        "an unset limit is absent",
    )
```

In `service.py`, in `_build_quotas`, directly before `        return quotas`, add:

```python
        try:
            from tldw_Server_API.app.core.Usage.quota_checks import llm_quota_summary

            quotas["llm"] = await llm_quota_summary(user_id)
        except Exception as exc:  # noqa: BLE001 - one unavailable block must not fail the quotas section
            logger.debug("LLM quotas unavailable for user {}: {}", user_id, exc)
```

`user_id` is the variable the other blocks in `_build_quotas` use; confirm the name there. The block catches `Exception` because a database error from a counter is not in `_PROFILE_NONCRITICAL_EXCEPTIONS`, and it should leave the rest of the section intact.

- [ ] **Step 6: Regenerate the OpenAPI fingerprint**

The schema gained a field, so the checked-in fingerprint changes:

```bash
$PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json
```

The first command must now print `[openapi-export] FAIL — OpenAPI contract drift detected.` Regenerate:

```bash
$PY Helper_Scripts/export_openapi_schema.py --fingerprint apps/packages/api-types/openapi.fingerprint.json
$PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json
```

The second must print `OpenAPI fingerprint matches the checked-in snapshot`. `git diff --stat apps/packages/api-types` must show only `openapi.fingerprint.json`. Each command imports the whole app and takes several minutes.

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_quota_settings_and_summary.py tldw_Server_API/tests/Services/test_startup_auth_runtime.py tldw_Server_API/tests/UserProfile`
Expected: no failures.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Usage/quota_checks.py tldw_Server_API/app/services/startup_auth_runtime.py \
  tldw_Server_API/app/core/UserProfiles/service.py tldw_Server_API/app/api/v1/schemas/user_profile_schemas.py \
  apps/packages/api-types/openapi.fingerprint.json tldw_Server_API/tests/Usage/test_llm_quota_settings_and_summary.py
git commit -m "feat(usage): warn when a setting defeats the LLM quotas; the profile reports LLM usage and limits (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Operator docs, the spec status and the backlog

**Files:**
- Modify: `Docs/Operations/Usage_Quotas.md` (the `limits.*` table near lines 63-75 and the section "What `limits.llm_tokens_per_month` counts", near lines 79-87)
- Modify: `Docs/Operations/Env_Vars.md` (the keys list near line 322), then refresh `Docs/Published`
- Modify: `Docs/Operations/Rate_Limits_Troubleshooting.md` (the `llm_tokens_month` row near line 19)
- Modify: `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (the Status line, one sentence in §6)
- Backlog: TASK-13437 notes

- [ ] **Step 1: The operator page**

Read `Docs/Operations/Usage_Quotas.md` first and match its style. Write plainly: short sentences, concrete names. Change only these parts.

- **The `limits.*` table.** The `limits.llm_tokens_per_month` row no longer says "enforced on `/chat/completions` only"; it says LLM tokens per calendar month (UTC), checked before LLM work at the entry points listed below. Add a row after it for each new key, with the same columns:
  - `limits.llm_cost_per_day_usd`: LLM spend per UTC day in US dollars; `llm_usage_log` costs summed per user over the day; HTTP 402 `{"error":"limit_exceeded","category":"llm_cost_day",...}`.
  - `limits.llm_cost_per_month_usd`: the same per calendar month; category `llm_cost_month`.
- **The section on what is counted** covers all three keys now. Keep its existing bullets and add:
  - **Where a refusal happens.** These entry points refuse with a 402 before they start any LLM work, streaming requests included: `/chat/completions`; the Anthropic-compatible Messages API (`/messages` and `/v1/messages`); character chat replies; RAG search, batch, batch resume and streaming, when the request generates an answer; document generation from a chat; the flashcards and quizzes study assistants; `/evaluations/geval`, `/evaluations/rag` and `/evaluations/response-quality`; `/evaluations/batch` of type `geval`, `rag` or `response_quality`; the `/media/process-*` endpoints and `/media/add`, when analysis is on.
  - **Not checked yet** (a later change refuses them at the call): background jobs (`/media/ingest-jobs`, evaluation recipe runs, audio jobs), workflow runs, WebSocket sessions (audio, realtime), MCP tools, and these HTTP endpoints: `/media/ingest-web-content` with analysis, `/media/{id}/insights`, the writing and manuscript analysis endpoints, prompt improvement and Prompt Studio, reading, study suggestions, paper search, chunking, sharing, and `/translate`. Their usage is recorded and counts toward the limits. MCP tools are not recorded (they carry no user).
  - **Not recorded today:** Messages API calls to Anthropic or llama.cpp take a native path that never reaches a provider adapter, so they write no usage row. The boundary check still refuses them. TASK-13610 covers recording them.
  - **How the cost keys compare.** A cost key refuses once today's or this month's spend has reached the limit. The token key refuses when this request's estimated tokens would take the month past it; outside `/chat/completions` the estimate is one token.
  - **An admitted request runs to completion,** so a user can pass a limit by one request's worth of usage. A media batch or an evaluation batch can be large.
  - **Local providers cost nothing by default:** llama.cpp, kobold, ooba, tabbyapi, vllm, local-llm, ollama, aphrodite and mlx. To give one a price, edit its `default` entry in `Config_Files/model_pricing.json`. A custom OpenAI-compatible endpoint is priced at the unknown-model rate ($0.01 per 1K prompt tokens, $0.03 per 1K completion tokens) unless you give it a price. A virtual-key USD budget no longer counts local calls either.
  - **Two settings defeat these limits, and the server warns at startup when quotas are on:** `LLM_USAGE_ENABLED` off (nothing is recorded, so nothing is refused), and `LLM_USAGE_LOG_RETENTION_DAYS` under 31 (the monthly counts lose rows).
  - **Failures let requests through.** If a limit cannot be read or a counter fails, the request is admitted and a warning is logged.
  - **The profile reports it.** `GET /api/v1/users/me/profile?sections=quotas` has an `llm` block with tokens used this month, spend today and this month, and each limit that is set.

- [ ] **Step 2: The other docs**

- `Env_Vars.md`, the keys list near line 322: replace "`limits.llm_tokens_per_month` (enforced on `/chat/completions`)" with the key alone, and add `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd` in the same list format. Then run `bash Helper_Scripts/refresh_docs_published.sh` and check `git status --short Docs/Published`: only `Docs/Published/Env_Vars.md` may change. If anything else changes, do not commit it and do not revert it with git; report the names.
- `Rate_Limits_Troubleshooting.md`, the row for `llm_tokens_month`: it now covers all three categories (`llm_tokens_month`, `llm_cost_day`, `llm_cost_month`). The source column names the entry points from Step 1 (or points to `Usage_Quotas.md` for the list), and the fix column says to raise or remove the user's `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd` or `limits.llm_cost_per_month_usd`.
- The spec: replace the Status line with `- **Status:** Approved; PR 1a (#3222) and PR 1b (#3224) merged; PR 2a in progress.` In the §4a table, add a row after `/chat/completions`: `| The Anthropic-compatible Messages API (`/messages`, `/v1/messages`) | Always. Its native path for Anthropic and llama.cpp never reaches a provider adapter, so only this check gates it. |`. Replace the table's following bullet "**The plan lists every other HTTP endpoint that reaches an LLM.** ..." with one that names the endpoints left to §4b: the same list as the operator page's "Not checked yet" bullet. In §6, in the bullet that begins "**`compute_costs` resolves the provider's canonical name first,**", change "`compute_costs` resolves" to "The catalog lookup resolves". The plan put the resolution in `get_rate_details`, which `compute_costs` and the other pricing helpers all call.

- [ ] **Step 3: The backlog**

Put the command in a small `sh` script in the scratch directory (the command guard refuses a command that starts with an environment variable) and run it once:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server/.claude/worktrees/core-review-fixes || exit 1
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13437 --append-notes "PR 2a: one LLM quota decision (llm_quota_refusal) and one 402 error (LLMQuotaExceededError, an HTTPException); limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd; boundary checks at /chat/completions, character chat, RAG with generation, document generation, the study assistants, the in-request evaluations and media analysis; local providers priced at zero in the catalog; startup warnings for LLM_USAGE_ENABLED off and retention under 31 days; the profile's quotas.llm block; the Messages API is checked at the boundary. RAG batch now returns spec 2's 402 instead of a 500. Next: PR 2b, the check at the call."
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Record LLM usage for the Messages API native path" --id TASK-13610 --labels quotas,backend --description "The Anthropic-compatible Messages API (api/v1/endpoints/messages.py) sends requests for the providers in MESSAGES_NATIVE_PROVIDERS (anthropic, llama.cpp) straight to the provider over HTTP (_native_post_json, _prepare_native_stream_iterator). Those calls never reach a ChatProvider adapter, so the provider base class (ADR-067) records nothing for them and a check at the call (LLM usage accounting PR 2b) cannot see them. PR 2a gates the endpoint at the request boundary. Parent: TASK-13437." --ac "A native Messages API call writes one llm_usage_log row for the user, streaming and not" --ac "The row's tokens come from the provider's usage block when it sends one"
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task normalize --check backlog/tasks/task-13437*.md backlog/tasks/task-13610*.md
echo "normalize exit=$?"
```

Expected: `normalize exit=0`, and `git status --short backlog` shows the modified TASK-13437 file and the new TASK-13610 file, nothing else. If another session has taken TASK-13610 by then, use the next free id with a gap of 20 above the highest in use, here and on the operator page.

- [ ] **Step 4: Run the docs tests**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider tldw_Server_API/tests/Docs`
Expected: no failures.

- [ ] **Step 5: Commit**

```bash
git add Docs/Operations/Usage_Quotas.md Docs/Operations/Env_Vars.md Docs/Published/Env_Vars.md \
  Docs/Operations/Rate_Limits_Troubleshooting.md Docs/Design/2026-10-09-llm-usage-accounting-design.md
git add -A backlog/tasks
git commit -m "docs(usage): the LLM cost keys, where an LLM quota refuses, and local-provider pricing (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Ship

- [ ] **Step 1: Rebase and re-check**

```bash
git fetch origin dev
git rebase origin/dev
$PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json
```

Expected: the fingerprint matches (Task 8 regenerated it). If `dev` moved the fingerprint again, regenerate it on the rebased branch.

- [ ] **Step 2: Run the sweep**

Every test file that references a changed symbol, plus the suites around the changed code, with `-n 4`. The repo's command guard refuses `$(...)`, so write the commands below into a small `sh` script in the scratch directory, with the file list read by the script, and run the script.

```bash
git grep -l -e quota_checks -e usage_quota_deps -e LLMQuotaExceededError -e llm_call_context -e pricing_catalog -e compute_costs \
  -e user_profile_catalog -e _build_quotas -e UserProfileQuotas -e startup_auth_runtime -e sum_user_llm \
  -- 'tldw_Server_API/tests/**/test_*.py' > <scratchpad>/pr2a_tests.txt
TLDW_TEST_NO_DOCKER=1 RUN_EVALUATIONS=1 $PY -m pytest -q -n 4 -p no:cacheprovider --timeout=600 $(cat <scratchpad>/pr2a_tests.txt) \
  tldw_Server_API/tests/Usage tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Character_Chat_NEW \
  tldw_Server_API/tests/Flashcards tldw_Server_API/tests/Quizzes tldw_Server_API/tests/RAG_NEW tldw_Server_API/tests/Evaluations \
  tldw_Server_API/tests/Chat tldw_Server_API/tests/Services tldw_Server_API/tests/Docs tldw_Server_API/tests/lint \
  tldw_Server_API/tests/Media tldw_Server_API/tests/MediaIngestion_NEW
```

Every failure needs evidence before it is called pre-existing: it fails the same way at `origin/dev` in a detached worktree, or it passes alone at both. Known on `dev`: `tests/Services/test_router_groups_contract.py` (10 tests), `tests/Audio/test_nemo_streaming_error_sentinel.py`, and the load flakes in Global Constraints.

- [ ] **Step 3: Static checks**

Bandit `-ll` on the changed app files: 0 issues. Ruff per-rule counts on the changed app files must not rise against the merge base; new test files are clean.

- [ ] **Step 4: Open the PR against `dev`**

The body covers: what is refused and where; the error type and why it is an `HTTPException`; local-provider pricing and its effect on reports and virtual-key budgets; that `default` is now a reserved key in the pricing catalog, and `huggingface` no longer lists it as a model; the RAG batch fix (spec 2's 402 and the billing 429); that a character chat request on the offline simulation path is refused too; that the Messages API's native path is gated but not recorded (TASK-13610); what is not checked until PR 2b; the review focus items and their tests; the sweep results with each failure's evidence; and the `## Change summary` waiver section. Queue it alone. After the merge, delete the branch.
