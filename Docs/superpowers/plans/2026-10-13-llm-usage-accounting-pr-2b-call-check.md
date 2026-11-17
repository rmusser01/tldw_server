# LLM Usage Accounting PR 2b (the check at the call) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With usage quotas on, an LLM call made for a user who is over an LLM quota, and that no request boundary admitted, is refused with `LLMQuotaExceededError` before the provider is called: background jobs, workflow runs, WebSocket sessions and every HTTP endpoint outside the §4a table. A refusal never looks like a provider outage to shared machinery.

**Architecture:** One new module, `core/Usage/llm_quota_gate.py`, decides whether a call is checked (`user_to_check`) and runs the existing decision (`llm_quota_refusal`) on the accounting loop from wherever the call runs (`check_user_sync`, `check_user_async`). The four wrapped provider methods call it before the provider method runs. The shared call layer calls it at the top of `perform_chat_api_call` and `perform_chat_api_call_async`, before their `try`, and marks its nested adapter call admitted. Shared machinery (the Evaluations circuit breaker, G-Eval's retry, the extraction retry, the chat pipeline, job retries, the RAG answer fallback) lets the refusal through unchanged. Seven sync LLM calls made on the server loop move to `asyncio.to_thread`, so they can be checked.

**Tech Stack:** Python 3.12, FastAPI, asyncio (`run_coroutine_threadsafe`, `to_thread`), `contextvars`, pytest (asyncio mode auto).

**Spec:** `Docs/Design/2026-10-09-llm-usage-accounting-design.md`: §4 (the decision, the error), §4b, §5, §Testing 5 and 6, and Delivery "PR 2b". Read it, and the owner decisions at its top, before any task. PR 2a (#3230) built the decision, the error, the boundary check and the `prechecked` mark this plan uses.

## Global Constraints

- **Quotas are off unless an admin turns them on.** Nothing is refused unless `USAGE_QUOTAS_ENABLED` is on and the user has a limit. `0` blocks, an unset limit is unlimited, and a lookup or counter failure admits, with a warning.
- **The public refusal is exactly PR 2a's:** `LLMQuotaExceededError`, an `HTTPException` 402 with a `Retry-After` header and the body `{"detail": {"error": "limit_exceeded", "category": ..., "current": ..., "limit": ..., "message": ...}}`. It is never a `ChatAPIError`, and nothing in this PR catches it and raises something else in its place.
- **A call is not checked** when it has no user, when its context is `exempt` or `prechecked`, when it is nested inside an accounted provider call, when quotas are off, or when the user's cached limits set none of the three LLM keys.
- **The decision runs on the accounting loop only** (spec §5): awaited directly there, handed off with `asyncio.run_coroutine_threadsafe` from anywhere else, never through `asyncio.run` and never on another loop. A check handed to the accounting loop that gets no answer within `CHECK_TIMEOUT_SECONDS = 2.0`, or that fails, admits the call with a warning (a check awaited directly on the accounting loop has no timeout, as §5 says). A sync call on the accounting loop's own thread is not checked (waiting would deadlock); a stream started there is checked at its first item.
- **Before the provider runs.** `chat` and `achat` check before calling the provider method; `stream` checks when it is called; `astream` checks before an async generator's body runs and before an awaitable result is awaited. A refused call records no usage row. A call is decided once: a nested provider method (an `achat` that calls its own `chat`) is not checked again, also when the call records itself.
- **The shared call layer checks above its `try`.** Inside it, `privacy_safe_chat_error` rebuilds any exception as a provider error.
- **An admitted request is never refused part-way.** The `prechecked` mark from PR 2a is honoured everywhere; a job or workflow root is never admitted by the request that started it.
- **Test-first.** Each task's tests are written and seen failing before its code, except tests the plan calls guards.
- **Every task was prototyped** in a scratch checkout of `origin/dev` (79aa6f00a4): the code and tests in Tasks 1 to 7 are the code that ran there, each new test was seen failing without its change and passing with it, and Task 8's backlog commands ran there. If an edit does not match the file, or a test does not behave as a step says, stop and report it. Do not improvise.
- **Edits are targeted.** Use the Edit tool for every source change. Never rewrite a whole file, and never run `ruff --fix` on a whole file: several files have unrelated findings that a fix would also move. Where a step says where an import goes, put it there (ruff's isort rules decide).
- **New test files go only where CI runs the directory whole:** `tldw_Server_API/tests/Usage` and `tldw_Server_API/tests/Media`. Elsewhere, add tests to existing files.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit ids, `git add -A backlog/tasks`.
- **Commits** end with exactly `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `git stash`. Never `--no-verify`.
- **Python:** `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (written `$PY` below). Tests run with `TLDW_TEST_NO_DOCKER=1`; the Evaluations tests also need `RUN_EVALUATIONS=1`.
- **Known failures on `dev` in this environment:** `tests/Services/test_router_groups_contract.py` (10 tests), `tests/Audio/test_nemo_streaming_error_sentinel.py` (no `pyaudio`), `tests/RAG_NEW/unit/test_pgvector_adapter_sanitizers.py::test_initialize_psycopg_connect_fallback_log_omits_raw_exception_and_uses_psycopg2`, and one flashrank test in `tests/RAG_NEW/unit/test_rag_profiles.py` (`TestSlidesSourceLocalRerankers`). **Under `-n 4` across many directories,** order-dependent 404s and 401s on the `/media` routes and four `test_chat_closes_runtime_*` tests fail on `dev` too, and so do five tests in `tests/RAG/test_knowledge_qa_live_regressions.py` and `tests/RAG/test_restricted_postgres_media_retrieval.py::test_owner_retrieves_chunked_media_through_standard_pipeline[sqlite]` under `-n 4`. Re-run any failure alone, and at `origin/dev`, before reporting it.

## Review Focus

Inputs a person would hit that the spec implies. Each has a test in the task named.

1. **The default: quotas off, on a busy server.** Expected: every LLM call costs one switch read, with no hand-off and no query. (Task 1)
2. **A user over a cost limit whose background job or workflow makes LLM calls.** Expected: the call is refused before the provider is called; a workflow `llm` step fails the run, which records the 402; a WorkerSDK job fails once with `llm_quota_exceeded` and is not retried. (Tasks 3, 6)
3. **A streaming call from a direct adapter caller, on an adapter that sends its request inside `stream()`.** Expected: refused before the request is sent, and no usage row. (Task 2)
4. **An admitted request that crosses its limit part-way** (a long chat, a media batch, an evaluation run). Expected: it is never refused part-way. (Tasks 2, 3, 5)
7. **An over-quota user whose `/chat/completions` request is auto-routed.** Expected: the 402, and the router's provider is not counted as failing for anyone. (Task 6)
5. **A slow or stuck accounting loop.** Expected: the call waits at most two seconds, then goes ahead, with a warning. (Task 1)
6. **A sync LLM call made on the server loop's thread by code this PR missed.** Expected: the call goes ahead, recorded but not checked, and a warning is logged the first time. (Task 2)

## File Structure

| File | Responsibility |
|---|---|
| `tldw_Server_API/app/core/Usage/llm_quota_gate.py` (new) | Whether a call is checked, and running the decision on the accounting loop. |
| `tldw_Server_API/app/core/Usage/quota_resolver.py` | `cached_limits`: the user's cached limits without I/O. |
| `tldw_Server_API/app/core/Usage/llm_roots.py` | A root is never admitted. |
| `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` | The check in the four wrapped provider methods. |
| `tldw_Server_API/app/core/Chat/chat_service.py` | The check at the top of the shared call layer; the chat pipeline lets a refusal through. |
| `tldw_Server_API/app/api/v1/endpoints/chat.py` | The auto-router call does not record a provider failure for a refusal. |
| `tldw_Server_API/app/core/exceptions.py` | `LLMQuotaExceededError.retryable` and `.failure_code` for job workers. |
| The modules in Task 4 | Seven sync LLM calls move off the loop. |
| `core/Infrastructure/circuit_breaker.py`, `core/Evaluations/*`, `endpoints/evaluations/evaluations_crud.py` | The breaker ignores a refusal; evaluators and G-Eval let it through; an evaluation run is checked when it is created. |
| `core/Web_Scraping/extraction/strategies/llm.py`, `core/RAG/rag_service/generation.py`, `endpoints/media/document_insights.py`, `core/Writing/manuscript_analysis.py` | No retry, no silent fallback, no 500, no stored failure. |
| `core/RAG/rag_service/unified_pipeline.py`, `agentic_execution.py`, `endpoints/rag_unified.py` | Optional RAG stages degrade; three handlers pass the 402 through. |
| `Helper_Scripts/ci/scope_predicate_baseline.txt`, `Docs/Design/web_scraping_refactor_import_inventory.json`, `Docs/Design/WebScraping_Refactor_Import_Inventory.md` | Line-number baselines, regenerated after Task 7 moves lines in `unified_pipeline.py`. |
| `tldw_Server_API/tests/helpers/llm_quota.py` | `refuse_llm_calls`, for tests of the check at the call. |
| `Docs/Operations/Usage_Quotas.md`, the spec, backlog | What each caller sees; TASK-13437 closes. |

## Rulings made while planning

These depart from the spec's wording or settle what it left open. Each is recorded in the spec in Task 8.

- **Web search aggregation.** The spec names `core/WebSearch/Web_Search.py`, which only tests import. The live `/research/websearch` path awaits `core/Web_Scraping/WebSearch_APIs.py::analyze_and_aggregate`, whose sync `aggregate_results(...)` (a summarize per chunk, then the final report) runs on the loop. Task 4 moves that call. The legacy module is left alone: nothing in the app calls it.
- **The RAG custom metric is dropped from the list.** `rag_custom_metrics.py:233` calls `analyze(prompt, "...")` without `custom_prompt_arg`, so it raises `TypeError` before any provider call and falls back to a heuristic. It spends nothing. Fixing the call needs a choice of grading provider; Task 8 files a follow-up.
- **Two more sync calls on the loop move too:** the Explainer job's `adapter.chat` (`core/Explainer/jobs.py`), and the article extractor, which has no caller in the app but is public and costs one line.
- **The workflow web-search step and cohere's `astream` stay.** The first runs on the workflow engine's own loop, where a sync check hands off and waits safely. The second is nested inside the checked `astream` call.
- **Jobs are not retried on a refusal.** A quota window does not reset in seconds, so `LLMQuotaExceededError` carries `retryable = False` and `failure_code = "llm_quota_exceeded"`, which WorkerSDK and the job workers that read `retryable` already honour. A worker that wraps every exception in its own type keeps its own retry (the recurring-question RAG adapter wraps it as retryable); the operator page says "job workers that read it".
- **Audio jobs are not changed.** Their `audio_analyze` stage calls `analyze(text, api_name)` without the required `custom_prompt_arg`, so it raises `TypeError` before any provider call and can never be refused. Task 8 files TASK-13637 to fix the call and honour `retryable` when it does.
- **A workflow `llm` step is retried once** by the engine's default policy, and is refused again without a provider call. That costs about a second and is left as is. Other workflow steps that make LLM calls (content generation, NLP, bibliography, image, summarize, translate) catch broadly and return an error output, so the run goes on; the operator page says so, and TASK-13638 lists them with the other callers below.
- **The `/chat/completions` auto-router call runs before the boundary check** (`chat.py`, routing at about line 3575, the check at about line 4288), so it is not admitted and can be refused at the call. Its `except Exception` would record a failure of the router's provider in the shared `ProviderManager`, which marks it unhealthy for every user after three. Task 6 lets the refusal through that clause; the routing runtime then falls back to rules routing (it treats any router error that way), and the boundary check returns the 402. Moving the boundary check ahead of routing was considered and not done: it would reorder it against the rate limiter and the billing pre-check PR 2a placed it after.
- **The chat pipeline changes are defence in depth.** Only `/chat/completions` reaches the pipeline, and its generation call is admitted at the boundary. The pipeline still must not rewrite a refusal as a provider outage or record a provider failure for it.
- **Evaluation runs are checked when they are created.** `POST /evaluations/{eval_id}/runs` starts the run in a task from the request (`EvaluationRunner`, `asyncio.create_task`), so it inherits the admitted mark. Checked at the call instead, a refused sample would be scored 0.0 and the run would complete. Task 5 adds the boundary check; the §4a table gains a row.
- **Manuscript analysis** stored a refusal as an `analysis_failed` result with a 200. Task 6 lets it through; the writing endpoints' catch tuple holds no `HTTPException`, so they answer 402.
- **`/media/{id}/insights`** turns any exception into a 500. It gains `except HTTPException: raise`, so it returns the 402. Other §4b paths that catch broadly report their own error or go on with degraded output, as the spec allows: claims verification (`claims_service.py` catches `HTTPException` and falls back to heuristic verdicts), Prompt Studio test runs and benchmarks (error outputs, 0.0 scores), character memory extraction (an empty result). The operator page names them and TASK-13638 tracks them.
- **Standalone worker processes** (a worker module run with `python -m`) never set the accounting loop, so their sync LLM calls are neither checked nor recorded. The check logs that once per process; the operator page says to run workers in the server process (the default).
- **The three RAG handlers** (`/rag/simple`, `/rag/advanced`, `/rag/ablate`) gain `except HTTPException: raise`. Gap analysis and the agentic planner stop letting a refusal escape the pipeline. This settles the open note from PR 2a's final review.
- **WebSocket frames.** `/audio/chat/stream` sends `{"type":"error","code":"provider_unavailable",...}` and the socket stays open; the realtime sockets send `{"type":"error","error":{"type":"server_error","code":"internal_error",...}}` then `response.done` with status `failed`. Both are documented, not changed.

---

### Task 1: The check at the call

**Files:**
- Create: `tldw_Server_API/app/core/Usage/llm_quota_gate.py`
- Modify: `tldw_Server_API/app/core/Usage/quota_resolver.py` (add `cached_limits` before `invalidate_user`, near line 99)
- Modify: `tldw_Server_API/app/core/Usage/llm_roots.py` (`llm_call_root`, near line 55)
- Modify: `tldw_Server_API/tests/helpers/llm_quota.py` (add `refuse_llm_calls`)
- Test: `tldw_Server_API/tests/Usage/test_llm_quota_gate.py` (new)

**Interfaces:**
- Consumes: `quota_checks.llm_quota_refusal(user_id, requested_tokens=1) -> LLMQuotaRefusal | None`, the three key constants and `as_quota_user_id` (PR 2a); `llm_accounting.get_accounting_loop()` (PR 1a); `llm_call_context.current_llm_call_context`, `current_llm_user_id`, `in_accounted_call` (PR 1a); `config.usage_quotas_enabled()`; `exceptions.LLMQuotaExceededError(*, category, current, limit, message, retry_after)` (PR 2a).
- Produces, in `llm_quota_gate`:
  - `CHECK_TIMEOUT_SECONDS: float = 2.0`
  - `user_to_check() -> int | None`: the user to check this call for, or None when it is not checked.
  - `check_user_sync(user_id: int) -> bool`: raises `LLMQuotaExceededError` on a refusal; returns False when the check cannot run here (on the accounting loop's thread, or with no accounting loop), True otherwise. Each of those two cases logs a warning the first time it happens in the process, and at debug after that.
  - `async check_user_async(user_id: int) -> None`: raises `LLMQuotaExceededError` on a refusal.
  - `check_llm_call_sync() -> bool` and `async check_llm_call_async() -> None`: the two above for the current context.
- Produces `quota_resolver.cached_limits(user_id: int) -> dict[str, int | float] | None`.
- Produces `tests/helpers/llm_quota.refuse_llm_calls(monkeypatch) -> None`, which later tasks use.

- [ ] **Step 1: Write the failing tests**

Create `tldw_Server_API/tests/Usage/test_llm_quota_gate.py`:

```python
"""The check at the call: who is checked, and where the decision runs (LLM usage accounting spec §4b, §5)."""

import asyncio
import threading
import time
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage import llm_accounting, llm_quota_gate, quota_resolver
from tldw_Server_API.app.core.Usage.llm_call_context import (
    accounted_call,
    llm_call_context,
    mark_llm_quota_prechecked,
)
from tldw_Server_API.app.core.Usage.llm_roots import llm_call_root
from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal

pytestmark = pytest.mark.unit

REFUSAL = LLMQuotaRefusal("llm_tokens_month", 100, 100, "Monthly LLM token limit reached", 60)


@pytest.fixture
def decisions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, asyncio.AbstractEventLoop]]:
    """Quotas on, no cached limits, and a decision that refuses user 7 and records where it ran."""
    seen: list[tuple[int, asyncio.AbstractEventLoop]] = []

    async def _decide(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Refuse user 7; admit everyone else."""
        seen.append((user_id, asyncio.get_running_loop()))
        return REFUSAL if user_id == 7 else None

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _decide)
    return seen


@pytest.fixture
async def accounting_loop(monkeypatch: pytest.MonkeyPatch) -> asyncio.AbstractEventLoop:
    """Make the test's loop the accounting loop."""
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(llm_accounting, "_accounting_loop", loop)
    return loop


async def test_a_refused_user_gets_the_402_from_a_coroutine_on_the_accounting_loop(decisions, accounting_loop) -> None:
    """Awaited directly on the accounting loop: the refusal is the public 402."""
    with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError) as refused:
        await llm_quota_gate.check_llm_call_async()
    assert refused.value.status_code == 402 and refused.value.headers == {"Retry-After": "60"}
    assert refused.value.detail["category"] == "llm_tokens_month"
    assert decisions == [(7, accounting_loop)]


async def test_a_sync_call_on_a_plain_thread_is_decided_on_the_accounting_loop(decisions, accounting_loop) -> None:
    """A thread with no loop hands the decision to the accounting loop and waits for it."""
    outcome: list[object] = []

    def _call() -> None:
        """Run the sync check as a provider call on a pool thread would."""
        with llm_call_context(user_id=7):
            try:
                llm_quota_gate.check_llm_call_sync()
            except LLMQuotaExceededError as error:
                outcome.append(error.status_code)

    await asyncio.to_thread(_call)
    assert outcome == [402] and decisions == [(7, accounting_loop)]


async def test_a_coroutine_on_another_loop_is_decided_on_the_accounting_loop(decisions, accounting_loop) -> None:
    """A workflow-engine-style thread with its own loop never runs the decision on that loop."""
    outcome: list[object] = []

    async def _other_loop_main() -> None:
        """Check from a second event loop."""
        with llm_call_context(user_id=8):
            await llm_quota_gate.check_llm_call_async()
        outcome.append("admitted")

    worker = threading.Thread(target=lambda: asyncio.run(_other_loop_main()))
    worker.start()
    await asyncio.to_thread(worker.join)
    assert outcome == ["admitted"] and decisions == [(8, accounting_loop)]


async def test_a_sync_call_on_the_accounting_loop_thread_is_not_checked(decisions, accounting_loop) -> None:
    """Waiting there would deadlock: the call is admitted unchecked and the caller is told."""
    with llm_call_context(user_id=7):
        assert llm_quota_gate.check_llm_call_sync() is False
    assert decisions == []


async def test_a_check_that_outlasts_the_timeout_admits_the_call(monkeypatch: pytest.MonkeyPatch, accounting_loop) -> None:
    """A slow decision lets a sync call through after the timeout, and says it settled."""

    async def _slow(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Take longer than the timeout."""
        await asyncio.sleep(1)
        return REFUSAL

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _slow)
    monkeypatch.setattr(llm_quota_gate, "CHECK_TIMEOUT_SECONDS", 0.05)

    def _call() -> bool:
        """Check from a pool thread."""
        with llm_call_context(user_id=7):
            return llm_quota_gate.check_llm_call_sync()

    started = time.monotonic()
    assert await asyncio.to_thread(_call) is True
    assert time.monotonic() - started < 0.9


async def test_a_failing_decision_admits_the_call(monkeypatch: pytest.MonkeyPatch, accounting_loop) -> None:
    """Anything but a refusal lets the call through."""

    async def _broken(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Fail like a broken pool."""
        raise RuntimeError("pool gone")

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _broken)
    with llm_call_context(user_id=7):
        await llm_quota_gate.check_llm_call_async()
        assert await asyncio.to_thread(llm_quota_gate.check_llm_call_sync) is True


async def test_with_no_accounting_loop_a_coroutine_uses_its_own_loop_and_sync_code_skips(
    decisions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the application lifespan: async checks on the running loop; sync calls go unchecked."""
    monkeypatch.setattr(llm_accounting, "_accounting_loop", None)
    with llm_call_context(user_id=7):
        with pytest.raises(LLMQuotaExceededError):
            await llm_quota_gate.check_llm_call_async()
        assert await asyncio.to_thread(llm_quota_gate.check_llm_call_sync) is False
    assert decisions == [(7, asyncio.get_running_loop())]


async def test_no_user_exempt_admitted_and_nested_calls_are_not_checked(decisions, accounting_loop) -> None:
    """Nothing is decided without a user, for an exempt or admitted context, or inside an accounted call."""
    await llm_quota_gate.check_llm_call_async()
    with llm_call_context(user_id=7, exempt=True):
        await llm_quota_gate.check_llm_call_async()
    with llm_call_context(user_id=7, prechecked=True):
        await llm_quota_gate.check_llm_call_async()
    with llm_call_context(user_id=7), accounted_call():
        await llm_quota_gate.check_llm_call_async()
    assert decisions == []


async def test_with_quotas_off_nothing_is_handed_off_or_queried(monkeypatch: pytest.MonkeyPatch, accounting_loop) -> None:
    """The common case costs one switch read in the calling thread."""

    def _no_hand_off(*args: Any, **kwargs: Any) -> None:
        """Fail the test if the check hands anything to the loop."""
        raise AssertionError("handed off")

    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)
    monkeypatch.setattr(llm_quota_gate, "get_accounting_loop", _no_hand_off)
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _no_hand_off)
    with llm_call_context(user_id=7):
        assert await asyncio.to_thread(llm_quota_gate.check_llm_call_sync) is True
        await llm_quota_gate.check_llm_call_async()


async def test_a_cached_user_with_no_llm_limit_is_not_handed_off(decisions, monkeypatch: pytest.MonkeyPatch, accounting_loop) -> None:
    """Quotas on, limits cached, none of them an LLM key: no hand-off; with an LLM key there is one."""

    async def _limits(user_id: int) -> dict:
        """A user with only a RAG limit, or user 9 with an LLM cost limit."""
        return {"limits.llm_cost_per_day_usd": 1.0} if user_id == 9 else {"limits.rag_queries_per_day": 5}

    monkeypatch.setattr(quota_resolver, "_load_limits", _limits)
    await quota_resolver.user_quota(7, "limits.rag_queries_per_day")
    await quota_resolver.user_quota(9, "limits.rag_queries_per_day")
    with llm_call_context(user_id=7):
        assert await asyncio.to_thread(llm_quota_gate.check_llm_call_sync) is True
    with llm_call_context(user_id=9):
        await asyncio.to_thread(llm_quota_gate.check_llm_call_sync)
    assert decisions == [(9, accounting_loop)]


async def test_a_root_is_never_admitted_by_the_request_that_started_it(decisions, accounting_loop) -> None:
    """A job or workflow started from an admitted request is still checked when it runs."""
    mark_llm_quota_prechecked()
    with llm_call_root(7), pytest.raises(LLMQuotaExceededError):
        await llm_quota_gate.check_llm_call_async()
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_quota_gate.py`
Expected: collection error, `ImportError: cannot import name 'llm_quota_gate' from 'tldw_Server_API.app.core.Usage'`.

- [ ] **Step 3: Add `cached_limits` to the resolver**

In `tldw_Server_API/app/core/Usage/quota_resolver.py`, insert before `def invalidate_user(user_id: int) -> None:`:

```python
def cached_limits(user_id: int) -> dict[str, int | float] | None:
    """The user's cached limits, read without I/O; None when not cached or expired (LLM usage accounting spec §4b)."""
    hit = _cache.get(user_id)
    if hit is None or hit[0] <= time.monotonic():
        return None
    return hit[1]


```

- [ ] **Step 4: Create the gate**

Create `tldw_Server_API/app/core/Usage/llm_quota_gate.py`:

```python
"""Check the LLM quota at the call, for work no request boundary admitted (LLM usage accounting spec §4b, §5).

``user_to_check`` decides whether a call is checked at all. ``check_user_sync``
and ``check_user_async`` run the decision on the accounting loop, wherever the
call runs, and raise ``LLMQuotaExceededError`` when it refuses. Anything else
that goes wrong admits the call: a check must never break an LLM call.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from loguru import logger

from tldw_Server_API.app.core.config import usage_quotas_enabled
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage.llm_accounting import get_accounting_loop
from tldw_Server_API.app.core.Usage.llm_call_context import (
    current_llm_call_context,
    current_llm_user_id,
    in_accounted_call,
)
from tldw_Server_API.app.core.Usage.quota_checks import (
    LLM_COST_PER_DAY_KEY,
    LLM_COST_PER_MONTH_KEY,
    LLM_TOKENS_PER_MONTH_KEY,
    LLMQuotaRefusal,
    as_quota_user_id,
    llm_quota_refusal,
)
from tldw_Server_API.app.core.Usage.quota_resolver import cached_limits

# How long a check handed to the accounting loop waits for its answer before admitting the call.
CHECK_TIMEOUT_SECONDS = 2.0
_LLM_LIMIT_KEYS = (LLM_TOKENS_PER_MONTH_KEY, LLM_COST_PER_DAY_KEY, LLM_COST_PER_MONTH_KEY)
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Log a warning the first time ``key`` happens in this process, and at debug after that."""
    if key in _WARNED:
        logger.debug(message)
        return
    _WARNED.add(key)
    logger.warning(message)


def user_to_check() -> int | None:
    """The user this call is checked for, or None when it is not checked.

    Not checked: no user, an exempt or admitted (``prechecked``) context, a call
    nested in an accounted call, quotas off, or a user whose cached limits set
    no LLM key.
    """
    try:
        if in_accounted_call():
            return None
        context = current_llm_call_context()
        if context.exempt or context.prechecked:
            return None
        user_id = as_quota_user_id(current_llm_user_id())
        if user_id is None or not usage_quotas_enabled():
            return None
        limits = cached_limits(user_id)
        if limits is not None and all(limits.get(key) is None for key in _LLM_LIMIT_KEYS):
            return None
        return user_id
    except Exception as error:  # noqa: BLE001 - a check must never break an LLM call
        logger.bind(exception_type=type(error).__name__).warning("LLM quota check could not start; call admitted")
        return None


def _raise_if_refused(refusal: LLMQuotaRefusal | None) -> None:
    """Raise the 402 for a refusal."""
    if refusal is not None:
        raise LLMQuotaExceededError(**asdict(refusal))


def _running_loop() -> asyncio.AbstractEventLoop | None:
    """The loop running on this thread, or None."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def check_user_sync(user_id: int) -> bool:
    """Check ``user_id`` from synchronous code; raise ``LLMQuotaExceededError`` when refused.

    Returns False when the check cannot run here, and the call goes ahead
    unchecked: on the accounting loop's own thread, where waiting would
    deadlock, or with no accounting loop. A check that times out or fails
    admits the call and returns True.
    """
    loop = get_accounting_loop()
    if loop is None:
        _warn_once(
            "no_loop",
            "No accounting loop is set, so synchronous LLM calls are not quota-checked "
            "(a standalone worker process must call set_accounting_loop)",
        )
        return False
    if _running_loop() is loop:
        _warn_once("on_loop", "A synchronous LLM call on the event loop's thread is not quota-checked")
        return False
    coro = llm_quota_refusal(user_id)
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as error:  # noqa: BLE001 - the loop closed between the two lines above
        coro.close()
        logger.bind(exception_type=type(error).__name__).warning("LLM quota check failed; call admitted")
        return True
    try:
        refusal = future.result(timeout=CHECK_TIMEOUT_SECONDS)
    except TimeoutError:
        future.cancel()
        logger.warning("LLM quota check timed out after {}s; call admitted", CHECK_TIMEOUT_SECONDS)
        return True
    except Exception as error:  # noqa: BLE001 - a check must never break an LLM call
        logger.bind(exception_type=type(error).__name__).warning("LLM quota check failed; call admitted")
        return True
    _raise_if_refused(refusal)
    return True


async def check_user_async(user_id: int) -> None:
    """Check ``user_id`` from a coroutine; raise ``LLMQuotaExceededError`` when refused.

    On the accounting loop, or with none captured, the decision is awaited
    here. From another loop it is handed to the accounting loop. A check that
    times out or fails admits the call.
    """
    try:
        running = asyncio.get_running_loop()
        target = get_accounting_loop() or running
        if target is running:
            refusal = await llm_quota_refusal(user_id)
        else:
            coro = llm_quota_refusal(user_id)
            try:
                future = asyncio.run_coroutine_threadsafe(coro, target)
            except Exception:
                coro.close()
                raise
            try:
                refusal = await asyncio.wait_for(asyncio.wrap_future(future), CHECK_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning("LLM quota check timed out after {}s; call admitted", CHECK_TIMEOUT_SECONDS)
                return
    except Exception as error:  # noqa: BLE001 - a check must never break an LLM call
        logger.bind(exception_type=type(error).__name__).warning("LLM quota check failed; call admitted")
        return
    _raise_if_refused(refusal)


def check_llm_call_sync() -> bool:
    """Check the current call from synchronous code. False when it could not be checked here."""
    user_id = user_to_check()
    return True if user_id is None else check_user_sync(user_id)


async def check_llm_call_async() -> None:
    """Check the current call from a coroutine."""
    user_id = user_to_check()
    if user_id is not None:
        await check_user_async(user_id)
```

- [ ] **Step 5: A root is never admitted**

In `tldw_Server_API/app/core/Usage/llm_roots.py`, in `llm_call_root`, replace:

```python
    An owner that is not a number sets no user, and never clears one that is
    already set.
    """
    overrides: dict[str, Any] = {}
```

with:

```python
    An owner that is not a number sets no user, and never clears one that is
    already set. A root is never admitted by the request that started it: its
    calls are quota-checked when they run (LLM usage accounting spec §4a, §4b).
    """
    overrides: dict[str, Any] = {"prechecked": False}
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_quota_gate.py tldw_Server_API/tests/Usage/test_llm_roots.py tldw_Server_API/tests/Usage/test_quota_resolver.py`
Expected: all pass (11 new).

- [ ] **Step 7: Add the test helper later tasks use**

In `tldw_Server_API/tests/helpers/llm_quota.py`, replace the first line and the import block:

```python
"""Test helpers for the LLM quota boundary checks (LLM usage accounting spec §4a)."""

from typing import Any
```

with:

```python
"""Test helpers for the LLM quota checks (LLM usage accounting spec §4a, §4b)."""

import asyncio
from typing import Any
```

and insert before `def assert_llm_quota_402(response: Any) -> None:`:

```python
def refuse_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """``refuse_llm_work``, and switch on the check at the call (spec §4b) for a test without the application lifespan.

    The call-site check reads the quotas switch itself, so it is turned on for that
    check only, and cached limits are dropped. Called from a coroutine, it also makes
    the running loop the accounting loop, which the lifespan would otherwise set.
    """
    from tldw_Server_API.app.core.Usage import llm_accounting, quota_resolver

    refuse_llm_work(monkeypatch)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.llm_quota_gate.usage_quotas_enabled", lambda: True)
    quota_resolver.invalidate_all()
    try:
        monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    except RuntimeError:
        return


```

Why a separate helper: `refuse_llm_work` patches `quota_checks.user_quota`, which the boundary check uses, and leaves `USAGE_QUOTAS_ENABLED` alone. The check at the call reads the switch itself, and a test app without the application lifespan has no accounting loop. Turning the switch on globally would wake other quota sites in unrelated tests.

- [ ] **Step 8: Lint and commit**

Run: `$PY -m ruff check --output-format concise tldw_Server_API/app/core/Usage/llm_quota_gate.py tldw_Server_API/app/core/Usage/quota_resolver.py tldw_Server_API/app/core/Usage/llm_roots.py tldw_Server_API/tests/Usage/test_llm_quota_gate.py tldw_Server_API/tests/helpers/llm_quota.py`
Expected: `All checks passed!`

```bash
git add tldw_Server_API/app/core/Usage/llm_quota_gate.py tldw_Server_API/app/core/Usage/quota_resolver.py tldw_Server_API/app/core/Usage/llm_roots.py tldw_Server_API/tests/Usage/test_llm_quota_gate.py tldw_Server_API/tests/helpers/llm_quota.py
git commit -m "feat(usage): the LLM quota check at the call, decided on the accounting loop (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The wrapped provider methods check before the provider runs

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` (module docstring; imports; `_wrap_chat`, `_wrap_achat`, `_wrap_stream`, `_wrap_astream`; three new helpers before `_accounted_async_stream`)
- Test: `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py` (append)

**Interfaces:**
- Consumes: `llm_quota_gate.user_to_check`, `check_user_sync`, `check_user_async` (Task 1).
- Produces: every `ChatProvider` subclass's own `chat`, `achat`, `stream` and `astream` raise `LLMQuotaExceededError` before the provider method runs, for a checked user who is refused. Shapes are unchanged: `stream` still returns an iterator, `astream` an async iterator or, when the provider's `astream` returns an awaitable, an awaitable.

- [ ] **Step 1: Write the failing tests**

Append to `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py` (it already imports `asyncio`, `inspect`, `AsyncIterator`, `Iterable`, `Any`, `pytest`, `llm_call_context`, and defines `REQUEST`, `USAGE`, `LINES`, `_FakeProvider`, `_AwaitableAstreamProvider`, the `rows` fixture and `_settled`):

```python
# --- The check at the call (LLM usage accounting spec §4b) ---


class _EagerStreamProvider(_FakeProvider):
    """Sends its request inside stream(), before any item is read, as moonshot, zai and cohere do."""

    name = "fake-eager-stream"
    sent: list[str] = []

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Record that the provider was called."""
        self.sent.append("chat")
        return super().chat(request, timeout=timeout)

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:  # type: ignore[override]
        """Record the request when stream() is called, then return the lines."""
        self.sent.append("stream")
        return iter(LINES)

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Record that the provider was called."""
        self.sent.append("achat")
        return {"choices": [{"message": {"content": "hello"}}], "usage": dict(USAGE)}

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        """Record when the provider code runs, then yield the lines."""
        self.sent.append("astream")
        for line in LINES:
            yield line


@pytest.fixture
def refused(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Quotas on and user 7 over a limit; returns the users the decision ran for."""
    from tldw_Server_API.app.core.Usage import llm_quota_gate, quota_resolver
    from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal

    seen: list[int] = []

    async def _decide(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Refuse user 7."""
        seen.append(user_id)
        if user_id == 7:
            return LLMQuotaRefusal("llm_tokens_month", 100, 100, "Monthly LLM token limit reached", 60)
        return None

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _decide)
    _EagerStreamProvider.sent = []
    return seen


async def test_a_refused_call_never_reaches_the_provider_and_records_nothing(rows, refused) -> None:
    """chat on a thread, achat, stream on a thread and astream all raise the 402 before the provider runs."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    provider = _EagerStreamProvider()
    with llm_call_context(user_id=7):
        with pytest.raises(LLMQuotaExceededError):
            await asyncio.to_thread(provider.chat, REQUEST)
        with pytest.raises(LLMQuotaExceededError):
            await provider.achat(REQUEST)
        with pytest.raises(LLMQuotaExceededError):
            await asyncio.to_thread(provider.stream, REQUEST)
        astream = provider.astream(REQUEST)
    with pytest.raises(LLMQuotaExceededError):
        await astream.__anext__()
    assert _EagerStreamProvider.sent == []
    assert await _settled(rows) == []
    assert refused == [7, 7, 7, 7]


async def test_an_admitted_user_is_checked_once_per_call_and_recorded(rows, refused) -> None:
    """A user under every limit goes through; nested provider methods are not checked again."""
    with llm_call_context(user_id=8):
        await asyncio.to_thread(_FakeProvider().chat, REQUEST)
        await _FakeProvider().achat(REQUEST)
        assert [line async for line in _FakeProvider().astream(REQUEST)] == LINES
    assert refused == [8, 8, 8]
    assert len(await _settled(rows)) == 3


async def test_a_call_that_records_itself_is_still_checked(rows, refused) -> None:
    """The claims engine records its own rows, and its calls are still refused."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    with llm_call_context(user_id=7, records_itself=True), pytest.raises(LLMQuotaExceededError):
        await asyncio.to_thread(_EagerStreamProvider().chat, REQUEST)
    assert _EagerStreamProvider.sent == []


async def test_an_admitted_request_is_not_refused_part_way(rows, refused) -> None:
    """After a boundary check admits the request, its calls are not checked, even for a refused user."""
    with llm_call_context(user_id=7, prechecked=True):
        await asyncio.to_thread(_FakeProvider().chat, REQUEST)
    assert refused == []
    assert len(await _settled(rows)) == 1


async def test_a_sync_call_on_the_loop_thread_is_recorded_but_not_checked(rows, refused) -> None:
    """Waiting on the accounting loop from its own thread would deadlock: the call goes through and is recorded."""
    with llm_call_context(user_id=7):
        assert _FakeProvider().chat(REQUEST)["usage"] == USAGE
    assert refused == []
    assert len(await _settled(rows)) == 1


async def test_a_stream_started_on_the_loop_thread_is_checked_at_its_first_item(rows, refused) -> None:
    """stream() called on the loop cannot check; the thread that reads the first item does, and records nothing."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    with llm_call_context(user_id=7):
        stream = _FakeProvider().stream(REQUEST)
    with pytest.raises(LLMQuotaExceededError):
        await asyncio.to_thread(next, stream)
    assert refused == [7]
    assert await _settled(rows) == []


async def test_an_awaitable_astream_is_checked_before_it_is_awaited(rows, refused) -> None:
    """An astream that returns an awaitable stays awaitable, and the refusal comes from awaiting it."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    with llm_call_context(user_id=7):
        pending = _AwaitableAstreamProvider().astream(REQUEST)
    assert inspect.isawaitable(pending)
    with pytest.raises(LLMQuotaExceededError):
        await pending
    assert await _settled(rows) == []


async def test_a_self_recorded_call_is_decided_once_even_when_achat_calls_chat(rows, refused) -> None:
    """The claims engine's calls record themselves; the nested sync chat is not checked a second time."""
    with llm_call_context(user_id=8, records_itself=True):
        await _FakeProvider().achat(REQUEST)
    assert refused == [8]
    assert await _settled(rows) == []


async def test_with_quotas_off_a_provider_call_hands_nothing_off(rows, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default costs a switch read: no hand-off to the accounting loop and no decision."""
    from tldw_Server_API.app.core.Usage import llm_quota_gate

    def _no_hand_off(*args: Any, **kwargs: Any) -> None:
        """Fail the test if the check hands anything to the loop."""
        raise AssertionError("handed off")

    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)
    monkeypatch.setattr(llm_quota_gate, "get_accounting_loop", _no_hand_off)
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _no_hand_off)
    with llm_call_context(user_id=7):
        assert (await asyncio.to_thread(_FakeProvider().chat, REQUEST))["usage"] == USAGE
        await _FakeProvider().achat(REQUEST)
    assert len(await _settled(rows)) == 2
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`
Expected: 6 failed, 27 passed. The failures: `test_a_refused_call_never_reaches_the_provider_and_records_nothing`, `test_an_admitted_user_is_checked_once_per_call_and_recorded`, `test_a_call_that_records_itself_is_still_checked`, `test_a_stream_started_on_the_loop_thread_is_checked_at_its_first_item`, `test_an_awaitable_astream_is_checked_before_it_is_awaited` and `test_a_self_recorded_call_is_decided_once_even_when_achat_calls_chat` (no refusal is raised, or no decision runs). `test_an_admitted_request_is_not_refused_part_way`, `test_a_sync_call_on_the_loop_thread_is_recorded_but_not_checked` and `test_with_quotas_off_a_provider_call_hands_nothing_off` are guards and pass.

- [ ] **Step 3: Update the module docstring and imports**

In `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`, replace:

```python
"""Record usage for every chat provider call made for a known user (LLM usage accounting spec §2, §3).

``instrument_provider_class`` wraps a provider class's own ``chat``, ``achat``,
``stream`` and ``astream``. Nothing here may raise into the provider call.
"""
```

with:

```python
"""Check and record every chat provider call made for a known user (LLM usage accounting spec §2, §3, §4b).

``instrument_provider_class`` wraps a provider class's own ``chat``, ``achat``,
``stream`` and ``astream``. The quota check runs before the provider method and
may raise ``LLMQuotaExceededError``; nothing else here may raise into the call.
"""
```

and add, directly after the `from tldw_Server_API.app.core.Usage.llm_call_context import (...)` block:

```python
from tldw_Server_API.app.core.Usage.llm_quota_gate import check_user_async, check_user_sync, user_to_check
```

- [ ] **Step 4: Add the three helpers**

Insert directly before `async def _accounted_async_stream(`:

```python
def _close_quietly(*streams: Any) -> None:
    """Close sync iterators, ignoring their errors."""
    for stream in streams:
        close = getattr(stream, "close", None)
        if callable(close):
            with suppress(Exception):
                close()


async def _aclose_quietly(*streams: Any) -> None:
    """Close async iterators, ignoring their errors."""
    for stream in streams:
        aclose = getattr(stream, "aclose", None)
        if callable(aclose):
            with suppress(Exception):
                await aclose()


def _checked_at_first_item(user_id: int, outer: Iterator[Any], inner: Any) -> Iterator[Any]:
    """Check on the thread that reads the first item, for a stream started where it could not be checked (§4b)."""
    try:
        check_user_sync(user_id)
    except BaseException:
        _close_quietly(outer, inner)
        raise
    yield from outer


async def _async_checked_at_first_item(user_id: int, outer: Any, inner: Any) -> AsyncIterator[Any]:
    """Check before the first item, which is before an async generator's provider code runs (§4b)."""
    try:
        await check_user_async(user_id)
    except BaseException:
        await _aclose_quietly(outer, inner)
        raise
    try:
        async for item in outer:
            yield item
    finally:
        await _aclose_quietly(outer)


```

`outer` is either the provider's own iterator or the recording generator around it. When a refusal is raised before the recording generator has started, closing it runs none of its body, so no usage row is written.

- [ ] **Step 5: Check in `chat` and `achat`**

In `_wrap_chat`, replace:

```python
    @functools.wraps(fn)
    def chat(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        plan = _plan_call()
        if plan is None:
            return fn(self, request, *args, **kwargs)
```

with:

```python
    @functools.wraps(fn)
    def chat(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        user_id = user_to_check()
        if user_id is not None:
            check_user_sync(user_id)
        plan = _plan_call()
        if plan is None:
            if user_id is None:
                return fn(self, request, *args, **kwargs)
            with accounted_call():  # checked here, so a nested provider method is not checked again
                return fn(self, request, *args, **kwargs)
```

In `_wrap_achat`, replace:

```python
    @functools.wraps(fn)
    async def achat(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        plan = _plan_call()
        if plan is None:
            return await fn(self, request, *args, **kwargs)
```

with:

```python
    @functools.wraps(fn)
    async def achat(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        user_id = user_to_check()
        if user_id is not None:
            await check_user_async(user_id)
        plan = _plan_call()
        if plan is None:
            if user_id is None:
                return await fn(self, request, *args, **kwargs)
            with accounted_call():  # checked here, so a nested provider method is not checked again
                return await fn(self, request, *args, **kwargs)
```

A call that records itself (the claims engine) has no recording plan, so without the `accounted_call()` block its `achat` would be checked, and then the `chat` it calls checked again: two hand-offs and two counter reads per call.

- [ ] **Step 6: Check in `stream`**

Replace the whole `_wrap_stream` function body from its docstring to `return _accounted_sync_stream(self, plan, request, inner, started)`:

```python
def _wrap_stream(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return a generator that records when the stream ends. The plan is fixed when stream() is called."""

    @functools.wraps(fn)
    def stream(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        plan = _plan_call()
        if plan is None:
            return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            inner = fn(self, request, *args, **kwargs)
        if not hasattr(inner, "__iter__"):
            return inner  # a broken provider's result reaches the caller's own check unchanged
        return _accounted_sync_stream(self, plan, request, inner, started)
```

with:

```python
def _wrap_stream(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Check before the provider method, then return a generator that records when the stream ends.

    The check, and the plan, are fixed when stream() is called. Where the check
    cannot run then (on the event loop's thread), it runs at the first item.
    """

    @functools.wraps(fn)
    def stream(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        user_id = user_to_check()
        unchecked = user_id is not None and not check_user_sync(user_id)
        plan = _plan_call()
        if plan is None and not unchecked:
            return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            inner = fn(self, request, *args, **kwargs)
        if not hasattr(inner, "__iter__"):
            return inner  # a broken provider's result reaches the caller's own check unchanged
        outer = inner if plan is None else _accounted_sync_stream(self, plan, request, inner, started)
        return _checked_at_first_item(user_id, outer, inner) if unchecked else outer
```

(The `return stream` line after it is unchanged.)

- [ ] **Step 7: Check in `astream`**

Replace the `_wrap_astream` function from its docstring to `return _accounted_async_stream(self, plan, request, inner, started)` (the `if inspect.iscoroutinefunction(fn):` lines after it are unchanged):

```python
def _wrap_astream(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return an async generator that records when the stream ends; an awaitable result stays awaitable."""

    @functools.wraps(fn)
    def astream(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        plan = _plan_call()
        if plan is None:
            return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            inner = fn(self, request, *args, **kwargs)
        if inspect.isawaitable(inner):

            async def _resolve() -> Any:
                """Await the provider's result, then wrap the iterator it resolves to."""
                with accounted_call():
                    resolved = await inner
                if not hasattr(resolved, "__aiter__"):
                    return resolved
                return _accounted_async_stream(self, plan, request, resolved, started)

            return _resolve()
        if not hasattr(inner, "__aiter__"):
            return inner  # a broken provider's result reaches the caller's own check unchanged
        return _accounted_async_stream(self, plan, request, inner, started)
```

with:

```python
def _wrap_astream(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Check before the provider code runs, and return an async generator that records when the stream ends.

    An awaitable result stays awaitable, and is checked before it is awaited.
    """

    @functools.wraps(fn)
    def astream(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        user_id = user_to_check()
        plan = _plan_call()
        if plan is None and user_id is None:
            return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            inner = fn(self, request, *args, **kwargs)
        if inspect.isawaitable(inner):

            async def _resolve() -> Any:
                """Check, await the provider's result, then wrap the iterator it resolves to."""
                if user_id is not None:
                    try:
                        await check_user_async(user_id)
                    except BaseException:
                        _close_quietly(inner)
                        raise
                with accounted_call():
                    resolved = await inner
                if plan is None or not hasattr(resolved, "__aiter__"):
                    return resolved
                return _accounted_async_stream(self, plan, request, resolved, started)

            return _resolve()
        if not hasattr(inner, "__aiter__"):
            return inner  # a broken provider's result reaches the caller's own check unchanged
        outer = inner if plan is None else _accounted_async_stream(self, plan, request, inner, started)
        return outer if user_id is None else _async_checked_at_first_item(user_id, outer, inner)
```

Calling an async generator function runs none of its body, so checking at the first item is checking before the provider code runs. A coroutine returned by `astream` is closed when it is refused, so it never runs and raises no "never awaited" warning.

- [ ] **Step 8: Run the tests to see them pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Usage -W error::RuntimeWarning`
Expected: all pass; 33 in `test_usage_accounting.py`. No `RuntimeWarning`.

- [ ] **Step 9: Run the neighbours**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider -n 4 tldw_Server_API/tests/LLM_Calls tldw_Server_API/tests/Chat/unit tldw_Server_API/tests/Chat/integration/test_chat_completions_marks_self_recorded.py tldw_Server_API/tests/Workflows/test_engine_llm_user.py tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py`
Expected: all pass apart from the load flakes named in Global Constraints (re-run one alone before reporting it). With quotas off, which is these tests' default, nothing changes.

- [ ] **Step 10: Lint and commit**

Run: `$PY -m ruff check --output-format concise tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`
Expected: `All checks passed!`

```bash
git add tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py
git commit -m "feat(usage): provider methods refuse an over-quota user before the provider runs (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The shared call layer checks above its `try`

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py` (imports near line 216; `perform_chat_api_call` near line 2624; `perform_chat_api_call_async` near line 2652)
- Test: `tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py` (imports; append)
- Test: `tldw_Server_API/tests/Workflows/test_engine_llm_user.py` (append)

**Interfaces:**
- Consumes: `llm_quota_gate.check_llm_call_sync() -> bool`, `check_llm_call_async()` (Task 1); `llm_call_context.llm_call_context(**overrides)` (PR 1a); `tests/helpers/llm_quota.refuse_llm_calls` (Task 1).
- Produces: `perform_chat_api_call` and `perform_chat_api_call_async` raise `LLMQuotaExceededError` itself, before their `try`, for a refused user; their adapter call runs with `prechecked=True` when the check ran, so it is decided once.

- [ ] **Step 1: Write the failing tests**

In `tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py`, replace the imports:

```python
import asyncio
from typing import Any
```

with:

```python
import asyncio
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any
```

and, after `from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context`, add:

```python
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
```

Then append to the file:

```python
# --- The shared call layer checks first (LLM usage accounting spec §4b) ---


class _ContextRecordingProvider(ChatProvider):
    """Records whether each call ran as already admitted."""

    name = "openai"
    seen: list[bool] = []

    def capabilities(self) -> dict[str, Any]:
        """Streaming is supported."""
        return {"supports_streaming": True}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Record the admitted mark and answer."""
        self.seen.append(current_llm_call_context().prechecked)
        return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Yield one line. Its body runs lazily, after the call layer has returned, so it records nothing."""
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'


_CALL_ARGS = {
    "api_provider": "openai",
    "messages": [{"role": "user", "content": "hi"}],
    "model": "gpt-4o-mini",
    "api_key": "test-key",
    "app_config": {},
}


@pytest.fixture
async def call_layer(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Route the shared call layer to the recording provider; refuse user 7; return who was decided."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Usage import llm_accounting, llm_quota_gate, quota_resolver
    from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal

    decided: list[int] = []

    async def _decide(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Refuse user 7."""
        decided.append(user_id)
        if user_id == 7:
            return LLMQuotaRefusal("llm_tokens_month", 100, 100, "Monthly LLM token limit reached", 60)
        return None

    adapter = _ContextRecordingProvider()
    _ContextRecordingProvider.seen = []
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: SimpleNamespace(get_adapter=lambda _name: adapter))
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _decide)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _no_write)
    return decided


async def _no_write(**_kwargs: Any) -> None:
    """Write nothing."""


async def test_the_shared_call_layer_refuses_before_its_try_even_with_privacy_safe_errors(
    call_layer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal reaches the caller as itself, not rebuilt as a provider error, and no provider runs."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    monkeypatch.setattr(chat_service, "_privacy_safe_provider_errors_enabled", lambda _request: True)
    with llm_call_context(user_id=7):
        with pytest.raises(LLMQuotaExceededError):
            await chat_service.perform_chat_api_call_async(**_CALL_ARGS)
        with pytest.raises(LLMQuotaExceededError):
            await asyncio.to_thread(lambda: chat_service.perform_chat_api_call(**_CALL_ARGS))
    assert _ContextRecordingProvider.seen == []
    assert call_layer == [7, 7]


async def test_an_admitted_call_runs_its_adapter_as_admitted_and_is_decided_once(call_layer) -> None:
    """The check at the top settles it: the adapter calls are not checked again, streams included."""
    from tldw_Server_API.app.core.Chat import chat_service

    with llm_call_context(user_id=8):
        await chat_service.perform_chat_api_call_async(**_CALL_ARGS)
        stream = await chat_service.perform_chat_api_call_async(**_CALL_ARGS, streaming=True)
        assert [line async for line in stream]
        await asyncio.to_thread(lambda: chat_service.perform_chat_api_call(**_CALL_ARGS))
    assert _ContextRecordingProvider.seen == [True, True]
    assert call_layer == [8, 8, 8]
    assert current_llm_call_context().prechecked is False


async def test_a_sync_call_on_the_loop_thread_leaves_its_adapter_call_unmarked(call_layer) -> None:
    """Where the top check cannot run, the adapter call is not marked admitted, so it can still be checked."""
    from tldw_Server_API.app.core.Chat import chat_service

    with llm_call_context(user_id=8):
        chat_service.perform_chat_api_call(**_CALL_ARGS)
    assert _ContextRecordingProvider.seen == [False]
    assert call_layer == []
```

Append to `tldw_Server_API/tests/Workflows/test_engine_llm_user.py` (it already imports `asyncio`, `threading`, `WorkflowsDatabase`, `engine_mod` and defines `_create_run`):

```python
async def test_a_refused_llm_call_fails_the_step_and_the_run_records_it(tmp_path, monkeypatch) -> None:
    """A run on its own thread and loop is checked at the call (spec §4b): no provider call, a failed run."""
    from types import SimpleNamespace
    from typing import Any

    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls

    refuse_llm_calls(monkeypatch)
    provider_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _name: SimpleNamespace(chat=provider_calls.append)),
    )
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = _create_run(db, "7")
    done = threading.Event()

    async def _adapter(_config, _context):
        """Make an LLM call through the shared call layer, as the workflow llm step does."""
        try:
            return await chat_service.perform_chat_api_call_async(
                api_provider="openai", messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini",
                api_key="test-key", app_config={},
            )
        finally:
            done.set()

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    engine = engine_mod.WorkflowEngine(db)
    engine_mod.WorkflowScheduler._spawn(engine.start_run(run_id))  # noqa: SLF001 - the thread hand-off under test
    assert await asyncio.to_thread(done.wait, 10)
    for _ in range(100):
        run = db.get_run(run_id)
        if run is not None and run.status == "failed":
            break
        await asyncio.sleep(0.05)
    assert run is not None and run.status == "failed"
    assert (run.error or "").startswith("402") and "limit_exceeded" in run.error
    assert provider_calls == []
```

The workflow test starts the run the way the scheduler does, on its own thread and loop, so the check hands the decision to the test's loop, which `refuse_llm_calls` made the accounting loop.

- [ ] **Step 2: Run the tests to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py tldw_Server_API/tests/Workflows/test_engine_llm_user.py`
Expected: 3 failed. `test_the_shared_call_layer_refuses_before_its_try_even_with_privacy_safe_errors` fails with `ChatAPIError: Provider request failed.`: the provider method's refusal is raised inside the `try` and rebuilt as a privacy-safe provider error. `test_an_admitted_call_runs_its_adapter_as_admitted_and_is_decided_once` fails on `seen`. `test_a_refused_llm_call_fails_the_step_and_the_run_records_it` fails with the run `succeeded` (its fake adapter is not a `ChatProvider`, so only the call layer can refuse it). `test_a_sync_call_on_the_loop_thread_leaves_its_adapter_call_unmarked` is a guard and passes.

- [ ] **Step 3: Import the check**

In `tldw_Server_API/app/core/Chat/chat_service.py`, replace:

```python
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage
```

with:

```python
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
from tldw_Server_API.app.core.Usage.llm_quota_gate import check_llm_call_async, check_llm_call_sync
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage
```

(`contextlib` is already imported at the top of the file.)

- [ ] **Step 4: Check at the top of `perform_chat_api_call`**

Replace:

```python
def perform_chat_api_call(**kwargs: Any) -> Any:
    """Adapter-backed replacement for chat_orchestrator.chat_api_call."""
    provider, request, internal = _build_adapter_request_from_chat_args(kwargs)
```

with:

```python
def perform_chat_api_call(**kwargs: Any) -> Any:
    """Adapter-backed replacement for chat_orchestrator.chat_api_call."""
    # LLM usage accounting spec §4b: refuse before the try below, which can rebuild any error as a provider error.
    checked = check_llm_call_sync()
    provider, request, internal = _build_adapter_request_from_chat_args(kwargs)
```

and, in the same function, replace:

```python
    try:
        with provider_error_privacy_scope(privacy_safe_errors):
            _validate_audited_call_policy_transport(registry, provider, request)
            if request.get("stream"):
```

with:

```python
    # A checked call's adapter method is not checked again; one that could not be checked here is.
    admitted = llm_call_context(prechecked=True) if checked else contextlib.nullcontext()
    try:
        with provider_error_privacy_scope(privacy_safe_errors), admitted:
            _validate_audited_call_policy_transport(registry, provider, request)
            if request.get("stream"):
```

- [ ] **Step 5: Check at the top of `perform_chat_api_call_async`**

Replace:

```python
async def perform_chat_api_call_async(**kwargs: Any) -> Any:
    """Async adapter-backed replacement for chat_orchestrator.chat_api_call_async."""
    provider, request, internal = _build_adapter_request_from_chat_args(kwargs)
```

with:

```python
async def perform_chat_api_call_async(**kwargs: Any) -> Any:
    """Async adapter-backed replacement for chat_orchestrator.chat_api_call_async."""
    # LLM usage accounting spec §4b: refuse before the try below, which can rebuild any error as a provider error.
    await check_llm_call_async()
    provider, request, internal = _build_adapter_request_from_chat_args(kwargs)
```

and, in the same function, replace:

```python
    try:
        with provider_error_privacy_scope(privacy_safe_errors):
            strict_policy = _validate_audited_call_policy_transport(
```

with:

```python
    try:
        with provider_error_privacy_scope(privacy_safe_errors), llm_call_context(prechecked=True):
            strict_policy = _validate_audited_call_policy_transport(
```

An async check always settles (it awaits the decision, or hands it off), so the adapter call is always marked. The mark reaches the pool thread because `BoundedDaemonPool` captures the call state when the work is started, inside this block. A lazily iterated stream is checked when `stream()` or `astream()` is called, also inside it.

- [ ] **Step 6: Run the tests to see them pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py tldw_Server_API/tests/Workflows/test_engine_llm_user.py`
Expected: all pass (6 and 6).

- [ ] **Step 7: Run the neighbours**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider -n 4 tldw_Server_API/tests/Chat tldw_Server_API/tests/Workflows tldw_Server_API/tests/LLM_Calls tldw_Server_API/tests/Usage`
Expected: all pass apart from the known failures and load flakes in Global Constraints.

- [ ] **Step 8: Lint and commit**

Run: `$PY -m ruff check --output-format concise tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py tldw_Server_API/tests/Workflows/test_engine_llm_user.py`
Expected: `All checks passed!`. For `chat_service.py`, which has findings on `dev`, compare per-rule tallies with the merge base: `$PY -m ruff check --output-format concise --statistics tldw_Server_API/app/core/Chat/chat_service.py` before and after must match.

```bash
git add tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py tldw_Server_API/tests/Workflows/test_engine_llm_user.py
git commit -m "feat(usage): the shared call layer refuses an over-quota user before its try (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Seven sync LLM calls move off the server loop

A sync LLM call made on the server loop's own thread blocks the server while the provider answers, and cannot be quota-checked there (spec §5). Each call below moves to `asyncio.to_thread`, with its arguments unchanged. `to_thread` copies the context, so the user and the `prechecked` mark travel with it, and it resolves the module-level name (`analyze`, `aggregate_results`) when it is called, so existing tests that patch those names still work.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/translate.py` (imports; the `analyze` call in `translate_text`, near line 100)
- Modify: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py` (`analyze_and_aggregate`, near line 718)
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py` (`scrape_and_summarize_multiple`, near line 416)
- Modify: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py` (`EnhancedWebScraper._summarize_content`, near line 1983)
- Modify: `tldw_Server_API/app/services/web_scraping_service.py` (`maybe_summarize_one`, near line 351)
- Modify: `tldw_Server_API/app/services/enhanced_web_scraping_service.py` (`WebScrapingService._summarize_content`, near line 736)
- Modify: `tldw_Server_API/app/core/Explainer/jobs.py` (imports; `_generator` in `make_configured_explainer_generator`, near line 142)
- Test (append to each): `tests/Translation/test_translate_service_prompt.py`, `tests/WebSearch/test_websearch_core.py`, `tests/WebScraping/test_jsonld_extraction.py`, `tests/Web_Scraping/test_enhanced_web_scraping_guards.py`, `tests/Web_Scraping/test_legacy_fallback_behavior.py`, `tests/Web_Scraping/test_enhanced_service_status_and_queue_retention.py`, `tests/Explainer/test_explainer_jobs.py` (all under `tldw_Server_API/`)

**Interfaces:**
- Consumes: `tests/helpers/llm_quota.refuse_llm_calls` and `assert_llm_quota_402` (Task 1); the provider-method check (Task 2).
- Produces: nothing new. `/translate` now returns the 402 from the call (it has no boundary check).

The new tests import locally inside the test function, as several tests in these files already do, so no import block changes.

- [ ] **Step 1: Write the failing tests**

Append to `tldw_Server_API/tests/Translation/test_translate_service_prompt.py`:

```python
@pytest.mark.asyncio
async def test_translation_analyze_runs_off_the_event_loop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sync analyze call runs on a worker thread, where it can be quota-checked (spec §4b)."""
    import threading

    seen: list[int] = []

    def capture_analyze(**kwargs: object) -> str:
        """Record the thread and answer."""
        seen.append(threading.get_ident())
        return "Bonjour"

    monkeypatch.setattr(translate_module, "analyze", capture_analyze)

    await translate_module.translate_text(_request(), current_user=SimpleNamespace(id=1), db=_PromptDatabase())

    assert seen and seen[0] != threading.get_ident()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_translate_returns_402_and_calls_no_provider_when_llm_quota_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/translate has no boundary check: the refusal comes from the provider call, and the handler passes it on."""
    from collections.abc import Iterable
    from typing import Any

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
    from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
    from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
    from tldw_Server_API.app.core.Usage.llm_call_context import bind_llm_call_user
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_calls

    refuse_llm_calls(monkeypatch)
    provider_calls: list[dict[str, Any]] = []

    class _CountingProvider(ChatProvider):
        """A provider that records every request it receives."""

        name = "openai"

        def capabilities(self) -> dict[str, Any]:
            """Streaming is supported."""
            return {"supports_streaming": True}

        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            """Record the call and answer."""
            provider_calls.append(request)
            return {"choices": [{"message": {"content": "Bonjour"}}]}

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Unused."""
            return iter(())

    registry = ChatProviderRegistry(include_defaults=False)
    registry.register_adapter("openai", _CountingProvider)
    monkeypatch.setattr(summarization_module, "get_registry", lambda: registry)

    async def _user() -> SimpleNamespace:
        """Authenticate user 1 and bind them as the LLM call user, as authentication does through the scope."""
        bind_llm_call_user(1)
        return SimpleNamespace(id=1)

    app = FastAPI()
    app.include_router(translate_module.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = _user
    app.dependency_overrides[get_prompts_db_for_user] = lambda: _PromptDatabase()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/translate",
            json={"text": "Hello", "target_language": "French", "provider": "openai", "model": "gpt-test"},
        )

    assert_llm_quota_402(response)
    assert provider_calls == []
```

The second test drives the real `analyze` and a real `ChatProvider` subclass (instrumentation is applied in `ChatProvider.__init_subclass__`, so a plain fake class would never refuse). The dependency override is `async` and binds the user, because a sync dependency runs in a copy of the context and the binding would be lost.

Append to `tldw_Server_API/tests/WebSearch/test_websearch_core.py`:

```python
@pytest.mark.asyncio
async def test_analyze_and_aggregate_runs_aggregation_off_the_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summaries and the final report run on a worker thread, where they can be quota-checked (spec §4b)."""
    import threading

    seen: dict[str, int] = {}

    async def fake_relevance(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """One relevant result."""
        return {"0": {"content": "Summary content", "reasoning": "Contains the answer"}}

    def fake_summarize(**_: Any) -> str:
        """Record the thread."""
        seen["summarize"] = threading.get_ident()
        return "Chunk summary"

    def fake_chat_api_call(**_: Any) -> str:
        """Record the thread."""
        seen["chat"] = threading.get_ident()
        return "Aggregated answer text."

    monkeypatch.setattr(web_search, "search_result_relevance", fake_relevance)
    monkeypatch.setattr(web_search, "summarize", fake_summarize)
    monkeypatch.setattr(web_search, "chat_api_call", fake_chat_api_call)

    result = await web_search.analyze_and_aggregate(
        {"results": []},
        {"main_goal": "q", "sub_questions": []},
        {"final_answer_llm": "fake-llm"},
    )

    assert result["final_answer"]["text"] == "Aggregated answer text."
    loop_thread = threading.get_ident()
    assert seen["summarize"] != loop_thread
    assert seen["chat"] != loop_thread
```

Append to `tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py`:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_legacy_scrape_summary_runs_off_the_event_loop_thread(monkeypatch):
    """The article summary runs on a worker thread, where it can be quota-checked (spec §4b)."""
    import threading

    seen = []

    async def fake_scrape_article(_url, custom_cookies=None, *, allow_llm_extraction=True):
        """One extracted article."""
        return {"url": "https://example.com", "title": "T", "content": "body", "extraction_successful": True}

    def fake_analyze(**_kwargs):
        """Record the thread."""
        seen.append(threading.get_ident())
        return "summary"

    monkeypatch.setattr(extractor_lib, "scrape_article", fake_scrape_article)
    monkeypatch.setattr(extractor_lib, "RateLimiter", lambda: None)
    monkeypatch.setattr(extractor_lib, "analyze", fake_analyze)

    result = await extractor_lib.scrape_and_summarize_multiple(
        urls="https://example.com",
        custom_prompt_arg=None,
        api_name="openai",
        api_key=None,
        keywords="",
        custom_article_titles=None,
        summarize_checkbox=True,
    )

    assert result[0]["summary"] == "summary"
    assert seen and seen[0] != threading.get_ident()
```

Append to `tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py`:

```python
@pytest.mark.asyncio
async def test_summarize_content_runs_analyze_off_the_event_loop_thread(monkeypatch):
    """The summary runs on a worker thread, where it can be quota-checked (spec §4b)."""
    import threading

    seen = []

    def fake_analyze(**_kwargs):
        """Record the thread."""
        seen.append(threading.get_ident())
        return "summary"

    monkeypatch.setattr(ews, "analyze", fake_analyze)
    scraper = ews.EnhancedWebScraper(config={})

    assert await scraper._summarize_content("body", api_name="openai") == "summary"
    assert seen and seen[0] != threading.get_ident()
```

Append to `tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py`:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_web_content_analysis_runs_off_the_event_loop_thread(monkeypatch):
    """The analysis runs on a worker thread, where it can be quota-checked (spec §4b)."""
    import threading

    seen = []

    async def fake_scrape_article(url, custom_cookies=None, *, allow_llm_extraction=None):
        """One extracted article."""
        return {"url": url, "title": "Example", "content": "article body", "extraction_successful": True}

    def fake_analyze(**kwargs):
        """Record the thread."""
        seen.append(threading.get_ident())
        return "summary"

    monkeypatch.setattr(ws_service, "scrape_article", fake_scrape_article, raising=True)
    monkeypatch.setattr(ws_service, "analyze", fake_analyze, raising=True)

    result = await ws_service.ingest_web_content_orchestrate(
        request=IngestWebContentRequest(urls=["https://example.com/a"], perform_analysis=True, api_name="openai"),
        db=object(),
        usage_log=_UsageLog(),
    )

    assert result[0]["analysis"] == "summary"
    assert seen and seen[0] != threading.get_ident()
```

Append to `tldw_Server_API/tests/Web_Scraping/test_enhanced_service_status_and_queue_retention.py`:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_summarize_content_runs_analyze_off_the_event_loop_thread(monkeypatch):
    """The summary runs on a worker thread, where it can be quota-checked (spec §4b)."""
    import threading

    from tldw_Server_API.app.services import enhanced_web_scraping_service as enhanced_service

    seen = []

    def fake_analyze(**kwargs):
        """Record the thread."""
        seen.append(threading.get_ident())
        return "summary"

    monkeypatch.setattr(enhanced_service, "analyze", fake_analyze)
    svc = WebScrapingService()

    assert await svc._summarize_content("body", None, "openai", None, None, 0.7) == "summary"
    assert seen and seen[0] != threading.get_ident()
```

Append to `tldw_Server_API/tests/Explainer/test_explainer_jobs.py`:

```python
@pytest.mark.asyncio()
async def test_configured_generator_calls_the_provider_off_the_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync provider call runs on a worker thread, where it can be quota-checked (spec §4b)."""
    import threading

    seen: list[int] = []

    class FakeAdapter:
        def chat(self, request, *, timeout=None):
            """Record the thread and answer with one child."""
            seen.append(threading.get_ident())
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"children":[{"title":"Child","body":"Body",'
                                '"kind":"explanation","intent":"explain","outside_knowledge_used":true}]}'
                            )
                        }
                    }
                ]
            }

    monkeypatch.setenv("EXPLAINER_GENERATOR_ENABLED", "1")
    monkeypatch.setenv("EXPLAINER_GENERATOR_PROVIDER", "fake-provider")
    monkeypatch.setenv("EXPLAINER_GENERATOR_MODEL", "fake-model")
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils

    monkeypatch.setattr(adapter_utils, "get_adapter_or_raise", lambda _provider: FakeAdapter())
    monkeypatch.setattr(adapter_utils, "resolve_provider_api_key_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter_utils, "ensure_app_config", lambda *_args, **_kwargs: {})

    generator = make_configured_explainer_generator()
    result = await generator(type("Prompt", (), {"as_messages": lambda self: [{"role": "user", "content": "Expand"}]})())

    assert result["children"][0]["title"] == "Child"
    assert seen and seen[0] != threading.get_ident()
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Translation/test_translate_service_prompt.py tldw_Server_API/tests/WebSearch/test_websearch_core.py tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py tldw_Server_API/tests/Web_Scraping/test_enhanced_service_status_and_queue_retention.py tldw_Server_API/tests/Explainer/test_explainer_jobs.py -k "off_the_event_loop or 402"`
Expected: 8 failed. Seven fail at the thread assertion (`seen[0] != threading.get_ident()`). `test_translate_returns_402_and_calls_no_provider_when_llm_quota_refuses` fails with status 200: on the loop's thread the check cannot run, so the call goes through.

- [ ] **Step 3: `/translate`**

In `tldw_Server_API/app/api/v1/endpoints/translate.py`, replace:

```python
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
```

with:

```python
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
```

and replace:

```python
        # Use the existing analyze function for LLM call
        result = analyze(
            api_name=provider,
```

with:

```python
        # Use the existing analyze function for LLM call, off the event loop: a sync LLM call
        # on the loop blocks the server and cannot be quota-checked (LLM usage accounting spec §4b).
        result = await asyncio.to_thread(
            analyze,
            api_name=provider,
```

(The other arguments and the closing parenthesis are unchanged.)

- [ ] **Step 4: Web search aggregation**

In `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py` (`asyncio` is already imported):

Replace:

```python
    final_answer = aggregate_results(
        relevant_results,
        sub_query_dict["main_goal"],
        sub_questions,
        search_params.get('final_answer_llm')
    )
```

with:

```python
    # Off the event loop: the summaries and the final report are sync LLM calls (LLM usage accounting spec §4b).
    final_answer = await asyncio.to_thread(
        aggregate_results,
        relevant_results,
        sub_query_dict["main_goal"],
        sub_questions,
        search_params.get('final_answer_llm'),
    )
```

This is the live `/research/websearch` path. `core/WebSearch/Web_Search.py` has the same code, but nothing in the app imports it; leave it alone.

- [ ] **Step 5: The article extractor**

In `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py` (`asyncio` is already imported):

Replace:

```python
                        summary = analyze(
                            input_data=content,
                            custom_prompt_arg=article_custom_prompt,
                            api_name=api_name,
                            api_key=api_key,
                            temp=temperature,
                            system_message=system_message_final
                        )
```

with:

```python
                        summary = await asyncio.to_thread(
                            analyze,
                            input_data=content,
                            custom_prompt_arg=article_custom_prompt,
                            api_name=api_name,
                            api_key=api_key,
                            temp=temperature,
                            system_message=system_message_final,
                        )
```

- [ ] **Step 6: Enhanced web scraping**

In `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py` (`asyncio` is already imported):

Replace:

```python
            summary = analyze(
                input_data=content,
                custom_prompt_arg=overrides.get('user', kwargs.get('custom_prompt', '')),
                api_name=kwargs.get('api_name', 'openai'),
                api_key=kwargs.get('api_key'),
                temp=kwargs.get('temperature', 0.7),
                system_message=overrides.get('system', kwargs.get('system_message', system_default))
            )
```

with:

```python
            summary = await asyncio.to_thread(
                analyze,
                input_data=content,
                custom_prompt_arg=overrides.get('user', kwargs.get('custom_prompt', '')),
                api_name=kwargs.get('api_name', 'openai'),
                api_key=kwargs.get('api_key'),
                temp=kwargs.get('temperature', 0.7),
                system_message=overrides.get('system', kwargs.get('system_message', system_default)),
            )
```

- [ ] **Step 7: Web content ingestion**

In `tldw_Server_API/app/services/web_scraping_service.py` (`asyncio` is already imported):

Replace:

```python
        analysis_results = analyze(
            input_data=content,
```

with:

```python
        analysis_results = await asyncio.to_thread(
            analyze,
            input_data=content,
```

(The remaining arguments and the closing parenthesis are unchanged.)

- [ ] **Step 8: The web scraping service**

In `tldw_Server_API/app/services/enhanced_web_scraping_service.py` (`asyncio` is already imported):

Replace:

```python
            summary = analyze(
                input_data=content,
                custom_prompt_arg=custom_prompt,
```

with:

```python
            summary = await asyncio.to_thread(
                analyze,
                input_data=content,
                custom_prompt_arg=custom_prompt,
```

(The remaining arguments and the closing parenthesis are unchanged.)

- [ ] **Step 9: The Explainer job**

In `tldw_Server_API/app/core/Explainer/jobs.py`:

Replace:

```python
from __future__ import annotations

import hashlib
```

with:

```python
from __future__ import annotations

import asyncio
import hashlib
```

and:

Replace:

```python
        response = adapter.chat(request, timeout=timeout)
        if inspect.isawaitable(response):
```

with:

```python
        # Off the event loop: a sync provider call blocks the server and cannot be quota-checked there (spec §4b).
        response = await asyncio.to_thread(adapter.chat, request, timeout=timeout)
        if inspect.isawaitable(response):
```

The existing `if inspect.isawaitable(response)` stays: an adapter whose `chat` returns a coroutine still gets it awaited on the loop.

- [ ] **Step 10: Run the tests to see them pass**

Run the Step 2 command without `-k`.
Expected: all pass.

- [ ] **Step 11: Run the neighbours**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider -n 4 tldw_Server_API/tests/Translation tldw_Server_API/tests/WebSearch tldw_Server_API/tests/WebScraping tldw_Server_API/tests/Web_Scraping tldw_Server_API/tests/Explainer tldw_Server_API/tests/Media/test_web_summary_service_prompt.py`
Expected: all pass apart from the `tests/Web_Scraping/test_phase4_safe_regex.py` load flake (re-run it alone). `tests/Media/test_web_summary_service_prompt.py` drives the real web summaries end to end through the HTTP endpoints, so it shows the hand-off keeps working.

- [ ] **Step 12: Lint and commit**

For each changed file, compare ruff's finding count with `origin/dev` (several of these files have findings on `dev`); the counts must match. `$PY -m ruff check --output-format concise <file> | grep -c ':[0-9]*:[0-9]*:'` against `git show origin/dev:<file> | $PY -m ruff check --output-format concise --stdin-filename <file> - | grep -c ':[0-9]*:[0-9]*:'`. On `dev` the counts are: `WebSearch_APIs.py` 3, `enhanced_web_scraping.py` 1, `enhanced_web_scraping_service.py` 1, `Explainer/jobs.py` 3, `test_websearch_core.py` 52, `test_explainer_jobs.py` 1, every other file 0.

```bash
git add tldw_Server_API/app/api/v1/endpoints/translate.py tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py tldw_Server_API/app/services/web_scraping_service.py tldw_Server_API/app/services/enhanced_web_scraping_service.py tldw_Server_API/app/core/Explainer/jobs.py tldw_Server_API/tests/Translation/test_translate_service_prompt.py tldw_Server_API/tests/WebSearch/test_websearch_core.py tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py tldw_Server_API/tests/Web_Scraping/test_enhanced_service_status_and_queue_retention.py tldw_Server_API/tests/Explainer/test_explainer_jobs.py
git commit -m "fix(usage): seven sync LLM calls leave the event loop, so they can be quota-checked (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Evaluations never count, retry or score a refusal

The Evaluations circuit breaker is shared by all users per provider and opens after 3 to 5 failures; one user's refusals must not stop evaluations for everyone. G-Eval retries a failed call up to 10 times with waits of up to 120 seconds, and its catch-all rewrites any other exception as `ValueError("Unsupported API endpoint")`. The response-quality evaluator scores any failed metric 0.0, so a refused evaluation would be stored as a real score. The RAG evaluator's first-metric probe would keep a refusal as a per-metric error and go on. And `POST /evaluations/{eval_id}/runs` has no boundary check: its run is a task started from the request, so checked at the call a refused sample would be scored 0.0 and the run would complete. It gets the boundary check the other evaluation endpoints have.

**Files:**
- Modify: `tldw_Server_API/app/core/Infrastructure/circuit_breaker.py` (`CircuitBreakerConfig`, near line 128; `CircuitBreaker.call` and `call_async`, the two `except BaseException as exc:` clauses near lines 822 and 867)
- Modify: `tldw_Server_API/app/core/Evaluations/circuit_breaker.py` (imports; `CircuitBreakerConfig._to_unified`; `CircuitBreaker.call`)
- Modify: `tldw_Server_API/app/core/Evaluations/ms_g_eval.py` (imports; the `Retrying` predicate near line 483; the catch-all near line 537)
- Modify: `tldw_Server_API/app/core/Evaluations/response_quality_evaluator.py` (imports; six re-raise clauses)
- Modify: `tldw_Server_API/app/core/Evaluations/rag_evaluator.py` (imports; the probe's re-raise clause near line 363)
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py` (imports; `create_run`, near line 324)
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py` (append)
- Test: `tldw_Server_API/tests/Evaluations/unit/test_circuit_breaker.py` (append)
- Test: `tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py` (append)

**Interfaces:**
- Produces: `Infrastructure.circuit_breaker.CircuitBreakerConfig.ignored_exceptions: type | tuple = ()`. Exceptions matching it are re-raised and recorded as neither a failure nor a success. The default changes nothing for any other breaker.

- [ ] **Step 1: Write the failing tests**

Append to `tldw_Server_API/tests/Evaluations/unit/test_circuit_breaker.py` (it imports `pytest`, `CircuitBreaker`, `CircuitBreakerConfig` and `CircuitState`):

```python
@pytest.mark.asyncio
async def test_llm_quota_refusal_never_opens_the_breaker():
    """A quota refusal propagates unchanged and counts as neither a failure nor a success (spec §4b)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    config = CircuitBreakerConfig(failure_threshold=2)
    cb = CircuitBreaker("quota", config)
    refusal = LLMQuotaExceededError(
        category="llm_tokens_month", current=100, limit=100, message="Monthly LLM token limit reached", retry_after=60
    )

    async def refused():
        """Raise the refusal."""
        raise refusal

    for _ in range(config.failure_threshold + 3):
        with pytest.raises(LLMQuotaExceededError) as exc_info:
            await cb.call(refused)
        assert exc_info.value is refusal

    assert cb.state == CircuitState.CLOSED
    assert (cb.stats.failed_calls, cb.stats.consecutive_failures, cb.stats.rejected_calls) == (0, 0, 0)

    async def broken():
        """A real provider failure."""
        raise RuntimeError("provider down")

    for _ in range(config.failure_threshold):
        with pytest.raises(RuntimeError):
            await cb.call(broken)
    assert cb.state == CircuitState.OPEN
```

Append to `tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py` (it imports `Any`, `pytest`, `ms_g_eval`, `rag_evaluator`, `response_quality_evaluator` and defines `_run_analyze_inline`):

```python
def _quota_refusal():
    """The refusal a provider call raises for a user over the LLM quota."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    return LLMQuotaExceededError(
        category="llm_tokens_month", current=10, limit=10, message="Monthly LLM token limit reached", retry_after=60
    )


def test_geval_quota_refusal_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """One adapter request, and the refusal reaches the caller as itself (spec §4b)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    calls = 0
    refusal = _quota_refusal()

    def refuse(**_kwargs: Any) -> str:
        """Refuse every call."""
        nonlocal calls
        calls += 1
        raise refusal

    monkeypatch.setattr(ms_g_eval, "_call_adapter_text", refuse)
    monkeypatch.setattr(ms_g_eval, "wait_random_exponential", lambda **_kwargs: (lambda _retry_state: 0.0))

    with pytest.raises(LLMQuotaExceededError) as exc_info:
        ms_g_eval.geval_summarization(
            "judge this summary",
            5,
            "openai",
            "some-key",
            model="model-a",
            app_config={"openai_api": {"model": "model-a"}},
            credentials_resolved=True,
        )

    assert calls == 1
    assert exc_info.value is refusal


@pytest.mark.asyncio
async def test_response_quality_quota_refusal_stops_the_evaluation_without_a_zero_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused metric raises the refusal; it is never scored 0.0 as a provider failure (spec §4b)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    calls = 0

    def refuse(api_name: str, *_args: Any, **_kwargs: Any) -> str:
        """Refuse every call."""
        nonlocal calls
        calls += 1
        raise _quota_refusal()

    async def call_inline(_provider: str, function, *args: Any, **kwargs: Any):
        """Run the metric without the breaker's thread hop."""
        return await _run_analyze_inline(function, *args, **kwargs)

    monkeypatch.setattr(response_quality_evaluator, "analyze", refuse)
    monkeypatch.setattr(response_quality_evaluator.llm_circuit_breaker, "call_with_breaker", call_inline)

    with pytest.raises(LLMQuotaExceededError):
        await response_quality_evaluator.ResponseQualityEvaluator().evaluate(
            prompt="prompt",
            response="response",
            api_name="openai",
            api_key="key",
            model="model-a",
            app_config={"openai_api": {"model": "model-a"}},
            credentials_resolved=True,
        )

    assert calls == 1


@pytest.mark.asyncio
async def test_rag_quota_refusal_stops_the_evaluation_at_the_first_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first refused metric raises the refusal, and no other metric runs (spec §4b)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    calls = 0

    def refuse(api_name: str, *_args: Any, **_kwargs: Any) -> str:
        """Refuse every call."""
        nonlocal calls
        calls += 1
        raise _quota_refusal()

    async def call_inline(_provider: str, function, *args: Any, **kwargs: Any):
        """Run the metric without the breaker's thread hop."""
        return await _run_analyze_inline(function, *args, **kwargs)

    monkeypatch.setattr(rag_evaluator, "analyze", refuse)
    monkeypatch.setattr(rag_evaluator.llm_circuit_breaker, "call_with_breaker", call_inline)

    evaluator = rag_evaluator.RAGEvaluator(
        embedding_provider=None,
        embedding_model=None,
        api_key="key",
        app_config={"openai_api": {"model": "model-a"}},
        credentials_resolved=True,
    )
    with pytest.raises(LLMQuotaExceededError):
        await evaluator.evaluate(
            query="query",
            contexts=["context"],
            response="response",
            metrics=["relevance", "faithfulness"],
            api_name="openai",
            model="model-a",
        )

    assert calls == 1
```

Append to `tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py` (it imports `pytest` and defines the `evals_crud_client` fixture):

```python
@pytest.mark.integration
def test_create_run_refuses_an_over_quota_user_before_the_run_starts(evals_crud_client, monkeypatch):
    """A run's samples are LLM calls made in a task the request starts, so the run is checked at creation (§4a)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    client, headers = evals_crud_client
    started: list[str] = []

    class _SvcStub:
        async def create_run(self, eval_id, target_model, **_kwargs):
            started.append(eval_id)
            return {"id": "run_1", "object": "run", "eval_id": eval_id, "status": "pending",
                    "target_model": target_model, "created": 1700000000}

    import tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_crud as crud

    monkeypatch.setattr(crud, "get_unified_evaluation_service_for_user", lambda uid: _SvcStub())
    refuse_llm_work(monkeypatch)

    r = client.post("/api/v1/evaluations/eval_abc/runs", json={"target_model": "gpt-4o-mini", "config": {}}, headers=headers)

    assert_llm_quota_402(r)
    assert started == []
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 RUN_EVALUATIONS=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Evaluations/unit/test_circuit_breaker.py tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py -k quota`
Expected: 5 failed. The breaker opens; G-Eval raises `ValueError` after 10 attempts; the response-quality evaluator returns 0.0 scores instead of raising; the RAG evaluator calls the provider twice; the run is created (202).

- [ ] **Step 3: The breaker's ignore list**

In `tldw_Server_API/app/core/Infrastructure/circuit_breaker.py`:

Replace:

```python
    expected_exception: type | tuple = Exception
    """Exception type(s) that count as circuit-breaker failures.
    Exceptions *not* matching this are always re-raised without recording."""
```

with:

```python
    expected_exception: type | tuple = Exception
    """Exception type(s) that count as circuit-breaker failures.
    Exceptions *not* matching this are always re-raised without recording."""

    ignored_exceptions: type | tuple = ()
    """Exception type(s) re-raised without being recorded as a failure or a success:
    they say nothing about the dependency's health (a per-user quota refusal, say)."""
```

Then make the same replacement in **both** `CircuitBreaker.call` and `CircuitBreaker.call_async` (the text is identical in the two, so replace both occurrences):

Replace:

```python
        except BaseException as exc:
            if isinstance(exc, self.config.expected_exception):
                with self._lock:
                    self._on_failure(exc)
            raise
```

with:

```python
        except BaseException as exc:
            if isinstance(exc, self.config.expected_exception) and not isinstance(
                exc, self.config.ignored_exceptions
            ):
                with self._lock:
                    self._on_failure(exc)
            raise
```

`isinstance(exc, ())` is False, so a breaker with no ignore list behaves as before. The `finally` that releases a half-open probe slot is unchanged, so the breaker's state stays where it was.

- [ ] **Step 4: The Evaluations breaker ignores a refusal**

In `tldw_Server_API/app/core/Evaluations/circuit_breaker.py`, add the import (it sorts before the `Infrastructure` imports):

Replace:

```python
from loguru import logger

```

with:

```python
from loguru import logger

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
```

and in `CircuitBreakerConfig._to_unified`:

Replace:

```python
            expected_exception=self.expected_exception,
            half_open_max_calls=1,
            category="evaluations",
```

with:

```python
            expected_exception=self.expected_exception,
            # A quota refusal is the user's limit, not the provider's health (LLM usage accounting spec §4b).
            ignored_exceptions=(LLMQuotaExceededError,),
            half_open_max_calls=1,
            category="evaluations",
```

and in `CircuitBreaker.call`, before the `except self.config.expected_exception:` clause:

Replace:

```python
        except self.config.expected_exception:
            self.stats.failed_calls += 1
```

with:

```python
        except LLMQuotaExceededError:
            raise  # the user's limit, not the provider's health: no stats change

        except self.config.expected_exception:
            self.stats.failed_calls += 1
```

The second change keeps this wrapper's own statistics (`failed_calls`, `consecutive_failures`) unchanged too. The service-wide breaker in `unified_evaluation_service.py` is built from the same config class, so it is covered.

- [ ] **Step 5: G-Eval does not retry a refusal**

In `tldw_Server_API/app/core/Evaluations/ms_g_eval.py`:

Replace:

```python
from tldw_Server_API.app.core.exceptions import raise_detached_error
```

with:

```python
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError, raise_detached_error
```

Replace:

```python
            retry=retry_if_not_exception_type(
                (ChatAuthenticationError, ChatConfigurationError, ChatBadRequestError)
            ),
```

with:

```python
            retry=retry_if_not_exception_type(
                (ChatAuthenticationError, ChatConfigurationError, ChatBadRequestError, LLMQuotaExceededError)
            ),
```

Replace:

```python
                except Exception:
                    raise ValueError(f"Unsupported API endpoint: {api_endpoint}") from None
```

with:

```python
                except LLMQuotaExceededError:
                    raise  # the user's limit: not retried, and not rewritten as an endpoint error
                except Exception:
                    raise ValueError(f"Unsupported API endpoint: {api_endpoint}") from None
```

With `reraise=True`, tenacity re-raises the original object when it does not retry.

- [ ] **Step 6: The evaluators let a refusal through**

In `tldw_Server_API/app/core/Evaluations/response_quality_evaluator.py`, add the import:

Replace:

```python
from tldw_Server_API.app.core.Evaluations.scoring import normalize_likert
```

with:

```python
from tldw_Server_API.app.core.Evaluations.scoring import normalize_likert
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
```

and replace each of the **six** occurrences of:

Replace:

```python
        except (SummaryProviderError, DaemonCapacityError):
            raise
```

with:

```python
        except (SummaryProviderError, DaemonCapacityError, LLMQuotaExceededError):
            raise
```

In `tldw_Server_API/app/core/Evaluations/rag_evaluator.py`:

Replace:

```python
from tldw_Server_API.app.core.exceptions import raise_detached_error
```

with:

```python
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError, raise_detached_error
```

Replace:

```python
                try:
                    probe_result: Any = await probe_factory()
                except (SummaryProviderError, DaemonCapacityError):
                    raise
```

with:

```python
                try:
                    probe_result: Any = await probe_factory()
                except (SummaryProviderError, DaemonCapacityError, LLMQuotaExceededError):
                    raise
```

The RAG evaluator's other clauses catch `_RAG_EVAL_NONCRITICAL_EXCEPTIONS`, which holds no `HTTPException`, so a refusal already passes them.

- [ ] **Step 6b: Evaluation runs are checked when they are created**

In `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py`:

Replace:

```python
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import rbac_rate_limit, TokenScopeGuard, User
```

with:

```python
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import rbac_rate_limit, TokenScopeGuard, User
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota
```

Replace:

```python
        target_model = request.target_model
        # Allow free-form config; convert Pydantic models if provided in future
```

with:

```python
        # Usage quotas (LLM usage accounting spec §4a): a run's samples are LLM calls, made in a task
        # started from this request, so the run is checked here and then runs to completion.
        await enforce_llm_quota(getattr(current_user, "id", None))
        target_model = request.target_model
        # Allow free-form config; convert Pydantic models if provided in future
```

The check comes after the idempotent-replay branch, which returns an existing run without new work, and inside the handler's `try`, which already has `except HTTPException: raise`.

- [ ] **Step 7: Run the tests to see them pass**

Run the Step 2 command without `-k quota`, and add `tldw_Server_API/tests/Evaluations/test_circuit_breaker.py`.
Expected: all pass.
Expected: all pass.

- [ ] **Step 8: Run the neighbours**

Run: `TLDW_TEST_NO_DOCKER=1 RUN_EVALUATIONS=1 $PY -m pytest -q -p no:cacheprovider -n 4 tldw_Server_API/tests/Evaluations tldw_Server_API/tests/Infrastructure`
Expected: all pass apart from the known failures. If `tldw_Server_API/tests/Infrastructure` does not exist, run `git grep -l "Infrastructure.circuit_breaker" -- 'tldw_Server_API/tests/**/test_*.py'` and run those files instead.

- [ ] **Step 9: Lint and commit**

Compare ruff finding counts with `origin/dev` as in Task 4. On `dev`: `Infrastructure/circuit_breaker.py` 0, `Evaluations/circuit_breaker.py` 3, `ms_g_eval.py` 0, `response_quality_evaluator.py` 0, `rag_evaluator.py` 1, `evaluations_crud.py` 1, `tests/Evaluations/unit/test_circuit_breaker.py` 9, `test_specialized_provider_auth_failures.py` 0, `test_evaluations_crud_create_run_api.py` 3.

```bash
git add tldw_Server_API/app/core/Infrastructure/circuit_breaker.py tldw_Server_API/app/core/Evaluations/circuit_breaker.py tldw_Server_API/app/core/Evaluations/ms_g_eval.py tldw_Server_API/app/core/Evaluations/response_quality_evaluator.py tldw_Server_API/app/core/Evaluations/rag_evaluator.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py tldw_Server_API/tests/Evaluations/unit/test_circuit_breaker.py tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py
git commit -m "fix(usage): evaluations never count, retry or score an LLM quota refusal; runs are checked at creation (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The chat pipeline and auto-router, jobs, extraction, the RAG answer fallback, insights and manuscript analysis let a refusal through

Seven small changes, each so that a refusal is not mistaken for a provider failure. One of them fixes a regression the check at the call would otherwise cause: the `/chat/completions` auto-router makes its LLM call before the boundary check, so for an over-quota user that call is refused at the call, and its broad `except` would record a failure of the router's provider for every user.

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py` (the `core.exceptions` import near line 171; `_is_trusted_local_chat_http_exception` near line 3480; the two queued processors near lines 5290 and 6896)
- Modify: `tldw_Server_API/app/core/exceptions.py` (`LLMQuotaExceededError`, near line 2075)
- Modify: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/llm.py` (imports; `call_llm_provider`, near line 181)
- Modify: `tldw_Server_API/app/core/RAG/rag_service/generation.py` (imports; `AnswerGenerator.generate`, near line 480)
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/document_insights.py` (the LLM call's handlers, near line 331)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (the `core.exceptions` import near line 242; `_execute_router_call` in `_select_auto_chat_llm_router_choice`, near line 1641)
- Modify: `tldw_Server_API/app/core/Writing/manuscript_analysis.py` (imports; `_run_structured_analysis`, near line 148)
- Test (append to each): `tests/Chat/unit/test_chat_service_fallback.py`, `tests/Chat/integration/test_chat_endpoint_auto_routing.py`, `tests/Jobs/test_worker_sdk_llm_user.py`, `tests/Web_Scraping/test_phase4_extraction_failures.py`, `tests/RAG_NEW/unit/test_generation_executor.py`, `tests/Media/test_document_insights.py`, `tests/Writing/test_manuscript_analysis_service.py` (all under `tldw_Server_API/`)

**Interfaces:**
- Consumes: `refuse_llm_calls` (Task 1), `check_llm_call_async` (Task 1).
- Produces: `LLMQuotaExceededError.retryable = False` and `.failure_code = "llm_quota_exceeded"` (class attributes). WorkerSDK and several job workers read `retryable` with `getattr`; WorkerSDK records `failure_code` as the job's `error_code`.

- [ ] **Step 1: Write the failing tests**

Append to `tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py` (it defines `_DummyMetrics`, `_DummyProviderManager`, `_ProcessorQueue`, `_nonstream_call_kwargs` and imports `execute_non_stream_call`, `chat_service`, `pytest`):

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
async def test_execute_non_stream_call_quota_refusal_is_not_a_provider_failure(monkeypatch, queued):
    """An LLM quota refusal reaches the caller as its 402, records no provider failure and starts no fallback (§4b)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    refusal = LLMQuotaExceededError(
        category="llm_tokens_month", current=10, limit=10, message="Monthly LLM token limit reached", retry_after=60
    )
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: _ProcessorQueue() if queued else None)

    def refused_llm_call():
        raise refusal

    with pytest.raises(LLMQuotaExceededError) as captured:
        await execute_non_stream_call(
            **_nonstream_call_kwargs(
                llm_call_func=refused_llm_call,
                provider_manager=provider_manager,
                metrics=metrics,
                refresh_provider_params=lambda _provider: ({}, None),
                queue_execution_enabled=queued,
            )
        )

    assert captured.value.status_code == 402
    assert captured.value.detail["error"] == "limit_exceeded"
    assert provider_manager.failure_records == []
    assert provider_manager.fallback_requests == []
```

Append to `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py` (it imports `patch`, `pytest`, `ChatCompletionRequest`, `ChatCompletionUserMessageParam`, `RoutingDecision`, and loads the `persona_chat_client`, `persona_chat_db` and `execution_scoped_provider_credentials` fixtures):

```python
@pytest.mark.integration
def test_a_refused_router_call_is_not_a_provider_failure_and_the_request_gets_the_402(
    persona_chat_client,
    persona_chat_db,
    execution_scoped_provider_credentials,
    monkeypatch,
):
    """The auto-router's LLM call runs before the boundary check; its refusal must not mark the provider unhealthy."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    authenticated_client, headers, _ = persona_chat_client
    refuse_llm_work(monkeypatch)
    request_data = ChatCompletionRequest(
        model="auto",
        api_provider="openrouter",
        messages=[ChatCompletionUserMessageParam(role="user", content="Summarize this")],
    )

    class _RecordingProviderManager:
        def __init__(self) -> None:
            self.failures: list[str] = []

        def record_failure(self, provider, error) -> None:
            self.failures.append(provider)

        def record_success(self, provider, latency) -> None:
            pass

        def get_available_provider(self, exclude=None):
            return None

    provider_manager = _RecordingProviderManager()
    dispatched: list[dict] = []

    async def refused_router_call(**kwargs):
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    def rules_route_model(**kwargs):
        return RoutingDecision(
            provider="openrouter",
            model="anthropic/claude-4.5-sonnet",
            canonical=True,
            decision_source="rules",
        )

    async def must_not_dispatch(**kwargs):
        dispatched.append(kwargs)
        raise AssertionError("dispatched")

    with (
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_configured_providers",
            return_value={
                "providers": [
                    {
                        "name": "openrouter",
                        "default_model": "anthropic/claude-4.5-sonnet",
                        "models_info": [
                            {"name": "anthropic/claude-4.5-sonnet", "tool_support": True, "vision_support": True, "quality_rank": 20},
                            {"name": "openai/gpt-4.1-mini", "tool_support": True, "vision_support": True, "quality_rank": 10},
                        ],
                    }
                ],
                "default_provider": "openrouter",
            },
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call_async",
            side_effect=refused_router_call,
            create=True,
        ),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.get_provider_manager", return_value=provider_manager),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.route_model", side_effect=rules_route_model),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.execute_non_stream_call", side_effect=must_not_dispatch),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", headers=headers, json=request_data.model_dump())

    assert_llm_quota_402(response)
    assert provider_manager.failures == []
    assert dispatched == []
```

Append to `tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py` (it imports `pytest`, `AsyncMock`, `patch` and defines `MODULE`):

```python
@pytest.mark.asyncio
async def test_an_llm_quota_refusal_is_raised_not_stored_as_a_failed_analysis():
    """The refusal reaches the endpoint, which answers 402, instead of becoming an analysis_failed result (§4b)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_pacing

    refusal = LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )
    with patch(f"{MODULE}.perform_chat_api_call_async", new_callable=AsyncMock, side_effect=refusal):
        with pytest.raises(LLMQuotaExceededError):
            await analyze_pacing("Some scene text.")
```

Append to `tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py` (it imports `asyncio`, `JobManager`, `ensure_jobs_tables` and defines `_sdk`):

```python
async def test_a_job_refused_by_the_llm_quota_fails_at_once_with_its_code(tmp_path, monkeypatch) -> None:
    """A refused handler is not retried, since a quota window does not reset in seconds (spec §4b)."""
    from tldw_Server_API.app.core.Usage.llm_quota_gate import check_llm_call_async
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls

    refuse_llm_calls(monkeypatch)
    db_path = tmp_path / "jobs_llm_quota.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(domain="writing", queue="default", job_type="t", payload={}, owner_user_id="7")
    sdk = _sdk(jm)

    async def handler(job_row):
        """Make a checked LLM call for the job's owner."""
        sdk.stop()
        await check_llm_call_async()
        return {"ok": True}

    await asyncio.wait_for(sdk.run(handler=handler, job_type="t"), timeout=5)

    row = jm.get_job(int(job["id"]))
    assert (row["status"], row["error_code"]) == ("failed", "llm_quota_exceeded")
```

Append to `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_failures.py` (it imports `Any`, `pytest`, `llm_strategy`, `extraction` and defines `_install_dependencies`):

```python
def test_provider_quota_refusal_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A quota refusal is final: one provider call, no backoff, and it reaches the caller (spec §4b)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    provider_calls = 0
    sleeps: list[float] = []
    refusal = LLMQuotaExceededError(
        category="llm_tokens_month", current=10, limit=10, message="Monthly LLM token limit reached", retry_after=60
    )

    def provider(**_kwargs: Any) -> dict[str, Any]:
        """Refuse every call."""
        nonlocal provider_calls
        provider_calls += 1
        raise refusal

    monkeypatch.setenv("EXTRACTOR_MAX_RETRIES", "3")
    monkeypatch.setenv("EXTRACTOR_RETRY_BASE_MS", "25")
    monkeypatch.setenv("EXTRACTOR_RETRY_JITTER_MS", "0")
    _install_dependencies(monkeypatch, llm_strategy, perform_chat_api_call=provider, sleep=sleeps.append)

    with pytest.raises(LLMQuotaExceededError) as exc_info:
        extraction.extract_llm_entities(
            "<p>Body</p>",
            "https://example.com/quota",
            llm_settings={"provider": "openai"},
        )

    assert exc_info.value is refusal
    assert provider_calls == 1
    assert sleeps == []
```

Append to `tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py` (it imports `Any`, `pytest`, `AnswerGenerator`):

```python
@pytest.mark.asyncio
async def test_answer_generator_never_replaces_a_quota_refusal_with_a_fallback_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a credential runtime, a provider failure falls back to an extractive answer; a refusal does not (§4b)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    async def refused_chat_call(**kwargs: Any) -> None:  # noqa: ARG001
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=10, limit=10, message="Monthly LLM token limit reached", retry_after=60
        )

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", refused_chat_call)

    with pytest.raises(LLMQuotaExceededError):
        await AnswerGenerator(provider="anthropic", model="claude-test").generate(query="question", context="evidence")
```

Append to `tldw_Server_API/tests/Media/test_document_insights.py` (it defines `app`, the `mock_user` and `mock_db` fixtures and imports `insights_mod`, `patch`, `AsyncClient`, `ASGITransport`, `get_request_user`, `get_media_db_for_user`):

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_document_insights_passes_an_llm_quota_refusal_through(mock_user, mock_db):
    """The refusal from the provider call keeps its 402; it is not reported as an LLM service error (spec §4b)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402

    class _RefusingAdapter:
        def chat(self, _request, **_kwargs):
            """Refuse, as the provider method does for a user over the LLM quota."""
            raise LLMQuotaExceededError(
                category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
            )

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    with (
        patch.object(insights_mod, "_get_adapter", return_value=_RefusingAdapter()),
        patch.object(insights_mod, "resolve_provider_api_key", return_value=("key", None)),
        patch.object(insights_mod, "provider_requires_api_key", return_value=False),
        patch.object(insights_mod, "_resolve_model", return_value="test-model"),
        patch.object(insights_mod, "get_cached_response", return_value=None),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/media/1/insights")

    app.dependency_overrides.clear()
    assert_llm_quota_402(response)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_failures.py tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py tldw_Server_API/tests/Media/test_document_insights.py tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py -k "quota or refused_router_call or refusal_is_raised_not_stored"`
Expected: 8 failed. The pipeline raises `SanitizedProviderStreamError` and records a provider failure (both parametrized cases); the auto-router records a failure for `openrouter` (the request still gets its 402, from the boundary check); the job is scheduled for a retry with error code `worker_exception`; the extraction retries four times; the generator returns a fallback answer; insights returns 500; manuscript analysis returns `analysis_failed` instead of raising.

- [ ] **Step 3: The chat pipeline**

In `tldw_Server_API/app/core/Chat/chat_service.py`:

Replace:

```python
from tldw_Server_API.app.core.exceptions import EgressPolicyError, raise_detached_error
```

with:

```python
from tldw_Server_API.app.core.exceptions import EgressPolicyError, LLMQuotaExceededError, raise_detached_error
```

Replace:

```python
def _is_trusted_local_chat_http_exception(exc: BaseException) -> bool:
    """Return whether an HTTP error was created and bounded by this service."""

    return isinstance(exc, UnsupportedMultiChoiceToolAutoexecError) or (
```

with:

```python
def _is_trusted_local_chat_http_exception(exc: BaseException) -> bool:
    """Return whether an HTTP error was created and bounded by this service.

    An LLM quota refusal is trusted: it is the user's limit, not a provider failure,
    and reaches the caller as its own 402 (LLM usage accounting spec §4b).
    """

    return isinstance(exc, (UnsupportedMultiChoiceToolAutoexecError, LLMQuotaExceededError)) or (
```

`_bound_provider_http_exception` and `_bound_provider_invocation_exception` return a trusted exception unchanged, so this one predicate keeps every rebinding site from turning the refusal into a provider outage. Then, in the streaming queued processor:

Replace:

```python
                    return result
                except Exception as proc_error:  # noqa: BLE001 - adapter trust boundary
                    latency = time.time() - local_start
```

with:

```python
                    return result
                except LLMQuotaExceededError:
                    raise  # the user's limit, not the provider's health: no failure, no fallback
                except Exception as proc_error:  # noqa: BLE001 - adapter trust boundary
                    latency = time.time() - local_start
```

and in the non-streaming queued processor:

Replace:

```python
                    return result
                except _CHAT_NONCRITICAL_EXCEPTIONS as proc_error:
                    latency = time.time() - local_start
```

with:

```python
                    return result
                except LLMQuotaExceededError:
                    raise  # the user's limit, not the provider's health: no failure, no fallback
                except _CHAT_NONCRITICAL_EXCEPTIONS as proc_error:
                    latency = time.time() - local_start
```

Those two record a provider failure before rebinding, so they need their own guard. Only `/chat/completions` reaches this pipeline, and it is admitted at its boundary, so none of this fires today; it keeps the pipeline honest if that ever changes.

- [ ] **Step 3b: The auto-router does not record a refusal as a provider failure**

In `tldw_Server_API/app/api/v1/endpoints/chat.py`:

Replace:

```python
from tldw_Server_API.app.core.exceptions import raise_detached_error
```

with:

```python
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError, raise_detached_error
```

and in `_execute_router_call` (inside `_select_auto_chat_llm_router_choice`):

Replace:

```python
        except asyncio.CancelledError:
            raise
        except Exception as router_error:  # noqa: BLE001 - provider trust boundary
            latency = time.time() - router_started_at
```

with:

```python
        except asyncio.CancelledError:
            raise
        except LLMQuotaExceededError:
            # The user's limit, not the router provider's health: no failure recorded. The routing
            # runtime falls back to rules routing, and the boundary check below returns the 402.
            raise
        except Exception as router_error:  # noqa: BLE001 - provider trust boundary
            latency = time.time() - router_started_at
```

The routing runtime (`core/LLM_Calls/routing/runtime.py`, `select_llm_router_choice`) treats any router error as "no LLM choice" and falls back to rules routing, so the request goes on to the boundary check, which refuses it with the 402 before any generation.

- [ ] **Step 4: Jobs do not retry a refusal**

In `tldw_Server_API/app/core/exceptions.py`, in `class LLMQuotaExceededError(HTTPException):`, replace:

```python
    through unchanged.
    """

    def __init__(self, *, category: str, current: float, limit: int | float, message: str, retry_after: int) -> None:
```

with:

```python
    through unchanged.
    """

    # Job workers read these: a quota window does not reset in seconds, so a refused job is not retried.
    retryable = False
    failure_code = "llm_quota_exceeded"

    def __init__(self, *, category: str, current: float, limit: int | float, message: str, retry_after: int) -> None:
```

- [ ] **Step 5: The extraction strategy does not retry a refusal**

In `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/llm.py`:

Replace:

```python
from loguru import logger

from ...observability import bounded_code, bounded_stage, sanitized_host
```

with:

```python
from loguru import logger

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

from ...observability import bounded_code, bounded_stage, sanitized_host
```

Replace:

```python
                dependencies.cancellation_checkpoint()
                return response, False
        except Exception as exc:  # noqa: BLE001 - provider SDKs use unrelated exception hierarchies
```

with:

```python
                dependencies.cancellation_checkpoint()
                return response, False
        except LLMQuotaExceededError:
            raise  # the user's limit: final, so not retried and not reported as a failed extraction
        except Exception as exc:  # noqa: BLE001 - provider SDKs use unrelated exception hierarchies
```

The extraction pipeline's own retry wrapper catches a tuple that holds no `HTTPException`, so the refusal reaches the scraping endpoint, which passes it on as a 402.

- [ ] **Step 6: The RAG answer generator never falls back on a refusal**

In `tldw_Server_API/app/core/RAG/rag_service/generation.py`:

Replace:

```python
from tldw_Server_API.app.core.LLM_Calls.sse import is_done_line
```

with:

```python
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.LLM_Calls.sse import is_done_line
```

Replace:

```python
        except Exception:
            logger.error("Error generating response")

            if kwargs.get("credential_runtime") is not None:
                raise
```

with:

```python
        except LLMQuotaExceededError:
            raise  # the user's limit: never replaced by a fallback answer (LLM usage accounting spec §4b)
        except Exception:
            logger.error("Error generating response")

            if kwargs.get("credential_runtime") is not None:
                raise
```

Without a credential runtime, any failure falls back to an extractive answer that looks like a real one. A refused user must see the refusal instead.

- [ ] **Step 7: Insights keeps the 402**

In `tldw_Server_API/app/api/v1/endpoints/media/document_insights.py`:

Replace:

```python
            detail="LLM provider configuration error",
        ) from e
    except Exception as e:
        logger.error("LLM call failed for document insights")
```

with:

```python
            detail="LLM provider configuration error",
        ) from e
    except HTTPException:
        raise  # an LLM quota refusal keeps its 402 (LLM usage accounting spec §4b)
    except Exception as e:
        logger.error("LLM call failed for document insights")
```

- [ ] **Step 7b: Manuscript analysis lets a refusal through**

In `tldw_Server_API/app/core/Writing/manuscript_analysis.py`:

Replace:

```python
from loguru import logger
```

with:

```python
from loguru import logger

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
```

Replace:

```python
    except json.JSONDecodeError:
        logger.warning("Failed to parse analysis JSON response")
        return {"error": "Failed to parse LLM response", "raw": content[:500] if content else ""}
    except Exception:
        logger.exception("Analysis LLM call failed")
```

with:

```python
    except json.JSONDecodeError:
        logger.warning("Failed to parse analysis JSON response")
        return {"error": "Failed to parse LLM response", "raw": content[:500] if content else ""}
    except LLMQuotaExceededError:
        raise  # the user's limit: the endpoint answers 402 instead of storing a failed analysis
    except Exception:
        logger.exception("Analysis LLM call failed")
```

The writing endpoints catch `_MANUSCRIPT_NONCRITICAL_EXCEPTIONS`, which holds no `HTTPException`, so they answer 402 instead of storing an `analysis_failed` analysis.

- [ ] **Step 8: Run the tests to see them pass**

Run the Step 2 command without `-k`.
Expected: all pass. The existing `test_execute_non_stream_call_untrusted_failures_never_fallback_or_leak[http_exception]` still passes: a plain `HTTPException` is still untrusted.

- [ ] **Step 9: Run the neighbours**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider -n 4 tldw_Server_API/tests/Chat tldw_Server_API/tests/Web_Scraping tldw_Server_API/tests/RAG_NEW/unit tldw_Server_API/tests/Media/test_document_insights.py tldw_Server_API/tests/Writing tldw_Server_API/tests/Usage`
and, alone (two Jobs files in that directory use random ids that collide under xdist): `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py`
Expected: all pass apart from the known failures and load flakes.

- [ ] **Step 10: Lint and commit**

Compare ruff finding counts with `origin/dev` as in Task 4. On `dev`: `chat_service.py` 1, `chat.py` 0, `exceptions.py` 0, `strategies/llm.py` 0, `generation.py` 1, `document_insights.py` 0, `manuscript_analysis.py` 1, `test_manuscript_analysis_service.py` 1, and each other test file 0.

```bash
git add tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/core/exceptions.py tldw_Server_API/app/core/Web_Scraping/extraction/strategies/llm.py tldw_Server_API/app/core/RAG/rag_service/generation.py tldw_Server_API/app/api/v1/endpoints/media/document_insights.py tldw_Server_API/app/core/Writing/manuscript_analysis.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_failures.py tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py tldw_Server_API/tests/Media/test_document_insights.py tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py
git commit -m "fix(usage): the chat pipeline and auto-router, jobs, extraction, the RAG answer fallback, insights and manuscript analysis pass a quota refusal through (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: RAG's optional stages degrade, and three handlers keep the 402

A RAG search with generation off is not checked at its boundary, so its optional LLM stages are checked at the call (spec §4a). Every stage but two already catches the refusal and degrades, recording `"failure_code": "provider_unavailable"` in its metadata: query classification and reformulation, the research loop, HyDE, evidence accumulation, document grading, evidence chains and the LLM reranker. Gap analysis and the agentic LLM planner catch narrow tuples and would let it escape the pipeline. Three handlers would then turn it into a 500. This settles the note from PR 2a's final review.

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` (imports near line 52; gap analysis near line 5197)
- Modify: `tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py` (imports near line 29; the planner near line 858)
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` (`rag_ablate` near line 932, `simple_search_endpoint` near line 1930, `advanced_search_endpoint` near line 2426)
- Regenerate: `Helper_Scripts/ci/scope_predicate_baseline.txt`, `Docs/Design/web_scraping_refactor_import_inventory.json`, `Docs/Design/WebScraping_Refactor_Import_Inventory.md` (line-number baselines that record lines in `unified_pipeline.py`)
- Test (append to each): `tests/RAG_NEW/unit/test_generation_executor.py`, `tests/RAG_NEW/unit/test_agentic_failures_and_fallbacks.py`, `tests/RAG_NEW/integration/test_rag_convenience_endpoint_credentials.py`, `tests/RAG_NEW/integration/test_rag_ablate_api.py` (all under `tldw_Server_API/`)

**Interfaces:** none new.

- [ ] **Step 1: Write the failing tests**

Append to `tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py` (it defines `_RecordingCredentialRuntime` and `_run_unified_bound_sgl_stage`, which runs one optional stage with generation off):

```python
@pytest.mark.parametrize(("stage", "metadata_key"), [("gap", "gap_analysis"), ("reranker", "reranking")])
@pytest.mark.asyncio
async def test_unified_optional_stage_degrades_when_llm_quota_refuses(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    metadata_key: str,
) -> None:
    """A search with generation off, for a user over the LLM quota, returns its results with the stage degraded (§4a)."""
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    attempts: list[str] = []

    def refused(*_args: Any, **_kwargs: Any) -> str:
        """The provider call is refused; analyze() does not catch an HTTPException."""
        attempts.append("refused")
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(sgl, "_dispatch_to_api", refused)
    runtime = _RecordingCredentialRuntime()

    result = await _run_unified_bound_sgl_stage(monkeypatch, stage, runtime)

    assert attempts  # nosec B101 - the stage really called the provider
    assert [doc["id"] for doc in result.documents] == [f"doc-{stage}"]  # nosec B101
    assert result.metadata[metadata_key]["failure_code"] == "provider_unavailable"  # nosec B101
    assert result.metadata[metadata_key]["verification_available"] is False  # nosec B101
    assert runtime.marked == []  # nosec B101 - a refused call is not credential use
```

The stub sits below the real `analyze`, so the test also shows that `analyze` lets an `HTTPException` through.

Append to `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_failures_and_fallbacks.py` (it imports `pytest`, `ac` and defines `_doc`):

```python
@pytest.mark.asyncio
async def test_planner_quota_refusal_falls_back_to_heuristics(monkeypatch):
    """A refused LLM planner degrades to heuristic planning instead of failing the search (spec §4a, §4b)."""
    import tldw_Server_API.app.core.RAG.rag_service.agentic_execution as agentic_execution
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return [_doc()]

    class RefusedPlanner:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, *args, **kwargs):  # noqa: ARG002
            raise LLMQuotaExceededError(
                category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
            )

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(agentic_execution, "AnswerGenerator", RefusedPlanner)

    cfg = ac.AgenticConfig(top_k_docs=1, enable_tools=True, use_llm_planner=True, time_budget_sec=2.0)
    res = await ac.agentic_rag_pipeline(
        query="explain attention",
        sources=["media_db"],
        search_mode="fts",
        agentic=cfg,
        enable_generation=False,
    )

    assert res.documents
```

`agentic_execution` keeps its own `AnswerGenerator` name, which is the one the planner uses, so the patch goes there.

Append to `tldw_Server_API/tests/RAG_NEW/integration/test_rag_convenience_endpoint_credentials.py` (it imports `Any`, `pytest`, `rag_endpoint`, `LLMQuotaExceededError` and defines `_CountingRuntime`, `_install_endpoint_fakes`, `_request`, `_user`, `_db`):

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_name", ["simple", "advanced"])
async def test_convenience_endpoint_passes_an_llm_quota_refusal_through(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_name: str,
) -> None:
    """A refusal raised inside the search keeps its 402; it is not turned into a 500 (spec §4b)."""
    runtime = _CountingRuntime()
    _install_endpoint_fakes(monkeypatch, runtime)

    async def refused(*_args: Any, **_kwargs: Any) -> Any:
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(rag_endpoint, f"{endpoint_name}_search", refused)
    endpoint = rag_endpoint.simple_search_endpoint if endpoint_name == "simple" else rag_endpoint.advanced_search_endpoint

    kwargs = {} if endpoint_name == "simple" else {"with_answer": False}
    with pytest.raises(LLMQuotaExceededError):
        await endpoint(
            request=_request(f"/api/v1/rag/{endpoint_name}"),
            query="q",
            current_user=_user(),
            media_db=_db("media.db"),
            chacha_db=_db("notes.db"),
            **kwargs,
        )

    assert runtime.close_calls == 1  # nosec B101
```

Append to `tldw_Server_API/tests/RAG_NEW/integration/test_rag_ablate_api.py` (it defines `_CountingRuntime`, `_install_direct_ablate_fakes`, `_request` and imports `rag_endpoint`, `User`, `pytest`):

```python
@pytest.mark.asyncio
async def test_ablate_passes_an_llm_quota_refusal_through(monkeypatch):
    """A refusal raised inside an ablation run keeps its 402; it is not turned into a 500 (spec §4b)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    runtime = _CountingRuntime()
    _install_direct_ablate_fakes(monkeypatch, runtime)

    async def refused(**_kwargs):
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(rag_endpoint, "unified_rag_pipeline", refused)
    monkeypatch.setattr(rag_endpoint, "agentic_rag_pipeline", refused)

    with pytest.raises(LLMQuotaExceededError):
        await rag_endpoint.rag_ablate(
            request_raw=_request(),
            request=rag_endpoint.AblationRequest(query="ablate", with_answer=False),
            current_user=User(id=1, username="tester", email=None, is_active=True),
            media_db=None,
            chacha_db=None,
        )

    assert runtime.close_calls == 1  # nosec B101
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_failures_and_fallbacks.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_convenience_endpoint_credentials.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_ablate_api.py -k "degrades_when_llm_quota or planner_quota or passes_an_llm_quota"`
Expected: 5 failed, 1 passed. Gap analysis and the planner let `LLMQuotaExceededError` escape; the three handlers raise `HTTPException: 500`. The `reranker` case is a guard: the LLM reranker already degrades.

- [ ] **Step 3: Gap analysis degrades**

In `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`, add the import (it sorts after the `DB_Management.scope_context` block):

Replace:

```python
    get_scope,
)
from tldw_Server_API.app.core.LLM_Calls.structured_output import (
```

with:

```python
    get_scope,
)
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.LLM_Calls.structured_output import (
```

and in gap analysis (the first of four `except (ByokResolutionError, SummaryProviderError) as exc:` clauses in the file; the other three are after generation, where the request was admitted, so leave them):

Replace:

```python
                        except (StructuredOutputParseError, TypeError, ValueError):
                            followups = [s.strip("- ") for s in llm_out.splitlines() if s.strip()]
                except (ByokResolutionError, SummaryProviderError) as exc:
```

with:

```python
                        except (StructuredOutputParseError, TypeError, ValueError):
                            followups = [s.strip("- ") for s in llm_out.splitlines() if s.strip()]
                except (ByokResolutionError, SummaryProviderError, LLMQuotaExceededError) as exc:
```

The clause's body already records `"failure_code": "provider_unavailable"` for anything that is not a `ByokResolutionError`, and leaves no follow-ups.

- [ ] **Step 4: The agentic planner degrades**

In `tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py`:

Replace:

```python
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
```

with:

```python
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
```

Replace:

```python
        except (ByokResolutionError, ChatAPIError) as exc:
            _record_planner_degraded(stage_metadata, exc)
```

with:

```python
        except (ByokResolutionError, ChatAPIError, LLMQuotaExceededError) as exc:
            _record_planner_degraded(stage_metadata, exc)
```

`_record_planner_degraded` records `provider_unavailable` for a 402, as for any status other than 401 and 403.

- [ ] **Step 5: Three handlers keep the 402**

In `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` (`HTTPException` is already imported):

Replace:

```python
    except _RAG_PROVIDER_FAILURES as exc:
        raise_detached_error(_rag_provider_http_exception(exc))
    except Exception as exc:  # noqa: BLE001 - convert unexpected endpoint failures
        logger.error("RAG ablation failed")
```

with:

```python
    except _RAG_PROVIDER_FAILURES as exc:
        raise_detached_error(_rag_provider_http_exception(exc))
    except HTTPException:
        raise  # an LLM quota refusal keeps its 402 (LLM usage accounting spec §4b)
    except Exception as exc:  # noqa: BLE001 - convert unexpected endpoint failures
        logger.error("RAG ablation failed")
```

Replace:

```python
    except _RAG_PROVIDER_FAILURES as exc:
        raise_detached_error(_rag_provider_http_exception(exc))
    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error("Simple search failed")
```

with:

```python
    except _RAG_PROVIDER_FAILURES as exc:
        raise_detached_error(_rag_provider_http_exception(exc))
    except HTTPException:
        raise  # an LLM quota refusal keeps its 402 (LLM usage accounting spec §4b)
    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error("Simple search failed")
```

Replace:

```python
    except _RAG_PROVIDER_FAILURES as exc:
        raise_detached_error(_rag_provider_http_exception(exc))
    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error("Advanced search failed")
```

with:

```python
    except _RAG_PROVIDER_FAILURES as exc:
        raise_detached_error(_rag_provider_http_exception(exc))
    except HTTPException:
        raise  # an LLM quota refusal keeps its 402 (LLM usage accounting spec §4b)
    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error("Advanced search failed")
```

The `finally` that closes the credential runtime is unchanged, so it still runs. `/rag/search` and `/rag/batch` already have `except HTTPException: raise`; the streaming search always generates, so it is admitted and never refused inside.

- [ ] **Step 6: Run the tests to see them pass**

Run the Step 2 command.
Expected: 6 passed.

- [ ] **Step 6b: Regenerate the two line-number baselines**

The new import moves every later line of `unified_pipeline.py` down by one, and two checked-in baselines record lines in that file: the scope-predicate ratchet (`tests/lint/test_scope_predicate_ratchet.py`) and the web scraping import inventory (`tests/WebScraping/test_refactor_import_inventory.py`). Regenerate both:

```bash
$PY Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline
$PY Helper_Scripts/web_scraping_refactor_inventory.py --json Docs/Design/web_scraping_refactor_import_inventory.json --markdown Docs/Design/WebScraping_Refactor_Import_Inventory.md
git diff --stat -- Helper_Scripts/ci/scope_predicate_baseline.txt Docs/Design/web_scraping_refactor_import_inventory.json Docs/Design/WebScraping_Refactor_Import_Inventory.md
```

Expected: the ratchet prints `Recorded 29 conditional scope predicates.`; the diff shows only line numbers in `unified_pipeline.py` moving by one (`unified_pipeline.py:3818 if user_id` becomes `:3819`, and three inventory rows `3313`/`3342` become `3314`/`3343`). Anything else in the diff means another change moved lines: stop and report it.

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py`
Expected: all pass.

- [ ] **Step 7: Run the neighbours**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider -n 4 tldw_Server_API/tests/RAG_NEW tldw_Server_API/tests/RAG`
Expected: all pass apart from the two known `RAG_NEW` failures and the six `tests/RAG` load flakes named in Global Constraints (re-run those alone; they pass).

- [ ] **Step 8: Lint and commit**

Compare ruff finding counts with `origin/dev` as in Task 4. On `dev`: `unified_pipeline.py` 0, `agentic_execution.py` 0, `rag_unified.py` 1 (a pre-existing `I001`), `test_agentic_failures_and_fallbacks.py` 1, the other test files 0.

```bash
git add tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py tldw_Server_API/app/api/v1/endpoints/rag_unified.py Helper_Scripts/ci/scope_predicate_baseline.txt Docs/Design/web_scraping_refactor_import_inventory.json Docs/Design/WebScraping_Refactor_Import_Inventory.md tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_failures_and_fallbacks.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_convenience_endpoint_credentials.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_ablate_api.py
git commit -m "fix(usage): RAG's optional LLM stages degrade on a quota refusal, and three handlers keep the 402 (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: The operator page, the spec and the backlog

**Files:**
- Modify: `Docs/Operations/Usage_Quotas.md` (the `limits.llm_tokens_per_month` row, line 68; the bullets at lines 90, 91 and 97)
- Modify: `Docs/Operations/Rate_Limits_Troubleshooting.md` (line 19)
- Modify: `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (Status; §4a table and RAG bullet; §4b; Risks; Review changes)
- Modify: `backlog/tasks/task-13437 - Gate-LLM-tokens-month-at-every-LLM-entry-point.md` (through backlog-py only)
- Create, through backlog-py: TASK-13635, TASK-13636, TASK-13637 and TASK-13638

Write plainly and match each document's style. Use the Edit tool. Neither operations page has a `Docs/Published` mirror. Every replacement below is given as a fenced block: copy its contents, not the fence.

- [ ] **Step 1: The operator page**

In `Docs/Operations/Usage_Quotas.md`:

1. In the `limits.llm_tokens_per_month` row, replace `checked before LLM work at the entry points listed below` with `checked before every LLM call (see below)`.
2. In the `- **Where a refusal happens.** ...` bullet (line 90), replace `` `/evaluations/batch` of type `geval`, `rag` or `response_quality`; `` with `` `/evaluations/batch` of type `geval`, `rag` or `response_quality`; evaluation runs (`POST /evaluations/{eval_id}/runs`); ``.
3. Replace the whole `- **Not checked yet** (a later change refuses them at the call): ...` bullet (line 91) with:

```markdown
- **Everything else is checked at the call.** Background jobs (`/media/ingest-jobs`, evaluation recipe runs), workflow runs, WebSocket sessions (audio, realtime) and every HTTP endpoint not listed above check each LLM call just before the provider is called, and a call for a user over a limit is refused without reaching the provider. What the caller gets back depends on the caller:
  - **An HTTP endpoint** that passes errors through, as most do, returns the same 402: `/translate`, `/media/ingest-web-content` with analysis, `/media/process-web-scraping` with summarization, `/media/{id}/insights` and the manuscript analysis endpoints among them. The web scraping endpoints discard that request's results. An endpoint that catches errors broadly reports its own error, usually a 500.
  - **A workflow:** an `llm` step fails, and so does the run, with `error` set to `402: {'error': 'limit_exceeded', ...}`; the step is retried once and refused again without a provider call. Other steps that make LLM calls (content generation, NLP, bibliography, image, summarize, translate) report the refusal in their outputs, and the run goes on.
  - **A job** whose worker reads the error's `retryable` flag (WorkerSDK workers and most others) fails at once with the error code `llm_quota_exceeded` and is not retried. A media ingest job marks the item `Error` (`Processing error: LLMQuotaExceededError: ...`); an audio or video batch marks every item in the batch. A media item that reuses an earlier transcript gets no summary.
  - **A WebSocket:** `/audio/chat/stream` sends `{"type":"error","code":"provider_unavailable",...}` for that turn and stays open. The realtime sockets send `{"type":"error","error":{"type":"server_error","code":"internal_error",...}}` and `response.done` with status `failed`, and stay open.
  - **A RAG search with generation off** still returns its results. The optional LLM stages it asked for (query classification, HyDE, gap analysis, LLM reranking and others) are skipped, and their entries in the response metadata say `"failure_code": "provider_unavailable"`.
  - **Shared machinery is not affected.** A refusal never opens the evaluations circuit breaker, is never scored 0.0 by an evaluator, and never marks a chat provider unhealthy, including the auto-router's call on `/chat/completions`, which runs before that endpoint's own check.
  - **Degraded, not refused:** claims verification falls back to heuristic verdicts; Prompt Studio test runs and benchmarks record error outputs or 0.0 scores; character memory extraction finds nothing. TASK-13638 tracks them.
  - **MCP tools** are neither recorded nor checked: they carry no user.
```

4. Replace the `- **Failures let requests through.** ...` bullet (line 97) with:

```markdown
- **Failures let requests through.** If a limit cannot be read or a counter fails, the request or call is admitted and a warning is logged. A check at the call that gets no answer within two seconds lets the call through, with a warning. A synchronous LLM call made on the server's event loop thread cannot be checked (waiting there would stall the server); it goes through, recorded, and the first one logs `A synchronous LLM call on the event loop's thread is not quota-checked`. A stream started there is checked when its first item is read.
- **Run workers inside the server.** A job worker started as its own process (`python -m ...`) has no accounting loop, so its synchronous LLM calls are neither checked nor recorded; the first one logs a warning. Workers run inside the server process by default.
- **What the check costs.** With quotas off, nothing. With quotas on and no LLM limit for the user, nothing once the user's limits are cached (for a minute). With a limit set, the counters are read for each call that a request boundary did not already admit, so a job that makes hundreds of calls makes hundreds of small indexed reads. While the user's limits cannot be read (the AuthNZ database is down), every call tries again and logs a warning.
```

- [ ] **Step 2: The troubleshooting row**

In `Docs/Operations/Rate_Limits_Troubleshooting.md`, line 19, replace:

```markdown
raised at the entry points listed in `Usage_Quotas.md` (`/chat/completions`, the Messages API, character chat, RAG with generation, and others)
```

with:

```markdown
raised before any LLM call: at the request boundary for the entry points listed in `Usage_Quotas.md`, and at the call (`core/Usage/llm_quota_gate.py`) for jobs, workflows, WebSockets and every other endpoint
```

- [ ] **Step 3: The spec**

In `Docs/Design/2026-10-09-llm-usage-accounting-design.md`:

1. **Status** (line 4): replace `PR 2a in progress.` with `PR 2a (#3230) merged; PR 2b in progress.`
2. **§4a table:** after the row `` | `/evaluations/batch` | When the evaluation type is `geval`, `rag` or `response_quality`. | `` add:

```markdown
| Evaluation runs (`POST /evaluations/{eval_id}/runs`) | Always. The run is a task the request starts, so it is admitted with the request; checked at the call, a refused sample would be scored 0.0 and the run would complete. |
```

3. **§4a**, the bullet that starts `- **A RAG search with answer generation off is not checked here and not marked.**`: replace its last sentence, `Until PR 2b they are unchecked, as today.`, with:

```markdown
Gap analysis and the agentic planner were the two stages that let a refusal escape; PR 2b made them degrade, and the three convenience handlers (`/rag/simple`, `/rag/advanced`, `/rag/ablate`) pass a 402 through.
```

4. **§4b**, replace the bullet `  - A WebSocket session reports it as it reports any failed reply. PR 2b's plan names the frame for the operator page.` with:

```markdown
  - A WebSocket session reports it as it reports any failed reply: `/audio/chat/stream` sends a `provider_unavailable` error for that turn, and the realtime sockets send an `internal_error` and a failed `response.done`. Both stay open.
```

5. **§4b**, under "It must not look like a provider outage to shared machinery", after the bullet that starts `  - retry loops do not retry it`, add:

```markdown
  - a job worker does not retry it: the error carries `retryable = False` and `failure_code = "llm_quota_exceeded"`;
  - the chat pipeline neither rewrites it as a provider error nor records a provider failure for it, and neither does the `/chat/completions` auto-router, whose LLM call runs before that endpoint's boundary check;
  - the RAG answer generator does not replace it with a fallback answer.
```

6. **§4b**, replace the last bullet (the one that starts `- **Sync LLM calls made on the server loop's own thread cannot be checked** (§5). Seven code paths`) with:

```markdown
- **Sync LLM calls made on the server loop's own thread cannot be checked** (§5). PR 2b moved seven code paths off the loop with `asyncio.to_thread`: `/translate`, web search aggregation (`WebSearch_APIs.aggregate_results`, the live `/research/websearch` path; the legacy `WebSearch/Web_Search.py` has no caller in the app), the article extractor, enhanced web scraping, web content ingestion, the web scraping service, and the Explainer job. The RAG custom metric named in an earlier draft makes no provider call: its `analyze` call is missing an argument and falls back to a heuristic (TASK-13635).
```

7. **Risks**, in the `- **Fail-open cases.**` bullet, replace `PR 2b moves the known ones off the loop.` with `PR 2b moved the known ones off the loop, and the first such call logs a warning.`
8. **Review changes**: append at the end of the file:

```markdown
Revision of 2026-10-13, while planning PR 2b:

- **The check at the call** lives in `core/Usage/llm_quota_gate.py`. It runs the same decision as §4a on the accounting loop, and a sync caller waits at most two seconds for it.
- **The seven sync calls** are the ones in §4b's last bullet. Web search aggregation is the live module, not the legacy one, and the Explainer job replaces the RAG custom metric, which never reaches a provider.
- **The `/chat/completions` auto-router** makes its LLM call before the endpoint's boundary check, so its refusal must not count as a failure of the router's provider.
- **Evaluation runs** join the §4a table: their samples run in a task the request starts.
- **Jobs** do not retry a refusal; a workflow `llm` step is retried once by the engine's default policy and refused again without a provider call.
- **Roots** are never admitted by the request that started them.
```

- [ ] **Step 4: The backlog**

Run each command from the worktree root, one at a time:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13437 --append-notes "PR 2b: the check at the call (core/Usage/llm_quota_gate.py). The wrapped provider methods and the shared call layer refuse an over-quota user before the provider runs, deciding on the accounting loop; a sync caller waits at most 2 s, and a sync call on the loop's thread is recorded but not checked. Seven sync LLM calls moved off the loop (/translate, web search aggregation, the article extractor, enhanced web scraping, web content ingestion, the web scraping service, the Explainer job). Evaluation runs are checked at creation. A refusal never opens the evaluations circuit breaker, is never retried by G-Eval, the extraction strategy or a job worker, never marks a chat provider unhealthy (the /chat/completions auto-router included) and never becomes a fallback RAG answer or a stored manuscript analysis; RAG's gap analysis and agentic planner degrade, and /rag/simple, /rag/advanced, /rag/ablate and /media/{id}/insights keep the 402. Follow-ups: TASK-13635, TASK-13636, TASK-13637, TASK-13638." --check-ac 1 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 6
```

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "RAG custom completeness metric never calls a provider" --id TASK-13635 --labels rag,evaluations,bug --description "RAGCustomMetrics.evaluate_response_completeness (core/RAG/rag_custom_metrics.py, near line 233) awaits analyze(prompt, \"...\"), but analyze is a sync function that needs api_name, input_data and custom_prompt_arg. The call raises TypeError before any provider call, and the except clause falls back to the heuristic score, so the LLM path is dead code. Fixing it needs a choice of grading provider and model, and the call must run off the event loop (asyncio.to_thread) so it can be quota-checked. Found while planning LLM usage accounting PR 2b (TASK-13437)." --ac "The completeness metric calls a configured provider through analyze with the right arguments, off the event loop" --ac "With no provider configured it keeps the heuristic, and says so in the result"
```

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Output summaries pass arguments analyze does not accept" --id TASK-13636 --labels outputs,bug --description "outputs_service._summarize_single_article (app/services/outputs_service.py, near line 666; the call near line 679) and _summarize_text_block (near line 869; the call near line 883) call analyze(input_data=..., custom_prompt=..., api_name=..., api_key=None, model=...). analyze takes custom_prompt_arg and model_override, so the call raises TypeError, the except clause logs a warning, and every summary falls back to the first 300 characters of the source text. Found while planning LLM usage accounting PR 2b (TASK-13437)." --ac "_summarize_single_article and _summarize_text_block call analyze with custom_prompt_arg and model_override and return the model's summary" --ac "A test drives each through the real analyze signature"
```

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Audio analyze job stage calls analyze with the wrong arguments" --id TASK-13637 --labels audio,jobs,bug --description "The audio_analyze stage in app/services/audio_jobs_worker.py (near line 354) calls analyze(text, api_name) under asyncio.to_thread. analyze's signature is (api_name, input_data, custom_prompt_arg, ...), so the call passes the transcript as the provider name and raises TypeError before any provider call; the stage fails and is retried until its attempts run out. When the call is fixed, the stage's failure path (fail_job with retryable=True, near line 384) should honour the exception's retryable flag and failure_code, so an LLM quota refusal (LLMQuotaExceededError, retryable = False, failure_code llm_quota_exceeded) fails the job at once. Found while planning LLM usage accounting PR 2b (TASK-13437)." --ac "audio_analyze calls analyze with the provider, the transcript and a prompt, and stores the analysis" --ac "A failure's retryable flag and failure_code reach fail_job"
```

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Callers that turn an LLM quota refusal into degraded output" --id TASK-13638 --labels quotas,backend --description "After LLM usage accounting PR 2b (TASK-13437) an LLM call for an over-quota user raises LLMQuotaExceededError at the call. These callers catch it broadly and go on with degraded output instead of reporting it: workflow adapters that return an error output (Workflows/adapters/content/generation.py, text/nlp.py, research/bibliography.py, content/image.py, content/summarize.py, llm/translate.py, which returns the input as a fallback translation; control/flow.py parallel drops it; control/state.py retry retries it); claims verification (Claims_Extraction/claims_service.py, whose _CLAIMS_NONCRITICAL_EXCEPTIONS holds HTTPException, falls back to heuristic verdicts); Prompt Studio test runs and evaluation (test_runner.py, evaluation_manager.py save error outputs and 0.0 scores) and benchmarks (evaluation_manager.py returns score 0.0); character memory extraction (character_memory_extraction.py returns an empty result). Decide for each whether to let the refusal through (except LLMQuotaExceededError: raise) or keep the degraded output and say so." --ac "Each listed caller either lets the refusal through or records that it was refused" --ac "The operator page lists what each one does"
```

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task normalize --check "backlog/tasks/task-13437 - Gate-LLM-tokens-month-at-every-LLM-entry-point.md" backlog/tasks/task-13635*.md backlog/tasks/task-13636*.md backlog/tasks/task-13637*.md backlog/tasks/task-13638*.md
```

Expected: the edit and the four creates succeed; `normalize --check` lists no file. DoD #4 (bandit), #5 (final summary) and the status change are made when the PR is opened, with the ship checks' results.

- [ ] **Step 5: Run the docs tests**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Docs`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add Docs/Operations/Usage_Quotas.md Docs/Operations/Rate_Limits_Troubleshooting.md Docs/Design/2026-10-09-llm-usage-accounting-design.md
git add -A backlog/tasks
git status --short
git commit -m "docs(usage): what each caller sees when an LLM call is refused at the call; four follow-ups (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

`git status --short` before the commit must show only the three docs and the five backlog files.
