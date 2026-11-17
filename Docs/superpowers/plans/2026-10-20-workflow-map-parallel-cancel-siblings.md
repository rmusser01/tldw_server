# Workflow map and parallel Cancel the Other Sub-Steps When One Fails (TASK-13863)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Review folded in (2026-10-20):** independent review and dry run, `review_plan_fu_workflow_cancel_siblings.md`. Verdict: ready with fixes, with 0 Blocker, 2 Important and 7 Minor, all folded in.
- **I1:** the tests count siblings that finished *or were still being cancelled* after the step raised, so a helper that cancels without waiting fails them. Measured: the `no_await` mutant now fails 4 tests.
- **I2:** a cancelled `media_ingest` sub-step left its yt-dlp download running with nothing to stop it. It now terminates the download, test-first. The other subprocess-starting adapters are listed and filed.
- **M1:** `run_step` collects a sub-step's own `CancelledError` when `fail_fast` is off.
- **M2:** Decision 1 states two more facts.
- **M3:** Review Focus 4 is reworded.
- **M4:** the tests gate the siblings on an `asyncio.Event` instead of timing margins.
- **M5:** line and count corrections.
- **M6:** three plans added to the conflicts table.
- **M7:** the in-flight LLM call undercount is in Review Focus 2.

Every changed step was re-run in a scratch worktree on `origin/dev` `df7c290fbf`.

**Goal:** When a sub-step of a `map` or `parallel` step raises, the step cancels its other sub-steps, waits for them to stop, and then fails with that error. A cancelled `media_ingest` sub-step stops the download it started. With `fail_fast` off, `parallel` still collects every other sub-step error, and an LLM quota refusal still fails the step.

**Architecture:**
- One helper in `Workflows/adapters/control/flow.py`, `_run_cancelling_siblings(coros)`, replaces the two `asyncio.gather` calls.
  - It creates a task per sub-step and gathers them.
  - On the first exception, or on its own cancellation, it cancels every task, waits for all of them, and re-raises.
  - The step therefore fails with the sub-step's own exception, as today, so the engine's retry classifier still sees `402` for a refusal.
  - Sub-steps queued behind the concurrency semaphore are tasks waiting on it; cancelling them removes them before they start.
- `run_parallel_adapter.run_step` decides what escapes:
  - `_FLOW_NONCRITICAL_EXCEPTIONS` are recorded as today, in both modes;
  - an `LLMQuotaExceededError` is raised in both modes (TASK-13638);
  - any other `Exception` is raised with `fail_fast` and recorded as `{"error": "parallel_step_error"}` without it;
  - a `CancelledError` is raised when `fail_fast` is on or the task itself is being cancelled. Otherwise it is the sub-step's own, and it is recorded.
- `media_ingest` polls its yt-dlp subprocess with `await asyncio.sleep(0.25)`. A cancel at that `await` now runs `terminate_process` on a worker thread (it blocks for its 5 s grace) and re-raises.
- The AC names `asyncio.TaskGroup` "or cancel the remaining tasks". The helper does the latter, for two reasons:
  - ruff targets py39 (`pyproject.toml:694`) and reports `BaseExceptionGroup` as F821 (measured), although the project requires Python 3.12;
  - the helper raises the sub-step's own exception, not an `ExceptionGroup` it would have to unwrap.

**Tech Stack:** Python 3.12 (`requires-python = ">=3.12"`, `pyproject.toml:15`), asyncio, pytest (`asyncio_mode = "auto"`).

**Spec:**
- `backlog/tasks/task-13863 - Workflow-map-and-parallel-leave-sibling-sub-steps-running-after-one-fails.md`
  - AC1: map and parallel run their sub-steps in an `asyncio.TaskGroup`, or cancel the remaining tasks, so the first raised error cancels the siblings, with a test that records which siblings finished after the step raised.
  - AC2: parallel with `fail_fast` off keeps collecting per-step errors, apart from an LLM quota refusal, which fails the step (TASK-13638).
- Parent plan: `Docs/superpowers/plans/2026-10-18-llm-quota-refusal-callers-rag-metric.md` (Architecture, "What a refusal does to a fan-out"; Review Focus 4). Operator page: `Docs/Operations/Usage_Quotas.md`.

## What was measured

Measured on `origin/dev`, first on `a6719bf24f` and `2b925562e5`, and after the review on `df7c290fbf`. Those commits change none of this plan's files. Line numbers below are `df7c290fbf`'s. The probes and logs are in the scratchpad: `fu_probe_fanout.py`, `fu_probe_engine_map.py`, `fu_probe_engine_spawned.py`, `fu_probe_asyncio_subprocess_cancel.py`, `fu_p2r_*`, and the reviewer's `rvcs_*`.

- **Python floor:** `requires-python = ">=3.12"`, and `.venv` is 3.12.13, so `TaskGroup` (3.11) is available.
  - Ruff's `target-version = "py39"` (`pyproject.toml:694`) flags `BaseExceptionGroup` as `F821 Undefined name`, and `except*` as invalid syntax.
  - No module under `tldw_Server_API/app` uses `TaskGroup`, `ExceptionGroup` or `except*`.
- **How the two steps bound concurrency:**
  - `map` builds `asyncio.Semaphore(concurrency)` (default 4, `flow.py:321-322`). Each item is a coroutine that acquires it inside `_run_one`, and all items are handed to one `asyncio.gather(..., return_exceptions=False)` (`flow.py:357`). `map` accepts only parallelizable sub-step types: 96 at runtime, from 97 `parallelizable=True` decorators.
  - `parallel` builds `asyncio.Semaphore(max_concurrency)` (default 5, `flow.py:385-388`) and accepts any step type. `run_step` catches `_FLOW_NONCRITICAL_EXCEPTIONS` per step (`flow.py:407-410`), and the steps are gathered with `return_exceptions=not fail_fast`. A refusal found in the outcomes is re-raised (`flow.py:412-416`, TASK-13638).
  - With `fail_fast` on, a noncritical error does not fail the step: it is recorded and appended to `errors`, as `test_control_adapters.py::test_parallel_sanitizes_substep_errors` pins. Only an exception outside the tuple escapes `run_step`.
- **What happens to the siblings on dev** (`fu_probe_fanout.py`: sub-step 0 raises a refusal after 10 ms, the others take 100 ms; concurrency 2):

  | Step | What the siblings did |
  |---|---|
  | `map` over 6 items | Items 1-5 finished after the step raised. |
  | `parallel` with `fail_fast` over 4 steps | Steps 1-3 finished after the step raised. |
  | `parallel` without `fail_fast` | The step waited for steps 1-3, then raised the refusal. |

- **What the engine records** (`fu_probe_engine_map.py`, run awaited on the caller's loop):
  - The run is `failed` with `error` `402: {...}`, and step run `m` is `failed`.
  - Events: `run_started, step_started, step_failed, run_failed`. Then the five siblings' `append_event` calls arrive **after** `run_failed`, and the siblings add 5 artifacts to the failed run.
  - Nothing is recorded per sibling. Sub-steps have no step-run records of their own: they are adapter calls inside the one step run. They reach the run only through hooks the engine puts in their context (`append_event`, `add_artifact`, `record_subprocess`, `engine.py:1844-1894`), under the parent step's `step_run_id`. These hooks are synchronous, so a cancel cannot split one of their DB writes.
- **The engine's own path** (`fu_probe_engine_spawned.py`; `asyncio.run` per run, as `WorkflowScheduler._spawn` does at `engine.py:2381-2390`):
  - With no `on_failure`, the run's loop shutdown cancelled the two running items once `start_run` returned, so nothing finished after `run_failed`.
  - With `on_failure` routing to a 400 ms `delay` step, all five siblings finished during the next step and wrote their events. The run ended `succeeded`, with `step_failed` for the map.
  - So siblings run to completion in these cases:
    - a step that routes `on_failure`;
    - a run awaited on the server loop: the endpoint's inline fallbacks at `endpoints/workflows.py:1808`, `:1835` and `:2238`, and test mode;
    - child runs (`adapters/control/orchestration.py:74`);
    - a step retry, whose attempt 2 overlaps attempt 1's siblings.
  - On the scheduler path without `on_failure`, siblings run only until the run's loop ends, which includes the completion webhook.
- **What a cancelled sub-step does about its own side effects:**
  - It stops at its next `await`, and nothing it already did is undone.
  - No adapter under `Workflows/adapters` catches `CancelledError` or `BaseException` (`git grep`).
  - Work that a sub-step started outside its own task can outlive it:
    - **`media_ingest`** (parallelizable). It starts yt-dlp with `start_process` in its own session (`adapters/media/ingest.py:325`) and polls with `await asyncio.sleep(0.25)` (`:352`). It calls `terminate_process` only on its own `timeout_seconds` or on `is_cancelled`.
      - On dev, a sibling `media_ingest` kept polling and stopped a hung download at its timeout.
      - With the helper alone, it would be cancelled at the poll and nothing would ever stop the process: the engine's kill paths (`engine.py:1637`, `:1947`) look only at running or stale step runs.
      - Reviewer's probe `rvcs_probe_ingest.py`, with the download replaced by `sleep 30` and `timeout_seconds: 5`: on dev the process was gone 6 s after the step raised, and with the helper alone it was still alive.
      - The engine's step timeout already leaks it this way on dev. Task 1 fixes both cases.
    - **`tts` and `multi_voice_tts`** (both parallelizable). They run ffmpeg through `asyncio.create_subprocess_exec` and `await asyncio.wait_for(proc.communicate(), timeout=...)`, and kill it only on `TimeoutError`: `adapters/audio/tts.py:235` and `multi_voice_tts.py:64`, `:153`, `:192`, `:267`, `:304`, `:398`, `:488`. A cancel leaves ffmpeg running to its end without that timeout. Probe `fu_probe_asyncio_subprocess_cancel.py`: the child was alive 1 s after the cancel, and again after `asyncio.run` closed the loop. These are bounded local jobs. Listed and filed (Task 2 Step 1), not fixed here.
    - **`workflow_call` with `wait: true`** (not parallelizable, but `parallel` does not check). A cancelled child run is left `running` for the orphan reaper, because the engine has no `CancelledError` handler (`grep -c CancelledError engine.py` gives 0). Its own `timeout_seconds` already does this on dev. Listed and filed.
    - `run_checked_async` (the audio and video processing adapters) kills and reaps its child on cancellation (`subprocess_utils.py:163`), so those adapters are fine.
    - `media_ingest` also returns as soon as the first file appears in its step directory, without waiting for yt-dlp to exit. That is not about cancellation; it is filed with the others.
- **A sub-step's own `CancelledError`** (reviewer's probe `rvcs_probe_selfcancel.py`):
  - On dev, `parallel` without `fail_fast` recorded it as `None`, and the step succeeded.
  - With the helper alone, it escaped the step as a `CancelledError` while nothing was cancelling the step. The engine catches only `Exception`, so the run would stay `running`.
  - `map` and `parallel` with `fail_fast` already raise it on dev.
- **A defect next to this one, not fixed here:** `map`'s `_run_one` wraps the tracing span and the sub-step call in one `try ... except _FLOW_NONCRITICAL_EXCEPTIONS`, whose fallback ("if tracing fails") calls the sub-step again (`flow.py:340-355`). So a sub-step that raises `AdapterError`, `RuntimeError`, `ValueError` or another member runs twice (probe: adapter calls `['a', 'a']`). Task 2 files it.
- **Tests and docs that pin today's behaviour:**
  - `Usage_Quotas.md:96`, last sentence: "In a `map` step or a `parallel` step with `fail_fast`, sub-steps that have started or are queued run on after the step fails; ...". Task 2 replaces it.
  - `Docs/Design/Workflows.md:114` describes `map` without saying what a failure does.
  - No test pins that siblings run on.
  - These tests pin behaviour that stays, and pass after the change: `test_parallel_sanitizes_substep_errors`, `test_parallel_does_not_drop_a_refused_step` (TASK-13638), `test_map_cancellation` and `test_parallel_cancellation` (the `is_cancelled` hook), and the timing tests for concurrency.
- **RED** (Task 1 Step 2) on unchanged source: `8 failed, 1 passed`.

  | Test | Failure on dev |
  |---|---|
  | map | `assert [1, 2, 3, 4, 5] == []` |
  | parallel with `fail_fast`, refusal and HTTP 500 | `assert [1, 2, 3] == []` each |
  | parallel without `fail_fast` | `TimeoutError` from the test's 5 s bound: the step waits for siblings that wait for the test's release |
  | error collection | `At index 1 diff: None != {'error': 'parallel_step_error'}` |
  | own `CancelledError` | `At index 0 diff: None != {'error': 'parallel_step_error'}` |
  | `media_ingest` | `assert [] == [call(SubprocessTask(...))]` |
  | engine | `'map_item_finished' not in [... 'map_item_finished', ...]` |

  The pass is the step-timeout guard, which pins behaviour dev already has.
- **Prototype** (this plan applied on `df7c290fbf`):
  - The 9 new tests with the existing map integration test: `10 passed`, three runs in a row, about 2 s each.
  - Mutants, each against the two test files: `no_await` (cancel without waiting) fails 4 tests; `no_cancelled_clause` fails 1; `no_ingest_terminate` fails 1.
  - `tests/Workflows`: `1730 passed, 6 skipped` (5 min 43 s).
  - `tests/Docs` with the ratchet test: `213 passed`. Test-quality triage `--enforce`: exit 0, with neither test file in its report.
  - Ruff: `flow.py` and the new test file are clean. `ingest.py` and `test_workflows_map_substeps_integration.py` keep the one I001 each has on dev.
  - `uvx bandit -q -r` on `flow.py` and `ingest.py`: no output, exit 0.

## Global Constraints

- **Python:** `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. **Tests:** `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`.
- **Branch** `fix/workflow-map-parallel-cancel-siblings` from `origin/dev`; the PR targets `dev`. Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **Test first:** Task 1's RED run happens on unchanged source and fails for the reasons above.
- **No `TaskGroup` and no `ExceptionGroup`** (ruff F821 under the py39 target). Do not change ruff's `target-version` in this PR: that would raise new pyupgrade findings across the repository.
- **Unchanged:**
  - `_FLOW_NONCRITICAL_EXCEPTIONS` and its members;
  - what `fail_fast` does with a noncritical error (Decision 1);
  - the output shapes `{"results", "count"}` and `{"results", "count", "errors"}`;
  - the `is_cancelled` hooks;
  - the engine;
  - every adapter except `media_ingest`'s poll.
- **`terminate_process` blocks** (`time.sleep` for its grace), so it runs through `asyncio.to_thread` and never on the loop.
- **Test-quality ratchet (required gate):** `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` exits 0, and `tldw_Server_API/tests/CI/test_test_quality_ratchet.py` gives `1 passed`. A test that asserts only on a list it appended to itself counts as `tautology_suspect`, so each new test also asserts on a result.
- **Lint comparison:** run `$PY -m ruff check --output-format concise <touched .py files>` here and on the same files from `origin/dev`. Export the dev copies with `git show origin/dev:<path>` into a temporary tree holding the repo's `pyproject.toml`, and diff the two outputs with line and column numbers stripped. Expected: only the I001 findings that `ingest.py` and `test_workflows_map_substeps_integration.py` already have. No `ruff --fix`.
- **Line-number baselines:** neither `flow.py` nor `ingest.py` is in `Helper_Scripts/ci/scope_predicate_baseline.txt`.
- **Docs:** `Docs/Operations/Usage_Quotas.md` and `Docs/Design/Workflows.md` have no `Docs/Published` mirror (`refresh_docs_published.sh:165` copies only `Operations/Env_Vars.md` from Operations, and nothing from `Docs/Design`). Edits are anchored on quoted text.
- **Global defaults:** none changes, so there is no symbol to sweep across the tests.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit ids for new tasks, `git add -A backlog/tasks`. Compute the next id right before `task create` (Task 2 Step 1); the companion plan `plan_fu_fva_adjudicator.md` and others in the scratchpad also file tasks.

## Decisions for the owner (recommendation first)

1. **`fail_fast` does not stop on most errors.**
   - The config field's description says "Stop on first error" (`control/_config.py:70`). With `fail_fast` on, a sub-step error in `_FLOW_NONCRITICAL_EXCEPTIONS` (`AdapterError`, `ValueError`, `RuntimeError`, `OSError`, `TimeoutError` and the rest) is recorded as `parallel_step_error`, and the step succeeds. `test_parallel_sanitizes_substep_errors` pins this.
   - **What fail_fast does stop on.** Any exception outside the tuple: provider and transport failures (`ChatAPIError` and its subclasses, `NetworkError`, httpx errors), an `HTTPException`, and the quota refusal. In short, it stops on provider and transport failures, not on adapter or validation errors.
   - **The "Stop on first error" text is not user-visible.** It exists only as the `ParallelConfig.fail_fast` Field description:
     - `/step-types` has no `parallel` schema, and falls back to `{"type": "object", "additionalProperties": true}`;
     - `registry.get_catalog()`, which would publish `config_schema`, has no endpoint caller;
     - the WebUI step registry has no `parallel` config form;
     - no doc mentions `fail_fast` except the `Usage_Quotas.md` sentence this plan replaces.
   - **Recommended:** leave it as it is in this PR.
     - The task is about what happens to the siblings of an error that escapes, and the plan keeps every escape path.
     - Changing what `fail_fast` stops on changes workflow outcomes beyond quotas, and breaks the pinned test.
     - Users cannot have been misled by a description they cannot see.
   - **Alternative:** with `fail_fast`, every sub-step error fails the step and cancels the rest. That needs:
     - `run_step`'s noncritical clause to raise when `fail_fast` is set;
     - `test_parallel_sanitizes_substep_errors` rewritten to expect the failure;
     - a decision on what the step's error text exposes, since today's output hides it behind `parallel_step_error`.

## Review Focus

1. **The sub-step that takes the freed slot starts.**
   - When the failing sub-step leaves the semaphore, the next queued one is woken before the cancel reaches it. It starts, runs to its first `await`, and is cancelled there.
   - Synchronous work it does before that `await` is kept, and the ones queued behind it never start.
   - Pinned by `_never_started(log, 6) == [3, 4, 5]` in `test_map_cancels_the_other_items_when_one_raises`.
2. **Work that outlives the cancel, and what it costs.**
   - A sub-step awaiting `asyncio.to_thread` is cancelled at the `await`, but the thread finishes its call. A claims provider call on the bounded pool (`drain_after_timeout=True`, used by the `claims_extract` step) holds the helper until it returns, up to its 60 s deadline. The step fails later than its first error, but nothing outlives it.
   - A cancelled `media_ingest` holds the helper while `terminate_process` sends SIGTERM and waits its 5 s grace before SIGKILL. A step whose sibling downloads were running therefore fails about 5 s after its first error.
   - The ffmpeg children of `tts` and `multi_voice_tts`, and a `workflow_call` child run, outlive the cancel. They are filed (Task 2 Step 1).
   - **An in-flight LLM call in a cancelled sibling is no longer recorded.** On dev, a sibling admitted before the refusal finished its call, and the call's usage was submitted to the accounting loop. Now a sibling cancelled inside an awaited provider request never gets the response, so nothing is submitted, though the provider may still bill it. A write already submitted survives, because the accounting write is fire-and-forget (`Usage/llm_accounting.py:45-60`). This is an undercount at exactly the limit. It is the counterpart of the overshoot `Usage_Quotas.md:106` documents.
   - Not tested here: these depend on threads, real processes or a provider. The drain and the accounting hand-off are tested where they live.
3. **`fail_fast` off, an error outside the noncritical tuple.**
   - `results[i]` is now `{"error": "parallel_step_error"}`, where dev left `None` because `gather` dropped the exception. A sub-step's own `CancelledError` gets the same marker (M1).
   - A downstream template that tested for `None` now sees the error dict.
   - Pinned by `test_parallel_without_fail_fast_collects_every_other_error` and `test_parallel_without_fail_fast_collects_a_sub_step_that_raises_cancelled_error`.
4. **The step itself is cancelled** (its `timeout_seconds`, or the run's cancel).
   - Dev's `gather` already cancelled the children and waited for them before the cancellation left the step. The reviewer's probe gave `[('cleanup_done', 0), ('cleanup_done', 1), ('step_raised', -1)]`.
   - The helper keeps that. It adds the cancel and the wait only on the path where a sub-step raised.
   - Pinned by `test_a_map_step_that_times_out_leaves_no_item_running`, which passes on dev and after.
5. **Nested fan-outs** (a `parallel` holding a `map` sub-step).
   - The inner step raises its sub-step's own exception, so the outer step treats it like any other. The reviewer's probe: the outer step raised the refusal, and every inner and outer sibling was cancelled.
   - Order and shape of `results` are unchanged, pinned by the existing `test_maps_over_list` and `test_executes_steps_in_parallel`.

## ADR assessment

Not needed. No ADR governs the workflow engine's fan-out (`Docs/ADR/README.md`, searched for workflow, concurrency and asyncio). The change adds no durable rule for module boundaries, persistence, security or worker ownership:
- cancelling a failed step's own sub-steps changes one adapter's behaviour;
- stopping a cancelled download changes another's.

ADR-064 and ADR-067 (quotas, accounting) are not touched.

## Conflicts with other plans

| Place | This plan | Other | How to merge |
|---|---|---|---|
| `Usage_Quotas.md:96` | Task 2 replaces the bullet's last sentence | `plan_fu_fva_adjudicator.md` edits lines 95 and 103 | Adjacent lines: git conflicts for whichever lands second. Keep the landed text and re-apply your own edit by its quoted text. No sentence is shared. |
| `Usage_Quotas.md:96` | as above | `plan_fu_keep_artifact_on_refusal.md` edits lines 93, 95 and 98, two of them next to 96 | Same rule. Its own conflicts table already names TASK-13863. |
| `Usage_Quotas.md` | as above | `plan_fu_anthropic_cache.md` inserts a bullet after line 107; `plan_fu_rg_redis_error.md` edits the Redis line (124) | Not adjacent: both apply cleanly in either order. If one does not, use the same rule. |
| `Usage_Quotas.md` | as above | Open PR #3084 (2300 files) touches the file | Same rule. |
| `flow.py`, `ingest.py`, `Docs/Design/Workflows.md`, the two test files | this plan | No open PR touches them (checked 2026-10-20, including the 11 open PRs with more than 100 files, paged) | None. |

---

### Task 1: map and parallel cancel the other sub-steps when one raises, and a cancelled media_ingest stops its download (AC1, AC2)

**Files:**
- Modify: `tldw_Server_API/app/core/Workflows/adapters/control/flow.py`
  - the imports at lines 8-12;
  - a helper after `_FLOW_NONCRITICAL_EXCEPTIONS` (which ends at line 51);
  - `run_map_adapter`'s `gather` at line 357;
  - `run_parallel_adapter`'s `run_step` clauses at 407-410 and its `gather` block at 412-418.
- Modify: `tldw_Server_API/app/core/Workflows/adapters/media/ingest.py`, the poll's `await asyncio.sleep(0.25)` at line 352.
- Create: `tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflows_map_substeps_integration.py` (import block at lines 1-6; append after line 50).

**Interfaces:**
- Consumes:
  - `exceptions.LLMQuotaExceededError` and `exceptions.AdapterError` (already imported in `flow.py`, line 17);
  - `flow.get_adapter(name)`, which the tests patch;
  - `subprocess_utils.SubprocessTask`, and `start_process` and `terminate_process` as `ingest` imports them (`ingest.py:34`), which the `media_ingest` test patches.
- Produces:
  - `flow._run_cancelling_siblings(coros: list[Coroutine[Any, Any, Any]]) -> list[Any]`, private to `flow.py`;
  - `run_map_adapter`, `run_parallel_adapter` and `run_media_ingest_adapter` keep their signatures and output shapes.

- [ ] **Step 0: start the task and the branch**

```bash
git fetch origin dev --quiet && git switch -c fix/workflow-map-parallel-cancel-siblings origin/dev
mkdir -p Docs/superpowers/plans
cp /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad/plan_fu_workflow_cancel_siblings.md Docs/superpowers/plans/2026-10-20-workflow-map-parallel-cancel-siblings.md
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13863 -s "In Progress" \
  --append-notes "Plan: Docs/superpowers/plans/2026-10-20-workflow-map-parallel-cancel-siblings.md. One helper in control/flow.py gathers the sub-steps as tasks and, on the first exception or its own cancellation, cancels and awaits the rest, then re-raises. Not TaskGroup: ruff targets py39 and flags BaseExceptionGroup as F821. A cancelled media_ingest terminates its yt-dlp download. Measured on dev: map items 1-5 and parallel(fail_fast) steps 1-3 finished after the step raised; through the engine with on_failure, the five siblings wrote events and artifacts to the run after step_failed."
git add -A backlog/tasks Docs/superpowers/plans/2026-10-20-workflow-map-parallel-cancel-siblings.md
```

- [ ] **Step 1: write the failing tests**

Create `tests/Workflows/adapters/test_control_fanout_cancellation.py`:

```python
"""map and parallel cancel their other sub-steps when one raises, and a cancelled sub-step stops its work (TASK-13863)."""

import asyncio
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.exceptions import AdapterError, LLMQuotaExceededError
from tldw_Server_API.app.core.Workflows.adapters.control import flow
from tldw_Server_API.app.core.Workflows.adapters.media import ingest
from tldw_Server_API.app.core.Workflows.subprocess_utils import SubprocessTask

pytestmark = pytest.mark.unit


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


def _upstream_failure() -> HTTPException:
    return HTTPException(status_code=500, detail="upstream failed")


def _recording_substep(log: list[tuple[str, int]], error: Exception, release: asyncio.Event):
    """Sub-step 0 raises ``error`` at its first await; every other one waits for ``release``. ``log`` says what each did."""

    async def substep(config, context):
        n = context.get("item", config.get("n"))
        log.append(("started", n))
        if n == 0:
            await asyncio.sleep(0)
            raise error
        try:
            await release.wait()
        except asyncio.CancelledError:
            log.append(("cancelled", n))
            raise
        log.append(("finished", n))
        return {"text": str(n)}

    return substep


async def _run_then_release(log: list[tuple[str, int]], error: Exception, release: asyncio.Event, call_step) -> Exception:
    """Run the step, mark when it raised, then release any sibling still waiting and let it run.

    The 5 s bound only matters on dev, where a parallel step without fail_fast waits for its siblings,
    and they wait for this release.
    """
    with pytest.raises(type(error)) as raised:
        await asyncio.wait_for(call_step(), timeout=5)
    log.append(("step_raised", -1))
    release.set()
    for _ in range(20):
        await asyncio.sleep(0)
    return raised.value


def _ran_after_the_step_raised(log: list[tuple[str, int]]) -> list[int]:
    """Siblings that finished, or were still being cancelled, after the step had raised."""
    raised_at = log.index(("step_raised", -1))
    return [n for kind, n in log[raised_at:] if kind in {"finished", "cancelled"}]


def _never_started(log: list[tuple[str, int]], total: int) -> list[int]:
    started = {n for kind, n in log if kind == "started"}
    return [n for n in range(total) if n not in started]


async def test_map_cancels_the_other_items_when_one_raises() -> None:
    """On dev, items 1-5 finished after the step had raised."""
    log: list[tuple[str, int]] = []
    error, release = _refusal(), asyncio.Event()
    config = {"items": [0, 1, 2, 3, 4, 5], "step": {"type": "summarize", "config": {}}, "concurrency": 2}

    with patch.object(flow, "get_adapter", return_value=_recording_substep(log, error, release)):
        raised = await _run_then_release(log, error, release, lambda: flow.run_map_adapter(config, {}))

    assert raised is error
    assert _ran_after_the_step_raised(log) == []
    # Item 2 takes the slot item 0 frees and is cancelled at its first await; 3-5 never start.
    assert _never_started(log, 6) == [3, 4, 5]


@pytest.mark.parametrize("error", [_refusal(), _upstream_failure()], ids=["llm_quota_refusal", "http_500"])
async def test_parallel_with_fail_fast_cancels_the_other_steps_when_one_raises(error: Exception) -> None:
    """On dev, steps 1-3 finished after the step had raised."""
    log: list[tuple[str, int]] = []
    release = asyncio.Event()
    config = {
        "steps": [{"type": "summarize", "config": {"n": n}} for n in range(4)],
        "max_concurrency": 2,
        "fail_fast": True,
    }

    with patch.object(flow, "get_adapter", return_value=_recording_substep(log, error, release)):
        raised = await _run_then_release(log, error, release, lambda: flow.run_parallel_adapter(config, {}))

    assert raised is error
    assert _ran_after_the_step_raised(log) == []
    assert _never_started(log, 4) == [3]


async def test_parallel_without_fail_fast_stops_the_other_steps_on_a_quota_refusal() -> None:
    """The refusal fails the step (TASK-13638); on dev the step waited for its siblings, so wait_for timed out."""
    log: list[tuple[str, int]] = []
    error, release = _refusal(), asyncio.Event()
    config = {"steps": [{"type": "summarize", "config": {"n": n}} for n in range(4)], "max_concurrency": 2}

    with patch.object(flow, "get_adapter", return_value=_recording_substep(log, error, release)):
        raised = await _run_then_release(log, error, release, lambda: flow.run_parallel_adapter(config, {}))

    assert raised is error
    assert _ran_after_the_step_raised(log) == []
    assert _never_started(log, 4) == [3]


async def test_parallel_without_fail_fast_collects_every_other_error() -> None:
    """A sub-step error outside the noncritical tuple is collected too; on dev it was dropped as None."""

    async def substep(config, _context):
        if config["n"] == 0:
            raise AdapterError("boom")
        if config["n"] == 1:
            raise _upstream_failure()
        return {"text": "ok"}

    config = {"steps": [{"type": "summarize", "config": {"n": n}} for n in range(3)]}
    with patch.object(flow, "get_adapter", return_value=substep):
        result = await flow.run_parallel_adapter(config, {})

    assert result["results"] == [{"error": "parallel_step_error"}, {"error": "parallel_step_error"}, {"text": "ok"}]
    assert result["errors"] is None


async def test_parallel_without_fail_fast_collects_a_sub_step_that_raises_cancelled_error() -> None:
    """A sub-step's own CancelledError is an error to collect; only the step's own cancellation stops the step."""

    async def substep(config, _context):
        if config["n"] == 0:
            raise asyncio.CancelledError()
        return {"text": "ok"}

    config = {"steps": [{"type": "summarize", "config": {"n": n}} for n in range(3)]}
    with patch.object(flow, "get_adapter", return_value=substep):
        result = await flow.run_parallel_adapter(config, {})

    assert result["results"] == [{"error": "parallel_step_error"}, {"text": "ok"}, {"text": "ok"}]


async def test_a_map_step_that_times_out_leaves_no_item_running() -> None:
    """The engine's step timeout cancels the step, and every item stops before it raises (dev's gather did this too)."""
    log: list[tuple[str, int]] = []
    release = asyncio.Event()  # never set: the items run until they are cancelled
    config = {"items": [1, 2, 3, 4], "step": {"type": "summarize", "config": {}}, "concurrency": 2}

    with patch.object(flow, "get_adapter", return_value=_recording_substep(log, _refusal(), release)):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(flow.run_map_adapter(config, {}), timeout=0.05)
        log.append(("step_raised", -1))
        for _ in range(20):
            await asyncio.sleep(0)

    assert _ran_after_the_step_raised(log) == []
    assert sorted(n for kind, n in log if kind == "cancelled") == [1, 2]


async def test_a_cancelled_media_ingest_terminates_its_download(tmp_path, monkeypatch) -> None:
    """Cancelled mid-poll, because a sibling failed or the step timed out, the step stops yt-dlp; on dev nothing did."""
    monkeypatch.setenv("WORKFLOWS_TMP", str(tmp_path))
    started = asyncio.Event()
    download = SubprocessTask(
        cmd=["yt-dlp"], workdir=tmp_path, stdout_path=tmp_path / "stdout.log", stderr_path=tmp_path / "stderr.log",
        pid=4242, pgid=4242,
    )

    def fake_start_process(_cmd, workdir, log_dir):
        started.set()
        return download

    terminate = MagicMock(return_value=(True, False))
    config = {"sources": [{"uri": "https://media.example.com/clip"}], "timeout_seconds": 60}
    with patch.object(ingest, "is_test_mode", return_value=False), patch.object(
        ingest, "is_url_allowed_for_tenant", return_value=True
    ), patch.object(ingest, "start_process", side_effect=fake_start_process), patch.object(
        ingest, "terminate_process", terminate
    ):
        step = asyncio.create_task(ingest.run_media_ingest_adapter(config, {"tenant_id": "default"}))
        await started.wait()  # the adapter is now in its poll's sleep
        step.cancel()
        with pytest.raises(asyncio.CancelledError):
            await step

    assert terminate.call_args_list == [call(download)]
```

The tests no longer depend on timing:
- every sibling waits on an `asyncio.Event` until the test releases it after the step has raised;
- the twenty `sleep(0)` yields let a sibling left running on dev finish and log it;
- `_ran_after_the_step_raised` also counts `"cancelled"`, so a helper that cancels the siblings without waiting for them fails, because they log their cancel after the step raised.

`summarize` is a registered parallelizable adapter, so `map` accepts it. The tests patch `flow.get_adapter`, which `map` and `parallel` use for their sub-steps.

The `media_ingest` test reaches the download path:
- it bypasses the test-mode skip (`is_test_mode`) and the egress check;
- `start_process` returns a fake task without starting anything;
- `WORKFLOWS_TMP` keeps the step directory under `tmp_path`;
- the adapter makes no `await` before its poll's `sleep`, so `started.wait()` returning means it is there.

In `tests/Workflows/test_workflows_map_substeps_integration.py`, replace the import block (lines 1-6) with:

```python
import asyncio
import json
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Workflows.adapters.control import flow
from tldw_Server_API.app.core.Workflows.engine import WorkflowEngine, RunMode
```

and append:

```python


@pytest.mark.asyncio
async def test_a_failed_map_step_leaves_no_item_running(tmp_path):
    """on_failure lets the run go on; on dev the failed map's other items ran on and wrote to the run (TASK-13863)."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    definition = {
        "name": "map-cancels-items",
        "version": 1,
        "steps": [
            {
                "id": "map1",
                "type": "map",
                "config": {"items": [0, 1, 2, 3, 4, 5], "step": {"type": "summarize", "config": {}}, "concurrency": 2},
                "on_failure": "after",
            },
            # The next step releases the map's items; on dev they are still running when it does.
            {"id": "after", "type": "parallel", "config": {"steps": [{"type": "summarize", "config": {"release": True}}]}},
        ],
    }
    run_id = "run-map-cancels-items"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )
    release = asyncio.Event()

    async def sub_step(config, context):
        if config.get("release"):
            release.set()
            for _ in range(20):
                await asyncio.sleep(0)
            return {"released": True}
        if context["item"] == 0:
            await asyncio.sleep(0)
            raise LLMQuotaExceededError(
                category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
            )
        await release.wait()
        context["append_event"]("map_item_finished", {"item": context["item"]})
        return {"text": str(context["item"])}

    with patch.object(flow, "get_adapter", return_value=sub_step):
        await WorkflowEngine(db).start_run(run_id, RunMode.SYNC)

    run = db.get_run(run_id)
    assert run is not None
    assert run.status == "succeeded"
    assert [(step["step_id"], step["status"]) for step in db.list_step_runs(run_id=run_id)] == [
        ("map1", "failed"),
        ("after", "succeeded"),
    ]
    assert "map_item_finished" not in [event["event_type"] for event in db.get_events(run_id)]
```

The engine resolves `map` and `parallel` through `Workflows.adapters.get_adapter` (`engine.py:21`), not through `flow.get_adapter`. So the patch reaches only the sub-steps: the map's items, and the `after` step's one sub-step, which releases them.

- [ ] **Step 2: run them on unchanged source**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py \
  "tldw_Server_API/tests/Workflows/test_workflows_map_substeps_integration.py::test_a_failed_map_step_leaves_no_item_running"
```

Expected (measured): `8 failed, 1 passed`, failing as the RED table in "What was measured" lists. The parallel-without-`fail_fast` test fails after its 5 s bound. The pass is `test_a_map_step_that_times_out_leaves_no_item_running`.

- [ ] **Step 3: implement**

`flow.py` imports: after `import types` add

```python
from collections.abc import Coroutine
```

After the closing `)` of `_FLOW_NONCRITICAL_EXCEPTIONS` and before `@registry.register(` for `"prompt"`:

```python
async def _run_cancelling_siblings(coros: list[Coroutine[Any, Any, Any]]) -> list[Any]:
    """Run sub-steps together and return their results in order.

    The first sub-step to raise cancels the others, those running and those
    still waiting for a concurrency slot, and its error is raised once they
    have stopped. ``asyncio.gather`` alone leaves them running (TASK-13863).
    """
    tasks = [asyncio.create_task(coro) for coro in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
```

`run_map_adapter`, replace

```python
    results = await asyncio.gather(*[_run_one(i, it) for i, it in enumerate(items)], return_exceptions=False)
```

with

```python
    results = await _run_cancelling_siblings([_run_one(i, it) for i, it in enumerate(items)])
```

`run_parallel_adapter`: replace everything from the noncritical clause in `run_step` to the end of the function, which is

```python
            except _FLOW_NONCRITICAL_EXCEPTIONS:
                results[idx] = {"error": "parallel_step_error"}
                if fail_fast:
                    errors.append("parallel_step_error")

    tasks = [run_step(i, step) for i, step in enumerate(steps)]
    outcomes = await asyncio.gather(*tasks, return_exceptions=not fail_fast)
    refusal = next((o for o in outcomes if isinstance(o, LLMQuotaExceededError)), None)
    if refusal is not None:
        raise refusal  # the user's limit fails the step; it is not dropped as a missing result

    return {"results": results, "count": len(results), "errors": errors if errors else None}
```

with

```python
            except _FLOW_NONCRITICAL_EXCEPTIONS:
                results[idx] = {"error": "parallel_step_error"}
                if fail_fast:
                    errors.append("parallel_step_error")
            except LLMQuotaExceededError:
                raise  # the user's limit fails the step, with or without fail_fast (TASK-13638)
            except Exception:
                if fail_fast:
                    raise
                results[idx] = {"error": "parallel_step_error"}
            except asyncio.CancelledError:
                if fail_fast or asyncio.current_task().cancelling():
                    raise  # the step, or a failed sibling, cancelled it
                results[idx] = {"error": "parallel_step_error"}  # the sub-step's own CancelledError

    await _run_cancelling_siblings([run_step(i, step) for i, step in enumerate(steps)])
    return {"results": results, "count": len(results), "errors": errors if errors else None}
```

How the clauses apply:
- `LLMQuotaExceededError` is an `HTTPException`, not a member of `_FLOW_NONCRITICAL_EXCEPTIONS`, so the clauses apply in the order written.
- `asyncio.CancelledError` is a `BaseException`, so only its own clause sees it.
- `cancelling()` is non-zero when the helper or the step's own cancellation cancelled this task. A `CancelledError` the sub-step raised itself leaves it at 0.

`ingest.py`, the poll loop: replace

```python
                await asyncio.sleep(0.25)
```

with

```python
                try:
                    await asyncio.sleep(0.25)
                except asyncio.CancelledError:
                    # A cancelled step (a map or parallel sibling failed, or the step timed out) stops the download.
                    await asyncio.to_thread(terminate_process, task)
                    raise
```

This `sleep` is the only `await` in the loop. `terminate_process` blocks for its grace period, so it runs on a worker thread. The thread finishes even if a second cancel interrupts the `await`.

- [ ] **Step 4: run the new tests, the mutants, then the neighbourhood**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py \
  tldw_Server_API/tests/Workflows/test_workflows_map_substeps_integration.py
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows
```

Expected:
- `10 passed`: the 9 new tests and the existing map integration test (measured, three runs, about 2 s each);
- then `1730 passed, 6 skipped` (measured on `df7c290fbf` with this task applied, 5 min 43 s).

Before attributing any other failure to this change, run it on unchanged `origin/dev`.

Check that the tests see what they claim. Edit a scratch copy, never the branch:
- delete the helper's `await asyncio.gather(*tasks, return_exceptions=True)`, so the siblings are cancelled but not awaited;
- delete `run_step`'s `CancelledError` clause;
- delete `ingest.py`'s `await asyncio.to_thread(terminate_process, task)` line.

Measured with `fu_p2r_mutants.py`: 4 tests fail, then 1, then 1.

Then the lint comparison (Global Constraints).

- [ ] **Step 5: commit**

```bash
git add -A backlog/tasks tldw_Server_API/app/core/Workflows/adapters/control/flow.py tldw_Server_API/app/core/Workflows/adapters/media/ingest.py \
  tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py tldw_Server_API/tests/Workflows/test_workflows_map_substeps_integration.py
git commit -m "fix(workflows): map and parallel cancel their other sub-steps when one raises, and a cancelled media_ingest stops its download (TASK-13863)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: the operator page, the workflow design note, two follow-ups, and close-out

**Files:** Modify `Docs/Operations/Usage_Quotas.md` and `Docs/Design/Workflows.md`. Backlog: TASK-13863 and two new tasks.

**Interfaces:**
- Consumes: Task 1's behaviour.
- Produces: the follow-up ids `$MAP_RERUN` and `$SUBPROC`. The operator page names `$SUBPROC`.

- [ ] **Step 1: file the follow-ups.** Compute the highest id with the AGENTS.md recipe, paged for open PRs with more than 100 files, then create both tasks:

```bash
git fetch origin dev --quiet
HIGHEST=$( {
  git ls-tree -r --name-only origin/dev backlog/
  gh pr list --state open --limit 500 --json files --jq '.[].files[].path'
  for n in $(gh pr list --state open --limit 500 --json number,changedFiles --jq '.[] | select(.changedFiles > 100) | .number'); do
    gh api --paginate "repos/rmusser01/tldw_server/pulls/$n/files" --jq '.[].filename'
  done
} | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1 )
echo "$HIGHEST"
MAP_RERUN="TASK-$((HIGHEST + 1))"; SUBPROC="TASK-$((HIGHEST + 2))"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Workflow map runs a sub-step twice when it raises a noncritical error" --id "$MAP_RERUN" -l workflows,backend \
  -d "run_map_adapter's _run_one (Workflows/adapters/control/flow.py) wraps the tracing span and the sub-step call in one try whose except _FLOW_NONCRITICAL_EXCEPTIONS branch, meant for a tracing failure, calls the sub-step again. A sub-step that raises AdapterError, RuntimeError, ValueError, OSError, TimeoutError or another member of the tuple therefore runs twice before the map step fails (probe while planning TASK-13863: adapter calls ['a', 'a']). A sub-step with a side effect, such as webhook or notify, repeats it." \
  --ac "A map sub-step that raises runs once, and the map step fails with its error, with a test" \
  --ac "A tracing failure still runs the sub-step once, with a test"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Cancelled or finished workflow sub-steps leave their subprocesses and child runs going" --id "$SUBPROC" -l workflows,backend \
  -d "Found while planning TASK-13863, which makes a failed map or parallel step cancel its other sub-steps and fixed media_ingest's poll. Still open: (1) tts (adapters/audio/tts.py near line 235) and multi_voice_tts (adapters/audio/multi_voice_tts.py, 7 sites) start ffmpeg with asyncio.create_subprocess_exec and kill it only on TimeoutError from wait_for(proc.communicate()), so a cancel (the engine's step timeout, or a failed sibling) leaves ffmpeg running to its end without its timeout; a probe showed the child alive after the cancel and after asyncio.run closed the loop. run_checked_async in subprocess_utils already kills and reaps on cancel. (2) workflow_call with wait: true (usable inside parallel, which does not check parallelizable): a cancelled child run is left running for the orphan reaper, because the engine has no CancelledError handler; its own timeout_seconds already does this. (3) media_ingest returns as soon as the first file appears in its step directory, without waiting for yt-dlp to exit." \
  --ac "A cancelled tts or multi_voice_tts sub-step kills the ffmpeg process it started, with a test" \
  --ac "A cancelled workflow_call child run ends in a terminal status instead of staying running, with a test" \
  --ac "media_ingest waits for yt-dlp to exit, or terminates it, before it returns, with a test"
git add -A backlog/tasks
```

Measured on 2026-10-20: the highest id was `13866` on `origin/dev` when drafting, and `13870` in the review's dry run. Recompute at execution time, because other plans file tasks too.

- [ ] **Step 2: `Usage_Quotas.md`, the workflow bullet (line 96).** Replace its last sentence

  > In a `map` step or a `parallel` step with `fail_fast`, sub-steps that have started or are queued run on after the step fails; any LLM call they make after the refusal is refused before it reaches the provider.

  with the following, writing the computed id in place of `$SUBPROC`:

  > In a `map` or `parallel` step, the refusal cancels the step's other sub-steps before the step fails: each one stops at its next wait, and what it had already done is kept. A download that a `media_ingest` sub-step started is stopped; an ffmpeg command that a TTS sub-step started runs to its end ($SUBPROC).

  The bullet's earlier sentence "A `parallel` step fails when one of its steps is refused." stays: with `fail_fast` off, the refusal is still the only error that fails the step.

- [ ] **Step 3: `Docs/Design/Workflows.md:114`.** Replace

  > - `map`: Fan-out over a list and apply a nested step with optional concurrency; returns a list of `results`.

  with

  > - `map`: Fan-out over a list and apply a nested step with optional concurrency; returns a list of `results`. If an item's step raises, the other items are cancelled and the map step fails with that error.

- [ ] **Step 4: the checks**

```bash
$PY Helper_Scripts/ci/test_quality_triage.py --enforce
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/Docs
uvx bandit -q -r tldw_Server_API/app/core/Workflows/adapters/control/flow.py tldw_Server_API/app/core/Workflows/adapters/media/ingest.py
```

Expected:
- the triage exits 0 with "OK — no offenses beyond the baseline";
- `213 passed` (both measured with Steps 2 and 3 applied);
- bandit prints nothing and exits 0 (measured). If `uvx` is unavailable, record the skip.

Then the lint comparison.

- [ ] **Step 5: close TASK-13863**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13863 \
  --append-notes "RED on origin/dev: 8 failed, 1 passed. map: items [1, 2, 3, 4, 5] ran after the step raised; parallel(fail_fast), refusal and HTTP 500: [1, 2, 3]; parallel without fail_fast: the step waited for its siblings (the test's 5 s bound expired); error collection: an HTTPException(500) left None in results, and so did a sub-step's own CancelledError; media_ingest: terminate_process never called after a cancel; engine with on_failure: map_item_finished events after step_failed. The pass is the step-timeout guard, which dev already met. GREEN: 10 passed (9 new and the existing map integration test). Mutants: cancel-without-await fails 4 tests, dropping the CancelledError clause 1, dropping media_ingest's terminate 1. tests/Workflows: 1730 passed, 6 skipped. Test-quality ratchet exit 0; tests/Docs 213 passed; ruff: nothing new; bandit: clean." \
  --append-notes "Decision 1, recommended option: fail_fast keeps today's meaning (a noncritical sub-step error is recorded, not raised). Not TaskGroup: ruff targets py39 and reports BaseExceptionGroup as F821; the helper gathers tasks and cancels the rest on the first exception, as AC1 allows. Follow-ups: $MAP_RERUN (map runs a sub-step twice on a noncritical error); $SUBPROC (ffmpeg in tts and multi_voice_tts, workflow_call child runs, media_ingest's early return)." \
  --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "map and parallel run their sub-steps through one helper. On the first exception, or its own cancellation, it cancels the other sub-steps, running or queued, waits for them to stop and re-raises, so no sub-step writes to the run after the step fails. A cancelled media_ingest terminates the download it started. parallel without fail_fast still collects sub-step errors, now including those outside the noncritical tuple and a sub-step's own CancelledError, which dev dropped as None; an LLM quota refusal fails the step and cancels the rest. ffmpeg started by the TTS adapters and workflow_call child runs still outlive a cancel ($SUBPROC). Usage_Quotas.md and the workflow design note say so." \
  -s Done
git add -A backlog/tasks
```

The tallies are the prototype's. If a run gives other numbers, write those, with the dev baseline for any failure. If the owner takes Decision 1's alternative, re-plan before Step 5: it changes Task 1.

- [ ] **Step 6: commit and PR**

```bash
git add Docs/Operations/Usage_Quotas.md Docs/Design/Workflows.md
git commit -m "docs(workflows): a failed map or parallel step cancels its other sub-steps; close TASK-13863" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push -u origin fix/workflow-map-parallel-cancel-siblings
gh pr create --base dev --title "Workflow map and parallel cancel their other sub-steps when one fails (TASK-13863)" --body-file /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad/pr_body_13863.md
```

The PR body needs:
- the measured before and after (the probe tables above);
- why the helper is not a `TaskGroup`;
- the `media_ingest` change;
- Decision 1 as taken, and the two follow-up ids;
- a "## Change summary" section. ADR-004: AI-authored PRs need the owner's own Change summary, and no recorded waiver names TASK-13863, so ask the owner for it;
- the attribution line `🤖 Generated with [Claude Code](https://claude.com/claude-code)` at the end.

Then:
- request a clean review (`/superpowers:requesting-code-review` in a fresh subagent) and address it before merging;
- check `gh variable get MERGE_QUEUE` and merge as `CLAUDE.md` says for that mode.

## Out of scope

- **The double run in `map`** on a noncritical sub-step error: filed in Task 2 Step 1.
- **ffmpeg in the TTS adapters, `workflow_call` child runs, and `media_ingest`'s early return:** filed in Task 2 Step 1.
- **What `fail_fast` stops on** (Decision 1).
- **`web_search`'s page fetch** (`adapters/rag/search.py:432`). It gathers without cancelling too, but each fetch catches its own errors, makes no LLM call and is not a sub-step.
- **Per-sibling records.** The engine records the failed step and its first error. Cancelled sub-steps get no event of their own, as no sub-step does today.
