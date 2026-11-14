# Workflow Sub-Step Cleanup and Detached-Run ChaCha Operations (TASK-13886, TASK-13887, TASK-13891, TASK-13897)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Re-review folded in (r3, 2026-10-22):** the scoped re-review of r2 is `<scratchpad>/rereview_plan_fu2_workflows.md`.
- **Verdict:** ready with fixes, with 0 Blocker, 1 Important and 2 Minor findings. All three are folded in. r2 is kept as `plan_fu2_workflows.r2.md`.
- **Re-measurement:** the changed tests were re-run on `e06eeec68c`. The r3 prototype is `<scratchpad>/fu2wf_prototype_r3.diff`.

Fixes:
- **I-1:** the engine-level budget test's step gets `timeout_seconds: 3`, and it asserts that the step directory holds only `clip.mp4.part` before the `.proc` check. With 1 s, a cold process timed out on dev too.
  - Measured with the file alone in a fresh process: dev `4 failed, 10 passed`, the budget test failing with `('succeeded', None) == ('failed', '')`; r2's app code and r3's both `14 passed`.
- **m-1:** Decision 1's Cost now names the WebUI limit, and weighs it against the alternatives. The editor has no step-level timeout control, so a WebUI author raises the budget only through the API or a JSON import. `Docs/Design/Workflows.md` says so.
- **m-2:** both clauses read `is_cancel_requested` under `contextlib.suppress`, defaulting to "cancelled", so a failed read propagates the `CancelledError` as on dev.
  - The suppressed set is `_CANCEL_FLAG_READ_ERRORS`: `_WF_NONCRITICAL_EXCEPTIONS`, plus `sqlite3.Error` and the backend `DatabaseError`.
  - Why the extra types: the ruling named `_WF_NONCRITICAL_EXCEPTIONS`, but that tuple holds neither the raw `sqlite3.Error` that `WorkflowsDatabase.is_cancel_requested` raises on SQLite nor the PostgreSQL backend's `DatabaseError`. With it alone, the stated outcome would not hold.
  - A new test, for both entries, makes the read raise `sqlite3.OperationalError("database is locked")`. It fails against r2's unguarded read (`2 failed`) and passes on dev and r3.

**Review folded in (r2, 2026-10-22):** the independent review and dry run are in `<scratchpad>/review_plan_fu2_workflows.md`.
- **Verdict:** not ready, with 1 Blocker, 3 Important and 11 Minor findings. All are folded in, with the coordinator's rulings. r1 is kept as `plan_fu2_workflows.r1.md`.
- **Re-measurement:** every changed step was re-run in the planner's worktree on `origin/dev` `e06eeec68c`, which had not moved. The r2 prototype is `<scratchpad>/fu2wf_prototype_r2.diff`.

Fixes:
- **B1, the scope-predicate baseline.** Task 6 shifts three `Workflows_DB.py` lines that `Helper_Scripts/ci/scope_predicate_baseline.txt` lists. Task 6 now regenerates the baseline and commits it, and `tests/lint/test_scope_predicate_ratchet.py` runs in Task 6 and Task 7. No other baseline under `Helper_Scripts/ci/` or `tests/` pins a touched file by line (measured).
- **I1, cancel through the run-cancel API.**
  - `engine.cancel()` cancels no task. So both new clauses also re-raise when `self.db.is_cancel_requested(run_id)`.
  - A test cancels through `engine.cancel()` for each entry point.
  - Review Focus 4 is corrected.
  - The pre-existing overwrite is filed as TASK-<N4>: any step error after an API cancel ends the run `failed`/`invariant_violation`.
- **I2, the `continue_run` guard.** Both AC 3 guards run for `start_run` and `continue_run`. The mutants now run once per clause, and each is killed by exactly one test.
- **I3, the download budget.** It is now Decision 1. Review Focus 1 and `Docs/Design/Workflows.md` describe what really happens, and an engine-level test pins it:
  - the budget is the step's `timeout_seconds`, shared by the step's sources;
  - a longer download fails the step with `step_timeout`.

Minors:
- **m1:** `task normalize --check` takes the touched task files, not the directory.
- **m2:** the id lock and the PR paging are spelled out (Setup).
- **m3:** the `plan_fu2_chacha_runtime.md` rows are added, and its effect on the voice command is stated.
- **m4:** the human-step timeout task is now created outside the operation, with a test.
- **m5:** a `media_ingest` download killed by the run's cancel reports `cancelled`.
- **m6:** the `pgrep` pattern is anchored.
- **m7:** the AST guard also catches `from asyncio import create_subprocess_exec` and `loop.subprocess_exec`.
- **m8:** the AC 3 probe now patches `ChaChaNotes_DB.current_connection_state` and has a positive control. The lookup count is not used as evidence.
- **m9:** mypy is recorded as a delta.
- **m10:** ADR-080's text names the notes-graph pattern.
- **m11:** the `Usage_Quotas.md` sentence is reworded.

**Goal:**
- A `map` item that raises runs once.
- A step that raises its own `CancelledError` fails the run instead of leaving it `running`.
- A cancelled `tts` or `multi_voice_tts` step kills its ffmpeg, and `media_ingest` returns only after yt-dlp has exited.
- A run that a detached engine task continues no longer inherits its creator's closed ChaCha operation, so a notes step in it works on PostgreSQL.

**Architecture:**
- **TASK-13886.**
  - `map`'s `_run_one` resolves the adapter first, then sets `started = True` inside the tracing span.
  - Its noncritical `except` re-raises once the sub-step has started.
  - It falls back to an untraced run only when the span failed to open.
- **TASK-13891.**
  - Both engine step loops (`_start_run_impl` and `_continue_run_impl`) gain an `except asyncio.CancelledError` clause.
  - It re-raises, as before, in two cases:
    - the run's task has cancel requests pending (`current_task().cancelling() > 0`: a cancelled run task, for example at shutdown);
    - the run was cancelled through the run-cancel API (`self.db.is_cancel_requested(run_id)`). `engine.cancel()` cancels no task, and on dev that case also kept the run `cancelled`.
  - The flag read is suppressed for `_CANCEL_FLAG_READ_ERRORS`: the noncritical tuple, plus `sqlite3.Error` and the backend `DatabaseError`. It defaults to "cancelled", so a failed read propagates the `CancelledError` as on dev.
  - The step timeout never reaches the clause: `wait_for` turns it into `TimeoutError`.
  - Otherwise the step raised `CancelledError` itself. The clause then records `AdapterError("step_raised_cancelled_error")` as the step's failure.
  - The fix is in the engine, not in `_run_cancelling_siblings`, because a plain step that raises its own `CancelledError` leaves the run `running` too (measured below).
- **TASK-13887 (1).**
  - The 8 direct `asyncio.create_subprocess_exec` sites in `tts.py` and `multi_voice_tts.py` call the existing `subprocess_utils.run_checked_async(cmd, timeout=..., check=False)`. It already kills and reaps its child on a timeout or a cancel.
  - `subprocess.TimeoutExpired` joins each module's noncritical tuple, so a timeout still returns the same failure value.
- **TASK-13887 (3).**
  - `SubprocessTask` keeps its `Popen` (`proc`).
  - `media_ingest` polls `task.proc.poll()` instead of "a file appeared", so it returns only after yt-dlp has exited.
  - An exit that the run's cancel caused (`is_cancelled()`) returns `{"__status__": "cancelled"}`.
  - A non-zero exit, or an exit with no file, is reported as `download_failed` with the exit code.
  - The download's budget is the step's `timeout_seconds` (Decision 1).
- **TASK-13887 (2).** `workflow_call` cannot create a child run on dev: it imports a module that does not exist. AC 2's premise is wrong, so the AC is reworded through backlog-py and its requirement moves to a new task that repairs `workflow_call`.
- **TASK-13897.**
  - `operation_scope.outside_chacha_operation()` is a context manager that clears the inherited operation (`_current_operation.set(None)`) and restores it on exit.
  - `WorkflowEngine.continue_run` runs inside it. Every detached path (the four routes, the human-step timeout, the orphan reaper, the research-wait bridge) goes through `continue_run`.
  - The human-step timeout task is created inside it too. `create_task` copies the context, so the task, which can sleep for days, holds no closed operation.
  - The run is not wrapped in an operation of its own: its concurrent branches would share one connection and lose writes (ADR-080, TASK-13861 plan F13).
  - AC 2 also needs a pre-existing PostgreSQL fix. `WorkflowsDatabase` stores step and run outputs with a plain `json.dumps`, and a notes step on PostgreSQL returns `datetime` values, so the run failed with `Object of type datetime is not JSON serializable`, detached or not. `update_run_status` and `complete_step_run` now pass `default=_json_default`, which writes a `date` or `datetime` as ISO text.
  - The 7 new lines shift three `Workflows_DB.py` entries in `scope_predicate_baseline.txt`, so Task 6 regenerates the baseline (B1).

**Tech Stack:** Python 3.12 (`requires-python = ">=3.12"`), asyncio (`Task.cancelling()`), psycopg_pool through the repo's `pg_database_config` fixture, pytest (`asyncio_mode = "auto"`).

**Spec:**
- `backlog/tasks/task-13886 - Workflow-map-runs-a-sub-step-twice-when-it-raises-a-noncritical-error.md` (AC 1 runs once and fails with its error; AC 2 a tracing failure still runs it once).
- `backlog/tasks/task-13887 - Cancelled-or-finished-workflow-sub-steps-leave-their-subprocesses-and-child-runs-going.md` (AC 1 ffmpeg killed; AC 2 cancelled `workflow_call` child terminal; AC 3 `media_ingest` waits for or terminates yt-dlp).
- `backlog/tasks/task-13891 - A-workflow-sub-step-that-raises-its-own-CancelledError-leaves-the-run-running.md` (AC 1 map; AC 2 parallel `fail_fast`; AC 3 real cancellations still propagate).
- `backlog/tasks/task-13897 - Workflows-engine-detached-run-tasks-inherit-the-caller-s-ChaCha-operation.md` (AC 1 no inherited operation; AC 2 PostgreSQL notes test; AC 3 RAG search audit).
- `Docs/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md`: Decision, and Consequences "Detached tasks". `Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md`: F13, and Review Focus 3.
- Source reviews: `<scratchpad>/clean3290_review.md` (M2, and TASK-13886 confirmed) and `<scratchpad>/clean3295_review.md` (I1). PR #3290 filed TASK-13886, 13887 and 13891; PR #3295 filed TASK-13897.

## What was measured

All on `origin/dev` **`e06eeec68c`** (merge of #3300), in this planner's isolated worktree, with Python 3.12.13. Line numbers are `e06eeec68c`'s. Probes and logs are in `<scratchpad>`: `fu2wf_probe_workflow_call.py`, `fu2wf_probe_rag_chacha.py`, `fu2wf_ruffcmp.py`, `fu2wf_mypy_dev/`, `fu2wf_branch_suite.log`. Every test below was written into the worktree, run RED on dev's sources, then run GREEN with this plan's code. The worktree is reset at the end.

The whole r3 prototype is `<scratchpad>/fu2wf_prototype_r3.diff` (21 files, +781/-137, including the regenerated baseline), for the dry run; r2's is `fu2wf_prototype_r2.diff` and r1's `fu2wf_prototype.diff`. Every `python` and `text` block in this plan appears verbatim in it or in dev's copies, apart from the one deliberate `<T>`/`<FAIL>` template in Task 3 Step 4 (`fu2wf_check_blocks.py`).

- **TASK-13886** (`adapters/control/flow.py:361-376`).
  - One `try` wraps the `workflows.map.item` span and the adapter call. Its `except _FLOW_NONCRITICAL_EXCEPTIONS` ("If tracing fails") calls the adapter again.
  - RED: a sub-step raising `RuntimeError` gives `assert [0, 0] == [0]`.
  - The span-failure guard passes on dev, which already falls back once.
- **TASK-13891** (`engine.py:787-840` and `:1357-1373`). Each step attempt's `try` around `asyncio.wait_for(self._run_step_adapter(...))` catches only `asyncio.TimeoutError`, `_WF_NONCRITICAL_EXCEPTIONS` and `Exception`.
  - A `CancelledError` the step raises itself leaves `start_run` or `continue_run`, and the run stays `running` (`probe3290_engine.py` on dev).
  - RED: `6 failed, 8 passed`. Each failure is `asyncio.exceptions.CancelledError` raised out of `start_run` or `continue_run`, for `map`, `parallel` with `fail_fast`, and a plain step.
    - The plain step was not in the task text. It is why the fix goes in the engine.
  - The 8 passes are the AC 3 guards, each run for both entries: the step timeout, a cancelled run task, a cancel through the API, and an unreadable cancel flag. Dev already satisfies them.
  - **An unreadable cancel flag (re-review m-2).**
    - r2 read `is_cancel_requested` unguarded inside the `except` clause. With the read raising `sqlite3.OperationalError("database is locked")`, the new guard failed for both entries (`2 failed`): the DB error replaced the `CancelledError`.
    - r3 suppresses `_CANCEL_FLAG_READ_ERRORS` around the read, defaulting to "cancelled", so the `CancelledError` propagates and the run stays `running`, as on dev.
    - The engine's `_WF_NONCRITICAL_EXCEPTIONS` alone does not cover the measured failure types: `sqlite3.Error` is raised raw on SQLite (`Workflows_DB.py:2254-2262`), and the PostgreSQL backend raises `backends.base.DatabaseError` subclasses (`postgresql_backend.py:1159-1172`). So r3 adds both.
  - **The run-cancel API (review I1).**
    - `engine.cancel()` (`engine.py:1625-1671`, behind `POST .../runs/{id}/control` `cancel`) sets `cancelled` and `cancel_requested`, and kills recorded subprocess groups. It cancels no asyncio task.
    - With r1's clause, which tested only `cancelling()`, the API-cancel guard failed for both entries (`2 failed, 10 passed`, `DID NOT RAISE CancelledError`): the run went `failed`/`invariant_violation`.
    - With the `is_cancel_requested` half, the run stays `cancelled`, as on dev.
  - **Already on dev (reviewer's `rv_fu2wf_probe_cancel_api.py`, re-run on r2):** after an API cancel, a step that raises `RuntimeError` gives `status=failed error=invariant_violation`, with events `['run_cancelled', 'step_failed', 'transition_rejected', 'run_failed']`.
    - The failure path's `_update_run_status_guarded(status="failed")` from `cancelled` is rejected, and its default `on_reject="fail_run"` overwrites the run.
    - Filed as TASK-<N4>, not fixed here.
- **TASK-13887 (1).** `tts.py:235-254` and `multi_voice_tts.py:64, :153, :192, :267, :304, :398, :488` call `asyncio.create_subprocess_exec` and kill only on `TimeoutError`.
  - These are the only direct subprocess calls under `Workflows/adapters`. Every other adapter uses `run_checked_async` (`subprocess_utils.py:134-169`), whose `except BaseException` reaps the child.
  - RED: `3 failed`:
    - the multi_voice_tts test: `assert not True  + where True = _is_running(83241)`;
    - the tts test: the same, with pid 83259;
    - the AST guard: `Left contains 8 more items, first extra item: 'audio/multi_voice_tts.py:64'`.
- **TASK-13887 (3)** (`adapters/media/ingest.py:340-362`).
  - The poll breaks on `any(step_dir.glob("*.*"))`, which matches yt-dlp's `.part` file, and the adapter returns `downloaded` with the download still running.
  - `start_process` (`subprocess_utils.py:26-71`) drops the `Popen` object, so nothing could poll for exit.
  - RED with `-k "media_ingest or download"`, in a fresh process: `3 failed, 2 passed`. The failures:
    - `['clip.mp4.part'] == ['clip.mp4']`;
    - `[('timeout', None)] == [('download_failed', 1)]`, which waits out `timeout_seconds`;
    - the engine-level budget test: `('succeeded', None) == ('failed', '')`. On dev the run "succeeded" at the `.part` file.
  - **The budget test's step has 3 s (re-review I-1).** With r2's 1 s, an engine's first run in a fresh process reached `start_process` after 1.1-1.5 s, past the deadline. Dev's run then also ended `step_timeout`, and the test failed on dev only at `.proc`, an attribute dev lacks: not the behaviour pinned.
    - With 3 s, each in a fresh process: the budget test alone gives `('succeeded', None) == ('failed', '')` on dev and passes with the fix; the whole file gives `4 failed, 10 passed` on dev and `14 passed` with r2's app code and with r3's.
    - The test also asserts the step directory holds only `clip.mp4.part`, so "returned before yt-dlp finished" does not rest on `.proc`.
  - The passes are TASK-13863's cancel test and the new run-cancel guard (m5). Dev also reports `cancelled` there, on the poll's next `is_cancelled()` check.
  - With r1's poll (no cancel check after the exit), the run-cancel guard failed: the source was recorded as `download_failed` with returncode -15.
  - **The download budget (review I3; `rv_fu2wf_probe_ingest_timeout.py` and the new engine test).**
    - The engine's `wait_for` uses the step-level `timeout_seconds` (default 300). `start_run` copies it into the config with `setdefault`, and `continue_run` overwrites the config's value with it.
    - The adapter's own `start_ts` (`ingest.py:87`) is set once, so all of a step's sources share the one budget. The engine's deadline starts first, so it fires first: a config-level `timeout_seconds` above the step's cannot extend it.
    - A download that outlasts the budget is cancelled at the poll. `terminate_process` sends SIGTERM, waits its 5 s grace, then SIGKILLs. The step fails with `step_timeout`, and so does the run unless `on_failure` routes around it.
    - Reviewer's probe, with step `timeout_seconds: 2`: `elapsed=7.0s status=failed`, events `['run_started', 'step_started', 'step_timeout', 'step_failed', 'run_failed']`.
- **TASK-13887 (2)** (`adapters/control/orchestration.py:48, :67, :74`). `fu2wf_probe_workflow_call.py`, real adapter, no injected module:
  - `import: No module named 'tldw_Server_API.app.core.Workflows.workflows_db'`
  - `create_run accepts status=: False`
  - `wait=True: {'error': 'workflow_call_error', 'result': None}`, and `wait=False`: the same.
  - The module has never existed in git history. The adapter dates from 2026-01-31 (`e982ffe41f`). Its unit tests inject a fake `sys.modules` entry.
  - The WebUI step registry offers the step (`apps/packages/ui/src/components/WorkflowEditor/step-registry.ts`).
  - So no child run is ever created, and AC 2 cannot be RED-tested. The repair (DB handle, definition snapshot, tenant and user check, call cycles) is its own task.
- **TASK-13897 AC 1.**
  - The detached `continue_run` tasks are at `engine.py:1807` (human-step timeout), `:2054` (orphan reaper, reached from `start_run` → `_reap_orphans` at `:688`), `endpoints/workflows.py:3503, :4234, :4358, :4575`, and `research_wait_bridge.py:58`. The bridge is not named in the task.
  - `Scheduler/handlers/workflows.py:157` awaits `continue_run` inline, in no operation.
  - RED: `4 failed`.
    - The request-task, human-step-timeout and orphan-reaper paths each fail with `('failed', 'The ChaCha operation has finished') == ('succeeded', None)`. The step joins with `chacha_operation()`.
    - The timeout-task test fails with `assert <...ChaChaOperation object ...> is None`: the task, which sleeps for the step's timeout, holds its creator's operation (review m4).
    - With r1's engine (`continue_run` fixed, the timeout task not), the timeout-task test still fails: `1 failed, 3 passed`.
- **TASK-13897 AC 2** (PostgreSQL, `tldw_postgres_test` through `pg_database_config`).
  - RED on dev: `('failed', 'Unexpected error initializing DB for user 1: The ChaCha operation has finished')`.
    - The per-call `CharactersRAGDB` raised inside `NotesInteropService._get_db`.
    - Its `CharactersRAGDBError` is outside `_KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS`, so the run failed. The task text expected a `notes_error` output.
  - With Task 5 only, the same test fails with `TypeError: Object of type datetime is not JSON serializable`, from `Workflows_DB.update_run_status` (`:2120`).
  - A control probe on dev's path, a SYNC `start_run` inside an open `chacha_operation(independent=True)`, gave `failed 'Object of type datetime is not JSON serializable'`, with step runs `['succeeded', 'running']`.
    - So any notes step that returns a note on PostgreSQL fails the run on dev.
    - Its step run stays `running`, because `complete_step_run`'s `json.dumps` (`:2513`) fails inside a `suppress`. The orphan reaper would later requeue it.
  - SQLite unit RED: `TypeError: Object of type datetime is not JSON serializable`.
- **TASK-13897 AC 3** (`adapters/rag/search.py:129`).
  - The adapter passes only `media_db_path`. `MultiDatabaseRetriever.__init__` (`RAG/rag_service/database_retrievers.py:4745-4815`) builds the notes, character, chat-history, world-book and dictionary retrievers only for `notes_db`, `character_cards_db`, `world_books_db` and `chat_dictionaries_db` paths, or a passed `chacha_db`.
  - `fu2wf_probe_rag_chacha.py` ran the real adapter with `sources` set to media_db, notes, chats, characters, world_books and dictionaries. It recorded `CharactersRAGDB constructed: 0`.
    - **Positive control:** the workflow notes adapter (`action: list`), through the same recorder, gave `constructed: 1`, so the recorder works.
    - **Not used as evidence:** the operation-lookup count. It now patches the name `ChaChaNotes_DB` imports, `ChaChaNotes_DB.current_connection_state` (review m8; r1 patched `operation_scope`'s, which intercepted nothing), but on SQLite with no open operation even the control records 0 lookups.
  - `RAGSearchConfig` has no `sources` field, and the step registry offers none. So the adapter reaches no ChaCha data, from any run.
- **GREEN, with this plan's code:**

  | Test set | Result |
  |---|---|
  | Task 1 Step 4 (`-k "map or Map"` over three files) | `15 passed` |
  | Task 2 Step 4: the engine file (14), the fan-out file as Task 2 leaves it (10), `test_workflows_map_substeps_integration.py` (3) | `27 passed` |
  | Task 3 tests, plus `test_audio_adapters.py`, `test_tts_workflow_normalization.py`, `Watchlists/test_audio_briefing_workflow.py` and `TTS/unit/test_audio_converter_sanitizer.py` | `140 passed` |
  | Task 4 Step 4: `test_control_fanout_cancellation.py` (14), plus `test_media_adapters.py`, `test_orphan_requeue_unit.py`, `CI/test_async_blocking_call_guard.py` and `Web_Scraping/test_phase3_preflight_external_tools.py` | `138 passed` |
  | Task 5 Step 4: the AC 1 file (4), `test_chacha_operation_scope.py`, `test_orphan_requeue_unit.py` | `24 passed` |
  | Task 6 Step 4 on PostgreSQL: the notes test, the AC 1 file, `test_workflows_db.py`, `lint/test_scope_predicate_ratchet.py` | `35 passed` |
  | `tests/Workflows` with the 18 outside paths that use these symbols, plus the scope ratchet (Task 7 Step 1's list) | `2 failed, 2422 passed, 7 skipped` (12 min 41 s, r3). r1's 2400, plus 9 new tests, plus the ratchet's 13. Both failures are already on dev; see below. 6 of the skips are `test_workflow_stress.py`, as TASK-13863's run recorded. The new PostgreSQL test ran. |
  | `tests/Docs`, `lint/test_scope_predicate_ratchet.py` and `CI/test_test_quality_ratchet.py`, after Tasks 4 and 6's doc edits, the mirror refresh and the baseline regeneration | `226 passed` |

- **The two failures in the wider run predate this plan.**
  - They are `Flashcards/test_flashcard_prompt_contract.py::test_endpoint_retains_real_claims_refusal_for_unsupported_drafts[nei-failed]` and `[refuted-failed]`.
  - With dev's 9 app files swapped back in (`git diff --stat origin/dev -- tldw_Server_API/app` empty), they fail the same way: `TypeError: string indices must be integers, not 'str'` at `raised.value.detail["code"]`.
  - The file was last changed by `eb0c8c9e52` (TASK-13260.278.18.83.45, the UAT repairs, In Progress).
- **Test-quality triage:** `--enforce` exits 0 ("no offenses beyond the baseline"), and none of the 5 new or extended test files appears in its report.
- **Mutants** (`fu2wf_mutate.py`, r3). Each clause's cancelling() half, flag half and read default are mutated separately, the other clause left as written. All six mutants are killed:
  - `start_run`'s `cancelling()` half replaced by `False`: `1 failed, 13 passed`, `test_a_cancelled_run_task_still_propagates_and_leaves_the_run_for_the_reaper[start_run]`;
  - `_continue_run_impl`'s: the same test's `[continue_run]` case. r1 had no test that killed this mutant (review I2);
  - `start_run`'s flag half dropped: `2 failed, 12 passed`, `test_a_run_cancelled_through_the_api_stays_cancelled_when_its_step_then_raises[start_run]` and `test_an_unreadable_cancel_flag_still_propagates_the_cancelled_error[start_run]`;
  - `_continue_run_impl`'s: the same two tests' `[continue_run]` cases;
  - `start_run`'s read default flipped to `False`: `1 failed, 13 passed`, `test_an_unreadable_cancel_flag_still_propagates_the_cancelled_error[start_run]`;
  - `_continue_run_impl`'s: that test's `[continue_run]` case.
  - Separately, skipping the notes adapter's `close_connections()` fails the PostgreSQL test: `assert 1 == 0` (outstanding checkouts).
- **Lint and types:**
  - Ruff 0.16.5 (the venv's) per file, branch against dev copies (`fu2wf_ruffcmp.py`): `new=none` for every touched `.py` file. Dev already has 2 BLE001 in `multi_voice_tts.py`, 1 I001 in `ingest.py` and 10 in `test_audio_adapters.py`.
  - mypy 1.20.1 (`uvx --offline`, CI's `--follow-imports=silent --ignore-missing-imports`), on the 9 app files, with dev's files swapped into the same tree:
    - the delta is **−1, with nothing new**: dev 67 and branch 66 here, where r1 recorded 65 and 64 (the reviewer's environment also gives 67/66; absolute counts vary by environment, so compare the delta);
    - the only differences are one line number inside a message (`Name "claimed" already defined on line 3168` → `3175`) and tts.py's `contextlib.suppress` arg-type error, which is gone;
    - the `meta_timeout: dict[str, Any]` annotation and the `exit_code` local were written before this measurement, to avoid a `union-attr` error on `task.proc` and a literal-inferred `dict[str, str]`. The version without them was not measured.
  - `uvx bandit -q -ll` on the 9 app files: 0 results.
- **Line-number baselines (review B1, corrected):**
  - `Helper_Scripts/ci/scope_predicate_baseline.txt` lists `Workflows_DB.py:1808 if tenant_id`, `:1811 if owner_id` and `:1960 if user_id`. Task 6's 7-line `_json_default` shifts them to `:1815`, `:1818` and `:1967`.
  - Before the baseline is regenerated, `tests/lint/test_scope_predicate_ratchet.py` gives `1 failed, 12 passed`. That test runs in `backend-required`.
  - `scope_predicate_ratchet.py --write-baseline` ("Recorded 28 conditional scope predicates") changes exactly those three lines.
  - No other baseline pins a touched file by line. `route_auth_baseline.txt`, `rls_coverage_baseline.txt` and `rg_route_map_lint_allowlist.txt` are keyed by route, table or path, and `test_quality_baseline.txt` by file and flag. A `grep -rnE '<touched file>\.py[:", ]+[0-9]{2,}'` over `tldw_Server_API/tests`, `Helper_Scripts` and `.github` finds only the three scope-predicate lines.
- **Mirrors:** of the docs touched, only `Docs/ADR/080-...md` has a `Docs/Published` copy. `refresh_docs_published.sh` changes only `Docs/Published/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md`.

## Global Constraints

- **Environment:**
  - Python: `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`.
  - Tests: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`. PostgreSQL tests also set `TLDW_TEST_POSTGRES_REQUIRED=1` and use the running `tldw_postgres_test` container through `pg_database_config`. Never roll your own database setup.
  - Never run the full suite.
  - Shell state does not persist between tool calls: start each command block with `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (zsh does not word-split variables, so commands spell out the backlog-py invocation).
- **Branch:** `fix/workflows-substep-cleanup-detached-chacha`, from `origin/dev`; the PR targets `dev`.
  - If `origin/dev` has moved past `e06eeec68c`, re-run each RED step on the new tip before the fix, and record the commit in the notes.
- **Commits:**
  - Every commit ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - Never `--no-verify`, never `git stash`, never force-push.
- **Test first:** each task's RED run happens before its fix, on the tree as the previous task left it, and fails for the reason above.
- **ChaCha operations (ADR-080, F13):**
  - Never wrap a workflow run, or a step that fans out, in one `chacha_operation`.
  - `outside_chacha_operation()` clears the inherited operation. It never opens one.
- **No real outbound requests:**
  - The tests' "ffmpeg" is a shell script that sleeps.
  - Their "yt-dlp" is `python -c` writing a `.part` file. The URL `https://media.example.com/clip` is never fetched: `start_process` is replaced.
- **Unchanged:**
  - `_FLOW_NONCRITICAL_EXCEPTIONS` and `_run_cancelling_siblings`;
  - the `parallel` adapter;
  - `run_checked_async`'s signature;
  - the HTTP routes;
  - `start_run`'s operation handling. A SYNC run started inside a request still joins the request's operation.
    - A voice command's SYNC run does the same today.
    - Once `plan_fu2_chacha_runtime.md` Task 4 lands, a voice command opens no operation of its own, and its SYNC run starts outside one. This plan does not depend on either shape (review m3).
- **Line-number baselines (B1):** after Task 6's `Workflows_DB.py` edit, run `$PY Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline`, and commit `Helper_Scripts/ci/scope_predicate_baseline.txt` with it.
  - If a rebase or a review fix moves lines in `Workflows_DB.py` again, regenerate it again.
  - `tests/lint/test_scope_predicate_ratchet.py` must pass before each push.
- **Test-quality ratchet (required gate):**
  - `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` exits 0, and `tldw_Server_API/tests/CI/test_test_quality_ratchet.py` passes.
  - In an `integration`-marked file, no test function makes 4 or more mock or patch calls.
  - Every test asserts on a result, not only on a list it appended to.
- **Lint and type comparison:**
  - Run ruff (`--output-format concise`) and mypy (CI's flags, `uvx --offline mypy`) on the touched files, here and with dev's copies.
  - For dev's copies: extract them with `git archive origin/dev <paths> -o <scratch>/dev.tar`, swap them into the tree, run, and restore yours from a tar.
  - Nothing new. No `ruff --fix`.
- **Docs:** `Docs/ADR/080-...md` changes, so `bash Helper_Scripts/refresh_docs_published.sh` runs, and its Docs/Published diff goes in the same commit. `Docs/Operations/Usage_Quotas.md` and `Docs/Design/Workflows.md` have no mirror. Edits are anchored on quoted text, not line numbers.
- **Global defaults:** none flips. The one shared-behaviour change, `Workflows_DB` serializing `date` and `datetime`, only turns a `TypeError` into stored text. Task 7 still runs every test file that uses `WorkflowsDatabase`, `WorkflowEngine` or the changed adapters.
- **Backlog:**
  - `tools/backlog-py` only, run as `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`.
  - `--append-notes` only, never `--notes`.
  - Explicit `--id` for new tasks. Setup computes all four ids (`<N1>` to `<N4>`) under the id lock and writes them to `<scratchpad>/fu2wf_ids.env`.
  - The lock is `mkdir <scratchpad>/id_lock`: if it exists, wait, and never break another session's lock. Release it with `rmdir <scratchpad>/id_lock` after the PR that carries the four task files is open and `gh pr view` lists them (review m2).
  - Stage with `git add -A backlog/tasks`.

## Decisions for the owner (recommendation first)

1. **The `media_ingest` download budget (review I3).**
   - **What changes:**
     - On dev, a network download "succeeded" as soon as its first file (often yt-dlp's `.part` file) appeared, and yt-dlp then ran on unsupervised.
     - With this plan, the step waits for yt-dlp to exit. That is the only correct reading of TASK-13887 AC 3: "waits for yt-dlp to exit, or terminates it".
     - The wait is bounded by the step's `timeout_seconds` (default 300), shared by all of the step's sources.
     - A download that outlasts it is stopped (SIGTERM, then SIGKILL after 5 s), and the step fails with `step_timeout`. So does the run, unless `on_failure` routes around it. The plan's engine-level test pins this.
   - **Recommended:** keep the wait and the 300 s default, and document how to raise it: the step-level `timeout_seconds`, beside `type`. Task 4 writes this in `Docs/Design/Workflows.md`.
     - **Cost:** workflows that ingest long media (a multi-hour podcast, say) worked by accident on dev. A later step read the download directory after yt-dlp finished, unsupervised. They now fail with `step_timeout` after 300 s plus the 5 s grace, until their author raises the step's `timeout_seconds`.
     - **Cost for WebUI authors (re-review m-1):** the WebUI editor has no control for a step-level `timeout_seconds`, so they can raise the budget only by editing the definition through the API or a JSON import.
       - `NodeConfigPanel.tsx` has no timeout field.
       - `workflow-editor-definition.ts:62` and `:97` only round-trip a step-level `timeout_seconds` that is already in the definition.
       - The `media_ingest` form in `step-registry.ts` edits `config` only. Its one `timeoutSeconds` field (`:373`) belongs to `wait_for_human`'s config.
       - **Weighed against the alternatives:** Alternative A needs no author action, at the price of an hour-long hold on a stuck download. A third way is to keep 300 s and add a step-level timeout control to the editor: a WebUI change outside this backend PR, which could be filed as a follow-up if the owner wants it. The recommendation stands because it keeps a hung download bounded at the default. WebUI authors with long media need the API or a JSON import until the editor gains the control.
     - None of the 4 shipped templates (`tldw_Server_API/Samples/Workflows/*.workflow.json`) uses `media_ingest` (measured). User-authored workflows cannot be checked.
   - **Alternative A: a larger default for `media_ingest` only,** for example 3600 s.
     - **Cost:** a per-type timeout default in the engine, which today has per-type defaults for retries (`_compute_max_retries_for_step`) but not for timeouts. That is about 10 lines and a test. And a stuck download would then hold its step, and its run, for an hour.
   - **Alternative B: no budget for downloads,** the step waiting as long as yt-dlp runs.
     - **Cost:** the engine's step timeout would need an exception for this step type. A hung yt-dlp would hold the run forever, which is the leak TASK-13887 set out to close.

What the ACs and the code settle, this plan resolves:
- TASK-13887 AC 2's premise is wrong. Task 4 rewords it through backlog-py and files the repair.
- A step's own `CancelledError` is a retriable step failure, like any other adapter error.
- A run cancelled through the API keeps `cancelled` when its step then raises `CancelledError`, as on dev. The broader overwrite, for any other step error, is filed as TASK-<N4>.

The choice between repairing `workflow_call` and removing it from the step registry belongs to the new task (TASK-<N1>), not to this PR.

The PR needs the owner's own `## Change summary` (ADR-004). No recorded waiver covers this round. It is an input, not a decision.

## Review Focus

1. **Long downloads now fail the step (Decision 1).**
   - On dev a `media_ingest` download "succeeded" as soon as its first file appeared. That file could be a partial `.part` file, and yt-dlp then ran on unsupervised.
   - Now the step waits for yt-dlp to exit, up to the step's `timeout_seconds` (default 300), which all of the step's sources share.
     - A download that outlasts it is cancelled at the poll and terminated: SIGTERM, then SIGKILL after 5 s, on a worker thread.
     - The step fails with `step_timeout`, and so does the run unless `on_failure` routes around it.
     - A config-level `timeout_seconds` cannot extend the step's: `continue_run` overwrites it, and in `start_run` the engine's deadline fires first.
   - An exit with an error, or an exit with no file, is `download_failed` with `returncode`. That covers `--max-filesize` refusing a file.
   - An exit that the run's own cancel caused (`engine.cancel()` kills the group) is reported `cancelled`, not `download_failed` (m5).
   - Pinned:
     - `test_media_ingest_returns_only_after_yt_dlp_has_exited`;
     - `test_media_ingest_reports_a_download_that_yt_dlp_failed`;
     - `test_media_ingest_whose_run_was_cancelled_reports_cancelled`;
     - `test_a_download_that_outlasts_the_steps_timeout_fails_the_step` (engine-level).
     - `Docs/Design/Workflows.md` says so (Task 4).
   - Not pinned: a yt-dlp that forks a child and exits early. `poll()` sees only yt-dlp's own exit.
2. **A continued run never joins a caller's operation.**
   - Every ChaCha call in it takes the thread-local fallback or owns its operation. On the event-loop thread, an adapter that does neither, and does not close its handle, would pin a loop-thread checkout: the leak class TASK-13861 fixed elsewhere.
   - The known adapters are covered:
     - notes closes its per-call handle, which the PostgreSQL test pins with 0 outstanding checkouts;
     - chatbooks and character_chat own independent operations;
     - RAG search reaches no ChaCha.
   - A new ChaCha adapter must follow ADR-080's rule.
   - The human-step timeout task is created outside the operation too (m4), so it holds no closed operation while it sleeps. Pinned by `test_a_human_steps_timeout_task_starts_outside_its_creators_operation`.
   - `start_run`, which routes run inline, still joins the request's open operation, as on dev. A voice command does the same until `plan_fu2_chacha_runtime.md` Task 4 removes the command's operation.
3. **A tracing span that fails on exit after the item returned.**
   - The item now fails with the tracing error. On dev it ran again.
   - `async_span`'s exit only deletes a dict key, and OpenTelemetry's exit does not raise, so this needs an OpenTelemetry fault to happen.
   - Not tested: the failure needs a fault injected into the tracer's `__aexit__`. The span-enter case is pinned by `test_a_map_item_still_runs_once_when_its_tracing_span_fails`.
4. **A step's own `CancelledError` is retriable.**
   - `_reason_code_from_error(AdapterError("step_raised_cancelled_error"))` is `step_raised_cancelled_error`, which is not in the non-retriable set. So a step with `retry` runs again, as after any `RuntimeError`.
   - **Two kinds of real cancel still propagate.**
     - A cancel of the run's asyncio task (shutdown, a cancelled route task) propagates because `cancelling() > 0`.
     - A cancel through the run-cancel API (`engine.cancel()`, which cancels no task) propagates because `is_cancel_requested(run_id)` is set, and the run stays `cancelled`, as on dev (I1).
   - **A failed read of that flag** (a locked SQLite file, a dropped PostgreSQL connection) counts as "cancelled". The step's `CancelledError` propagates and the run stays `running` for the reaper, as on dev, rather than the DB error escaping (m-2).
     - The flip side: a step's own `CancelledError` that coincides with a failed flag read is not converted into a step failure.
     - That needs two faults at once, and it lands on dev's behaviour.
   - **Still open (TASK-<N4>):** after an API cancel, a step that raises any other exception ends the run `failed`/`invariant_violation`. That is dev behaviour, measured.
   - **Pinned, each for `start_run` and `continue_run`:**
     - `test_a_cancelled_run_task_still_propagates_and_leaves_the_run_for_the_reaper`;
     - `test_a_run_cancelled_through_the_api_stays_cancelled_when_its_step_then_raises`;
     - `test_an_unreadable_cancel_flag_still_propagates_the_cancelled_error`;
     - `test_a_step_timeout_is_still_a_step_timeout`.
   - **A spurious `uncancel()`** by third-party code inside a step could make a real task cancel look like the step's own. The reviewer found no Workflows adapter path to the two app callers (`Notes_Graph`, `Web_Scraping/article_browser.py`).
5. **ffmpeg under a second cancel, and the stored outputs.**
   - `run_checked_async._reap` kills the child, then awaits `communicate()`. A second cancel during that await leaves a killed child for the loop's child watcher to reap. The child is not left running.
   - `tts.py` no longer passes `cwd=out_dir`. Both paths in its command are absolute (`resolve_artifacts_dir` resolves them), and loudnorm writes no side file.
   - `Workflows_DB` now converts only `date` and `datetime` to ISO text. Any other non-JSON output still raises `TypeError` and fails the run, as before.
   - Pinned: `test_run_and_step_outputs_store_datetimes_as_iso_text`.

## ADR assessment

- **No new ADR.** ADR-080 governs.
  - Its Consequences bullet "Detached tasks" already named this approach: "TASK-13897 tracks the root cause: starting those tasks outside the caller's operation".
  - This PR implements it, and it keeps the Decision's F13 rule: no operation around a run.
- **ADR-080 is updated in place, as status facts only.**
  - The bullet's last three sentences describe a state that this PR ends: "inherit their caller's operation today ... The notes adapter does not yet".
  - Task 6 replaces them with the new facts. The Decision text and the rationale stay as they are, so README rule 8 is kept.
  - The new text names the pattern the notes adapter follows (m10): "returns its checkout after each step outside an operation, as the notes-graph job does". That is an existing Consequence of ADR-080, so the bullet does not read as an exception to the Decision.
  - There is precedent: `b3c1a1fa56` (2026-10-22) added two Consequences bullets to the accepted ADR-080.
  - The README table row for ADR-080 does not mention detached tasks, so it is unchanged.
- **Other ADRs:**
  - ADR-004: the Change summary is the owner's.
  - ADR-006: Bandit covers the touched scope.
  - ADR-059: backlog-py.
  - No ADR governs the workflow engine's fan-out or subprocess handling (README searched for workflow, subprocess and cancel).

## PR split

**One PR, with one commit per task:**
- **The same file.** TASK-13891 edits both step loops in `engine.py`, and TASK-13897 edits `continue_run` between them. Two PRs would merge serially through `dev`'s seven required statuses, and the second would need a rebase and a full CI cycle once the first lands.
- **Neither holds the other up.** All four are small: about 90 app lines changed, measured, and GREEN together, and the PostgreSQL test uses the existing fixture. A split buys nothing.
- **One review and one summary.** The PR gets one clean review and one owner Change summary, not two of each.
- **Reverts stay targeted.** A revert can still undo one task's commit.

**Branch:** `fix/workflows-substep-cleanup-detached-chacha`, from `origin/dev`, PR to `dev`.

**Commit plan** (each commit ends with the Co-Authored-By trailer):

| # | Task | Commit subject | Contents |
|---|---|---|---|
| 1 | 1 | `fix(workflows): a map item that raises runs once (TASK-13886)` | `flow.py`, 2 tests, the plan copy under `Docs/superpowers/plans/`, TASK-13886 closed |
| 2 | 2 | `fix(workflows): a step that raises its own CancelledError fails the run (TASK-13891)` | `engine.py` step loops, new test file, TASK-<N4> filed, TASK-13891 closed |
| 3 | 3 | `fix(workflows): cancelled tts and multi_voice_tts steps kill their ffmpeg (TASK-13887)` | `tts.py`, `multi_voice_tts.py`, new test file, one test re-pointed |
| 4 | 4 | `fix(workflows): media_ingest waits for yt-dlp to exit (TASK-13887)` | `subprocess_utils.py`, `ingest.py`, 4 tests, `Usage_Quotas.md`, `Workflows.md`, TASK-<N1> and TASK-<N2> filed, TASK-13887 closed |
| 5 | 5 | `fix(workflows): a continued run leaves its creator's ChaCha operation (TASK-13897)` | `operation_scope.py`, `engine.py` (`continue_run` and the timeout task's creation), new test file |
| 6 | 6 | `fix(workflows): a notes step works on PostgreSQL in a continued run (TASK-13897)` | `Workflows_DB.py`, `Helper_Scripts/ci/scope_predicate_baseline.txt`, `search.py` comment, 2 tests, ADR-080 with its mirror, TASK-<N3> filed, TASK-13897 closed |

Review fixes, if any, go in further commits on the same branch.

## Conflicts with other plans and open PRs

| Place | This plan | Other | How to merge |
|---|---|---|---|
| `Docs/Operations/Usage_Quotas.md:96` | Task 4 replaces the bullet's last two sentences | `plan_fu2_ingest_openapi.md` edits line 93 | Lines 94-95 are untouched between the two edits, so git merges them. If not, keep the landed text and re-apply yours by its quoted text. |
| `tldw_Server_API/app/core/Workflows/engine.py` | hunks at `:838`, `:1143`, `:1382`, `:1829` | Open PR #3269 (1168 files, paged) changes one hunk at `:1914` (`_run_step_adapter`) | Not adjacent: clean in either order. |
| `Docs/ADR/080-...md` and `Docs/Published/ADR/080-...md` | Task 6 replaces the "Detached tasks" bullet (line 51) | `plan_fu2_chacha_runtime.md` Task 5 edits the "Cost" bullet (line 52) and "Concurrent transactions" (line 56) | Lines 51 and 52 are adjacent, so expect a textual conflict. Whoever merges second keeps both edits, re-runs `bash Helper_Scripts/refresh_docs_published.sh` and `tests/Docs`, and commits the merged Published copy. |
| `tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py` | Task 5 adds `outside_chacha_operation` after `chacha_operation` (`:215`) | `plan_fu2_chacha_runtime.md` Task 1 edits `ClosedChaChaOperationError` (`:22`) and `ConnectionState` (`:26-92`) | Not adjacent: clean in either order. |
| Voice-command operation (behaviour, no shared file) | Review Focus 2 and Global Constraints describe today's voice command | `plan_fu2_chacha_runtime.md` Task 4 removes the command's operation | No code conflict. Its Task 4 makes the voice command's SYNC run start outside an operation, which this plan already handles. If it lands first, keep this plan's wording as written: both shapes are covered. |
| `Helper_Scripts/ci/scope_predicate_baseline.txt` | Task 6 regenerates three `Workflows_DB.py` lines | Any other PR that shifts a listed file | Whoever merges second re-runs `scope_predicate_ratchet.py --write-baseline` after the rebase. |
| `tldw_Server_API/app/core/Workflows/subprocess_utils.py` | adds `SubprocessTask.proc` | Open PR #854 (managed vLLM, base `dev`) touches the file | Re-check at execution. Field additions merge unless #854 rewrote the dataclass. |
| Every other touched file | this plan | No open PR among 44 (the 10 PRs with more than 100 files were paged), and no other `plan_fu2_*` in the scratchpad, apart from the rows above | None. |

---

## Setup (once)

```bash
cd <your worktree>
git fetch origin dev --quiet
git switch -c fix/workflows-substep-cleanup-detached-chacha origin/dev
git log -1 --format='%h %s'   # record it; the plan measured e06eeec68c
export PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
mkdir -p Docs/superpowers/plans
cp <scratchpad>/plan_fu2_workflows.md Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13886 -s "In Progress"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13887 -s "In Progress"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13891 -s "In Progress"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13897 -s "In Progress"
```

The plan copy goes in Task 1's commit. Check it has no trailing whitespace: `grep -c ' $' Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md` prints `0`.

**Take the id lock and allocate the four follow-up ids** (review m2). The lock is `mkdir <scratchpad>/id_lock`. If it already exists, another session holds it: wait and retry, and never remove it. Then compute the highest id across `origin/dev` and every open PR. This is AGENTS.md's command, paging the PRs with more than 100 files, whose file lists `gh pr list` truncates:

```bash
mkdir <scratchpad>/id_lock || { echo "id lock held by another session: wait"; exit 1; }
git fetch origin dev --quiet
HIGHEST=$( {
  git ls-tree -r --name-only origin/dev backlog/
  gh pr list --state open --limit 500 --json files --jq '.[].files[].path'
  for n in $(gh pr list --state open --limit 500 --json number,changedFiles --jq '.[] | select(.changedFiles > 100) | .number'); do
    gh api --paginate "repos/rmusser01/tldw_server/pulls/$n/files" --jq '.[].filename'
  done
} | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1 )
printf 'N1=TASK-%s\nN2=TASK-%s\nN3=TASK-%s\nN4=TASK-%s\n' $((HIGHEST+1)) $((HIGHEST+2)) $((HIGHEST+3)) $((HIGHEST+4)) | tee <scratchpad>/fu2wf_ids.env
```

On 2026-10-22 the highest id was 13901. The reviewer paged #3269 and #3298 and found nothing higher. Shell state does not persist between tool calls, so each later step that files a task starts with `source <scratchpad>/fu2wf_ids.env`, and writes `"$N1"` to `"$N4"` where this plan writes `TASK-<N1>` to `TASK-<N4>`. In prose and commit messages, write the computed ids. Hold the lock until the PR is open (Task 7 Step 4).

---

### Task 1: a `map` item that raises runs once (TASK-13886)

**Files:**
- Modify: `tldw_Server_API/app/core/Workflows/adapters/control/flow.py:361-376` (`run_map_adapter._run_one`)
- Test: `tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py` (append; it is `unit`-marked)

**Interfaces:**
- Consumes: `flow.get_adapter`, `flow._start_span` (the module's `start_async_span` alias), `flow.run_map_adapter(config, context)`.
- Produces: nothing new. `_run_one` returns the adapter's result, or `{"error": "unsupported_substep:<type>"}`, as before.

- [ ] **Step 1: Write the failing test and its guard**

Add `import contextlib` after `import asyncio` at the top of the test file. Then append:

```python
async def test_a_map_item_that_raises_runs_once_and_fails_the_step() -> None:
    """On dev the tracing fallback ran the item again before the step failed: calls == [0, 0] (TASK-13886)."""
    calls: list[int] = []
    error = RuntimeError("webhook failed")

    async def substep(_config, context):
        calls.append(context["item"])
        raise error

    with patch.object(flow, "get_adapter", return_value=substep):
        with pytest.raises(RuntimeError) as raised:
            await flow.run_map_adapter({"items": [0], "step": {"type": "summarize", "config": {}}}, {})

    assert raised.value is error
    assert calls == [0]


async def test_a_map_item_still_runs_once_when_its_tracing_span_fails() -> None:
    """A span that fails to open costs the item its trace, not its run (TASK-13886)."""
    calls: list[int] = []

    async def substep(_config, context):
        calls.append(context["item"])
        return {"text": str(context["item"])}

    @contextlib.asynccontextmanager
    async def broken_span(*_args, **_kwargs):
        raise RuntimeError("tracer unavailable")
        yield  # pragma: no cover - never reached

    with patch.object(flow, "get_adapter", return_value=substep), patch.object(flow, "_start_span", broken_span):
        result = await flow.run_map_adapter({"items": [0], "step": {"type": "summarize", "config": {}}}, {})

    assert result == {"results": [{"text": "0"}], "count": 1}
    assert calls == [0]
```

- [ ] **Step 2: Run them on unchanged source**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py -k "map_item"`
Expected: `1 failed, 1 passed`. The failure is `test_a_map_item_that_raises_runs_once_and_fails_the_step` with `assert [0, 0] == [0]`. The pass is the span guard: dev already falls back once.

- [ ] **Step 3: Run the sub-step once**

In `flow.py`, replace this block in `_run_one`:

```python
            try:
                async with _start_span("workflows.map.item", attributes={
                    "index": int(idx),
                    "sub_type": sub_type,
                    "item_preview": preview,
                }):
                    adapter = get_adapter(sub_type)
                    if adapter is not None:
                        return await adapter(sub_cfg, sub_ctx)
                    return {"error": f"unsupported_substep:{sub_type}"}
            except _FLOW_NONCRITICAL_EXCEPTIONS:
                # If tracing fails, still attempt the sub-step
                adapter = get_adapter(sub_type)
                if adapter is not None:
                    return await adapter(sub_cfg, sub_ctx)
                return {"error": f"unsupported_substep:{sub_type}"}
```

with:

```python
            adapter = get_adapter(sub_type)
            if adapter is None:
                return {"error": f"unsupported_substep:{sub_type}"}
            started = False
            try:
                async with _start_span("workflows.map.item", attributes={
                    "index": int(idx),
                    "sub_type": sub_type,
                    "item_preview": preview,
                }):
                    started = True
                    return await adapter(sub_cfg, sub_ctx)
            except _FLOW_NONCRITICAL_EXCEPTIONS:
                if started:
                    raise  # the sub-step's own error: running it again would repeat its side effects (TASK-13886)
            # The span failed to open: still run the sub-step, untraced
            return await adapter(sub_cfg, sub_ctx)
```

- [ ] **Step 4: Run them, and the map tests, again**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py tldw_Server_API/tests/Workflows/test_workflows_map_substeps_integration.py tldw_Server_API/tests/Workflows/adapters/test_control_adapters.py -k "map or Map"`
Expected: `15 passed`.

- [ ] **Step 5: Close TASK-13886 and commit**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13886 \
  --append-notes "RED on dev <commit>: test_a_map_item_that_raises_runs_once_and_fails_the_step failed with assert [0, 0] == [0]; the span guard passed (dev already falls back once). 2 tests, 1 failed, 1 passed." \
  --append-notes "Fix: _run_one resolves the adapter first and sets started inside the workflows.map.item span; the noncritical except re-raises once the sub-step has started, and the untraced fallback runs only when the span failed to open. GREEN: both tests pass, with the map tests in test_control_fanout_cancellation.py, test_workflows_map_substeps_integration.py and test_control_adapters.py." \
  --append-notes "Behaviour change: a span that fails on exit after the item returned now fails the item with the tracing error instead of running it again (plan Review Focus 3). Docs: Docs/Design/Workflows.md's map bullet stays true; no change. Bandit -ll on flow.py: 0 results. Plan: Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md." \
  --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A map sub-step that raises now runs once, and the map step fails with its error; on dev the tracing fallback ran it a second time, repeating side effects such as a webhook. A span that fails to open still runs the sub-step once, untraced. Pinned by two tests in test_control_fanout_cancellation.py." \
  -s Done
git add tldw_Server_API/app/core/Workflows/adapters/control/flow.py \
  tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py \
  Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md
git add -A backlog/tasks
git commit -m "fix(workflows): a map item that raises runs once (TASK-13886)

The tracing fallback in map's _run_one caught the sub-step's own error
and ran the sub-step again, repeating its side effects. It now falls
back only when the span failed to open.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

`--final-summary` and `--append-notes` run before `-s Done` in the same call. Run `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task TASK-13886` and check that the 2 ACs, the 6 DoD items and `status: Done` are all set.

---

### Task 2: a step that raises its own `CancelledError` fails the run (TASK-13891)

**Files:**
- Modify: `tldw_Server_API/app/core/Workflows/engine.py`:
  - the imports (`sqlite3`, and the backend `DatabaseError`);
  - a module-level `_CANCEL_FLAG_READ_ERRORS` after `_WF_NONCRITICAL_EXCEPTIONS` (`:26-37`);
  - `_start_run_impl`'s step attempt (`:836-840`) and `_continue_run_impl`'s (`:1371-1373`).
- Create: `tldw_Server_API/tests/Workflows/test_engine_step_own_cancelled_error.py`

**Interfaces:**
- Consumes:
  - `WorkflowEngine(db).start_run(run_id, RunMode.SYNC)`, `WorkflowEngine.continue_run(run_id, after_step_id=...)`;
  - `WorkflowEngine.cancel(run_id) -> str`, the run-cancel route's call: it sets `cancelled` and `cancel_requested`, and cancels no task;
  - `engine_module.get_adapter`, `flow.get_adapter`;
  - `WorkflowsDatabase.update_run_status(run_id, status=...)` and `WorkflowsDatabase.is_cancel_requested(run_id)`. On SQLite that raises `sqlite3.Error`; on the PostgreSQL backend, a `backends.base.DatabaseError` subclass.
- Produces: a step failure whose error is `step_raised_cancelled_error`, an `AdapterError`. It is retriable.

- [ ] **Step 1: Write the failing tests and the AC 3 guards**

Create `tldw_Server_API/tests/Workflows/test_engine_step_own_cancelled_error.py`. Every case runs through both entry points, because `_continue_run_impl`'s clause is a separate copy (review I2):

```python
"""A step that raises its own CancelledError fails the run; a real cancellation still propagates (TASK-13891).

Each case runs through both step loops: ``start_run``'s and ``continue_run``'s, which is the
loop every detached task (a route, a human step's timeout, the orphan reaper) runs.
"""

import asyncio
import contextlib
import sqlite3
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows import engine as engine_module
from tldw_Server_API.app.core.Workflows.adapters.control import flow
from tldw_Server_API.app.core.Workflows.engine import RunMode, WorkflowEngine

pytestmark = pytest.mark.integration

_REAL_GET_ADAPTER = engine_module.get_adapter
_ENTRIES = ["start_run", "continue_run"]

_MAP = {"id": "s", "type": "map", "config": {"items": [0, 1], "step": {"type": "summarize", "config": {}}}}
_PARALLEL_FAIL_FAST = {
    "id": "s",
    "type": "parallel",
    "config": {"fail_fast": True, "steps": [{"type": "summarize", "config": {"n": n}} for n in range(2)]},
}
_PLAIN = {"id": "s", "type": "summarize", "config": {"n": 0}}


def _run(db: WorkflowsDatabase, engine: WorkflowEngine, entry: str, step: dict):
    """Create a run whose step under test is ``step``, and return the coroutine that runs it."""
    first = [] if entry == "start_run" else [{"id": "first", "type": "summarize", "config": {"n": 1}}]
    db.create_run(
        run_id="run",
        tenant_id="default",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "run", "version": 1, "steps": [*first, step]},
    )
    if entry == "start_run":
        return engine.start_run("run", RunMode.SYNC)
    db.update_run_status("run", status="waiting_human")  # e.g. approved: the route continues the run
    return engine.continue_run("run", after_step_id="first")


@contextlib.contextmanager
def _summarize_is(sub_step):
    """Route every ``summarize`` step and sub-step to ``sub_step``; other step types stay real."""
    with patch.object(flow, "get_adapter", return_value=sub_step), patch.object(
        engine_module, "get_adapter", side_effect=lambda t: sub_step if t == "summarize" else _REAL_GET_ADAPTER(t)
    ):
        yield


def _events(db: WorkflowsDatabase) -> list[str]:
    return [event["event_type"] for event in db.get_events("run")]


async def _raises_its_own_cancelled_error(config, context):
    """Item or step 0 raises CancelledError itself; nothing is cancelling it."""
    if context.get("item", config.get("n")) == 0:
        raise asyncio.CancelledError()
    return {"ok": True}


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("step", [_MAP, _PARALLEL_FAIL_FAST, _PLAIN], ids=["map", "parallel_fail_fast", "plain"])
async def test_a_step_that_raises_its_own_cancelled_error_fails_the_run(tmp_path, step, entry) -> None:
    """On dev the CancelledError left the engine, and the run stayed 'running' for the orphan reaper."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run = _run(db, WorkflowEngine(db), entry, step)

    with _summarize_is(_raises_its_own_cancelled_error):
        await asyncio.wait_for(run, timeout=10)

    finished = db.get_run("run")
    assert (finished.status, finished.error) == ("failed", "step_raised_cancelled_error")
    assert _events(db)[-2:] == ["step_failed", "run_failed"]


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_a_step_timeout_is_still_a_step_timeout(tmp_path, entry) -> None:
    """The step's own timeout cancels it, and wait_for turns that into TimeoutError, as on dev."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run = _run(db, WorkflowEngine(db), entry, {"id": "s", "type": "summarize", "timeout_seconds": 1, "config": {}})

    async def never_returns(_config, _context):
        await asyncio.Event().wait()

    with _summarize_is(never_returns):
        await asyncio.wait_for(run, timeout=10)

    finished = db.get_run("run")
    assert finished.status == "failed"
    assert "step_timeout" in _events(db)


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_a_cancelled_run_task_still_propagates_and_leaves_the_run_for_the_reaper(tmp_path, entry) -> None:
    """Cancelling the run's task (e.g. at shutdown) still raises CancelledError and leaves the run 'running', as on dev."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run = _run(db, WorkflowEngine(db), entry, {"id": "s", "type": "summarize", "config": {}})
    started = asyncio.Event()

    async def runs_until_cancelled(_config, _context):
        started.set()
        await asyncio.Event().wait()

    with _summarize_is(runs_until_cancelled):
        task = asyncio.create_task(run)
        await asyncio.wait_for(started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert db.get_run("run").status == "running"
    assert "run_failed" not in _events(db)


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_a_run_cancelled_through_the_api_stays_cancelled_when_its_step_then_raises(tmp_path, entry) -> None:
    """engine.cancel() (the run-cancel route) cancels no task, so cancelling() is 0; dev kept 'cancelled', and so must this."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    engine = WorkflowEngine(db)
    run = _run(db, engine, entry, {"id": "s", "type": "summarize", "config": {}})
    started, release = asyncio.Event(), asyncio.Event()

    async def raises_after_the_cancel(_config, _context):
        started.set()
        await release.wait()
        raise asyncio.CancelledError()

    with _summarize_is(raises_after_the_cancel):
        task = asyncio.create_task(run)
        await asyncio.wait_for(started.wait(), timeout=10)
        cancelled = engine.cancel("run")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    finished = db.get_run("run")
    assert (cancelled, finished.status, finished.error) == ("applied", "cancelled", None)
    assert "run_failed" not in _events(db)


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_an_unreadable_cancel_flag_still_propagates_the_cancelled_error(tmp_path, entry) -> None:
    """If the clause cannot read the cancel flag, the step's CancelledError propagates and the run stays 'running', as on dev."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run = _run(db, WorkflowEngine(db), entry, {"id": "s", "type": "summarize", "config": {}})
    read_flag, stepped = db.is_cancel_requested, asyncio.Event()

    def locked_once_the_step_runs(run_id):
        if stepped.is_set():
            raise sqlite3.OperationalError("database is locked")
        return read_flag(run_id)

    async def raises_cancelled_error(_config, _context):
        stepped.set()
        raise asyncio.CancelledError()

    with _summarize_is(raises_cancelled_error), patch.object(db, "is_cancel_requested", side_effect=locked_once_the_step_runs):
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, timeout=10)

    assert db.get_run("run").status == "running"
    assert "run_failed" not in _events(db)
```

- [ ] **Step 2: Run them on unchanged source**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/test_engine_step_own_cancelled_error.py`
Expected: `6 failed, 8 passed`.
- The failures are the six `test_a_step_that_raises_its_own_cancelled_error_fails_the_run` cases, each with `asyncio.exceptions.CancelledError` raised out of `start_run` or `continue_run`.
- The passes are the four AC 3 guards, each for both entries. Dev already behaves this way, and the guards keep it:
  - the step timeout;
  - a cancelled run task;
  - a cancel through the API;
  - an unreadable cancel flag.

- [ ] **Step 3: Catch a step's own `CancelledError` in both step loops**

In `engine.py`, change `import re\nimport threading` to `import re\nimport sqlite3\nimport threading`. Above `from tldw_Server_API.app.core.DB_Management.DB_Manager import ...`, add:

```python
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
```

(Task 5 later adds the `operation_scope` import between these two lines.) After the `_WF_NONCRITICAL_EXCEPTIONS` tuple, add:

```python
# Reading the cancel flag inside a CancelledError handler: a DB failure there must not replace the
# CancelledError (SQLite raises sqlite3.Error; the PostgreSQL backend wraps its driver's in DatabaseError).
_CANCEL_FLAG_READ_ERRORS = (*_WF_NONCRITICAL_EXCEPTIONS, sqlite3.Error, BackendDatabaseError)
```

In `_start_run_impl`, after this clause:

```python
                        except Exception as e:  # noqa: BLE001 - adapter/provider failures must terminally fail the step
                            err = e
                            error_reason_code = _reason_code_from_error(e)
                            with contextlib.suppress(*_WF_NONCRITICAL_EXCEPTIONS):
                                record_span_exception(e)
```

add, at the same indentation:

```python
                        except asyncio.CancelledError:
                            current = asyncio.current_task()
                            run_cancelled = True  # if the flag cannot be read, the CancelledError propagates, as on dev
                            with contextlib.suppress(*_CANCEL_FLAG_READ_ERRORS):
                                run_cancelled = self.db.is_cancel_requested(run_id)
                            if (current is not None and current.cancelling()) or run_cancelled:
                                raise  # the run task, or the run itself, was cancelled: propagate, as before
                            err = AdapterError("step_raised_cancelled_error")  # the step raised it itself (TASK-13891)
                            error_reason_code = _reason_code_from_error(err)
```

In `_continue_run_impl`, after this clause:

```python
                except Exception as e:  # noqa: BLE001 - adapter/provider failures must terminally fail the step
                    err = e
                    error_reason_code = _reason_code_from_error(e)
```

add:

```python
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    run_cancelled = True  # if the flag cannot be read, the CancelledError propagates, as on dev
                    with contextlib.suppress(*_CANCEL_FLAG_READ_ERRORS):
                        run_cancelled = self.db.is_cancel_requested(run_id)
                    if (current is not None and current.cancelling()) or run_cancelled:
                        raise  # the run task, or the run itself, was cancelled: propagate, as before
                    err = AdapterError("step_raised_cancelled_error")  # the step raised it itself (TASK-13891)
                    error_reason_code = _reason_code_from_error(err)
```

Notes on this code:
- `AdapterError` is already imported in `engine.py`, and `run_id` is in scope in both loops.
- The step timeout never reaches the clause: `wait_for` turns it into `TimeoutError` first.
- **Why the flag is read (review I1).** `is_cancel_requested` covers `engine.cancel()`, which cancels no task. Without it, a run the user cancelled ended `failed`/`invariant_violation`; measured with an earlier draft's clause, `2 failed, 10 passed`.
- **Why the read is guarded (re-review m-2).**
  - The read follows the file's `contextlib.suppress` pattern, and defaults to "cancelled". A failed read (a locked SQLite file, a dropped PostgreSQL connection) therefore propagates the `CancelledError`, as dev did, instead of replacing it with a DB error.
  - `_WF_NONCRITICAL_EXCEPTIONS` alone is not enough. It holds neither `sqlite3.Error`, which `WorkflowsDatabase.is_cancel_requested` raises raw on SQLite (`Workflows_DB.py:2254-2262`), nor the `backends.base.DatabaseError` that the PostgreSQL backend raises (`postgresql_backend.py:1159-1172`). Hence `_CANCEL_FLAG_READ_ERRORS`.
  - Measured against the unguarded read: `test_an_unreadable_cancel_flag_still_propagates_the_cancelled_error` failed for both entries with `sqlite3.OperationalError: database is locked`.
- Inside the `except` block, a bare `raise` after the suppressed read still re-raises the `CancelledError`. The suppressed error's handling has ended (pinned by the unreadable-flag test).

- [ ] **Step 4: Run them, the fan-out tests, and the mutants**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/test_engine_step_own_cancelled_error.py tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py tldw_Server_API/tests/Workflows/test_workflows_map_substeps_integration.py`
Expected: `27 passed`: 14, plus the fan-out file's 10 as Task 1 left it, plus 3.

Mutant checks. Each mutant rewrites one clause, with the other clause left as written. Run only `test_engine_step_own_cancelled_error.py` for each, then restore the file and confirm with `git diff --stat`.

| Mutant | Change | Expected |
|---|---|---|
| `start_run`'s task half | `if False or run_cancelled:` | `1 failed, 13 passed`: `test_a_cancelled_run_task_still_propagates_and_leaves_the_run_for_the_reaper[start_run]` |
| `_continue_run_impl`'s task half | the same, in the second clause | `1 failed, 13 passed`: the same test's `[continue_run]` case |
| `start_run`'s flag half | `if (current is not None and current.cancelling()):` | `2 failed, 12 passed`: `test_a_run_cancelled_through_the_api_stays_cancelled_when_its_step_then_raises[start_run]` and `test_an_unreadable_cancel_flag_still_propagates_the_cancelled_error[start_run]` |
| `_continue_run_impl`'s flag half | the same, in the second clause | `2 failed, 12 passed`: the same two tests' `[continue_run]` cases |
| `start_run`'s read default | `run_cancelled = False  # ...` | `1 failed, 13 passed`: `test_an_unreadable_cancel_flag_still_propagates_the_cancelled_error[start_run]` |
| `_continue_run_impl`'s read default | the same, in the second clause | `1 failed, 13 passed`: that test's `[continue_run]` case |

`<scratchpad>/fu2wf_mutate.py <clause 0|1> <task|api|default>` writes each mutant from a saved copy of the fixed `engine.py`.

- [ ] **Step 5: File TASK-<N4>, close TASK-13891 and commit**

```bash
source <scratchpad>/fu2wf_ids.env
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Workflow run cancelled through the API ends failed/invariant_violation when its step then fails" --id "$N4" -l workflows,backend,bug \
  -d "WorkflowEngine.cancel() (engine.py, behind POST /api/v1/workflows/runs/{id}/control with action cancel) sets the run to cancelled and cancel_requested, and kills recorded subprocess groups, but cancels no asyncio task. If the step in flight then raises (for example a RuntimeError from its killed subprocess), the step loop's failure path calls _update_run_status_guarded(status='failed') from cancelled. That transition is not allowed, and its default on_reject='fail_run' overwrites the run with status failed and error invariant_violation. Events: run_cancelled, step_failed, transition_rejected, run_failed. Measured on origin/dev e06eeec68c and with TASK-13891's fix (reviewer probe rv_fu2wf_probe_cancel_api.py). TASK-13891 keeps 'cancelled' for a step's own CancelledError after an API cancel, as dev did, and leaves this case as it was." \
  --ac "A run cancelled through the API stays cancelled, with no run_failed event, when its in-flight step then raises any exception, in start_run and continue_run, with a test" \
  --ac "A run that was not cancelled still fails as before when a step raises"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13891 \
  --append-notes "RED on dev <commit>: test_engine_step_own_cancelled_error.py gave 6 failed, 8 passed. Each failure was CancelledError raised out of start_run or continue_run, for map, parallel with fail_fast, and a plain step (a plain step was not in the task text, which is why the fix is in the engine rather than in _run_cancelling_siblings). The passes are the AC 3 guards, each for start_run and continue_run: the step timeout, a cancelled run task, a cancel through the run-cancel API, and an unreadable cancel flag." \
  --append-notes "Fix: both engine step loops catch asyncio.CancelledError. They re-raise when the run's task has cancel requests pending (current_task().cancelling() > 0) or the run was cancelled through the API (db.is_cancel_requested(run_id); engine.cancel() cancels no task). The flag read is suppressed for _CANCEL_FLAG_READ_ERRORS (the engine's noncritical tuple plus sqlite3.Error and the backend DatabaseError) and defaults to cancelled, so a failed read propagates the CancelledError as on dev. Otherwise the step failed with AdapterError('step_raised_cancelled_error'), which is retriable like any adapter error. GREEN: 14 passed; with the fan-out and map integration files, 27 passed. Six mutants (each clause's task half, flag half and read default): each is killed." \
  --append-notes "Plan reviews: an earlier draft re-raised on cancelling() only, so an API cancel followed by the step's own CancelledError ended failed/invariant_violation instead of dev's cancelled; the flag half and its test fix that. The same overwrite for any other exception after an API cancel is pre-existing and filed as $N4. Docs: none describe the engine's handling of a step's CancelledError; no change. Bandit -ll on engine.py: 0 results. Plan: Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A step that raises its own CancelledError (a map item, a parallel sub-step with fail_fast, or a plain step) now fails the step and the run, with error step_raised_cancelled_error, in both start_run and continue_run. On dev the run stayed running for the orphan reaper. A real cancellation still propagates as before: the run task cancelled (for example at shutdown) leaves the run for the reaper, a cancel through the run-cancel API keeps the run cancelled, an unreadable cancel flag propagates the CancelledError, and the step timeout is still a step_timeout. Follow-up: $N4 (an API cancel followed by any other step error ends invariant_violation)." \
  -s Done
git add tldw_Server_API/app/core/Workflows/engine.py tldw_Server_API/tests/Workflows/test_engine_step_own_cancelled_error.py
git add -A backlog/tasks
git commit -m "fix(workflows): a step that raises its own CancelledError fails the run (TASK-13891)

The engine's step loops caught only TimeoutError and Exception, so a
CancelledError that a step raised itself left start_run and the run
stayed running. A real cancel, with requests pending on the task or
the run cancelled through the API, still propagates, and so does the
CancelledError when the cancel flag cannot be read.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: a cancelled `tts` or `multi_voice_tts` step kills its ffmpeg (TASK-13887 AC 1)

**Files:**
- Modify: `tldw_Server_API/app/core/Workflows/adapters/audio/tts.py` (imports, `_WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS`, `:234-254`)
- Modify: `tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py` (imports, `_MULTI_TTS_NONCRITICAL_EXCEPTIONS`, and the 7 sites at `:64`, `:153`, `:192`, `:267`, `:304`, `:398`, `:488`)
- Modify: `tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py:1818` and `:1845-1849` (`test_system_tts_wrapper_converts_requested_format`'s patch target)
- Create: `tldw_Server_API/tests/Workflows/adapters/test_audio_subprocess_cancellation.py`

**Interfaces:**
- Consumes: `subprocess_utils.run_checked_async(cmd, *, timeout, check=True, capture_output=True) -> subprocess.CompletedProcess[bytes]`. It raises `subprocess.TimeoutExpired` on a timeout, and kills and reaps the child on any `BaseException`, including cancellation.
- Produces: no signature changes. Each helper returns what it returned on dev, for success, failure and timeout.

- [ ] **Step 1: Write the failing tests and the guard**

Create `tldw_Server_API/tests/Workflows/adapters/test_audio_subprocess_cancellation.py`:

```python
"""A cancelled tts or multi_voice_tts step kills the ffmpeg it started (TASK-13887)."""

import ast
import asyncio
import contextlib
import os
import signal
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Workflows.adapters.audio import multi_voice_tts
from tldw_Server_API.app.core.Workflows.adapters.audio import tts as tts_adapter

pytestmark = pytest.mark.unit

_ADAPTERS_DIR = Path(tts_adapter.__file__).resolve().parents[1]


@pytest.fixture
def slow_ffmpeg(tmp_path, monkeypatch):
    """An "ffmpeg" that writes its pid, then runs until it is killed."""
    pid_file = tmp_path / "ffmpeg.pid"
    script = tmp_path / "ffmpeg"
    script.write_text(f'#!/bin/sh\necho $$ > "{pid_file}"\nexec sleep 30\n')
    script.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda name: str(script) if name == "ffmpeg" else None)
    yield pid_file
    with contextlib.suppress(FileNotFoundError, ProcessLookupError, ValueError):
        os.kill(int(pid_file.read_text()), signal.SIGKILL)  # dev leaves it running


async def _cancel_once_ffmpeg_runs(step, pid_file: Path) -> int:
    """Run the step until its ffmpeg has started, cancel it, and return ffmpeg's pid."""
    task = asyncio.create_task(step)
    for _ in range(500):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return int(pid_file.read_text())


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_a_cancelled_multi_voice_tts_step_kills_its_ffmpeg(slow_ffmpeg, tmp_path) -> None:
    """On dev the cancel left ffmpeg running to its end, past the step's timeout."""
    pid = await _cancel_once_ffmpeg_runs(
        multi_voice_tts._normalize_audio(tmp_path / "in.mp3", tmp_path / "out.mp3"), slow_ffmpeg
    )

    assert not _is_running(pid)


async def test_a_cancelled_tts_step_kills_its_ffmpeg(slow_ffmpeg, tmp_path, monkeypatch) -> None:
    """On dev the cancel during post-process normalization left ffmpeg running."""
    monkeypatch.chdir(tmp_path)

    class _Speech:
        async def generate_speech(self, _request, provider=None, **_kwargs):  # noqa: ARG002
            yield b"audio"

    async def _service():
        return _Speech()

    monkeypatch.setattr(tts_adapter, "get_tts_service_v2", _service)
    config = {
        "input": "Hello",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "mp3",
        "post_process": {"normalize": True},
    }

    pid = await _cancel_once_ffmpeg_runs(tts_adapter.run_tts_adapter(config, {}), slow_ffmpeg)

    assert not _is_running(pid)


_SPAWNS = {"create_subprocess_exec", "create_subprocess_shell", "subprocess_exec", "subprocess_shell"}


def _spawns_directly(node: ast.AST) -> bool:
    """``asyncio.create_subprocess_exec(...)``, ``loop.subprocess_exec(...)``, or ``from asyncio import create_subprocess_exec``."""
    if isinstance(node, ast.Attribute):
        return node.attr in _SPAWNS
    return isinstance(node, ast.ImportFrom) and any(alias.name in _SPAWNS for alias in node.names)


def test_workflow_adapters_start_commands_only_through_subprocess_utils() -> None:
    """run_checked_async kills and reaps its child on cancel; a direct asyncio spawn does not (dev: 8 sites)."""
    direct = [
        f"{path.relative_to(_ADAPTERS_DIR)}:{node.lineno}"
        for path in sorted(_ADAPTERS_DIR.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if _spawns_directly(node)
    ]

    assert direct == []
```

- [ ] **Step 2: Run them on unchanged source**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/adapters/test_audio_subprocess_cancellation.py`
Expected: `3 failed`.
- The two behavioural tests fail with `assert not True` (`where True = _is_running(<pid>)`).
- The guard fails with `Left contains 8 more items, first extra item: 'audio/multi_voice_tts.py:64'`.
- The fixture kills the leftover processes.

- [ ] **Step 3: Route `tts.py`'s ffmpeg through `run_checked_async`**

In `tts.py`, replace the imports

```python
import asyncio
import contextlib
import uuid
```

with

```python
import subprocess
import uuid
```

After `from tldw_Server_API.app.core.Workflows.adapters.audio._config import TTSConfig`, add:

```python
from tldw_Server_API.app.core.Workflows.subprocess_utils import run_checked_async
```

In `_WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS`, after `TimeoutError,`, add `subprocess.TimeoutExpired,`.

Replace:

```python
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=str(out_dir),
                    )
                    try:
                        await asyncio.wait_for(proc.communicate(), timeout=120)
                    except asyncio.TimeoutError:
                        proc.kill()
                        with contextlib.suppress(_WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS):
                            await proc.communicate()
                    else:
                        if proc.returncode == 0:
                            normalized = True
                            normalized_path = norm_out
                        else:
                            normalized = False
```

with:

```python
                    # Kills and reaps ffmpeg on a timeout or a cancel (TASK-13887)
                    proc = await run_checked_async(cmd, timeout=120, check=False)
                    if proc.returncode == 0:
                        normalized = True
                        normalized_path = norm_out
```

`normalized` is already `False` before this block. `asyncio` and `contextlib` have no other use in `tts.py`: confirm with `grep -n "asyncio\.\|contextlib\." tldw_Server_API/app/core/Workflows/adapters/audio/tts.py`, which should print nothing.

- [ ] **Step 4: Route `multi_voice_tts.py`'s seven sites through `run_checked_async`**

Imports:
- Replace `import asyncio\nimport contextlib\nimport shutil\nimport tempfile` with `import contextlib\nimport shutil\nimport subprocess\nimport tempfile`. `contextlib` stays: `contextlib.suppress(OSError)` is used at four other places.
- After `from tldw_Server_API.app.core.Workflows.adapters.audio._config import MultiVoiceTTSConfig`, add `from tldw_Server_API.app.core.Workflows.subprocess_utils import run_checked_async`.
- In `_MULTI_TTS_NONCRITICAL_EXCEPTIONS`, after `TimeoutError,`, add `subprocess.TimeoutExpired,`.

Each site has this shape:

```python
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=<T>)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.communicate()
            return <FAIL>
```

Replace each with `proc = await run_checked_async(cmd, timeout=<T>, check=False)`, and keep the lines after it:

| Function | `<T>` | `<FAIL>` | Line kept after it |
|---|---|---|---|
| `_generate_silence` | 30 | `False` | `return proc.returncode == 0` |
| `_convert_wav_to_format` | 120 | `False` | `return proc.returncode == 0 and output_path.exists()` |
| `_synthesize_section_with_system_tts` | 120 | `0` | `if proc.returncode != 0 or not synth_path.exists():` |
| `_concat_files` | 600 | `False` | `return proc.returncode == 0` |
| `_normalize_audio` | 120 | `False` | `return proc.returncode == 0` |
| `_probe_duration_seconds` | 30 | `None` | `if proc.returncode != 0:` |
| `_mix_background_track` | 180 | `False` | `return proc.returncode == 0 and output_path.exists()` |

`_probe_duration_seconds` read `stdout` from `communicate()`. In it, replace

```python
        text = (stdout or b"").decode("utf-8", errors="ignore").strip()
```

with

```python
        text = (proc.stdout or b"").decode("utf-8", errors="ignore").strip()
```

On a timeout, `run_checked_async` raises `subprocess.TimeoutExpired`. Each site's existing `except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:` catches it, and returns the same `<FAIL>` value as dev's timeout branch.

- [ ] **Step 5: Re-point the one test that patched `multi_voice_tts.asyncio`**

`multi_voice_tts` no longer imports `asyncio`. In `test_audio_adapters.py::TestMultiVoiceTTSAdapter::test_system_tts_wrapper_converts_requested_format`:
- add `from tldw_Server_API.app.core.Workflows import subprocess_utils` above the test's `from tldw_Server_API.app.core.Workflows.adapters.audio import multi_voice_tts`;
- replace `multi_voice_tts.asyncio,` with `subprocess_utils.asyncio,` in its `monkeypatch.setattr(..., "create_subprocess_exec", create_process)`.

`run_checked_async` calls `asyncio.create_subprocess_exec(*cmd, stdout=..., stderr=...)`, so the mock still sees the same command and the test's assertions are unchanged.

- [ ] **Step 6: Run them and every test that drives these adapters**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/adapters/test_audio_subprocess_cancellation.py tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py tldw_Server_API/tests/Workflows/test_tts_workflow_normalization.py tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/TTS/unit/test_audio_converter_sanitizer.py`
Expected: `140 passed`.

Then run `pgrep -fl '^sleep 30$'`. Expect no process younger than this run:
- check any hit's age and parent with `ps -o pid,ppid,etime,command -p <pid>`;
- other sessions' polling loops also run `sleep 30`, so leave those alone.

- [ ] **Step 7: Commit** (TASK-13887 is closed in Task 4)

```bash
git add tldw_Server_API/app/core/Workflows/adapters/audio/tts.py \
  tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py \
  tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py \
  tldw_Server_API/tests/Workflows/adapters/test_audio_subprocess_cancellation.py
git commit -m "fix(workflows): cancelled tts and multi_voice_tts steps kill their ffmpeg (TASK-13887)

The eight ffmpeg, ffprobe and espeak calls killed their child only on
their own timeout, so a cancelled step (a failed sibling, or the
engine's step timeout) left it running. They now go through
run_checked_async, which kills and reaps on any exit.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `media_ingest` waits for yt-dlp to exit; `workflow_call`'s AC moves to its repair (TASK-13887 AC 2 and AC 3)

**Files:**
- Modify: `tldw_Server_API/app/core/Workflows/subprocess_utils.py:15-23` (`SubprocessTask`) and `:62-70` (`start_process`)
- Modify: `tldw_Server_API/app/core/Workflows/adapters/media/ingest.py:339-390` (the poll, the run-cancel check after it, and its failure branch)
- Modify: `Docs/Operations/Usage_Quotas.md:96`, `Docs/Design/Workflows.md:106`
- Test: `tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py` (append; it is `unit`-marked)

**Interfaces:**
- Consumes: `subprocess_utils.start_process(cmd, workdir, log_dir) -> SubprocessTask`; `WorkflowEngine(db).start_run(run_id, RunMode.SYNC)`; the step-level `timeout_seconds`, which the engine's `wait_for` uses.
- Produces:
  - `SubprocessTask.proc: subprocess.Popen[bytes] | None`. `start_process` sets it; other constructors leave it `None`.
  - `media_ingest` metadata gains the status `download_failed`, with `returncode`.
  - An exit caused by the run's cancel returns `{"__status__": "cancelled"}`.

- [ ] **Step 1: Write the failing tests**

In the test file's imports:
- add `import functools`, `import os`, `import signal` and `import sys` after `import contextlib`;
- add `from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase` before `from tldw_Server_API.app.core.exceptions import ...`;
- add `from tldw_Server_API.app.core.Workflows import subprocess_utils` before `from tldw_Server_API.app.core.Workflows.adapters.control import flow`;
- add `from tldw_Server_API.app.core.Workflows.engine import RunMode, WorkflowEngine` after `from tldw_Server_API.app.core.Workflows.adapters.media import ingest`.

Then append:

```python
_SOURCES = [{"uri": "https://media.example.com/clip"}]  # never fetched: start_process is replaced


@contextlib.contextmanager
def _fake_yt_dlp(script: str):
    """Run ``script`` (Python, in the step directory) in place of yt-dlp; yield the downloads it started."""
    started: list[SubprocessTask] = []

    def start(_cmd, workdir, log_dir):
        started.append(subprocess_utils.start_process([sys.executable, "-c", script], workdir, log_dir))
        return started[-1]

    with patch.object(ingest, "is_test_mode", return_value=False), patch.object(
        ingest, "is_url_allowed_for_tenant", return_value=True
    ), patch.object(ingest, "start_process", side_effect=start):
        yield started


async def _ingest_with_fake_yt_dlp(script: str, timeout_seconds: int, context=None) -> tuple[dict, SubprocessTask]:
    """Run the media_ingest adapter itself with ``script`` in place of yt-dlp."""
    config = {"sources": _SOURCES, "timeout_seconds": timeout_seconds}
    with _fake_yt_dlp(script) as started:
        result = await ingest.run_media_ingest_adapter(config, context or {"tenant_id": "default"})
    return result, started[0]


def _stop(download: SubprocessTask) -> bool:
    """Kill the download if it is still running (dev leaves it), and say whether it was there to kill."""
    try:
        os.kill(download.pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    return True


async def test_media_ingest_returns_only_after_yt_dlp_has_exited(tmp_path, monkeypatch) -> None:
    """On dev it returned as soon as yt-dlp's .part file appeared, with the download still running (TASK-13887)."""
    monkeypatch.setenv("WORKFLOWS_TMP", str(tmp_path))
    script = (
        "import pathlib, time\n"
        "pathlib.Path('clip.mp4.part').write_bytes(b'x')\n"
        "time.sleep(1)\n"
        "pathlib.Path('clip.mp4.part').rename('clip.mp4')\n"
    )

    result, download = await _ingest_with_fake_yt_dlp(script, timeout_seconds=30)
    files = sorted(path.name for path in download.workdir.glob("*.*"))
    was_running = _stop(download)

    assert [entry["status"] for entry in result["metadata"]] == ["downloaded"]
    assert files == ["clip.mp4"]
    assert was_running is False


async def test_media_ingest_reports_a_download_that_yt_dlp_failed(tmp_path, monkeypatch) -> None:
    """yt-dlp exited with an error and wrote nothing; on dev the poll waited out the step's timeout instead."""
    monkeypatch.setenv("WORKFLOWS_TMP", str(tmp_path))

    result, download = await _ingest_with_fake_yt_dlp("raise SystemExit(1)", timeout_seconds=2)
    _stop(download)

    assert [(entry["status"], entry.get("returncode")) for entry in result["metadata"]] == [("download_failed", 1)]


async def test_media_ingest_whose_run_was_cancelled_reports_cancelled(tmp_path, monkeypatch) -> None:
    """engine.cancel() kills the download, so yt-dlp exits -15 before the poll sees the cancel: a cancel, not a failure."""
    monkeypatch.setenv("WORKFLOWS_TMP", str(tmp_path))
    checks = iter([False])  # the poll's first is_cancelled() passes; every later one sees the run's cancel
    context = {"tenant_id": "default", "is_cancelled": lambda: next(checks, True)}
    killed_by_the_cancel = "import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n"

    result, download = await _ingest_with_fake_yt_dlp(killed_by_the_cancel, timeout_seconds=30, context=context)
    _stop(download)

    assert result == {"__status__": "cancelled"}


async def test_a_download_that_outlasts_the_steps_timeout_fails_the_step(tmp_path, monkeypatch) -> None:
    """The step's timeout_seconds is the download's budget; on dev the run 'succeeded' at the .part file (TASK-13887)."""
    monkeypatch.setenv("WORKFLOWS_TMP", str(tmp_path))
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    step = {
        "id": "ingest",
        "type": "media_ingest",
        # The step's budget, the engine's wait_for. 3 s, not 1: an engine's first run in a fresh process spends
        # about 1.1-1.5 s before the adapter starts the download, and dev's poll must get to see the .part file.
        "timeout_seconds": 3,
        "config": {"sources": _SOURCES, "timeout_seconds": 60},  # a larger adapter budget cannot extend it
    }
    db.create_run(
        run_id="run",
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "run", "version": 1, "steps": [step]},
    )
    slow_download = "import pathlib, time\npathlib.Path('clip.mp4.part').write_bytes(b'x')\ntime.sleep(30)\n"
    quick_terminate = functools.partial(subprocess_utils.terminate_process, grace_ms=100)  # the real grace is 5 s

    with _fake_yt_dlp(slow_download) as started, patch.object(ingest, "terminate_process", quick_terminate):
        await asyncio.wait_for(WorkflowEngine(db).start_run("run", RunMode.SYNC), timeout=30)
    try:
        run = db.get_run("run")
        assert (run.status, run.error) == ("failed", "")
        assert [event["event_type"] for event in db.get_events("run")][-3:] == ["step_timeout", "step_failed", "run_failed"]
        assert sorted(path.name for path in started[0].workdir.glob("*.*")) == ["clip.mp4.part"]  # yt-dlp never finished
        assert started[0].proc.poll() is not None  # the download was stopped, not left running
    finally:
        _stop(started[0])
```

Notes on these tests:
- `_stop` kills by pid, not by group: on macOS, `killpg` on a group whose only member is a zombie raises `PermissionError` (measured). A reaped child raises `ProcessLookupError`.
- The engine-level test (review I3) shortens `terminate_process`'s grace to 100 ms. The adapter's `CancelledError` branch calls it through `asyncio.to_thread`, and the real 5 s grace would only slow the test. Its `finally` kills dev's still-running download.

- [ ] **Step 2: Run them on unchanged source**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py -k "media_ingest or download"`
Expected: `3 failed, 2 passed`. The failures:
- `assert ['clip.mp4.part'] == ['clip.mp4']`;
- `assert [('timeout', None)] == [('download_failed', 1)]`, which takes about 2 s, waiting out `timeout_seconds`;
- `assert ('succeeded', None) == ('failed', '')`: on dev the run "succeeded" at the `.part` file.

**Why the budget test's step has 3 s (re-review I-1).** With 1 s, an engine's first run in a fresh process spent 1.1-1.5 s before the adapter reached `start_process`. Dev's run then also ended `step_timeout`, and the test failed only at `.proc`, which dev's `SubprocessTask` lacks. Measured with 3 s, each in a fresh process:
- on dev, both `-k "media_ingest or download"` and the budget test alone fail with `('succeeded', None) == ('failed', '')`, and the whole file gives `4 failed, 10 passed`;
- with the fix (r2's app code and r3's), the whole file gives `14 passed`, and the budget test alone passes.

The `clip.mp4.part`-only assert pins that the step returned before yt-dlp finished, independently of `.proc`.

The passes are TASK-13863's `test_a_cancelled_media_ingest_terminates_its_download`, and the run-cancel guard: on dev the poll's next `is_cancelled()` also returns `cancelled`.

- [ ] **Step 3: Keep the `Popen`, poll it, and check the run's cancel after it exits**

In `subprocess_utils.py`, add a field to `SubprocessTask` after `started_at: float = 0.0`:

```python
    proc: subprocess.Popen[bytes] | None = None
```

In `start_process`, add `proc=proc,` after `started_at=time.time(),` in the `SubprocessTask(...)` call.

In `ingest.py`, replace:

```python
            # Poll with timeout
            exited = False
            while time.time() - start_ts < timeout_seconds:
```

with:

```python
            # Poll with timeout
            exit_code: int | None = None
            while time.time() - start_ts < timeout_seconds:
```

Replace:

```python
                # Check if any file downloaded
                if any(step_dir.glob("*.*")):
                    exited = True
                    break
            if not exited:
                terminate_process(task)
```

with:

```python
                # A file appears long before the download ends (yt-dlp's .part): wait for yt-dlp to exit (TASK-13887)
                exit_code = task.proc.poll() if task.proc is not None else None
                if exit_code is not None:
                    break
            exited = exit_code is not None
            if exited and callable(context.get("is_cancelled")) and context["is_cancelled"]():
                return {"__status__": "cancelled"}  # the run's cancel (engine.cancel) killed the download
            failed = exited and (exit_code != 0 or not any(step_dir.glob("*.*")))
            if not exited or failed:
                if not exited:
                    terminate_process(task)
```

Replace:

```python
                meta_timeout = {
                    "source": uri,
                    "status": "timeout",
                }
```

with:

```python
                meta_timeout: dict[str, Any] = {
                    "source": uri,
                    "status": "download_failed" if failed else "timeout",
                }
                if failed:
                    meta_timeout["returncode"] = exit_code
```

Notes on this code:
- `Any` is already imported in `ingest.py`.
- The `exit_code` local and the annotation keep mypy at dev's count. `task.proc.returncode` on an Optional would add a `union-attr` error, and a literal-inferred `dict[str, str]` would reject the int.
- The cancel check after the exit is review m5. `engine.cancel()` kills the recorded group, so `poll()` sees -15 before the loop's next `is_cancelled()` check. Without the check, that source was recorded as `download_failed` (measured with r1's poll).
- The rest of the branch (log tails, `step_log_tail`, `continue`) is unchanged.

- [ ] **Step 4: Run them and the neighbours**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py tldw_Server_API/tests/Workflows/adapters/test_media_adapters.py tldw_Server_API/tests/Workflows/test_orphan_requeue_unit.py tldw_Server_API/tests/CI/test_async_blocking_call_guard.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_external_tools.py`
Expected: `138 passed`: the fan-out file's 14, plus the 124 neighbours.

- [ ] **Step 5: Update the two docs**

In `Docs/Operations/Usage_Quotas.md`, in the "A workflow" bullet, replace

```text
A download that a `media_ingest` sub-step is still waiting on is stopped. A download whose first file has already appeared keeps running, and so does an ffmpeg command that a TTS sub-step started (both TASK-13887).
```

with (review m11)

```text
A download that a `media_ingest` sub-step started is stopped: the sub-step now waits for the download, so a cancel always finds it running or finished. So is an ffmpeg command that a `tts` or `multi_voice_tts` sub-step started (TASK-13887).
```

In `Docs/Design/Workflows.md`, at the end of the `media_ingest` bullet, after "and artifact persistence of downloaded files.", add (review I3; adjust to the owner's answer to Decision 1):

```text
 A network download finishes before the step returns: the step waits for yt-dlp to exit, up to the step's `timeout_seconds` (default 300), which all of the step's sources share. A download that outlasts it is stopped (SIGTERM, then SIGKILL after 5 s), and the step fails, as does the run unless `on_failure` routes around it, with `step_timeout`. For long media, raise the step's own `timeout_seconds` (the step-level key, beside `type`, not under `config`; the WebUI editor has no control for it, so set it in the definition's JSON). yt-dlp exiting with an error, or writing no file, is recorded as `download_failed` with its `returncode`.
```

Neither file has a `Docs/Published` mirror. Run `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs`: expected `0 failed`.

- [ ] **Step 6: File the two follow-ups that this task finds**

The ids come from Setup (`<scratchpad>/fu2wf_ids.env`).

```bash
source <scratchpad>/fu2wf_ids.env
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Workflow workflow_call cannot run a sub-workflow" --id "$N1" -l workflows,backend,bug \
  -d "run_workflow_call_adapter (Workflows/adapters/control/orchestration.py) never creates a child run. It imports get_workflows_db from tldw_Server_API.app.core.Workflows.workflows_db (:48), a module that has never existed, so every call returns {'error': 'workflow_call_error'} (probe while planning TASK-13887, origin/dev e06eeec68c). Behind that: it calls create_run(status='pending') (:67), which has no status parameter; it creates the child without a definition_snapshot, so the child would have no steps; and it passes mode='sync' as a string (:74). Its unit tests inject a fake module through sys.modules. The WebUI step registry offers the step (apps/packages/ui/src/components/WorkflowEditor/step-registry.ts). The child run also needs a tenant and user check on get_workflow, and a guard against a workflow calling itself. This task's plan decides whether to repair the step or remove it from the registry. TASK-13887 AC 2 (a cancelled or timed-out child run ends terminal) moved here." \
  --ac "A workflow_call step with wait true runs the named workflow's definition as a child run and returns its status and outputs, with a test that injects no fake module" \
  --ac "A cancelled or timed-out workflow_call child run ends in a terminal status instead of staying running, with a test (from TASK-13887 AC 2)" \
  --ac "workflow_call runs only workflows its run's tenant and user may run, and refuses a call cycle, with tests"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "media_ingest and the orphan reaper block the event loop for terminate_process's grace" --id "$N2" -l workflows,backend,performance \
  -d "terminate_process (Workflows/subprocess_utils.py) sends SIGTERM, then time.sleep()s its 5 s grace before SIGKILL. Three callers run it on the event loop and stall every request for 5 s: media_ingest's cooperative-cancel branch and its timeout branch (Workflows/adapters/media/ingest.py, the two plain terminate_process(task) calls in the poll; the CancelledError branch already uses asyncio.to_thread), and WorkflowEngine._reap_orphans (engine.py, terminate_process(task, grace_ms=grace_ms)), which every SYNC start_run awaits before its first step. engine.cancel() also calls it synchronously for each recorded subprocess. Found while planning TASK-13887." \
  --ac "media_ingest's cooperative-cancel and timeout branches terminate the download off the event loop, with a test that the loop keeps running during the grace" \
  --ac "The orphan reaper and engine.cancel() terminate a step's subprocess off the event loop, with a test"
```

- [ ] **Step 7: Reword AC 2, close TASK-13887 and commit**

`--remove-ac 2` runs before `--ac`, so the new criterion becomes #3: #1 is ffmpeg, #2 is media_ingest, #3 is the reworded workflow_call AC. The reviewer's dry run confirmed this numbering.

```bash
source <scratchpad>/fu2wf_ids.env
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13887 \
  --remove-ac 2 \
  --ac "workflow_call's cancel cleanup moves to $N1, which repairs the adapter: on dev it creates no child run to leave running (it imports a module that does not exist, Workflows.workflows_db, and passes status= to create_run, which has no such parameter), recorded with the probe" \
  --append-notes "AC 2 reworded (premise wrong, measured on <commit>): the real adapter, with no injected module, returns {'error': 'workflow_call_error', 'result': None} for wait true and false, because import tldw_Server_API.app.core.Workflows.workflows_db fails ('No module named ...'), and WorkflowsDatabase.create_run has no status parameter. The module has never existed (git log --all). No child run is ever created, so there is none to leave running and no RED test is possible. $N1 repairs workflow_call and carries the requirement (its AC 2)." \
  --append-notes "AC 1 RED on <commit>: test_audio_subprocess_cancellation.py gave 3 failed: ffmpeg still running after the cancel, for multi_voice_tts._normalize_audio and run_tts_adapter's normalization; and 8 direct asyncio spawn sites (tts.py:235, multi_voice_tts.py:64, :153, :192, :267, :304, :398, :488). Fix: all 8 call run_checked_async(cmd, timeout=..., check=False), which kills and reaps on cancel; subprocess.TimeoutExpired joins each module's noncritical tuple, so a timeout returns what dev returned. tts.py drops cwd=out_dir (both paths in its command are absolute). The AST guard also flags from-imports of the spawn functions and loop.subprocess_exec/subprocess_shell. GREEN: 140 passed with test_audio_adapters.py, test_tts_workflow_normalization.py, Watchlists/test_audio_briefing_workflow.py and TTS/unit/test_audio_converter_sanitizer.py; test_system_tts_wrapper_converts_requested_format now patches subprocess_utils.asyncio." \
  --append-notes "AC 3 RED on <commit>: -k 'media_ingest or download' gave 3 failed, 2 passed: ['clip.mp4.part'] == ['clip.mp4'] (returned at yt-dlp's .part file with yt-dlp running); [('timeout', None)] == [('download_failed', 1)] (a failed yt-dlp waited out timeout_seconds); and, through the engine, ('succeeded', None) == ('failed', '') (the run succeeded at the .part file). Fix: SubprocessTask keeps its Popen (proc); the poll waits for proc.poll(), so media_ingest returns only after yt-dlp exits; an exit the run's cancel caused returns cancelled; an exit with an error or with no file is download_failed with returncode. GREEN: 138 passed with test_media_adapters.py, test_orphan_requeue_unit.py, CI/test_async_blocking_call_guard.py and Web_Scraping/test_phase3_preflight_external_tools.py." \
  --append-notes "Download budget (owner Decision 1 in the plan; record the owner's answer here): the step-level timeout_seconds (default 300), shared by the step's sources, bounds the wait; a longer download is stopped and fails the step with step_timeout. Pinned by test_a_download_that_outlasts_the_steps_timeout_fails_the_step." \
  --append-notes "Docs: Usage_Quotas.md's 'A workflow' bullet and Docs/Design/Workflows.md's media_ingest bullet updated (no Published mirror); tests/Docs 0 failed. Bandit -ll on tts.py, multi_voice_tts.py, ingest.py and subprocess_utils.py: 0 results. Follow-ups: $N1 (workflow_call), $N2 (terminate_process on the event loop). Plan: Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A cancelled tts or multi_voice_tts step now kills and reaps the ffmpeg (or ffprobe, espeak) it started: all eight call sites go through run_checked_async, and an AST guard keeps direct asyncio spawns out of the workflow adapters. media_ingest now returns only after yt-dlp has exited, within the step's timeout_seconds, so a cancel at any point of a download stops it; a download that outlasts the budget fails the step with step_timeout, a failed download is reported as download_failed, and a download killed by the run's cancel reports cancelled. workflow_call creates no child run on dev at all (it imports a missing module), so its cancel requirement moved to $N1, which repairs the adapter." \
  -s Done
git add tldw_Server_API/app/core/Workflows/subprocess_utils.py \
  tldw_Server_API/app/core/Workflows/adapters/media/ingest.py \
  tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py \
  Docs/Operations/Usage_Quotas.md Docs/Design/Workflows.md
git add -A backlog/tasks
git commit -m "fix(workflows): media_ingest waits for yt-dlp to exit (TASK-13887)

The poll ended at the first file in the step directory, which is
yt-dlp's .part file, so the step returned with the download still
running and a partial artifact. It now polls the process itself, within
the step's timeout, and reports a failed or cancelled download.
workflow_call's part of the task moves to <N1>: on dev it never
creates a child run.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

In the commit message, write the computed id in place of `<N1>`.

---

### Task 5: a continued run runs outside its creator's ChaCha operation (TASK-13897 AC 1)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py`: add `outside_chacha_operation` after `chacha_operation`, before `_F = TypeVar(...)`.
- Modify: `tldw_Server_API/app/core/Workflows/engine.py`: the import block, `continue_run` at `:1131-1143`, and `_schedule_human_timeout`'s `create_task` at `:1828-1831`.
- Create: `tldw_Server_API/tests/Workflows/test_engine_detached_tasks_chacha_operation.py`

**Interfaces:**
- Consumes: `chacha_operation(*, independent=False)`. With no current operation it owns one. Inside a closed one it raises `ClosedChaChaOperationError("The ChaCha operation has finished")`.
- Produces: `outside_chacha_operation() -> contextmanager[None]` in `operation_scope`. Inside it, `_current_operation.get()` is `None`. On exit the caller's operation is restored with the contextvar token.

- [ ] **Step 1: Write the failing test**

Create `tldw_Server_API/tests/Workflows/test_engine_detached_tasks_chacha_operation.py`:

```python
"""A run that a detached engine task continues does not inherit its creator's ChaCha operation (TASK-13897).

The creator is an HTTP request (ChaChaOperationMiddleware) or a voice command, each inside
``chacha_operation(independent=True)``. That operation has closed by the time the detached
task runs a step, so a step that joins it, as any ChaCha owner without ``independent=True``
does, raised ClosedChaChaOperationError.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.DB_Management.chacha import operation_scope
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows import engine as engine_module
from tldw_Server_API.app.core.Workflows.engine import RunMode, WorkflowEngine

pytestmark = pytest.mark.integration

_REAL_GET_ADAPTER = engine_module.get_adapter
_JOINS = {"id": "s", "type": "summarize", "config": {}}
_FIRST = {"id": "first", "type": "log", "config": {"message": "first"}}


async def _joins_the_current_operation(_config, _context):
    """A ChaCha step that does not open an independent operation."""
    with chacha_operation():
        return {"joined": True}


def _create_run(db: WorkflowsDatabase, run_id: str, steps: list[dict]) -> None:
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": run_id, "version": 1, "steps": steps},
    )


async def _continued_by_a_request(db: WorkflowsDatabase, engine: WorkflowEngine) -> str:
    """POST /runs/{id}/retry, approve or reject: the route starts continue_run and returns."""
    _create_run(db, "run", [_FIRST, _JOINS])
    db.update_run_status("run", status="waiting_human")
    with chacha_operation(independent=True):
        asyncio.create_task(engine.continue_run("run", after_step_id="first"))  # as the route does
    return "run"


async def _continued_after_a_human_step_timed_out(db: WorkflowsDatabase, engine: WorkflowEngine) -> str:
    """A SYNC run reaches a human step, whose timeout task later continues it."""
    human = {
        "id": "h",
        "type": "wait_for_human",
        "config": {"assigned_to_user_id": "user", "timeout_seconds": 0.1},
        "on_timeout": "s",
    }
    _create_run(db, "run", [human, _JOINS])
    with chacha_operation(independent=True):
        await engine.start_run("run", RunMode.SYNC)
    return "run"


async def _continued_by_the_orphan_reaper(db: WorkflowsDatabase, engine: WorkflowEngine) -> str:
    """A SYNC run's start reaps another user's stale run and continues it."""
    _create_run(db, "orphan", [_JOINS])
    db.create_step_run(
        step_run_id="orphan:s:1", tenant_id="default", run_id="orphan", step_id="s", name="s", step_type="summarize"
    )
    stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    db._conn.execute(
        "UPDATE workflow_step_runs SET heartbeat_at = ?, status = 'running' WHERE step_run_id = ?",
        (stale, "orphan:s:1"),
    )
    db._conn.commit()
    _create_run(db, "run", [_FIRST])
    with chacha_operation(independent=True):
        await engine.start_run("run", RunMode.SYNC)
    return "orphan"


@pytest.mark.parametrize(
    "continue_run",
    [_continued_by_a_request, _continued_after_a_human_step_timed_out, _continued_by_the_orphan_reaper],
    ids=["request_task", "human_step_timeout", "orphan_reaper"],
)
async def test_a_run_continued_by_a_detached_task_runs_outside_its_creators_operation(tmp_path, continue_run) -> None:
    """On dev the step raised ClosedChaChaOperationError ('The ChaCha operation has finished') and the run failed."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    engine = WorkflowEngine(db)
    with patch.object(
        engine_module,
        "get_adapter",
        side_effect=lambda t: _joins_the_current_operation if t == "summarize" else _REAL_GET_ADAPTER(t),
    ):
        run_id = await continue_run(db, engine)
        for _ in range(200):  # the detached task runs after its creator has returned
            if db.get_run(run_id).status in {"succeeded", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.05)

    finished = db.get_run(run_id)
    assert (finished.status, finished.error) == ("succeeded", None)
    assert finished.outputs_json == '{"joined": true}'


async def test_a_human_steps_timeout_task_starts_outside_its_creators_operation(tmp_path) -> None:
    """The task sleeps for the step's timeout, which can be days; on dev it held its creator's operation all along."""
    engine = WorkflowEngine(WorkflowsDatabase(str(tmp_path / "wf.db")))
    before = asyncio.all_tasks()
    with chacha_operation(independent=True):  # the request or voice command whose SYNC run reached the human step
        engine._schedule_human_timeout("run", "h", 3600, None)
    [timeout_task] = [task for task in asyncio.all_tasks() - before if task.get_coro().__name__ == "_timeout_task"]
    try:
        inherited = timeout_task.get_context().get(operation_scope._current_operation)
    finally:
        timeout_task.cancel()

    assert inherited is None
```

- `db._conn.execute(...)` backdates the step's heartbeat, as `test_orphan_requeue_unit.py` already does. `WorkflowsDatabase` has no method that sets an old heartbeat.
- The request case keeps no reference to its task, as the routes do: the step's pending awaits hold it.
- The timeout-task test (review m4) reads the task's own copied context (`Task.get_context()`, Python 3.11+). On dev it holds the closed `ChaChaOperation` for the whole sleep. It uses two private names, `_schedule_human_timeout` and `_current_operation`: no public API exposes either the task or the context variable.

- [ ] **Step 2: Run it on unchanged source**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/test_engine_detached_tasks_chacha_operation.py`
Expected: `4 failed`:
- the three paths, each `assert ('failed', 'T...has finished') == ('succeeded', None)`;
- the timeout-task test: `assert <...ChaChaOperation object at ...> is None`.

- [ ] **Step 3: Add `outside_chacha_operation` and run `continue_run` inside it**

In `operation_scope.py`, after `chacha_operation` and before `_F = TypeVar("_F", bound=Callable[..., Any])`, add:

```python
@contextmanager
def outside_chacha_operation() -> Iterator[None]:
    """Run without the caller's operation, for work that may outlive it, such as a detached task.

    ChaCha calls inside take the thread-local fallback or open their own operations. Never
    wrap such work in one operation of its own instead: concurrent branches would share
    one connection and lose writes (ADR-080).
    """
    token = _current_operation.set(None)
    try:
        yield
    finally:
        _current_operation.reset(token)
```

In `engine.py`, add the import above `from tldw_Server_API.app.core.DB_Management.DB_Manager import ...`:

```python
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import outside_chacha_operation
```

Replace `continue_run`'s body

```python
        """Resume a run, with its owner as the user for LLM usage accounting."""
        with llm_call_root(self._llm_user_for_run(run_id), operation="workflow"):
```

with

```python
        """Resume a run, with its owner as the user for LLM usage accounting.

        Callers start it as a detached task (a route, a human step's timeout, the orphan
        reaper), which outlives their ChaCha operation, so the run never joins it (TASK-13897).
        """
        with outside_chacha_operation(), llm_call_root(self._llm_user_for_run(run_id), operation="workflow"):
```

The `await self._continue_run_impl(...)` call inside the block is unchanged.

In `_schedule_human_timeout`, replace

```python
        try:
            asyncio.create_task(_timeout_task())
        except _WF_NONCRITICAL_EXCEPTIONS:
            return
```

with

```python
        try:
            with outside_chacha_operation():  # the task copies this context, and can outlive its creator by days
                asyncio.create_task(_timeout_task())
        except _WF_NONCRITICAL_EXCEPTIONS:
            return
```

`create_task` copies the current context when it is called. So the task starts with no operation, and its 45-line body needs no re-indent. `_current_operation.reset(token)` then restores the caller's operation.

- [ ] **Step 4: Run it and the operation-scope tests**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/test_engine_detached_tasks_chacha_operation.py tldw_Server_API/tests/DB_Management/test_chacha_operation_scope.py tldw_Server_API/tests/Workflows/test_orphan_requeue_unit.py`
Expected: `24 passed`. The new file gives 4 of them.

- [ ] **Step 5: Commit** (TASK-13897 is closed in Task 6)

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py tldw_Server_API/app/core/Workflows/engine.py \
  tldw_Server_API/tests/Workflows/test_engine_detached_tasks_chacha_operation.py
git commit -m "fix(workflows): a continued run leaves its creator's ChaCha operation (TASK-13897)

continue_run is started as a detached task by the routes, a human
step's timeout and the orphan reaper, and copied the creator's ChaCha
operation, which had closed by the time a step ran. It now runs outside
any inherited operation, and so does the human-step timeout task. The
run is not wrapped in one of its own: concurrent branches would share
its connection (ADR-080).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: a notes step works on PostgreSQL in a continued run; RAG audit; ADR-080 (TASK-13897 AC 2 and AC 3)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Workflows_DB.py`: the `datetime` import, a new `_json_default` after `_utcnow_iso` (`:252`), and the outputs `json.dumps` in `update_run_status` (`:2120`) and `complete_step_run` (`:2513`).
- Modify (regenerated): `Helper_Scripts/ci/scope_predicate_baseline.txt`. Its three `Workflows_DB.py` lines, `:1808`, `:1811` and `:1960`, become `:1815`, `:1818` and `:1967` (review B1).
- Modify: `tldw_Server_API/app/core/Workflows/adapters/rag/search.py:127` (a comment above the `unified_rag_pipeline` call).
- Modify: `Docs/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md` (Consequences, "Detached tasks"), plus its `Docs/Published` copy.
- Test: `tldw_Server_API/tests/Workflows/test_workflows_db.py` (append; and add `from datetime import datetime, timezone` to its imports).
- Create: `tldw_Server_API/tests/Workflows/test_notes_step_continued_after_its_creator_postgres.py`

**Interfaces:**
- Consumes: `outside_chacha_operation` (Task 5); `pg_database_config` (`tests/_plugins/postgres.py:298`); `DatabaseBackendFactory.create_backend(config)`; the module attribute `ChaChaNotes_DB.get_content_backend`, which `CharactersRAGDB` calls when given no backend.
- Produces: `Workflows_DB._json_default(value) -> str`, private. A `date` or `datetime` becomes `value.isoformat()`; anything else raises `TypeError` as `json.dumps` did.

- [ ] **Step 1: Write the PostgreSQL test and the SQLite unit test**

Create `tldw_Server_API/tests/Workflows/test_notes_step_continued_after_its_creator_postgres.py`:

```python
"""On PostgreSQL, a notes step in a run continued after its creator returned succeeds (TASK-13897).

An approval request or a voice command starts ``continue_run`` as a detached task inside its
ChaCha operation, then returns. The notes adapter opens a ChaCha handle per call on the event
loop; on dev that handle joined the closed operation, and the step failed.
"""

import asyncio
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows.adapters.knowledge import crud
from tldw_Server_API.app.core.Workflows.engine import WorkflowEngine

pytestmark = pytest.mark.integration


@pytest.fixture
def notes_on_postgres(pg_database_config, tmp_path, monkeypatch):
    """Every ChaCha handle the notes adapter opens uses the PostgreSQL content backend, whose checkouts are counted."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    taken, returned = Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    monkeypatch.setattr(chacha_module, "get_content_backend", lambda _config: backend)
    monkeypatch.setattr(crud, "is_test_mode", lambda: False)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    try:
        yield SimpleNamespace(outstanding=lambda: sum((taken - returned).values()))
    finally:
        pool.close_all()


async def test_a_notes_step_in_a_run_continued_after_its_creator_returned(notes_on_postgres, tmp_path) -> None:
    """On dev the handle's constructor raised 'The ChaCha operation has finished': the run failed, with no note."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    note = {"action": "create", "title": "From a resumed run", "content": "Saved after the request returned."}
    db.create_run(
        run_id="run",
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "run",
            "version": 1,
            "steps": [
                {"id": "review", "type": "log", "config": {"message": "approved"}},
                {"id": "note", "type": "notes", "config": note},
            ],
        },
    )
    db.update_run_status("run", status="waiting_human")
    with chacha_operation(independent=True):  # the approval request, or the voice command
        task = asyncio.create_task(WorkflowEngine(db).continue_run("run", after_step_id="review"))
    await asyncio.wait_for(task, timeout=60)  # the task's first step runs after the operation closed

    finished = db.get_run("run")
    assert (finished.status, finished.error) == ("succeeded", None)
    saved = json.loads(finished.outputs_json)["note"]
    assert (saved["title"], saved["content"]) == (note["title"], note["content"])
    assert notes_on_postgres.outstanding() == 0
```

The patches live in the fixture, so the test function makes none: the mock-density rule counts test functions only. The note's datetimes come back from PostgreSQL, so this test also covers the serialization fix.

Add `from datetime import datetime, timezone` after `from concurrent.futures import ThreadPoolExecutor` in `test_workflows_db.py`. Then append:

```python
def test_run_and_step_outputs_store_datetimes_as_iso_text(tmp_path):
    """A notes step on PostgreSQL returns datetimes; on dev storing them raised TypeError and failed the run (TASK-13897)."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    db.create_run(
        run_id="run",
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "run", "version": 1, "steps": []},
    )
    db.create_step_run(step_run_id="run:s:1", tenant_id="default", run_id="run", step_id="s", name="s", step_type="notes")
    outputs = {"note": {"title": "t", "created_at": datetime(2026, 10, 22, 17, 45, tzinfo=timezone.utc)}}

    db.complete_step_run(step_run_id="run:s:1", outputs=outputs)
    db.update_run_status("run", status="succeeded", outputs=outputs)

    stored = {"note": {"title": "t", "created_at": "2026-10-22T17:45:00+00:00"}}
    assert json.loads(db.get_run("run").outputs_json) == stored
    assert json.loads(db.list_step_runs(run_id="run")[0]["outputs_json"]) == stored
```

- [ ] **Step 2: Run them on the tree as Task 5 left it**

Run: `TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/test_notes_step_continued_after_its_creator_postgres.py tldw_Server_API/tests/Workflows/test_workflows_db.py -k "notes_step or datetimes"`
Expected: `2 failed`, each `TypeError: Object of type datetime is not JSON serializable`.
- In the PostgreSQL test it comes from `Workflows_DB.update_run_status`.
- Without Task 5, the PostgreSQL test fails earlier, with `('failed', 'Unexpected error initializing DB for user 1: The ChaCha operation has finished')`. That was measured on dev.
- If the PostgreSQL test skips, the container is not reachable. Stop and report it; do not treat the skip as RED.

- [ ] **Step 3: Store dates and datetimes as ISO text**

In `Workflows_DB.py`, change `from datetime import datetime, timezone` to `from datetime import date, datetime, timezone`. After `_utcnow_iso`, add:

```python
def _json_default(value: Any) -> str:
    """Store a date or datetime as ISO text: PostgreSQL rows carry them where SQLite rows carry text."""
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
```

In `update_run_status`, replace `json.dumps(outputs) if outputs is not None else None,` with `json.dumps(outputs, default=_json_default) if outputs is not None else None,`.

In `complete_step_run`, replace `json.dumps(outputs or {}),` with `json.dumps(outputs or {}, default=_json_default),`.

**Regenerate the scope-predicate baseline (review B1).** The 7 new lines shift three lines that the baseline pins.
- Before regenerating, `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py` gives `1 failed, 12 passed`. The ratchet prints the three new positions as new, and the three old ones as baseline entries that no longer match.
- Then run:

```bash
$PY Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline
git diff Helper_Scripts/ci/scope_predicate_baseline.txt
```

Expected: `Recorded 28 conditional scope predicates.`, and a diff of exactly three lines:
- `Workflows_DB.py:1808 if tenant_id` → `:1815`;
- `:1811 if owner_id` → `:1818`;
- `:1960 if user_id` → `:1967`.

Any other change in that diff means the tree is not as this plan expects. Stop and re-measure.

- [ ] **Step 4: Run them, the AC 1 test, the scope ratchet, and the checkout mutant**

Run: `TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/test_notes_step_continued_after_its_creator_postgres.py tldw_Server_API/tests/Workflows/test_engine_detached_tasks_chacha_operation.py tldw_Server_API/tests/Workflows/test_workflows_db.py tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py`
Expected: `35 passed`: 1, 4, 17 and 13.

Mutant check: in `Workflows/adapters/knowledge/crud.py`, replace the `close_connections()` call in the `finally` with `pass`, then run only the PostgreSQL test.
- Expected: `1 failed` with `assert 1 == 0`. Without its close, the notes adapter leaves its loop-thread checkout out.
- Restore with `git checkout -- tldw_Server_API/app/core/Workflows/adapters/knowledge/crud.py`.

- [ ] **Step 5: Record the RAG search audit where the next editor will see it (AC 3)**

In `search.py`, add this comment directly above `result = await unified_rag_pipeline(`:

```python
    # Media DB only: with no notes, character or ChaCha handle passed, the pipeline opens no
    # ChaCha retriever, so this step needs no ChaCha operation. Passing one needs one (ADR-080, TASK-13897).
```

- [ ] **Step 6: Update ADR-080's "Detached tasks" bullet, and refresh the mirror**

In `Docs/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md`, in the Consequences bullet that starts `- **Detached tasks.**`, replace

```text
The Workflows engine's detached run tasks (`engine.continue_run`, the human-step timeout and the orphan reaper) inherit their caller's operation today. So a wrapped owner they can reach uses `independent=True`, as the chatbooks and character_chat adapters do. The notes adapter does not yet, and TASK-13897 tracks the root cause: starting those tasks outside the caller's operation.
```

with

```text
The Workflows engine's detached tasks (`engine.continue_run`, which a route, the human-step timeout and the orphan reaper start, and the human-step timeout task itself) run outside their caller's operation instead, through `outside_chacha_operation()` (TASK-13897). The run is not wrapped in an operation of its own, because its concurrent branches would share one connection. So each ChaCha call in a continued run either owns its operation with `independent=True`, as the chatbooks and character_chat adapters do, or returns its checkout after each step outside an operation, as the notes-graph job does: the notes adapter closes its per-call handle.
```

The bullet's first sentence (the rule) and the Decision are unchanged. The new text names an existing ADR-080 pattern, the notes-graph job's (review m10).

```bash
bash Helper_Scripts/refresh_docs_published.sh
git status --short Docs
```

Expected: `Docs/ADR/080-...md` and `Docs/Published/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md` are modified, and nothing else under `Docs/Published` changes. Run `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs`: expected `0 failed`.

- [ ] **Step 7: File the follow-up, close TASK-13897 and commit**

The id comes from Setup (`<scratchpad>/fu2wf_ids.env`).

```bash
source <scratchpad>/fu2wf_ids.env
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Workflow notes steps on SQLite use a nested ChaChaNotes.db, not the user's notes database" --id "$N3" -l workflows,notes,bug \
  -d "run_notes_adapter (Workflows/adapters/knowledge/crud.py:177-181) passes DatabasePaths.get_user_base_directory(user), which is <base>/<id>, as NotesInteropService's base directory. NotesInteropService._get_db (Notes/Notes_Library.py) appends <id>/ChaChaNotes.db again, so on SQLite a workflow notes step reads and writes <base>/<id>/<id>/ChaChaNotes.db, while the Notes API uses DatabasePaths.get_chacha_db_path(id), <base>/<id>/ChaChaNotes.db. Seen in TASK-13897's PostgreSQL test log: 'Creating or loading DB for user_id 1 at path: .../user_dbs/1/1/ChaChaNotes.db'. On PostgreSQL the path does not choose the data, so only SQLite deployments are affected." \
  --ac "On SQLite, a note that a workflow notes step creates is visible to the Notes API for the same user, with a test" \
  --ac "Notes already written to the nested file are migrated, or the decision not to is recorded"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13897 \
  --append-notes "AC 1 RED on <commit>: test_engine_detached_tasks_chacha_operation.py gave 4 failed: ('failed', 'The ChaCha operation has finished') for a route's continue_run task, a human step's timeout, and the orphan reaper (a step joining with chacha_operation()); and the human-step timeout task itself held its creator's closed ChaChaOperation in its context for its whole sleep. Detached callers found: engine.py:1807 and :2054, endpoints/workflows.py:3503, :4234, :4358 and :4575, and research_wait_bridge.py:58 (not in the task text); all go through continue_run. Fix: operation_scope.outside_chacha_operation() clears the inherited operation; continue_run runs inside it, and the timeout task is created inside it; the run is not wrapped in an operation (ADR-080, F13). GREEN: 4 passed." \
  --append-notes "AC 2 RED on <commit> (PostgreSQL, pg_database_config): ('failed', 'Unexpected error initializing DB for user 1: The ChaCha operation has finished'). The per-call CharactersRAGDB raised a CharactersRAGDBError, which is outside the adapter's noncritical tuple, so the run failed (the task text expected a notes_error output). With AC 1 fixed, a second, pre-existing failure: TypeError: Object of type datetime is not JSON serializable, from WorkflowsDatabase.update_run_status. A control probe on dev (a SYNC run inside an open operation) failed the same way, with the notes step run left running, because complete_step_run's json.dumps failed inside a suppress. Fix: update_run_status and complete_step_run store a date or datetime as ISO text (_json_default); the 7 new lines shifted three Workflows_DB.py entries in Helper_Scripts/ci/scope_predicate_baseline.txt, regenerated with --write-baseline. GREEN: 35 passed with the AC 1 file, test_workflows_db.py and lint/test_scope_predicate_ratchet.py. Mutant (the notes adapter skips its close): assert 1 == 0 outstanding checkouts." \
  --append-notes "AC 3: the rag_search adapter passes only media_db_path; MultiDatabaseRetriever builds a ChaCha retriever only for notes, character, world-book or dictionary paths, or a passed chacha_db. Probe (real adapter, sources media_db, notes, chats, characters, world_books, dictionaries): CharactersRAGDB constructed 0 times; positive control, the workflow notes adapter through the same recorder: 1. (The operation-lookup count is not evidence: on SQLite with no open operation even the control records 0.) RAGSearchConfig has no sources field. Documented in a comment at the pipeline call and here; no ChaCha operation needed." \
  --append-notes "Docs: ADR-080's Consequences bullet 'Detached tasks' updated to the new state (status facts only; it names the notes-graph pattern the notes adapter follows; precedent b3c1a1fa56), with Docs/Published refreshed; tests/Docs 0 failed. Bandit -ll on operation_scope.py, engine.py, Workflows_DB.py and search.py: 0 results. Follow-up: $N3 (workflow notes on SQLite use a nested ChaChaNotes.db). Plan: Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A workflow run continued by a detached task (a route's retry, approve or reject, a human step's timeout, the orphan reaper, a research checkpoint) now runs outside its creator's ChaCha operation, through outside_chacha_operation() in continue_run; it is never wrapped in one operation of its own. On PostgreSQL a notes step in such a run now succeeds and returns its checkout, which also needed WorkflowsDatabase to store PostgreSQL's datetimes as ISO text (a pre-existing failure for any notes step on PostgreSQL). The RAG search adapter reaches no ChaCha data, documented at the call. ADR-080's detached-task bullet describes the new state." \
  -s Done
git add tldw_Server_API/app/core/DB_Management/Workflows_DB.py tldw_Server_API/app/core/Workflows/adapters/rag/search.py \
  Helper_Scripts/ci/scope_predicate_baseline.txt \
  tldw_Server_API/tests/Workflows/test_workflows_db.py \
  tldw_Server_API/tests/Workflows/test_notes_step_continued_after_its_creator_postgres.py \
  Docs/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md \
  Docs/Published/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md
git add -A backlog/tasks
git commit -m "fix(workflows): a notes step works on PostgreSQL in a continued run (TASK-13897)

With continue_run outside its creator's operation, the notes step's
per-call handle takes the thread-local path and closes it. The run then
failed on PostgreSQL's datetimes, which WorkflowsDatabase could not
store as JSON; outputs now store a date or datetime as ISO text, and the
scope-predicate baseline follows the three Workflows_DB.py lines that
moved. The RAG search adapter reaches no ChaCha data. ADR-080's
detached-task bullet describes the new state.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: verification, review and PR

- [ ] **Step 1: The wider run**

```bash
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Workflows \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_native_turn_content_wire.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_chat_workflows_deps.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_workflows_control_permissions_claims.py \
  tldw_Server_API/tests/CI/test_async_blocking_call_guard.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py \
  tldw_Server_API/tests/Chat_Workflows \
  tldw_Server_API/tests/DB_Management/test_chacha_per_call_handles_postgres.py \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/tests/DB_Management/test_postgres_returning_and_workflows.py \
  tldw_Server_API/tests/DB_Management/test_sqlite_policy_integrations.py \
  tldw_Server_API/tests/DB_Management/test_chacha_operation_scope.py \
  tldw_Server_API/tests/Flashcards/test_flashcard_answer_grounding.py \
  tldw_Server_API/tests/Flashcards/test_flashcard_pair_transport_contract.py \
  tldw_Server_API/tests/Flashcards/test_flashcard_prompt_contract.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py \
  tldw_Server_API/tests/StudyPacks/test_generation_service_imports.py \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py \
  tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py \
  tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py
```

- The list is every test file outside `tests/Workflows` that names `WorkflowEngine`, `continue_run`, `complete_step_run`, `WorkflowsDatabase`, `run_tts_adapter`, `multi_voice_tts`, `run_media_ingest_adapter`, `run_rag_search_adapter`, `run_map_adapter`, `SubprocessTask`, `Workflows.adapters` or `Workflows.engine` (`git grep -l`, measured on `e06eeec68c`). It adds the scope-predicate ratchet (review B1). Re-run the grep at execution and add any new file.
- Expected: `2422 passed, 7 skipped`, plus the two `test_flashcard_prompt_contract.py::test_endpoint_retains_real_claims_refusal_for_unsupported_drafts` cases. Those fail on dev too (What was measured).
- For any other failure, re-run that test with `origin/dev`'s copies of the touched app files before calling it unrelated:
  - export them with `git archive origin/dev <paths> -o <scratch>/dev.tar`;
  - keep yours in a tar, extract dev's, run, and restore yours;
  - confirm with `git diff --stat`.
- If the two flashcard cases still fail, say so in the PR body. File a task only if no open task or PR covers them: check TASK-13260.278.18.83.45 and PR #3269 first.

- [ ] **Step 2: The gates**

```bash
$PY Helper_Scripts/ci/test_quality_triage.py --enforce
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py tldw_Server_API/tests/Docs tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py
source <scratchpad>/fu2wf_ids.env
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task normalize --check backlog/tasks/task-13886\ *.md backlog/tasks/task-13887\ *.md backlog/tasks/task-13891\ *.md backlog/tasks/task-13897\ *.md backlog/tasks/task-${N1#TASK-}\ *.md backlog/tasks/task-${N2#TASK-}\ *.md backlog/tasks/task-${N3#TASK-}\ *.md backlog/tasks/task-${N4#TASK-}\ *.md
uvx bandit -q -ll tldw_Server_API/app/core/DB_Management/Workflows_DB.py tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py tldw_Server_API/app/core/Workflows/adapters/audio/tts.py tldw_Server_API/app/core/Workflows/adapters/control/flow.py tldw_Server_API/app/core/Workflows/adapters/media/ingest.py tldw_Server_API/app/core/Workflows/adapters/rag/search.py tldw_Server_API/app/core/Workflows/engine.py tldw_Server_API/app/core/Workflows/subprocess_utils.py
grep -rn ' $' Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md | wc -l
```

Expected:
- the triage exits 0;
- the pytest run gives `0 failed`;
- `normalize --check` exits 0. Pass the touched task files, as here (review m1): given the directory, it raises `IsADirectoryError`, and with no path it exits 1 on 14 non-canonical files already on dev. `test_backlog_task_format_ratchet.py` also covers this;
- Bandit prints no `Issue:` lines;
- the whitespace count is `0`.

Then run the ruff and mypy comparisons from Global Constraints. Expected: nothing new.
- **Ruff:** the per-file counts match dev's: 2 BLE001 in `multi_voice_tts.py`, 1 I001 in `ingest.py`, 10 in `test_audio_adapters.py`. If the CI ruff (0.15.10, `ci.yml:268`) differs from the venv's 0.16.5, compare with both.
- **mypy:** the branch has one error fewer than dev on the same 9 files, and no new message (review m9). Compare the delta, not the totals: they vary by environment, 67/66 in this planner's and the reviewer's runs.

- [ ] **Step 3: Clean review before the PR merges**

Run `/superpowers:requesting-code-review` in a clean subagent. Point it at:
- this plan's Review Focus;
- ADR-080;
- `<scratchpad>/clean3290_review.md` M2;
- `<scratchpad>/clean3295_review.md` I1.

Fix every Critical and Important finding test-first. Record the outcome in the PR body. Hold the merge until the findings are addressed.

- [ ] **Step 4: Push and open the PR against `dev`**

```bash
git push -u origin fix/workflows-substep-cleanup-detached-chacha
gh pr create --base dev --title "Workflow sub-step cleanup and detached-run ChaCha operations (TASK-13886, 13887, 13891, 13897)" --body-file <scratchpad>/pr_body_fu2_workflows.md
```

The PR body has these sections:
- **Summary:** one bullet per task, saying what a user sees.
- **`## Change summary`:** the owner's own words (ADR-004). Leave the placeholder `Owner to write` and ask the owner. Do not write it for them.
- **Test plan:** the RED and GREEN counts from each task, the mutants, Task 7's run, and the gates.
- **Review focus:** this plan's five items, and the owner's answer to Decision 1.
- **Follow-ups filed:** TASK-<N1>, TASK-<N2>, TASK-<N3> and TASK-<N4>, one line each, with the computed ids.
- **Clean review:** the outcome.
- The last line is `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Once `gh pr view --json files` lists the four new task files, release the id lock: `rmdir <scratchpad>/id_lock`.

Check the merge-queue mode with `gh variable get MERGE_QUEUE` and follow CLAUDE.md's "Merging into dev". Confirm `MERGED` before deleting the branch.

## Follow-ups (filed at execution)

| Id | Title | Filed in | Why |
|---|---|---|---|
| TASK-<N1> | Workflow workflow_call cannot run a sub-workflow | Task 4 Step 6 | It imports a module that does not exist, so it never creates a child run. It carries TASK-13887's AC 2. Repair or removal is that task's decision. |
| TASK-<N2> | media_ingest and the orphan reaper block the event loop for terminate_process's grace | Task 4 Step 6 | Three on-loop `terminate_process` calls stall every request for 5 s each. |
| TASK-<N3> | Workflow notes steps on SQLite use a nested ChaChaNotes.db, not the user's notes database | Task 6 Step 7 | `<base>/<id>/<id>/ChaChaNotes.db` instead of `<base>/<id>/ChaChaNotes.db`. Measured in this plan's PostgreSQL run log. |
| TASK-<N4> | Workflow run cancelled through the API ends failed/invariant_violation when its step then fails | Task 2 Step 5 | `engine.cancel()` cancels no task. A step that then raises anything except `CancelledError` takes the failure path, whose rejected `cancelled`→`failed` transition overwrites the run. This is dev behaviour (review I1). |

## Out of scope

- **The voice-assistant on-loop checkouts.** That is TASK-13896, in `plan_fu2_chacha_runtime.md`.
- **A guard against concurrent transactions inside one operation.** That is TASK-13899, in the same plan.
- **`parallel`'s `fail_fast` semantics.** The owner kept them in TASK-13863.
- **The overwrite of an API-cancelled run by any other step error.** That is TASK-<N4>.
