# Postgres Chat Import Pool Leak and Timing-Flaky Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Independent review and dry run (2026-10-18): 1 Blocker, 2 Important, 9 Minor; all folded in.

**Goal:** Make `test_another_owner_reusing_ids_never_touches_the_first_owners_chat[postgres]` pass by removing the connection leak that causes its `PoolTimeout`, and by ending the test's own open transaction before teardown closes the pool (Review B1). Make the flaky Postgres image test and the two admin health-timeout tests deterministic without weakening what they check.

**Architecture:** All three causes are in the tests, not in product code, and each fix changes one test file:
- TASK-13841: the import test builds its app without the `ChaChaOperationMiddleware` that `main.py` installs. Every request therefore leaks one PostgreSQL checkout. Once that is fixed, the test's `owners` fixture closes the shared pool while the test's own main-thread reads still hold a transaction open, which can hang teardown in `DROP OWNED BY` (Review B1).
- TASK-13842, image test: the test reads a returned connection while psycopg_pool is still resetting it on a pool worker.
- TASK-13842, admin tests: the tests give the health check a 50 ms budget. That budget also covers 5-50 ms of synchronous pre-dispatch work on the event loop, and CPU load stretches that work past 50 ms (Review I2).

The follow-ups are filed as one task (widened by Review I1):
- legacy thread-local checkouts, which leak when their thread exits and pin a pool slot for as long as a long-lived thread holds one outside an operation;
- `CharactersRAGDB.close_all_connections()`, which closes a psycopg pool before its queued reset of the returned checkout runs; the MCP modules call it on every call;
- the WebSocket `/chat/stream` persistence path, which runs outside any operation;
- 30 other test apps that also omit the middleware.

**Tech Stack:** pytest 9.1.1, pytest-asyncio 1.4.0, FastAPI TestClient, psycopg 3.3.6, psycopg-pool 3.3.2, PostgreSQL 18 (Docker container `tldw_postgres_test`), backlog-py.

**Spec:** `backlog/tasks/task-13841 - Postgres-chat-import-returns-500-when-a-second-owner-reuses-chat-ids.md` and `backlog/tasks/task-13842 - Two-tests-flaky-in-full-sweeps-Postgres-character-image-reads-and-admin-health-adapter-timeout.md` on origin/dev.

All measurements below were taken on origin/dev `e1f7770953` in a scratch worktree, against the running Docker container `tldw_postgres_test` (postgres:18 on 5432), on an 18-core Mac. Other sessions were running pytest at the same time, so the 1-minute load average was between 16 and 56 throughout. Each figure gives the load at which it was measured.

The review re-ran the plan on origin/dev `32eb11ae03` (load 10-73). This revision re-measured the changed parts on origin/dev `7c94a2efb1` (load 6-48): Task 1's fixture fix, Task 2's fallback observation, the task-id recipe and the ratchets. None of the three test files, `tests/_plugins/postgres.py`, `ChaChaNotes_DB.py`, `postgresql_backend.py`, `chacha/`, `admin_llm_providers.py` or `byok_testing.py` changed between `e1f7770953` and `7c94a2efb1` (measured with `git diff --stat`). Figures marked "(review)" are the reviewer's.

## Global Constraints

- Change tests only. No product code changes (`tldw_Server_API/app/**` stays untouched).
- TASK-13841 AC2: fix the cause of the pool timeout. Never raise `pool_timeout` or `pool_size`, and never change `pg_restricted_backend`.
- TASK-13842 AC2: a flaky test becomes deterministic "without weakening what it checks". Every assertion keeps its subject. Where a value changes (0.05 s to 0.5 s), the plan says why the check is unchanged.
- Fix defects test-first. A flaky test's RED is a run that fails on unchanged dev for the same reason as in the sweeps, made deterministic with a scratch-only pytest plugin, plus the repeated-run counts below.
- Postgres steps run with `TLDW_TEST_POSTGRES_REQUIRED=1`, so that a missing Postgres fails the run instead of silently skipping it. Do not set `TLDW_TEST_NO_DOCKER=1` on Postgres steps. The fixture reuses a reachable server and only starts Docker when none is reachable.
- Edit backlog tasks only with backlog-py: `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`. Use `--append-notes`, never `--notes`. Stage task files with `git add -A backlog/tasks`.
- One PR, based on `dev`, from a branch cut from `origin/dev`.
- Every commit ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never use `--no-verify`.
- Ruff's isort rule is on for tests. In the import block, `...DB_Management.chacha.operation_scope` sorts before `...DB_Management.ChaChaNotes_DB` (measured: ruff I001 fires otherwise).
- None of the three touched files is listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` (39 lines, measured again on `7c94a2efb1`), so shifted line numbers need no baseline regeneration.
- The test-quality ratchet (`6a6fcf7533`, TASK-13407) is new on dev and is a required step in backend-required ("Enforce tenant isolation ratchets"). Task 4 Step 1 runs it, with its CI test, on the finished branch (Review M6).
- No mirrored doc under `Docs/Published` is touched.
- No global default is flipped.

### ADR assessment

No ADR is needed. These are test-only changes that create no durable policy (`Docs/ADR/README.md`: "test-only changes usually do not need ADRs"). Governing ADRs that apply to the work:
- ADR-005: Bandit on the touched paths.
- ADR-059: backlog-py.
- ADR-004: the AI-authored PR needs the owner's Change summary.

ADR-020 (DB management) is unaffected because no product DB code changes.

AGENTS.md section 0a also asks for the assessment in the backlog task, so Task 1 Step 9 and Task 3 Step 6 each append it to their task's notes (Review M9).

## Findings (what the investigation measured)

### TASK-13841: the PoolTimeout is a per-request checkout leak in the test app

1. **Reproduced.** The test fails on dev every time: `1 failed, 1 passed` (the sqlite variant passes), `assert 500 == 201` at `test_chat_import_local_chats.py:798`, and the trace records `error.type psycopg_pool.PoolTimeout`. The run takes about 40 s because the last request waits out the 30 s pool timeout. The whole file gives `1 failed, 270 passed` in 423 s.
2. **Pool size.** `pg_restricted_backend` (`tests/_plugins/postgres.py:392`) sets `pool_size=5`. `PostgreSQLConnectionPool` passes `max_size = pool_size` (`postgresql_backend.py:207`) and `timeout = pool_timeout or 30.0` (`:208`); `max_overflow` is ignored.
3. **Who holds the checkouts.** A scratch probe wrapped `PostgreSQLConnectionPool.get_connection`, `return_connection` and `discard_connection`. It showed 5 checkouts and 0 returns before the 6th checkout blocked, with psycopg_pool stats at `pool_size: 5, pool_available: 0`. The holders were `AnyIO worker thread: DEAD` (3), `asyncio-portal-…` (1) and `MainThread` (1).
4. **Why they are never returned.** `_client` (`:131-151`) builds a bare `FastAPI()` without `ChaChaOperationMiddleware`, which `main.py:2796` installs on the real app. Without an operation, `CharactersRAGDB._connection_state()` (`ChaChaNotes_DB.py:8337-8343`) falls back to the legacy `threading.local` state, and `_get_connection_for_state` (`:8652-8660`) checks out a pooled connection that belongs to that thread. A TestClient that is not used as a context manager runs each request on a new portal, which has its own AnyIO worker threads. When a request ends, its threads exit, the thread-local is dropped, and the checkout is never returned.
5. **Why the 6th checkout times out.** The test makes 5 database-touching requests plus main-thread reads across two `CharactersRAGDB` handles that share the 5-connection pool. The 6th checkout therefore waits 30 s, raises `PoolTimeout`, and the request returns 500.
6. **Why only this test fails.** The other Postgres tests in the file use `pg_database_config` with the default `pool_size=10`, so they leak too but never reach the limit.
7. **The middleware fixes the 500.** After adding `app.add_middleware(ChaChaOperationMiddleware)` to `_client`:
   - The target test gives `2 passed` in 5 s.
   - The probe shows every request checkout returned by `chacha_operation`'s close, with a peak of 3 outstanding (2 main-thread handles plus 1 request).
   - The whole file gives `271 passed` (1098 s at load ~40; review: 1770 s at load ~38).
   - The body never returned 500 again in 98 runs with the middleware (review).
8. **The middleware alone leaves a teardown hang (Review B1).** At planning time 1 of 14 runs hit pytest-timeout (300 s) with no saved stack. The review repeated the target with the middleware alone: **19 of 52 runs hung** (37%; 19 of 47 in file order), at load 10-73, including 4 of 8 at load 10-17. This revision measured, on `7c94a2efb1` with `-o timeout=60`:
   - 2 loops of 10 runs side by side: **9 of 20 hung** (load 22-39).
   - 4 loops of 5 side by side: **11 of 20 hung** (load 8-31).
   - One run at a time: 0 of 10 hung (load 8-10). A quiet machine rarely shows the hang, so the repeat steps below run two loops side by side.

   Every hang has the same stack, in teardown, after the test body passed:
   ```
   tests/_plugins/postgres.py, line 409, in pg_restricted_backend
       conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
   psycopg/connection.py, line 495, in wait
   ```
   The mechanism (the review's `hang_probe` captured it in every hanging run):
   1. With the 500 gone, the body reaches `:804-806`. Those lines read through `owners.first` on the main thread, outside any operation, so they use the legacy thread-local checkout and leave it `INTRANS`. The last query is the `message_images` read in `get_message_by_id`. On dev the test fails at `:798`, before these reads, which is why dev never hangs. `:780-781` and `:793-794` do the same through `owners.second`.
   2. The `owners` teardown (`:764-766`) calls `first.close_all_connections()`. That returns the `INTRANS` checkout through `putconn`, which queues psycopg_pool's asynchronous `ReturnConnection`, and then at once closes the **shared** pool (`CharactersRAGDB.close_all_connections()` → `PostgreSQLConnectionPool.close_all()` → `ConnectionPool.close()`).
   3. `MaintenanceTask.run` quietly discards a task whose pool is closed ("Pool is no more working. Quietly discard the operation."). The connection is neither rolled back nor closed. (`second.close_all_connections()` runs after the pool is closed, so `putconn` closes its checkout synchronously; that one never hangs.)
   4. When the discarded connection stays referenced (the probe found a `BackendConnectionWrapper`, a `BackendCursorWrapper`, a frame and a psycopg `Cursor`, surviving `gc.collect()`), its session stays `idle in transaction` and keeps its lock on `message_images`. psycopg's `Connection.__del__` returns early for pooled connections, so nothing ends it. When the object is freed instead, libpq closes the socket and the run passes; that is why the hang is intermittent.
   5. `pg_restricted_backend`'s teardown `DROP OWNED BY <role>` (`tests/_plugins/postgres.py:409`) waits for that lock until pytest-timeout fires at 300 s.
   6. `timeout_method = "thread"`, so pytest-timeout `os._exit`s the whole process. In a sweep the rest of the session is lost, which is worse than today's deterministic failure.

   Review probe: of 14 runs, the 5 hangs had exactly one discarded `INTRANS` return each, and the 9 passes had none.
9. **Teardown fix verified (Review B1).** The `owners` fixture rolls back both handles' main-thread connections while the shared pool is still open, then closes them as before (Task 1 Step 5).
   - Made deterministic with the scratch `late_return` plugin, which starts each queued return 0.5 s late and keeps its connection referenced: the middleware alone hangs at `postgres.py:409` (one discarded `INTRANS` return, three discarded `IDLE`); with the fixture fix, `2 passed` (every discarded return `IDLE`). Measured on `7c94a2efb1`, load 9-22.
   - Repeated in separate processes, 0 hangs in 106 runs:
     - review: 46/46 passed (load 18-56);
     - this revision, 2 loops of 10 side by side: 20/20 passed (load 31-48), against 9/20 hung for the middleware alone under the same setup;
     - this revision, one run at a time: 20/20 passed (4.5-9.0 s each, load 10-15). The middleware alone also passed 10/10 at that load, so this set alone does not discriminate.
   - With 4 loops side by side, 9 of 20 fixed runs passed the test but errored in teardown with `psycopg.errors.OutOfMemory: out of shared memory` at the same `DROP OWNED BY` (the server runs `max_locks_per_transaction = 64`, `max_connections = 100`, and is shared with other sessions). This is environmental: unchanged `test_core_chat_owner_isolation.py::test_postgres_inline_tool_call_backfill_consumes_native_result`, which also uses `pg_restricted_backend`, errored the same way in 9 of 20 runs under the same 4-loop setup. It did not appear in the 40 fixed runs with one or two loops.
   - A discarded return of an `IDLE` connection still leaks an idle session, but it holds no lock, so teardown completes.
   - Whole file with both fixes: `271 passed` in 1555 s (load 20-37, run alongside the image re-measurements).
10. **Not changed here (follow-up task, widened by Review I1).** Production exposure is likely, not just possible:
    - psycopg's `Connection.__del__` returns early for pooled connections, so a dropped pooled checkout is never returned and its slot is lost. AnyIO worker threads exit after `MAX_IDLE_TIME = 10` s idle (`anyio/_backends/_asyncio.py:982`), so a legacy checkout taken on one leaks permanently, even in a long-running server.
    - A long-lived thread outside an operation (a default executor thread, the event-loop thread) holds its legacy checkout until the handle closes, which can exhaust the pool without any thread exiting.
    - `ChaChaOperationMiddleware` only wraps `scope["type"] == "http"`. The WebSocket route `/chat/stream` (`audio_streaming.py:1947`) persists chats through `_shim_get_chacha_db_for_user_id` (`:2415`) with no `chacha_operation`. The only product callers of `chacha_operation(` are `ChaCha_Notes_DB_Deps.py` (3), `conversation_enrichment.py` (2), the repair scope in `ChaChaNotes_DB.py` (1) and `study_pack_jobs_worker.py` (1).
    - `CharactersRAGDB.close_all_connections()` on psycopg_pool returns the checkout and then closes the pool before the queued reset runs (B1's product half). Product code calls it per call in many places, for example the MCP modules, which open a `CharactersRAGDB` and close it on every call (`flashcards_module.py` x13, `notes_module.py` x10).

    None of these was exercised here; `/chat/stream` is the first caller to verify.
11. **Sibling test apps (also follow-up).** A grep found 30 other test files that build a bare `FastAPI()` over ChaCha with Postgres params and no middleware (review re-ran the grep: 30). None of them fails today. Only this test uses `owners`.

### TASK-13842 (a): the image test reads a connection that psycopg_pool is still resetting

The test is `test_character_image_reads_leave_event_loop_and_release_operation[postgres-*]`.

1. **Failure.** Every failure is `assert raw.info.transaction_status.name == "IDLE"` at `test_character_chat_images.py:378`, reading `ACTIVE` (with an injected reset delay, also `INTRANS`).
2. **Mechanism.** The backend's pool is built with `reset=_reset_pooled_connection` (`postgresql_backend.py:211-224`). With a reset hook, `psycopg_pool.ConnectionPool._putconn` hands the return to a pool worker (`run_task(ReturnConnection(...))`). That worker rolls back, runs `RESET ROLE` / `RESET SESSION AUTHORIZATION` / `RESET ALL` / `COMMIT`, and only then calls `_add_to_pool`. The test reads `raw.info` from the test thread while that is in progress. This is the same race TASK-13412 documented for the lifecycle tests.
3. **State at hand-back.** Recorded inside `return_connection`, the request hands its checkout back `INTRANS`: its read transaction is left for the pool to roll back, which is the policy in `operation_scope.py`'s docstring. So "IDLE at hand-back" is not the contract. The contract is "the pool cleans it up and puts it back into service".
4. **Counts on dev** (postgres cases only; sqlite cases 6/6 pass):
   - One run of 6 cases at load ~50: 6/6 failed.
   - `REP_N=4` (24 cases) at load ~40-50: 24/24 failed, all `'ACTIVE' == 'IDLE'` at `:378`.
   - With the scratch `slow_reset` plugin (holds the reset open 0.3 s): 6/6 failed.
   - The sweep report describes it as intermittent. On this machine at load 40-50 it failed every time, so how often it fails depends on load.
   - Shuffled-order runs were **not** done for this test. Order cannot matter here (the race is within one request), and file order already failed 30/30.
5. **Fix verified.** Wait until the pool republishes the request's checkout, record its status at that moment, and require exactly one republication, `["IDLE"]`. The observation points are the ones `test_chacha_postgres_http_operation_lifecycle.py` uses: a hook on `pool._pool._add_to_pool` (`:133`) for psycopg_pool, and an `ObservedFreeList` on the fallback pool's `_free` (`:142-151`), which observes after the synchronous rollback and before the checkout is published (Review M8).
   - First draft (planning time; same psycopg path): `REP_N=4` 24/24 passed; `slow_reset` 6/6; `fallback_pool` 6/6; with `bad_reset` (the reset raises, so the checkout is never put back), the fixed test fails with "The pool never put the request's checkout back into service". So it still catches a checkout the pool cannot clean up.
   - First draft (review): `slow_reset` 6 passed; `REP_N=4` 24 passed in file order (load 25) and shuffled (`SHUFFLE_SEED=2001`, load 55); `fallback_pool` 6 passed; `bad_reset` 1 failed with that message; whole file 108 passed (load 34).
   - Revised code (`7c94a2efb1`, load 30-40): `fallback_pool` 6 passed (227 s); `slow_reset` 6 passed (235 s); `bad_reset` 1 failed with that message; `REP_N=4` 24 passed (264 s, load ~36); the 6 sqlite cases 6 passed; ruff clean; Bandit B101 x40.

### TASK-13842 (b): the admin health tests' 50 ms budget also covers pre-dispatch work

The main test is `test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers`.

1. **Failure.** `assert pool.active_count == 1` at `test_admin_llm_provider_test_runtime.py:1481` fails with `0 == 1`: the adapter was never dispatched.
2. **What the 0.05 s covers.** The test patches `_provider_health_timeout_seconds` to 0.05 (`:1472`). The endpoint wraps the *whole* `test_provider` call in `asyncio.wait_for(..., timeout=0.05)` (`admin_llm_providers.py:265-272`).
3. **What takes the time (corrected by Review I2).** Inside that window, `test_provider` does 5-50 ms of synchronous work on the event loop before it reaches `_run_bounded_health_call`: the real `load_server_config_snapshot()` (3-10 ms), the rest of `test_provider`, and event-loop scheduling. CPU load stretches that work past 50 ms. The first draft blamed a cold AuthNZ `Settings()` build of 0.12-0.16 s. That was wrong:
   - `Settings()` itself took 3.6 ms (review `sibling_probe`).
   - In the main test, the first call is cold in every run (an autouse fixture leaves the Settings singleton empty), so `resolve_byok_credentials` → `_can_use_base_url_override` → `resolve_byok_base_url_allowlist` → `get_settings()` builds it. That is one small part of the pre-dispatch work.
   - The multi-provider sibling never builds `Settings`, because `_install_batch_service_boundary` stubs `resolve_byok_credentials` (`:792`), and it fails just as often.
4. **Measured pre-dispatch latency** (time from `test_provider` entry to `_run_bounded_health_call`):
   - Planning probe: 6-31 ms when it passes (36 samples, load ~24-32); about 120-160 ms in the failing run (load ~50).
   - Review `predispatch_probe`: passing runs 5-48 ms cold and 0.4-51 ms warm. Failing dev runs were cancelled after 175 ms and 231 ms (main test) and after 56-159 ms (sibling). With the 0.5 s fix the worst observed was 108 ms, and every call was dispatched.
5. **Why the adapter is never called.** When pre-dispatch takes more than 50 ms, `wait_for` cancels before `_run_bounded_health_call` is reached. The probe confirmed no `dispatch-enter` in the failing run, while the process-wide admission semaphores were full (8/8 global, 2/2 ollama), so no admission leaked.
6. **A third flaky test with the same cause** was found in the same file: `test_admin_health_multi_provider_timeouts_retain_real_worker_admission_and_chat_headroom` (`timeout_seconds=0.05` at `:960`). It failed at `:973` (`health_adapter.call_count == 1`, got 0) and at `:995` (`recovery["healthy"] == 3`, got 2). This plan fixes it under TASK-13842, because it is the same cause in the same file and would keep the file flaky.
7. **Counts on dev for the whole file `test_admin_llm_provider_test_runtime.py` (34 tests).** pytest-randomly is not installed, so `-p no:randomly` is a no-op; "shuffled" means the scratch `rep.py` plugin with seeds 1001-1020.

| Run | Main test | Multi-provider sibling |
|---|---|---|
| Isolated, `REP_N=50` in one process (load ~21) | 0/50 failed | not run |
| Isolated, `REP_N=50` in one process, twice (review, load 47-56) | 7/50 and 6/50 failed, all at `:1481` | not run |
| File order, 23 runs (load 16-35) | 2/23 failed | 0/23 failed |
| File order, 20 runs (review, load 25-58) | 2/20 failed (`:1481`) | 3/20 failed (`:973` x2, `:995` `assert 1 == 3`) |
| Shuffled, 20 seeds (load 45-56) | 1/20 failed (seed 1009) | 4/20 failed (seeds 1007, 1009, 1011, 1012) |
| Shuffled, seeds 1001-1020 (review, load 32-62) | 0/20 failed | 0/20 failed |
| Stress: 4 loops side by side, 40 runs (review, load 38-57) | 4/40 failed | 3/40 failed |
| With the scratch `slow_predispatch` plugin (dispatch delayed 0.15 s, inside the 56-231 ms that failing runs spent before dispatch) | fails at `:1481` | fails at `:973` |

   So the main test is not stable in isolation either: it fails whenever load stretches pre-dispatch work past 50 ms (Review I2).
8. **Fix verified.** Use 0.5 s in both tests: still sub-second, so the adapter's `api_timeout` still rounds up to 1.
   - Planning time (load 36-53): file order 20/20 passed, shuffled 20/20 passed, and both pass under `slow_predispatch`.
   - Review: 130/130 fixed runs passed (file order 20/20, shuffled 20/20, main test `REP_N=50` 50/50, stress 40/40 at load 38-57).
   - Review mutation: with the health timeout made ineffective (`timeout=30.0` at `admin_llm_providers.py:271` and `byok_testing.py:309`), both fixed tests fail (main at `:1482` with `'healthy' == 'error'`; sibling at `:970` with `TimeoutError`). The fixed tests still catch a timeout that does not bound the call.
9. **A similar test that is structurally immune (corrected by Review M1).** `test_individual_admin_provider_timeout_stays_prompt_but_retains_worker_admission` also uses 0.05 s (`PROVIDER_CREDENTIAL_VALIDATION_TIMEOUT_SECONDS`, `:1085`), but not around pre-dispatch work. `admin_test_llm_provider` (`admin_llm_providers.py:205-213`) has no outer `wait_for`; its 0.05 s clock starts inside `_run_bounded_health_call` (`admission_deadline`, `byok_testing.py:259`), after pre-dispatch. It cannot hit this race, which is consistent with 0 failures in 43 file runs. It is not changed.

## Decisions for the owner

None. These were resolved in the plan, each with its reason:
- The multi-provider sibling test is fixed under TASK-13842: same cause, same file, measured flaky.
- The legacy thread-local leak, and B1's product half (`close_all_connections()` closing a psycopg pool before its queued reset), go to the follow-up task instead of product code here: HTTP traffic is already covered by `main.py`'s middleware, the test-only fixture fix removes the hang (Review B1), and none of the product paths was exercised (Review I1).
- TASK-13841's AC #1 is amended with backlog-py (`--remove-ac 1` plus `--ac`) rather than checked while its literal text is false (Review M4): on PostgreSQL the reused ids get 409 by design, and the 201 is the second owner's own-id import at `:798`.

The PR still needs the owner's own `## Change summary` (ADR-004) before merge. That is an input, not a decision.

## Review Focus

1. **Non-HTTP PostgreSQL callers (widened by Review I1).** Not exercised here; all go to the follow-up task in Task 1. Expected: no leak.
   - WebSocket routes run outside the middleware. `/chat/stream` (`audio_streaming.py:1947`) persists chats through `_shim_get_chacha_db_for_user_id` (`:2415`) with no `chacha_operation`; it is the first caller to verify. `persona.py` and `voice_assistant.py` are next.
   - A legacy checkout taken on a thread that exits (an AnyIO worker idle for 10 s) is never returned. One held by a long-lived thread (a default executor thread, the event-loop thread) pins a pool slot until its handle closes.
   - `CharactersRAGDB.close_all_connections()` on psycopg_pool returns the checkout, then closes the pool before the queued reset runs, so the connection is neither reset nor closed (B1's product half). The MCP modules call it per call (`flashcards_module.py` x13, `notes_module.py` x10).
2. **Other test apps without the middleware.** 30 test files build bare `FastAPI()` apps over Postgres ChaCha without `ChaChaOperationMiddleware`. Each leaks one checkout per request and passes only because the default pool has 10 connections. Adding requests, or a 5-connection pool, will reproduce the 30 s `PoolTimeout`. This also goes to the follow-up task.
3. **Private psycopg_pool hook.** The image test hooks `pool._pool._add_to_pool`, a private psycopg_pool method. If an upgrade renames it, the test fails loudly at setup (`AttributeError`) rather than passing silently. This is pinned by Task 2 Step 4 (the hook is installed unconditionally whenever psycopg_pool is in use).
4. **Fallback pool.** With psycopg-pool not installed (it is only in the `db_postgres` extra), the backend uses its fallback pool, which rolls back synchronously. The image test observes it through an `ObservedFreeList` on `_free`, after the rollback and before the checkout is published, as the lifecycle test does (Review M8). Pinned by Task 2 Step 6 (the `fallback_pool` run).
5. **Remaining short budgets.**
   - `test_individual_admin_provider_timeout_stays_prompt_but_retains_worker_admission` keeps 0.05 s. Its clock starts after pre-dispatch, inside `_run_bounded_health_call`, so it cannot hit this race (Review M1).
   - The two fixed admin tests now need dispatch within 0.5 s. That is about 2x the worst pre-dispatch latency observed (231 ms, review, at load ~50-56; Review M3). An event-loop stall longer than 0.5 s would still fail them. The product's outer `wait_for` deliberately bounds the whole check, including pre-dispatch, so no in-test wall-clock budget is fully load-independent.
   - The sibling's outer guards (`asyncio.wait_for(..., timeout=1.0)` at dev `:969`, first batch, and `:988-993`, recovery) had about 0.95 s of headroom over a 0.05 s budget and now have about 0.5 s. That is no tighter than the new inner margin, and 0 of 80 fixed sibling runs failed (review). They are left at 1.0 s (Review M2).
6. **Restricted-fixture teardown (Review B1, I1).** Any session the test leaves `idle in transaction` makes `pg_restricted_backend`'s `DROP OWNED BY` wait until pytest-timeout `os._exit`s the whole session after 300 s. Twelve test files use the fixture. The follow-up task may harden it (for example `SET lock_timeout` before `DROP OWNED BY`, and report the role's open sessions on timeout).

---

## Setup (once, before Task 1)

```bash
git fetch origin dev --quiet
git worktree add <scratchpad>/test-reliability -b fix/test-reliability-13841-13842 origin/dev
cd <scratchpad>/test-reliability
export PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
export P=<scratchpad>/plugins      # scratch pytest plugins, never committed
mkdir -p "$P/runs"                 # Task 1 Steps 2 and 4 write logs here before runs.sh first runs
docker ps --format '{{.Names}} {{.Ports}}' | grep tldw_postgres_test   # expect: tldw_postgres_test 0.0.0.0:5432->5432/tcp
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13841 -s "In Progress"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13842 -s "In Progress"
```

Create these scratch plugins in `$P`. They are investigation tools, not repo files.

`$P/rep.py`: repeats tests and shuffles collection order.

```python
"""Scratch pytest plugin: repeat matching tests REP_N times; shuffle all items with SHUFFLE_SEED."""
import os
import random

N = int(os.environ.get("REP_N", "1"))
MATCH = os.environ.get("REP_MATCH", "")


def pytest_generate_tests(metafunc):
    if N > 1 and MATCH and MATCH in metafunc.function.__name__:
        metafunc.fixturenames.append("_rep_idx")
        metafunc.parametrize("_rep_idx", range(N), ids=lambda i: f"rep{i}")


def pytest_collection_modifyitems(session, config, items):
    seed = os.environ.get("SHUFFLE_SEED")
    if seed:
        random.Random(int(seed)).shuffle(items)
```

`$P/runs.sh`: runs a file N times and prints one summary line per run.

```bash
#!/bin/zsh
# usage: runs.sh <label> <count> <shuffle:0|1> <pytest args...>   (run from the repo root)
label=$1; count=$2; shuffle=$3; shift 3
mkdir -p "$P/runs"
for i in $(seq 1 $count); do
  log=$P/runs/${label}_$i.log
  if [[ $shuffle == 1 ]]; then export SHUFFLE_SEED=$((1000 + i)); else unset SHUFFLE_SEED; fi
  PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p rep -q "$@" > $log 2>&1
  echo "$label run $i seed=${SHUFFLE_SEED:-none}: $(grep -E '^=+ .*(passed|failed|error).* =+$' $log | tail -1)"
  grep -E "^(FAILED|ERROR) " $log | cut -c1-220
  # pytest-timeout os._exits a hung run before any summary line (Review B1).
  grep -q "+++ Timeout" $log && echo "$label run $i: TIMEOUT at $(grep -oE 'postgres.py", line [0-9]+' $log | head -1)"
done
```

`$P/slow_reset.py`: holds psycopg_pool's asynchronous reset open, the way a loaded machine does.

```python
"""Scratch pytest plugin: hold psycopg_pool's async reset open, as a loaded machine does."""
import time

from tldw_Server_API.app.core.DB_Management.backends import postgresql_backend as pgb

_orig_reset = pgb._reset_pooled_connection


def _slow_reset(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT 1")  # the reset's implicit transaction is now open (INTRANS)
    time.sleep(0.3)
    _orig_reset(conn)


pgb._reset_pooled_connection = _slow_reset
```

`$P/bad_reset.py`: a reset that fails, so the pool never puts the checkout back.

```python
"""Scratch pytest plugin: a reset that fails, so psycopg_pool discards the returned checkout."""
from tldw_Server_API.app.core.DB_Management.backends import postgresql_backend as pgb


def _bad_reset(conn):
    raise RuntimeError("reset refused")


pgb._reset_pooled_connection = _bad_reset
```

`$P/fallback_pool.py`: builds pools as if psycopg-pool were not installed.

```python
"""Scratch pytest plugin: build PostgreSQL pools as if psycopg-pool were not installed."""
from tldw_Server_API.app.core.DB_Management.backends import postgresql_backend as pgb

pgb.psycopg_pool = None
```

`$P/slow_predispatch.py`: delays each health dispatch the way CPU load stretches pre-dispatch work (Review I2).

```python
"""Scratch pytest plugin: delay each health dispatch the way CPU load stretches pre-dispatch work."""
import asyncio

from tldw_Server_API.app.core.AuthNZ import byok_testing

_orig = byok_testing._run_bounded_health_call


async def _late(call, *, provider, timeout_seconds):
    await asyncio.sleep(0.15)  # measured: failing runs spent 56-231 ms before dispatch at load ~50
    return await _orig(call, provider=provider, timeout_seconds=timeout_seconds)


byok_testing._run_bounded_health_call = _late
```

`$P/late_return.py`: makes the TASK-13841 teardown hang deterministic (Review B1). It starts each queued psycopg_pool return 0.5 s late, so a pool closed right after a return discards it, and it keeps the returned connection referenced, as the hanging runs did.

```python
"""Scratch pytest plugin: run psycopg_pool's queued returns late and keep their connections referenced.

A return queued just before the pool closes is then discarded without its reset, and the
connection outlives the pool, as in the teardown hangs (wrapper, cursor and frame referrers).
"""
import os
import time

from psycopg_pool import pool as psycopg_pool_module

_orig_run = psycopg_pool_module.ReturnConnection.run
_kept = []
_log = os.environ.get("LATE_RETURN_LOG")


def _late_run(self):
    _kept.append(self.conn)
    time.sleep(0.5)  # close_all() closes the pool well within this; run() then discards the return
    pool = self.pool()
    if _log and (pool is None or pool.closed):
        with open(_log, "a") as fh:
            fh.write(f"discarded status={self.conn.info.transaction_status.name}\n")
    _orig_run(self)


psycopg_pool_module.ReturnConnection.run = _late_run
```

Make the runner executable: `chmod +x "$P/runs.sh"`.

---

### Task 1: Install the production operation middleware in the chat-import test app, and end the test's own transactions before teardown (TASK-13841)

**Files:**
- Modify: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py:45-46` (import), `:139` (`_client`) and `:764-766` (the `owners` fixture's `finally`, Review B1). These are dev line numbers; edit by content, because earlier hunks shift later lines.
- Backlog: `backlog/tasks/task-13841 - …md`, plus one new follow-up task

**Interfaces:**
- Consumes: `ChaChaOperationMiddleware` from `tldw_Server_API.app.core.DB_Management.chacha.operation_scope`, an ASGI middleware that opens `chacha_operation(independent=True)` per HTTP request. It is unchanged, and `main.py:2796` already installs it.
- Consumes: `CharactersRAGDB.get_connection()` (`ChaChaNotes_DB.py:8664`), which returns the current thread's connection wrapper (`BackendConnectionWrapper` on PostgreSQL, whose `rollback()` delegates to the raw connection), and `close_all_connections()` (`:8755`), which returns this thread's checkout and then closes the shared pool. Both unchanged.
- Produces: `_client(db, monkeypatch, *, user_id=1, sync_service=None, limiter=None) -> TestClient` keeps its signature. Every app it builds now returns each request's PostgreSQL checkout when the response ends.
- Produces: the `owners` fixture keeps its yield value; on PostgreSQL its teardown leaves no session in a transaction.

- [ ] **Step 1: Confirm the failing test fails on dev for the right reason (RED)**

The test already exists and is the regression test. It fails on dev every time.

Run:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  "tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat" \
  2>&1 | grep -E "^E  |passed|failed|error.type" | head
```
Expected (about 40 s; the last request waits out the 30 s pool timeout):
```
E   AssertionError: assert 500 == 201
        "error.type": "psycopg_pool.PoolTimeout"
==== 1 failed, 1 passed, … ====
```

- [ ] **Step 2: Record the file baseline on dev**

Run (7 min at load ~16; 18 min at load ~40):
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py \
  > "$P/runs/import_base.log" 2>&1; grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)" "$P/runs/import_base.log"
```
Expected:
```
FAILED …test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat[postgres]
==== 1 failed, 270 passed, … ====
```

- [ ] **Step 3: Add the middleware (GREEN)**

In the import block, insert the new import between the `backends.factory` and `ChaChaNotes_DB` imports, because ruff's isort orders it there:

```python
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ChaChaOperationMiddleware
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
```

In `_client`, directly after `app = FastAPI()`:

```python
    app = FastAPI()
    # As in main.py: each request owns its PostgreSQL checkout and returns it when the response ends.
    app.add_middleware(ChaChaOperationMiddleware)
    app.include_router(chats.router, prefix="/api/v1/chats")
```

- [ ] **Step 4: Show that the middleware alone can hang teardown (RED, Review B1)**

With the 500 gone, the body reaches its own main-thread reads, which leave both handles' checkouts in a transaction. The `owners` teardown then closes the shared pool before psycopg_pool's queued reset runs (Findings, TASK-13841 item 8). Check the body and lint first:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  "tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat" \
  2>&1 | grep -E "^=+ .*(passed|failed|error)|\+\+\+ Timeout"
$PY -m ruff check tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py
```
Expected: `2 passed` (5-30 s depending on load), or a `+++ Timeout +++` line after 300 s if this run hits the hang; then ruff `All checks passed!`.

Make the hang deterministic with `late_return`:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 LATE_RETURN_LOG="$P/runs/late_red.discards" PYTHONPATH=$P \
  $PY -m pytest -p no:cacheprovider -p late_return -o timeout=90 -q \
  "tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat" \
  > "$P/runs/late_red.log" 2>&1
grep -E "\+\+\+ Timeout|postgres.py\", line|DROP OWNED" "$P/runs/late_red.log" | head -3
sort "$P/runs/late_red.discards" | uniq -c
```
Expected (about 100 s; measured on `7c94a2efb1` at load 9-22):
```
.+++++++++++++++++++++++++++++++++++ Timeout ++++++++++++++++++++++++++++++++++++
  File "…/tldw_Server_API/tests/_plugins/postgres.py", line 409, in pg_restricted_backend
    conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
      3 discarded status=IDLE
      1 discarded status=INTRANS
```
The `INTRANS` entry is the checkout `first.close_all_connections()` returned just before it closed the pool.

Then count it without the plugin. A quiet machine rarely shows the hang (0 of 10 one at a time at load 8-10), so run two loops side by side:
```bash
for l in 1 2; do
  TLDW_TEST_POSTGRES_REQUIRED=1 $P/runs.sh import_red_l$l 10 0 -o timeout=60 \
    "tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat" \
    > "$P/runs/import_red_l$l.txt" 2>&1 &
done; wait
cat "$P"/runs/import_red_l*.txt | grep -c "TIMEOUT at postgres.py\", line 409"; uptime
```
Expected: a count above 0; every other run prints `2 passed`. Measured: 9 of 20 hung (load 22-39); the review measured 19 of 52 (load 10-73). Record the count and load. If it prints 0, the `late_return` run above is the RED.

- [ ] **Step 5: End the test's own transactions while the pool is open (GREEN, Review B1)**

Replace the `owners` fixture's `finally` block (dev `:764-766`):

```python
    try:
        yield SimpleNamespace(first=first, second=second, postgres=request.param == "postgres")
    finally:
        try:
            if backend is not None:
                # The test reads through both handles on this thread, outside any operation, so each
                # handle's checkout is left in a transaction. End it while the shared pool is open:
                # close_all_connections() closes the pool before psycopg_pool's queued reset runs, and
                # a transaction left open then blocks pg_restricted_backend's DROP OWNED BY.
                for handle in (first, second):
                    handle.get_connection().rollback()
        finally:
            first.close_all_connections()
            second.close_all_connections()
```

Why this is enough, and its limits:
- On this thread, `get_connection()` returns the same legacy checkout the reads used. `rollback()` ends its transaction and releases the `message_images` lock while psycopg_pool can still reset and republish the connection.
- The inner `try/finally` makes `close_all_connections()` run even if a rollback raises.
- If a returned connection is still discarded (pool closed first), it is `IDLE`: it leaks an idle session that holds no lock, so `DROP OWNED BY` completes.
- `get_connection()` takes a fresh checkout for a handle that has none. On a failing run with an exhausted pool, that waits for the 30 s pool timeout. This only happens after the test has already failed.

- [ ] **Step 6: Run under the RED condition, repeated, and the whole file**

Run:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 LATE_RETURN_LOG="$P/runs/late_green.discards" PYTHONPATH=$P \
  $PY -m pytest -p no:cacheprovider -p late_return -o timeout=90 -q \
  "tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat" \
  2>&1 | grep -E "^=+ .*(passed|failed|error)|\+\+\+ Timeout"
sort "$P/runs/late_green.discards" | uniq -c
```
Expected: `2 passed` (measured: 7.9 s), then `5 discarded status=IDLE`: every discarded return is `IDLE`.

Repeat with the same two loops as Step 4:
```bash
for l in 1 2; do
  TLDW_TEST_POSTGRES_REQUIRED=1 $P/runs.sh import_fix_l$l 10 0 -o timeout=60 \
    "tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat" \
    > "$P/runs/import_fix_l$l.txt" 2>&1 &
done; wait
cat "$P"/runs/import_fix_l*.txt | grep -c "2 passed"; cat "$P"/runs/import_fix_l*.txt | grep -cE "TIMEOUT|FAILED|ERROR"; uptime
```
Expected: `20`, then `0`. Measured: 20/20 passed (load 31-48). The review's candidate passed 46/46 (load 18-56), and one run at a time passed 20/20 (load 10-15).

Do not run more than two loops: with four, 9 of 20 runs errored in teardown with `psycopg.errors.OutOfMemory: out of shared memory` at the same `DROP OWNED BY`. That is the shared server's lock table (`max_locks_per_transaction = 64`), not this change: unchanged `tldw_Server_API/tests/DB_Management/test_core_chat_owner_isolation.py::test_postgres_inline_tool_call_backfill_consumes_native_result` errored in 9 of 20 runs under the same four loops.

Then the whole file:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py \
  > "$P/runs/import_fix.log" 2>&1; grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)" "$P/runs/import_fix.log"
$PY -m ruff check tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py
```
Expected: `271 passed` with no FAILED or ERROR lines (measured with both fixes: 1555 s at load 20-37); then ruff `All checks passed!`.

- [ ] **Step 7: Bandit on the touched file**

Bandit is not installed in the venv; run it through uvx:
```bash
uvx bandit -q -r tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py -f json -o "$P/runs/bandit_13841.json"
$PY -c "import json,collections;d=json.load(open('$P/runs/bandit_13841.json'));print(collections.Counter(r['test_id'] for r in d['results']))"
```
Expected: `Counter({'B101': 290, 'B105': 2})` (measured again with the fixture fix: it adds no `assert`). These are identical to the dev file, so there are no new findings.

- [ ] **Step 8: File the follow-up task (widened by Review I1)**

Compute a free id with the AGENTS.md recipe (Backlog section), paging any open PR with more than 100 files: `gh pr list --json files` stops at 100 files per PR.
```bash
git fetch origin dev --quiet
{ git ls-tree -r --name-only origin/dev backlog/
  gh pr list --state open --limit 500 --json number,changedFiles,files \
    --jq '.[] | if .changedFiles > 100 then "PR#\(.number)" else .files[].path end' \
    | while IFS= read -r line; do
        if [[ $line == PR#* ]]; then
          gh api --paginate "repos/rmusser01/tldw_server/pulls/${line#PR#}/files" --jq '.[].filename'
        else
          printf '%s\n' "$line"
        fi
      done
} | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1
```
At planning time (dev `7c94a2efb1`) this printed `13857`: dev's highest is 13853, the open PRs' 13857, and the 9 open PRs with more than 100 files were paged (their highest is 13598). Use the printed value + 1 as `<N>`.

Then:
```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create \
  "Legacy thread-local PostgreSQL ChaCha checkouts leak (thread exit, long-lived threads, close_all_connections before reset)" \
  --id TASK-<N> -l chat,postgres,bug \
  -d "Found while fixing TASK-13841 (measured on origin/dev 7c94a2efb1). (1) Thread exit: outside a chacha_operation, CharactersRAGDB._connection_state() falls back to threading.local and _get_connection_for_state checks out a pooled PostgreSQL connection that belongs to that thread (ChaChaNotes_DB.py:8337-8343, :8652-8660). psycopg's Connection.__del__ returns early for pooled connections, so when the thread exits the checkout is never returned and its pool slot is lost (measured: holders 'AnyIO worker thread: DEAD', pool_available 0). AnyIO worker threads exit after MAX_IDLE_TIME = 10 s idle, so this leaks even in a long-running server. (2) Long-lived threads: a default executor thread or the event-loop thread outside an operation holds its legacy checkout until the handle closes, which can exhaust the pool without any thread exiting. (3) close_all_connections before reset: on psycopg_pool, CharactersRAGDB.close_all_connections() returns the checkout through putconn, which queues an asynchronous ReturnConnection, and then closes the pool at once; MaintenanceTask.run discards the task, so the connection is neither reset nor closed. In TASK-13841's test the discarded connection stayed referenced with an open transaction and blocked pg_restricted_backend's DROP OWNED BY until pytest-timeout (review: 19 of 52 runs). Product code calls close_all_connections() per call, for example the MCP modules (flashcards_module.py x13, notes_module.py x10). (4) Callers outside the middleware: ChaChaOperationMiddleware only wraps scope type 'http'. The WebSocket route /chat/stream (audio_streaming.py:1947) persists chats through _shim_get_chacha_db_for_user_id (:2415) with no chacha_operation; the persona.py and voice_assistant.py WebSocket routes come next. The only product callers of chacha_operation( are ChaCha_Notes_DB_Deps.py (3), conversation_enrichment.py (2), the repair scope in ChaChaNotes_DB.py (1) and study_pack_jobs_worker.py (1). None of these paths was exercised on PostgreSQL. (5) Test apps: 30 test files build a bare FastAPI() over ChaCha with Postgres params and no middleware (git grep -l 'app = FastAPI()' -- tldw_Server_API/tests | xargs grep -l 'CharactersRAGDB\|get_chacha_db_for_user' | xargs grep -l postgres | xargs grep -L ChaChaOperationMiddleware); each leaks one checkout per request and passes only because the default pool has 10 connections. Optional: pg_restricted_backend's teardown (12 test files) waits 300 s on a leaked idle-in-transaction session and then os._exits the session; SET lock_timeout before DROP OWNED BY, plus a report of the role's open sessions, would make it fail fast." \
  --ac "/chat/stream persistence is exercised on PostgreSQL first, and every production caller that reaches the legacy thread-local path on PostgreSQL is listed, with evidence" \
  --ac "No PostgreSQL checkout outlives its thread or pins a pool slot outside an operation: such callers run inside chacha_operation or return their checkout" \
  --ac "After CharactersRAGDB.close_all_connections() on psycopg_pool, no checkout it returned is left open without its reset" \
  --ac "Test apps over PostgreSQL ChaCha install ChaChaOperationMiddleware, or are shown not to leak"
```
Expected (Review M7; dry-run on `7c94a2efb1`): `TASK-<N> - Legacy thread-local PostgreSQL ChaCha checkouts leak (thread exit, long-lived threads, close_all_connections before reset) (To Do)`, and a new file `backlog/tasks/task-<N> - Legacy-thread-local-PostgreSQL-ChaCha-checkouts-leak-thread-exit-long-lived-threads-close-all-connections-before-reset.md` with 4 ACs and 6 DoD items.

- [ ] **Step 9: Close TASK-13841**

Fill the bracketed counts with the values recorded in Steps 4 and 6.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13841 \
  --append-notes "Cause: the test app in _client was a bare FastAPI() without the ChaChaOperationMiddleware that main.py:2796 installs. Without an operation, CharactersRAGDB falls back to a threading.local checkout. TestClient runs each request on a new portal whose AnyIO worker threads exit when the request ends, so each request leaked one PostgreSQL checkout. pg_restricted_backend sets pool_size=5 (max_size; max_overflow is ignored), so the 6th checkout waited the 30 s pool_timeout and raised PoolTimeout, giving a 500 at line 798. A probe of get_connection/return_connection showed 5 checkouts, 0 returns, pool_available 0, and holders 'AnyIO worker thread: DEAD' x3, portal x1, MainThread x1. Fix: _client installs ChaChaOperationMiddleware, so each request's checkout is returned when the response ends (probe: peak 3 outstanding). Pool size and timeout are unchanged. Product code is unchanged; HTTP in production already has the middleware." \
  --append-notes "Teardown hang uncovered by the fix: with the 500 gone, the body's own main-thread reads (lines 780-781, 793-794, 804-806) run outside any operation and leave both handles' legacy checkouts INTRANS. The owners teardown called first.close_all_connections(), which returns the checkout through putconn (queuing psycopg_pool's asynchronous ReturnConnection) and then closes the shared pool; MaintenanceTask.run discards a task on a closed pool, so the connection was neither rolled back nor closed. When it stayed referenced, its session stayed idle in transaction with a lock on message_images, and pg_restricted_backend's DROP OWNED BY (tests/_plugins/postgres.py:409) waited until pytest-timeout os._exited the session after 300 s. Fix: the owners teardown rolls back both handles' connections while the pool is open, inside try/finally around close_all_connections()." \
  --append-notes "RED (dev e1f7770953, re-checked on 7c94a2efb1): target 1 failed (postgres), 1 passed (sqlite); assert 500 == 201 at :798, error.type psycopg_pool.PoolTimeout. Whole file 1 failed, 270 passed. Middleware alone: the body passes but teardown hangs at postgres.py:409 DROP OWNED BY; deterministic with a scratch plugin that starts queued returns 0.5 s late and keeps them referenced (1 discarded INTRANS return); without it, two loops side by side: [N]/20 hung at load [L] (review: 19/52). GREEN with the teardown fix: the scratch plugin gives 2 passed (every discarded return IDLE); two loops side by side [N]/20 passed, 0 hangs, at load [L] (review: 46/46); whole file [result]. Ruff clean. Bandit (uvx bandit) on the file: B101 x290 and B105 x2, identical to dev, so no new findings." \
  --append-notes "AC #1 amended (Review M4): its old text said a second owner reusing the first owner's ids gets 201 on Postgres, which is false by design. PostgreSQL ids share one key space, so reused chat or message ids get 409 (lines 782-791); the 201 is the second owner's import with ids of its own (line 798), and the first owner's chat and message stay untouched (lines 804-808). Removed with --remove-ac 1 and re-added with that wording; it is now AC #2, and the old AC #2 is now #1." \
  --append-notes "ADR: not needed; a test-only change that creates no durable policy (Docs/ADR/README.md). Governing: ADR-005 (Bandit on the touched paths), ADR-059 (backlog-py), ADR-004 (the owner's Change summary on the AI-authored PR)." \
  --append-notes "Follow-up: TASK-<N>, for legacy thread-local checkouts (thread exit, long-lived threads), close_all_connections() closing a psycopg pool before its queued reset (MCP modules call it per call), the WebSocket /chat/stream persistence path, and the 30 other test apps without the middleware." \
  --remove-ac 1 \
  --ac "On PostgreSQL a second owner reusing the first owner's chat or message ids gets 409 (shared key space), imports with ids of its own get 201, and the first owner's chat and messages are untouched" \
  --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "The PoolTimeout was a per-request checkout leak in the test app, not in the import endpoint. The test built its app without ChaChaOperationMiddleware, so each request took a thread-local PostgreSQL checkout on an AnyIO worker thread that exited without returning it. The 5-connection restricted pool ran out at the 6th checkout. The test app now installs the same middleware as main.py. That let the body reach its own main-thread reads, whose open transaction could outlive the pool when teardown closed it, hanging DROP OWNED BY; the owners fixture now rolls those back while the pool is open. The target test passes without hangs and the whole file is 271 passed (dev: 1 failed, 270 passed). No timeout or pool size changed. AC #1 was amended to the 409/201 contract the test checks. Docs: not relevant (test-only). Known follow-up: TASK-<N>." \
  -s Done
```
Expected: `TASK-13841 - Postgres chat import returns 500 when a second owner reuses chat ids (Done)`. The ACs then read `- [x] #1 The cause of the pool timeout is fixed, not hidden by a longer timeout` and `- [x] #2 On PostgreSQL a second owner reusing …` (dry-run on `7c94a2efb1`).

- [ ] **Step 10: Commit**

```bash
git add tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py
git add -A backlog/tasks
git commit -m "$(cat <<'EOF'
test(chat): give the chat-import test app the operation middleware main.py installs (TASK-13841)

Without ChaChaOperationMiddleware every request took a thread-local
PostgreSQL checkout on an AnyIO worker thread that exited without
returning it, so the 5-connection restricted pool ran out and the sixth
request hit PoolTimeout (500).

With the 500 gone, the test's own main-thread reads left a transaction
open, and teardown closed the shared pool before psycopg_pool's queued
reset ran; DROP OWNED BY then waited on that session until
pytest-timeout. The owners fixture now rolls both handles back while
the pool is open. Files TASK-<N> for the product's legacy path.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Read the image test's returned checkout after the pool republishes it (TASK-13842, image test)

**Files:**
- Modify: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py:330-332` (state lists), `:347-355` (pool hooks) and `:375-378` (final assertions). These are dev line numbers; edit by content, because earlier hunks shift later lines.

**Interfaces:**
- Consumes: `PostgreSQLConnectionPool.return_connection(raw)`, `PostgreSQLConnectionPool._use_psycopg_pool: bool` and `PostgreSQLConnectionPool._pool._add_to_pool(conn)`. The last is psycopg_pool's private publish step; it runs after the reset hook and only for a connection that is put back into service.
- Consumes (fallback pool, Review M8): `PostgreSQLConnectionPool._free: list`. The fallback `return_connection` rolls back synchronously and then calls `self._free.append(connection)` under `_fallback_lock` (`postgresql_backend.py:332-353`); only a connection that is `IDLE` after the rollback is appended.
- Produces: none. This is a change to a single test.

- [ ] **Step 1: Make the dev failure deterministic (RED)**

Run:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p slow_reset -q \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py \
  -k "test_character_image_reads_leave_event_loop_and_release_operation and postgres" \
  > "$P/runs/img_red.log" 2>&1
grep -E "^=+ .*(passed|failed)" "$P/runs/img_red.log"; grep -h "^E   AssertionError" "$P/runs/img_red.log" | sort | uniq -c
```
Expected (94 s at load ~30):
```
==== 6 failed, … ====
      3 E   AssertionError: assert 'ACTIVE' == 'IDLE'
      3 E   AssertionError: assert 'INTRANS' == 'IDLE'
```
The split between `ACTIVE` and `INTRANS` is incidental: the review got 4 and 2 (250 s, load 28-42). Every case must fail at `:378`.

- [ ] **Step 2: Record repeated-run counts on dev without the plugin (AC1 evidence)**

Run:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 REP_N=4 REP_MATCH=test_character_image_reads_leave_event_loop_and_release_operation \
  PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p rep -q \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py \
  -k "test_character_image_reads_leave_event_loop_and_release_operation and postgres" > "$P/runs/img_dev_rep4.log" 2>&1
grep -E "^=+ .*(passed|failed)" "$P/runs/img_dev_rep4.log"; uptime
```
Expected: some or all of the 24 fail, all `'ACTIVE' == 'IDLE'` at `:378`. At planning time, load 40-50, the result was `24 failed` in 388 s; the review got `24 failed` in 284 s at load 33. Record the actual count and load.

- [ ] **Step 3: Record the whole-file baseline on dev**

This was not run at planning time. Run:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py \
  > "$P/runs/img_file_base.log" 2>&1; grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)" "$P/runs/img_file_base.log"
```
Expected: 108 cases. The only failures should be the `[postgres-*]` cases of the target test. Record any other failure as a dev baseline before going on. (The review's whole-file run on the first-draft fix gave `108 passed in 1034s` at load 34, so no other failure exists on `32eb11ae03`.)

- [ ] **Step 4: Change the test (GREEN)**

Replace the state lists (dev `:330-332`):

```python
    owners = []
    reads = []
    returned = []
    republished = []
```

Replace the PostgreSQL pool hooks (dev `:347-355`). Both pools are observed after their own cleanup and before the checkout is published for reuse, as `test_chacha_postgres_http_operation_lifecycle.py:128-151` does: psycopg_pool through `_add_to_pool`, the fallback pool through an `ObservedFreeList` on `_free` (Review M8; the first draft observed the fallback after `return_connection` returned, when the checkout was already in `_free`).

```python
    if character_db.backend_type == BackendType.POSTGRESQL:
        pool = character_db.backend.get_pool()
        original_return = pool.return_connection
        handback = threading.Condition()

        def settled(raw: Any) -> None:
            """Record a checkout this request handed back, once the pool has cleaned it up."""
            with handback:
                if any(connection is raw for connection in returned):
                    republished.append((raw, raw.info.transaction_status.name))
                    handback.notify_all()

        def return_connection(raw: Any) -> None:
            with handback:
                returned.append(raw)
            original_return(raw)

        monkeypatch.setattr(pool, "return_connection", return_connection)
        # Observe each checkout after the pool's own cleanup, just before it is published for reuse.
        if pool._use_psycopg_pool:
            # psycopg_pool resets a returned checkout on its own worker, then publishes it.
            original_publish = pool._pool._add_to_pool

            def publish(raw: Any) -> None:
                settled(raw)
                original_publish(raw)

            monkeypatch.setattr(pool._pool, "_add_to_pool", publish)
        else:

            class ObservedFreeList(list):
                """The fallback pool rolls back synchronously, then appends the checkout to its free list."""

                def append(self, raw: Any) -> None:
                    settled(raw)
                    super().append(raw)

            monkeypatch.setattr(pool, "_free", ObservedFreeList(pool._free))
```

Replace the final PostgreSQL assertions (dev `:375-378`):

```python
    if character_db.backend_type == BackendType.POSTGRESQL:
        assert raw is not None
        assert sum(connection is raw for connection in returned) == 1
        with handback:
            assert handback.wait_for(lambda: any(item is raw for item, _ in republished), timeout=5.0), (
                "The pool never put the request's checkout back into service"
            )
        assert [status for item, status in republished if item is raw] == ["IDLE"]
```

Why this does not weaken the check:
- The old assertion meant "after the response, the returned checkout is clean". It read that state at an arbitrary moment, sometimes in the middle of the pool's reset.
- The new one reads it at the moment the pool puts the checkout back into service. It also requires that this happens exactly once.
- `settled` records only checkouts already in `returned`. A stale republication of the same connection from an earlier request's return (one that finished before the measured request borrowed it) is therefore never counted.
- A checkout the pool cannot clean up is never republished, so the new test fails with a clear message where the old one sometimes passed by chance (Step 6). On dev, `bad_reset` also fails, but at `:378` with `'ACTIVE' == 'IDLE'`: the race masks whether the old check would ever catch a bad reset (review).
- On psycopg_pool, `_reset_connection` discards any connection that is not `IDLE` after the reset, so there `== ["IDLE"]` reduces to "republished exactly once"; the status part bites on the fallback path, which appends only `IDLE` connections to `_free` (review). Republication is strictly stronger than a point-in-time `IDLE` read, and the earlier assertions (worker thread, HTTP owner, `retiring` and `conn is None`, returned exactly once) are untouched.
- Lock order is safe: `ObservedFreeList.append` runs under `_fallback_lock` and then takes `handback`; the test thread holds `handback` only while waiting, never `_fallback_lock`. `_add_to_pool` takes the pool lock only after the hook.
- The 5 s wait is a bound for the failure case, not a race: it returns as soon as publication happens.

- [ ] **Step 5: Run the target cases under the RED condition and repeated (GREEN)**

Run:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p slow_reset -q \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py \
  -k "test_character_image_reads_leave_event_loop_and_release_operation and postgres" 2>&1 | grep -E "^=+ .*(passed|failed|error)"
TLDW_TEST_POSTGRES_REQUIRED=1 REP_N=4 REP_MATCH=test_character_image_reads_leave_event_loop_and_release_operation \
  PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p rep -q \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py \
  -k "test_character_image_reads_leave_event_loop_and_release_operation and postgres" 2>&1 | grep -E "^=+ .*(passed|failed|error)"
```
Expected: `6 passed`, then `24 passed`. Measured on the code above (`7c94a2efb1`): `6 passed` in 235 s (load ~30) and `24 passed` in 264 s (load ~36). The first draft (identical psycopg path) gave the same at planning time and in review (`24 passed` in file order at load 25, and shuffled with `SHUFFLE_SEED=2001` at load 55).

These steps grep the summary line instead of `tail -1`: loguru can print after pytest's summary at interpreter exit, and one `tail -1` here printed `}` (measured).

- [ ] **Step 6: Show the check still bites, and covers the fallback pool**

Run:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p bad_reset -q \
  "tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py::test_character_image_reads_leave_event_loop_and_release_operation[postgres-complete-context]" \
  2>&1 | grep -E "^E   AssertionError|passed|failed"
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p fallback_pool -q \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py \
  -k "test_character_image_reads_leave_event_loop_and_release_operation and postgres" 2>&1 | grep -E "^=+ .*(passed|failed|error)"
```
Expected:
- The first command prints `E   AssertionError: The pool never put the request's checkout back into service` and `1 failed` (measured on the code above: 44 s, load ~37).
- The second prints `6 passed` (measured on the code above: 227 s, load ~33-40). The `ObservedFreeList` is the only path that records a status here, so a pass shows it fired.

- [ ] **Step 7: Whole file, lint, Bandit**

Run:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py \
  > "$P/runs/img_file_fix.log" 2>&1; grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)" "$P/runs/img_file_fix.log"
$PY -m ruff check tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py
uvx bandit -q -r tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py -f json -o "$P/runs/bandit_img.json"
$PY -c "import json,collections;d=json.load(open('$P/runs/bandit_img.json'));print(collections.Counter(r['test_id'] for r in d['results']))"
```
Expected:
- The whole file passes, apart from any baseline failure recorded in Step 3 (review, first draft: `108 passed in 1034s`, load 34).
- Ruff prints `All checks passed!` (measured on the code above).
- Bandit prints `Counter({'B101': 40})` (measured on the code above): dev has 39 B101 findings, and the one extra is the new test `assert`.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_images.py
git commit -m "$(cat <<'EOF'
test(chat): read the image request's checkout when the pool republishes it (TASK-13842)

psycopg_pool resets a returned connection on its own worker, so reading
raw.info right after the response raced the reset (ACTIVE/INTRANS). The
test now waits for the pool to put that checkout back and requires it to
be IDLE exactly then, which also fails if the pool cannot clean it up.
The fallback pool is observed the same way, when it appends the rolled-back
checkout to its free list.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Give the admin health-timeout tests a budget that always reaches dispatch (TASK-13842, admin tests)

**Files:**
- Modify: `tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py:960` (`timeout_seconds=0.05`), `:1472` (`lambda: 0.05`) and `:1483` (`[0.05]`). These are dev line numbers; edit by content, because the added comment lines shift later lines.
- Backlog: `backlog/tasks/task-13842 - …md`

**Interfaces:**
- Consumes: the `_install_real_health_chat_capacity_boundary(..., timeout_seconds: float)` helper (`:805`), which patches `admin_llm_providers._provider_health_timeout_seconds` to return `timeout_seconds`. `byok_testing._app_config_with_health_timeout` maps that timeout to `api_timeout = max(1, ceil(timeout))`.
- Produces: none.

- [ ] **Step 1: Make both dev failures deterministic (RED)**

Run:
```bash
TLDW_TEST_NO_DOCKER=1 PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p slow_predispatch -q \
  tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py \
  -k "test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers or test_admin_health_multi_provider_timeouts_retain_real_worker_admission_and_chat_headroom" \
  2>&1 | grep -E "runtime.py:[0-9]+: in|^E   assert|passed|failed"
```
Expected:
```
…test_admin_llm_provider_test_runtime.py:973: in test_admin_health_multi_provider_timeouts_retain_real_worker_admission_and_chat_headroom
E   assert 0 == 1
…test_admin_llm_provider_test_runtime.py:1481: in test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers
E   assert 0 == 1
==== 2 failed, 32 deselected, … ====
```
These are the same lines that failed in the unpatched sweeps.

- [ ] **Step 2: Record repeated-run counts on dev (AC1 evidence)**

Run:
```bash
TLDW_TEST_NO_DOCKER=1 REP_N=50 REP_MATCH=test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers \
  PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p rep -q \
  "tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py::test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers" 2>&1 | grep -E "^=+ .*(passed|failed|error)"
TLDW_TEST_NO_DOCKER=1 $P/runs.sh adm_dev_order 20 0 tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py | grep -E "failed|FAILED"
TLDW_TEST_NO_DOCKER=1 $P/runs.sh adm_dev_shuffle 20 1 tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py | grep -E "failed|FAILED"
uptime
```
Expected: failures, if any, only in the two tests named above, at `:1481` (main) and `:973` or `:995` (sibling). How many depends on load, in isolation too (Review I2). Counts so far:

| Run | Planning time | Review |
|---|---|---|
| Main test isolated, `REP_N=50` | 0/50 failed (load ~21) | 7/50 and 6/50 failed (load 47-56) |
| File order, 20-23 runs | main 2/23, sibling 0/23 (load 16-35) | 3/20 runs failed: main 2, sibling 3 (load 25-58) |
| Shuffled, seeds 1001-1020 | main 1/20 (seed 1009), sibling 4/20 (seeds 1007, 1009, 1011, 1012) (load 45-56) | 0/20 (load 32-62) |

Record the actual counts and load. Zero failures at low load does not contradict Step 1.

- [ ] **Step 3: Change the two budgets (GREEN)**

At `:960`, inside `test_admin_health_multi_provider_timeouts_retain_real_worker_admission_and_chat_headroom`:

```python
        admission=admission,
        # Sub-second, yet long enough that every provider reaches its admission check on a loaded machine.
        timeout_seconds=0.5,
    )
```

At `:1469-1474`, inside `test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers`:

```python
    monkeypatch.setattr(
        admin_llm_providers,
        "_provider_health_timeout_seconds",
        # Sub-second, so api_timeout still rounds up to 1; long enough that the call is dispatched first.
        lambda: 0.5,
        raising=False,
    )
```

At `:1483`:

```python
        assert adapter.timeouts == [0.5]
```

Leave the line after it (`adapter.requests[0]["app_config"]["ollama_api"]["api_timeout"] == 1`) unchanged. With 0.5 it still proves that a sub-second health timeout overrides the configured 300 and rounds up to 1.

Why this does not weaken the checks:
- Every assertion keeps its subject:
  - the first check errors while its blocked worker holds the only pool slot (`active_count == 1`, `call_count == 1`);
  - the second check is refused without calling the adapter;
  - capacity recovers and the third check is healthy.
- The blocked adapters still outlast the budget: `_BlockingAdapter` waits up to 2.0 s and `_AlwaysBlockingAdapter` up to 5.0 s, and both are released only after the timed-out call. So the first call still times out on every run.
- The only thing removed is the race between a 50 ms budget and pre-dispatch work the test does not control: 5-50 ms of synchronous work on the event loop, which CPU load stretched to 56-231 ms in failing runs (Review I2).
- Margin (Review M3): 0.5 s is about 2x the worst pre-dispatch latency observed (231 ms, review, load ~50-56). With the fix the worst observed was 108 ms. An event-loop stall longer than 0.5 s would still fail these tests.
- Mutation (review): with the health timeout made ineffective (`timeout=30.0` at `admin_llm_providers.py:271` and `byok_testing.py:309`), both fixed tests fail. They still catch a timeout that does not bound the call.
- The sibling's outer `asyncio.wait_for(..., timeout=1.0)` guards (dev `:969`, first batch; `:988-993`, recovery) are left at 1.0 s (Review M2). Their headroom over the inner budget drops from about 0.95 s to about 0.5 s. That is no tighter than the new inner margin, and they passed 40/40 at planning time (load 36-53) and 80/80 fixed sibling runs in review.

- [ ] **Step 4: Rerun the RED condition and the repeated runs**

Run:
```bash
TLDW_TEST_NO_DOCKER=1 PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p slow_predispatch -q \
  tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py \
  -k "test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers or test_admin_health_multi_provider_timeouts_retain_real_worker_admission_and_chat_headroom" 2>&1 | grep -E "^=+ .*(passed|failed|error)"
TLDW_TEST_NO_DOCKER=1 REP_N=50 REP_MATCH=test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers \
  PYTHONPATH=$P $PY -m pytest -p no:cacheprovider -p rep -q \
  "tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py::test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers" 2>&1 | grep -E "^=+ .*(passed|failed|error)"
TLDW_TEST_NO_DOCKER=1 $P/runs.sh adm_fix_order 20 0 tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py | grep -cE "34 passed"
TLDW_TEST_NO_DOCKER=1 $P/runs.sh adm_fix_shuffle 20 1 tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py | grep -cE "34 passed"
```
Expected: `2 passed, 32 deselected`, then `50 passed`, then `20`, then `20`. Planning time measured these on an identical copy (except the isolated run). The review measured `2 passed, 32 deselected`, `50 passed` (load 47-56, where dev failed 7/50 and 6/50), 20/20 file order and 20/20 shuffled, plus 40/40 in a 4-loop stress run at load 38-57 (dev 7/40).

- [ ] **Step 5: Neighbouring admin and BYOK files, lint, Bandit**

Run:
```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py \
  tldw_Server_API/tests/Admin/test_admin_llm_providers_health.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_validation_boundary.py 2>&1 | grep -E "^=+ .*(passed|failed|error)"
$PY -m ruff check tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py
uvx bandit -q -r tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py -f json -o "$P/runs/bandit_adm.json"
$PY -c "import json,collections;d=json.load(open('$P/runs/bandit_adm.json'));print(collections.Counter(r['test_id'] for r in d['results']))"
```
Expected:
- `155 passed` (review, on the fix). Record a dev baseline for any failure before going on.
- Ruff prints `All checks passed!` (measured again on `7c94a2efb1`).
- Bandit prints `Counter({'B101': 142, 'B105': 8})`, identical to dev (measured again on `7c94a2efb1`).

- [ ] **Step 6: Close TASK-13842**

Fill the bracketed counts with the values recorded in Task 2 Steps 2-7 and Task 3 Steps 2-5. Do not reuse the planning-time values unless they match.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13842 \
  --append-notes "Image test, test_character_image_reads_leave_event_loop_and_release_operation[postgres-*]: flaky, and how often depends on load. Cause: the backend pool has reset=_reset_pooled_connection, so psycopg_pool returns a checkout through a pool worker (rollback, then RESET ROLE / SESSION AUTHORIZATION / ALL and COMMIT) and only then republishes it with _add_to_pool. The test read raw.info.transaction_status from the test thread during that reset and saw ACTIVE (or INTRANS). The request itself hands back INTRANS by design; the pool rolls it back. This is the same race as TASK-13412. Dev counts: REP_N=4 (24 cases) [N] failed at load [L], all 'ACTIVE' == 'IDLE' at :378. With an injected 0.3 s reset (scratch slow_reset plugin): 6/6 failed. Sqlite cases pass." \
  --append-notes "Image fix: the test observes the request's checkout after the pool's own cleanup and just before the pool publishes it for reuse, and requires exactly one such publication, IDLE at that moment. psycopg_pool is observed through its _add_to_pool step; the fallback pool (no psycopg-pool) through an ObservedFreeList on _free, after its synchronous rollback, as test_chacha_postgres_http_operation_lifecycle.py does. GREEN: slow_reset 6/6 passed; REP_N=4 [N] passed; fallback pool 6/6 passed; whole file [result]. The check still bites: with a reset that raises, the fixed test fails with 'The pool never put the request's checkout back into service'." \
  --append-notes "Admin test, test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers: flaky in file order, shuffled order and isolation, and how often depends on CPU load. Cause: the test gave the health check 0.05 s, and the endpoint's asyncio.wait_for covers the whole test_provider call, including 5-50 ms of synchronous pre-dispatch work on the event loop (the real load_server_config_snapshot() at 3-10 ms, the rest of test_provider, event-loop scheduling, and in this test one cold AuthNZ Settings() build of about 4 ms). CPU load stretches that work past 50 ms: failing dev runs were cancelled after 175 and 231 ms, before dispatch, so pool.active_count was 0 at :1481. Admission semaphores were full at test start (no leak). Dev counts: isolated REP_N=50 [N]/50 failed at load [L] (review: 7/50 and 6/50 at load 47-56); file order [N]/20; shuffled seeds 1001-1020 [N]/20. With a 0.15 s injected pre-dispatch delay (scratch slow_predispatch plugin): fails at :1481 every time." \
  --append-notes "Third test, same cause and file, fixed under this task: test_admin_health_multi_provider_timeouts_retain_real_worker_admission_and_chat_headroom (0.05 s at :960) failed at :973 (call_count 0 == 1) and :995 (recovery healthy 2 == 3). It never builds Settings (the test stubs resolve_byok_credentials), and failing runs were cancelled after 56-159 ms of pre-dispatch work. Dev: file order [N]/20, shuffled [N]/20; slow_predispatch fails at :973. Fix for both: a 0.5 s budget, about 2x the worst pre-dispatch latency observed (231 ms). It is still sub-second, so api_timeout still rounds up to 1, and the blocked adapters (2 s / 5 s, released only after the timed-out call) still outlast it, so every assertion keeps its subject; with the timeout made ineffective both fixed tests fail. The sibling's outer 1.0 s wait_for guards keep about 0.5 s of headroom (was about 0.95 s). GREEN: slow_predispatch 2 passed; isolated REP_N=50 [N]/50 passed; file order 20/20; shuffled 20/20. Not changed: test_individual_admin_provider_timeout_stays_prompt_but_retains_worker_admission also uses 0.05 s, but admin_test_llm_provider has no outer wait_for and its clock starts inside _run_bounded_health_call (admission_deadline, byok_testing.py:259), after pre-dispatch, so it cannot hit this race (0 failures in 43 planning-time runs)." \
  --append-notes "pytest-randomly is not installed, so -p no:randomly is a no-op; shuffled order came from a scratch collection hook seeded 1001-1020. Bandit (uvx): image file B101 x40 (dev 39; +1 is the new assert); admin file B101 x142, B105 x8 (same as dev). Ruff clean." \
  --append-notes "ADR: not needed; a test-only change that creates no durable policy (Docs/ADR/README.md). Governing: ADR-005 (Bandit on the touched paths), ADR-059 (backlog-py), ADR-004 (the owner's Change summary on the AI-authored PR)." \
  --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Both tests were flaky, plus a third with the same cause. The Postgres image test read a returned connection while psycopg_pool was still resetting it on a pool worker; it now reads the connection when the pool republishes it, and must find it IDLE then. The admin health tests gave a 50 ms budget to a call that first does 5-50 ms of synchronous pre-dispatch work, which CPU load stretched past 50 ms (up to 231 ms observed), so the call was cancelled before dispatch; a 0.5 s sub-second budget keeps every assertion and the api_timeout round-up. With injected delays that reproduce each race, dev fails every time and the fixed tests pass; repeated, isolated and shuffled runs are recorded in the notes. Docs: not relevant (test-only). Known: a 0.5 s budget can still be beaten by an event-loop stall longer than 0.5 s; the individual-provider 0.05 s test was left alone because its clock starts after pre-dispatch." \
  -s Done
```
Expected: `TASK-13842 - Two tests flaky in full sweeps: … (Done)`.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/tests/Admin/test_admin_llm_provider_test_runtime.py
git add -A backlog/tasks
git commit -m "$(cat <<'EOF'
test(admin): give health-timeout tests a budget that always reaches dispatch (TASK-13842)

The 50 ms health timeout also covered 5-50 ms of synchronous pre-dispatch
work on the event loop, and CPU load stretched that work past 50 ms (up
to 231 ms observed), so the call was sometimes cancelled before the
adapter was dispatched. 0.5 s is still sub-second (api_timeout still
rounds up to 1) and still shorter than the blocked adapters, so every
assertion keeps its subject.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Review and open the PR

**Files:** none changed.

**Interfaces:**
- Consumes: the branch `fix/test-reliability-13841-13842` with the three commits above.
- Produces: one PR into `dev`.

- [ ] **Step 1: Check the branch diff and the ratchets**

Run:
```bash
git diff --stat origin/dev...HEAD
git diff --check origin/dev...HEAD && echo "no trailing whitespace"
$PY Helper_Scripts/ci/test_quality_triage.py --enforce 2>&1 | grep "test-triage\] OK"
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
  $PY -m pytest -q --disable-warnings -p pytest_asyncio.plugin -p no:cacheprovider \
  tldw_Server_API/tests/CI/test_test_quality_ratchet.py 2>&1 | grep -E "^=+ .*(passed|failed|error)"
BACKLOG_TASK_FORMAT_BASE=$(git rev-parse origin/dev) PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
  $PY -m pytest -q --disable-warnings -p pytest_asyncio.plugin -p no:cacheprovider \
  tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py 2>&1 | grep -E "^=+ .*(passed|failed|error)"
```
Expected:
- 6 files: the 3 test files, plus `task-13841`, `task-13842` and the new `task-<N>` under `backlog/tasks/`.
- `no trailing whitespace`. `git diff --check` replaces the earlier `grep -nE "^\+.*[[:space:]]$"`, which flags the `+++ b/backlog/tasks/task-… .md` header lines: a header for a path with spaces ends in a TAB (Review M5; reproduced on `7c94a2efb1`, where it printed the three task headers).
- `[test-triage] OK — no offenses beyond the baseline.` The test-quality ratchet is a required backend step on dev since `6a6fcf7533` (Review M6).
- `1 passed` for `test_test_quality_ratchet.py` (measured: 85 s).
- `4 passed` for the task-format ratchet.

Measured on `7c94a2efb1` with all three tasks' changes and the backlog-py commands applied (staged, not committed): `git diff --cached --check` clean; triage `OK`; `test_test_quality_ratchet.py` `1 passed`; `backlog_py task normalize --check` on the three task files exit 0. The task-format ratchet diffs `base...HEAD`, so it only sees the task files once they are committed; the review ran it on a committed tree (`BACKLOG_TASK_FORMAT_BASE=32eb11ae03`) with the scope-predicate, private-coercion and route-auth ratchets: `34 passed`.

- [ ] **Step 2: Independent review**

Run `/superpowers:requesting-code-review` in a clean subagent (Qodo is out of credits until next month). Give it the branch diff and both task files. Address the findings before opening the PR.

- [ ] **Step 3: Push and open the PR against dev**

```bash
git push -u origin fix/test-reliability-13841-13842
gh pr create --base dev --title "test: fix the Postgres chat-import pool leak and three timing-flaky tests (TASK-13841, TASK-13842)" --body "$(cat <<'EOF'
## Summary
- TASK-13841: the chat-import test app now installs ChaChaOperationMiddleware, as main.py does. Without it every request leaked a thread-local PostgreSQL checkout from an AnyIO worker thread that exited, and the 5-connection restricted pool timed out on the 6th request (500). No pool size or timeout changed.
- TASK-13841: the owners fixture rolls back the test's own main-thread transactions while the shared pool is still open. Otherwise close_all_connections() closed the pool before psycopg_pool's queued reset ran, and the open transaction hung DROP OWNED BY in teardown until pytest-timeout.
- TASK-13842: the Postgres image test reads the request's checkout when the pool republishes it (psycopg_pool) or appends it to its free list (fallback pool), instead of during the pool's async reset.
- TASK-13842: two admin health tests use a 0.5 s budget instead of 0.05 s. The 50 ms also had to cover 5-50 ms of synchronous pre-dispatch work, which CPU load stretched past 50 ms (up to 231 ms observed).
- Follow-up TASK-<N>: legacy thread-local checkouts (thread exit, long-lived threads), close_all_connections() closing a psycopg pool before its queued reset (MCP modules call it per call), the WebSocket /chat/stream persistence path, and 30 other test apps without the middleware.

## Change summary
<!-- The owner writes this section (ADR-004). -->

## Test plan
- [ ] test_chat_import_local_chats.py on Postgres: 271 passed (dev: 1 failed, 270 passed)
- [ ] chat-import target: teardown hang deterministic with an injected late return (middleware alone hangs; fixture fix passes); two loops side by side, middleware alone hung 9/20, fixed 20/20
- [ ] image test postgres cases: deterministic RED with an injected reset delay; GREEN repeated; fallback pool covered
- [ ] admin runtime file: deterministic RED with an injected dispatch delay; GREEN 20 ordered + 20 shuffled runs
- [ ] ruff clean; Bandit: no new findings beyond test asserts; test-quality ratchet OK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: the PR URL.

- [ ] **Step 4: Before merge**

- Ask the owner for the `## Change summary` (ADR-004). This PR is not covered by a recorded waiver.
- Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.
- Confirm the PR is `MERGED` before deleting the branch or the worktree: `git worktree remove --force <scratchpad>/test-reliability`.
