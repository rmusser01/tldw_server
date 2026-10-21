# ChaCha PostgreSQL Checkouts Outside HTTP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Review folded in (revision 2, 2026-10-20).** The independent review (`<scratchpad>/review_plan_fu_chacha_pg_checkouts.md`, verdict "not ready": 2 Blocker, 5 Important, 7 Minor) and the coordinator's rulings are folded in. Every changed step was re-measured on PostgreSQL 18 against origin/dev `e521605198`.
> - **The split changed: 3 PRs and 12 tasks became 4 PRs and 16 tasks.** v1's PR B (WebSockets plus a central `WorkerSDK` wrap) is now PR B (WebSockets and persona) and PR C (Jobs and other background work, per worker). v1's PR C (test apps) is now PR D.
> - **Blocker 1 (central wrap broke notes-graph).** C1 makes the notes-graph cleanup operation-aware and adds a PostgreSQL `WorkerSDK.run` test of the real job. The release contract that `test_chacha_operation_scope.py` pins is unchanged. The "close, then reuse" audit of every Jobs handler and callee is F11: notes-graph is the only one.
> - **Blocker 2 (the collection-time return is GC-timed).** The guard is now described everywhere as a thread-exit fix plus a GC-timed backstop. C3 makes the five never-closed per-call sites own their checkouts, tested with garbage collection disabled. The review also listed MCP `chats_module` and `rpg_module`: both already close on the thread that used the handle (F2).
> - **Important 1 (Decision 1).** Decision 1 is restated with the September note's words and the measured costs: Blocker 1, and a checkout held through the LLM call. On that evidence the plan now recommends per-worker operations at each job owner, the note's own proposal. The non-SDK runners are covered in C3.
> - **Important 2 (persona).** The measured numbers are in F12 and Decision 2. B4 adds the per-call wrap, a `chacha_operation(...)` around each `asyncio.to_thread` callable, as recommended. It costs about 0.8 ms per call.
> - **Important 3.** C2 adds one PostgreSQL `WorkerSDK` test per ChaCha-touching worker family (F11), and C1 covers notes-graph.
> - **Important 4.** The ADR is "ADR-NNN" until C4 Step 1 numbers it with a paged scan, taking max(dev, open PRs) + 1. The sibling plan `plan_fu_rg_redis_error.md` numbers its ADR the same way. The two execute serially, so whichever goes second takes the next number.
> - **Important 5.** `_LegacyCheckout.__del__` is lock-free: it puts `(backend, conn)` on a module-level `queue.SimpleQueue`, which `_open_new_connection` and `close_connection` drain. A2 has a test for it, and it is Review Focus 2.
> - **Minor 1-7** are folded in at A3 (impact), B6/C4 (RED on top of earlier PRs), C4 Step 4 (stage before the docs tests), the Global Constraints (pre-existing ruff findings; the voice test's F811 is fixed), A4 Step 1, D2 Step 4 (six and six), A2 (the closed-connection branch is tested) and D1 (the guard's stated limit).

> **Second review folded in (revision 3, 2026-10-20).** The re-review (`<scratchpad>/rereview_plan_fu_chacha_pg_checkouts.md`, verdict "not ready": 1 Blocker, 2 Important, 8 Minor) and the coordinator's rulings are folded in. Every changed step was re-measured on PostgreSQL 18 against origin/dev `2cae53929f`.
> - **The split is unchanged at 4 PRs. The task count goes from 16 to 17:** PR B gains Task B5, which wraps the persona session's on-loop calls; the old B5 is now B6.
> - **Blocker 1 (one operation around the Chat_Macros run lost writes): option (a).**
>   - `handle_chat_macro_job` is no longer wrapped. In the executor, each repository call and the macro loader run in their own independent operation. In `jobs.py`, the executor build, the post-back and the failure-path status write each own their operation (C2).
>   - New test `test_a_macro_run_whose_branches_write_concurrently`. It runs the real job of a 4-branch macro through `WorkerSDK.run` on PostgreSQL and asserts the saved run status and branch rows.
>     - On revision 2's PR C (the plan as written) it fails with `('running', [])`: every branch row and the final status were rolled back.
>     - On dev it fails on the pinned checkouts (`asyncio_0` … `asyncio_3`).
>     - With option (a) it passes.
>   - F11 gains a "concurrent ChaCha transactions inside one owner" column for every wrapped owner. F13 audits HTTP routes for the same hazard: none overlaps managed transactions on one handle. So C4 files option (b), serializing transactions per `ConnectionState`, as a follow-up guard. Decision 1 states the cost.
> - **Important 1 (persona): the recommendation changed to Alternative 1.**
>   - Measured after B4: 11 live sessions on a 10-connection pool. 10 held a checkout idle in a transaction on the event-loop thread, and the 11th raised `PoolTimeout` after blocking the whole event loop for `pool_timeout` (F12).
>   - B5 wraps the 14 on-loop call sites, 13 sync and 1 async, with an AST guard for direct calls. The cost is counted in round trips. (Revision 4 narrows the async one to the skill's registry lookup; see below.)
>   - The persona follow-up task is filed only if the owner declines.
> - **Important 2.** The family test file stays at 3 patches or fewer per test: the Chatbooks cache reset moved into a fixture. The triage now reports `OK — no offenses beyond the baseline`, with the baseline untouched.
> - **Minor 1-8:**
>   - M1: A4 Step 1 expects 108.
>   - M2: the Global Constraints list all 13 touched product files with pre-existing ruff findings.
>   - M3: Task A2 and Review Focus 2 say a queued return waits for the next ChaCha call. No drainer thread is added.
>   - M4: the ADR number is "whatever the scan prints" (#3288 holds 078); the `Docs/ADR/README.md` conflict with #3288 is noted; `adr_next.sh` matches `[0-9]{3}`; PR A's note stays as written.
>   - M5: costs are stated in round trips.
>   - M6: `run_chatbooks_adapter` uses `@owns_chacha_checkouts(independent=True)`, with a test; Review Focus 3.
>   - M7: C4 Step 5 states the union's build order and its count.
>   - M8: `--check-ac 2` depends on Decision 2.

> **Third review folded in (revision 4, 2026-10-20).** The round-3 re-review (`<scratchpad>/rereview3_plan_fu_chacha_pg_checkouts.md`, "ready with minors": 6 Minor) and the coordinator's rulings are folded in. Changed tests were re-run on PostgreSQL 18 against origin/dev `3ad31cba25`. The split (4 PRs) and the task count (17) are unchanged.
> - **Ruling 1: B5's residual is stated, not claimed as fixed** (B5, F12, Decision 2, the ADR's cost line).
>   - A wrapped on-loop call still checks out on the event-loop thread. When other work holds the pool, it stalls the loop for up to `pool_timeout`: 2.0 s measured by the re-review.
>   - B6 Step 2 now always files TASK-<N>, "Persona: run the 10 on-loop ChaCha calls inside async handlers off the event loop". The calls are not moved in this plan.
> - **Ruling 2: narrowed.** `handle_skill_tool_call` now owns only its registry lookup, in `context_integration.py`. `persona.py` no longer wraps the whole call, so no checkout spans a forked skill's LLM call.
>   - The skill's ChaCha work separates cleanly: `SkillExecutor` gets no handle, the change is two lines, and `persona.py` is the only production caller.
>   - New PostgreSQL test `test_no_checkout_is_held_while_the_skill_runs`: on dev it fails with `[['asyncio_0']]`, and with B5 it passes.
>   - B5's guard now counts 13 wrapped sites and lists the skill call as self-owning.
> - **Ruling 3:** C3's snippet shows `@owns_chacha_checkouts(independent=True)` between `@registry.register(...)` and `async def`, and the Workflows cases run `registry.get_adapter("chatbooks")`. Measured: with the decorator above `@registry.register`, both cases fail.
> - **Ruling 4:** test (d)'s branch runner records the checkouts out during each branch, and the test asserts `MainThread` holds none. Measured: it fails with the executor-build and post-back operations removed, and passes with them.
> - **Ruling 5:** B4's and B6's lists add `persona/stream`, and B6's adds Skills' `context_integration` for ruling 2.
>   - B4's list is 16 files and B6's is 120. The re-review's 118 predates ruling 2.
>   - C4's list is 117 and the union is 334.
>   - The 9 added files ran with all four PRs, with no failure.
> - **Ruling 6:**
>   - F11's `jobs.py` lines are `:102` and `:313`;
>   - PR D names the ADR "the number C4 Step 1 assigned";
>   - C4 computes `$ADR` from the scan.
> - **New counts:**
>   - 13 new test files and 32 new tests;
>   - RED: 30 failed and 2 pins passing (A 9+1, B 8, C 11+1, D 2);
>   - GREEN: 32 passed.

> **Fourth review folded in (revision 5, 2026-10-20).** The round-4 scoped review (`<scratchpad>/rereview4_plan_fu_chacha_pg_checkouts.md`, "ready with minors": 5 Minor) is folded in. Revision 4 is kept as `plan_fu_chacha_pg_checkouts.r4.md`. The split (4 PRs), the task count (17) and the test counts (13 new files, 32 tests: RED 30 failed and 2 pins passing, GREEN 32) are unchanged.
> - **Ruling 1:** B4 Step 4 and B5 Step 4 run `git add -N <the new test>` before their `git grep`, which reads tracked files only. They now print the stated `16` and `18`.
> - **Ruling 2:** TASK-<N>'s description (B6 Step 2) no longer says the skill lookup "now" checks out on the loop. It says `SkillsService.__init__` already runs 4 table-setup statements through `backend.execute` on the loop, outside any operation, at every skill call, on dev and with B5. It gains AC 3, "The skill tool call's SkillsService setup and lookup make no ChaCha checkout on the event-loop thread". The Alternative 2 task gets the same AC, and its description is corrected the same way. F12 and B5 say the same.
> - **Ruling 3:** the skill lookup uses `chacha_operation(independent=True)`, so it never joins a caller's operation and cannot raise on a closed inherited one. B5's Interfaces, code, rationale and Review Focus 7 say so. Re-run on `3c4d734981`: `test_no_checkout_is_held_while_the_skill_runs` fails on dev (`[['asyncio_0']] == [[]]`); with the change, it and `test_skill_mcp_integration.py` give `14 passed`.
> - **Ruling 4:** C4 Step 7's B5 residual is appended only in the recommendation branch, together with `--check-ac 2`.
> - **Ruling 5:** C2 Step 3 says 29 product files, and C4 Step 7 says `<P>` is above `<N>`.

**Goal:** On PostgreSQL:
- no ChaCha checkout is lost when its thread exits;
- none stays pinned by a long-lived thread after a WebSocket step, a persona call, a job or a background pass;
- `CharactersRAGDB.close_all_connections()` neither closes the shared content pool nor leaves a returned checkout open without its reset;
- every PostgreSQL ChaCha test app runs requests the way `main.py` does. Doing so also surfaces, and this plan fixes, a locking-read defect that only shows under the middleware.

**Architecture:** Option C (a fallback fix plus ownership at each entry point), in four pull requests, executed in order:
- **PR A** (`fix/chacha-pg-legacy-checkouts`): `ChaChaNotes_DB.py` and one line in `chacha/persona_state_store.py`. It fixes the legacy thread-local path itself, and the code that confuses it with an operation's state:
  - `close_all_connections()` never closes the shared content pool. Before closing a caller's pool, it closes this thread's checkout.
  - When a legacy checkout's thread exits, its return is queued without taking a lock, and the next ChaCha checkout or `close_connection()` in the process makes it. A collected handle is queued the same way, but only when the cyclic GC runs. That second path is a backstop, not a fix.
  - The workspace and persona locking reads see an operation's transaction.
- **PR B** (`fix/chacha-pg-websocket-operations`): WebSockets and the persona session own their checkouts:
  - one operation per `/audio/chat/stream` persistence step;
  - one per voice-assistant step;
  - one per persona session call, off the loop (B4) and on it (B5, Decision 2), with the skill tool call owning only its registry lookup;
  - a fix for the default-character insert race.
- **PR C** (`fix/chacha-pg-background-operations`): background work owns its checkouts:
  - a new `owns_chacha_checkouts` helper wraps each ChaCha-touching job owner and cancel action that runs its ChaCha work one call at a time;
  - the Chat_Macros run, whose branches write concurrently, is not wrapped; each of its ChaCha calls owns its own operation instead;
  - the notes-graph cleanup becomes operation-aware;
  - the chatbooks core Jobs worker, the chatbooks cleanup loop, the ingestion-sources worker, the Workflows chatbooks adapter and the chatbook notification each run in an operation;
  - ADR-NNN records the rule.
- **PR D** (`test/chacha-pg-test-app-middleware`, tests only): installs `ChaChaOperationMiddleware` in the 30 PostgreSQL ChaCha test apps that lack it, guards that, makes `pg_restricted_backend`'s teardown fail fast on a leaked session, and closes TASK-13861.

**Tech Stack:** Python 3.12.13, pytest 9.1.1, pytest-asyncio 1.4.0, pytest-timeout 2.4.0, psycopg 3.3.6, psycopg-pool 3.3.2, anyio 4.14.2, FastAPI 0.142.1, Starlette 1.6.0, PostgreSQL 18 (Docker container `tldw_postgres_test`), backlog-py.

**Spec:** `backlog/tasks/task-13861 - Legacy-thread-local-PostgreSQL-ChaCha-checkouts-leak-thread-exit-long-lived-threads-close-all-connections-before-reset.md` on origin/dev.
- Background: `Docs/superpowers/plans/2026-10-18-test-reliability-pg-import-and-admin-health.md` (TASK-13841, #3275).
- The September HTTP-ownership design note: `output/playwright/cycle5-repair-verification-2026-09-16/followup181-http-ownership-reviewed/author-DESIGN181-http-ownership.md` (commit `84a2c51934`).

**Where this was measured.** All runs used the Docker container `tldw_postgres_test` (postgres:18 on 5432) on an 18-core Mac, whose 1-minute load was 6-28 while other sessions ran.
- **Revision 5** re-ran the skill tests in `fu5` at origin/dev `3c4d734981` (no change to the touched files since `3ad31cba25`).
- **Revision 4** re-ran the tests the third review's rulings touched, and the mutation checks, in `fu4` (all four PRs) and `fu4_dev` (unchanged dev plus the new tests) at origin/dev `3ad31cba25`. `git diff --stat 2cae53929f 3ad31cba25` shows no change to any touched file.
- **Revision 3** ran in two scratch worktrees at origin/dev `2cae53929f`:
  - `fu3`: all four PRs applied.
  - `fu3_dev`: unchanged dev plus the new test files.
  - For the "against the plan as written" REDs, revision 2's PR C was applied to `fu3_dev` and then removed again.
- **Revision 2** ran at `e521605198` (worktrees `fu2`, `fu2_dev`).
- **v1's measurements** were taken on `a6719bf24f` and `2b925562e5`.

`git diff --stat` shows no change to any touched file between those commits and `2cae53929f`. Facts kept from earlier revisions say where they were measured. All the worktrees have been removed.

The prototype's tracked changes are kept at `<scratchpad>/fu4_prototype_r4.diff` (revision 4, `3ad31cba25`: 66 files, +535/-117) and `<scratchpad>/fu3_prototype_r3.diff` (revision 3), each including the docs from C4 numbered 078. Every new test file is embedded in its task below.

## Global Constraints

- TASK-13861 AC1: "/chat/stream persistence is exercised on PostgreSQL first, and every production caller that reaches the legacy thread-local path on PostgreSQL is listed, with evidence".
- TASK-13861 AC2: "No PostgreSQL checkout outlives its thread or pins a pool slot outside an operation: such callers run inside chacha_operation or return their checkout".
- TASK-13861 AC3: "After CharactersRAGDB.close_all_connections() on psycopg_pool, no checkout it returned is left open without its reset".
- TASK-13861 AC4: "Test apps over PostgreSQL ChaCha install ChaChaOperationMiddleware, or are shown not to leak".
- Fix defects test-first: each task's test fails, for the reason given, on a branch that has the earlier PRs but not this task. Then it passes with the fix.
- **Do not change the operation release contract.** Inside an operation, `close_connection()` retires the handle's state (`operation_scope.py:85-90`), and later use raises `ClosedChaChaOperationError`. `test_chacha_operation_scope.py` (`:35-57`, `:140-160`) pins this. Code that runs inside an operation must not close and then reuse a handle. C1 makes the one cleanup that did so operation-aware.
- **Never run ChaCha transactions concurrently inside one operation.**
  - All the threads of an operation share one connection, and `BackendManagedTransaction` keeps the transaction depth on that shared state without a lock: `__enter__` reads it (`ChaChaNotes_DB.py:731`), and `__exit__` writes back the value it read (`:785`).
  - Two overlapping `db.transaction()` blocks leave the depth wrong. The rest of the operation's writes then never commit, and the operation's close rolls them back (F13).
  - Code that fans out ChaCha work (`asyncio.gather` over `asyncio.to_thread`) gives each call its own `chacha_operation(independent=True)`. The Chat_Macros executor does this after C2.
- Never raise `pool_size` or `pool_timeout` to hide a leak. Tests may lower `pool_timeout`, so that a pinned pool fails in 5 s rather than 30 s.
- Postgres steps run with `TLDW_TEST_POSTGRES_REQUIRED=1` and without `TLDW_TEST_NO_DOCKER`. The fixture reuses a reachable server on 5432 and only starts Docker when there is none. Database-free steps may set `TLDW_TEST_NO_DOCKER=1`.
- **Operations and threads.**
  - `asyncio.to_thread` copies the context, so the callable joins the caller's operation.
  - `run_in_executor` does not. The default executor that `llm_call_root` installs (`Usage/llm_roots.py:32-37`) carries only the LLM call state. For `run_in_executor`, open the operation inside the callable.
  - `@chacha_operation(...)` works only on a plain function. On an `async def` it closes before the coroutine runs. So async code uses `with chacha_operation(...):` or PR C's `@owns_chacha_checkouts`.
- Backlog tasks change only through backlog-py: `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`. Use `--append-notes`, never `--notes`. Pass explicit ids. Stage with `git add -A backlog/tasks`.
- Every PR targets `dev`, from a branch cut from `origin/dev` after the previous PR of this series has merged. Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never use `--no-verify`.
- Mirrored docs: PR C edits `Docs/ADR/` and `Docs/Code_Documentation/Databases/ChaChaNotes_DB.md`, both mirrored under `Docs/Published/`. Run `bash Helper_Scripts/refresh_docs_published.sh`, stage everything, and only then run the Docs tests (C4 Step 4).
- None of the touched files is listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` (13 paths, 39 lines, rechecked on `e521605198`). Shifted line numbers need no baseline regeneration.
- **Ruff is non-blocking in CI (`ci.yml:269`). The rule here is no new findings.**
  - These touched files already fail ruff on dev (measured on `2cae53929f`, dev line numbers). Do not run `ruff --fix` on them, and leave their import blocks as they are:

    | File (`tldw_Server_API/…`) | Findings on dev |
    |---|---|
    | `app/api/v1/endpoints/persona.py` | I001 `:4`, SIM114 `:2494` |
    | `app/api/v1/endpoints/voice_assistant.py` | I001 `:9` |
    | `app/core/Chat/chat_helpers.py` | I001 `:5` |
    | `app/core/DB_Management/chacha/persona_state_store.py` | I001 `:3`, C420 `:2115` |
    | `app/core/Data_Tables/jobs_worker.py` | BLE001 `:882` |
    | `app/core/Evaluations/recipe_runs_jobs_worker.py` | I001 `:3` |
    | `app/core/Persona/visual_jobs_worker.py` | I001 `:1` |
    | `app/core/Workflows/adapters/integration/messaging.py` | I001 `:9`, F401 `:14`, I001 `:814` |
    | `app/services/core_jobs_worker.py` | B023 `:156` |
    | `app/services/ingestion_sources_worker.py` | I001 `:1`, F841 `:505` |
    | `app/services/research_workspace_output_jobs_worker.py` | I001 `:3` |
    | `app/services/workspace_file_inventory_jobs_worker.py` | UP035 `:8`, BLE001 `:153` |
    | `app/services/writing_annotation_review_jobs_worker.py` | I001 `:3` |
    | `tests/_plugins/postgres.py` | 10 findings |
  - All 29 product files the four PRs touch were compared, file by file, against dev, and none gains a finding. `Skills/context_integration.py` (B5) has no finding on dev.
  - Every new test file passes ruff (measured).
- Global defaults:
  - PR A changes `close_all_connections()` on PostgreSQL, and every legacy checkout.
  - PRs B and C wrap named functions and leave every other caller unchanged.
  - Each PR sweeps every test file that reaches what it changed (A4, B6, C4, D2).

### ADR assessment

**ADR required: yes.** PR C adds `Docs/ADR/NNN-chacha-postgres-checkouts-owned-by-entry-points.md`, where C4 Step 1 assigns NNN.
- **Why.** The plan sets a durable rule for worker ownership and persistence: every non-HTTP entry point that touches ChaCha owns its PostgreSQL checkouts. That means a WebSocket step, a persona call, a job owner, a cancel action or a background pass. `Docs/ADR/README.md` requires an ADR for "worker ownership" and "persistence" rules. The review also found that per-worker operations "would still merit an ADR".
- **No ADR governs ChaCha connection handling today.** `grep -rn "chacha_operation\|threading.local" Docs/ADR/` finds only an unrelated ACP inventory line.
- **Related ADRs.**
  - ADR-020 (shared PostgreSQL content backend, cached and validated at startup) supports PR A: a per-call handle must not close that shared pool.
  - ADR-021 (services lifecycle) is unaffected. Content-backend shutdown keeps closing the pool (`shutdown_post_runtime_cleanup.py:96-101`).
  - ADR-006 (Bandit on touched scope; it supersedes ADR-005) and ADR-059 (backlog-py) apply to every PR.
  - ADR-004 (the owner's Change summary on AI-authored PRs) applies to every PR.
- **Number.** It is written ADR-NNN in every step, body and note until C4 Step 1 assigns it. That step takes max(dev, open PRs) + 1 from the paged scan, whatever the scan prints, and replaces NNN everywhere it appears in PR C.
  - The sibling plan `plan_fu_rg_redis_error.md` is open as #3288 and holds ADR-078 (the re-review's paged scan). If it is still open or merged when C4 runs, this ADR gets 079 or later. The two plans execute serially.
  - #3288 also appends a row after the last row of `Docs/ADR/README.md`. Whichever PR merges second rebases and keeps both rows, in number order (C4 Step 2).
  - PR A's backlog note says "ADR-NNN, numbered in PR C". Backlog notes are append-only, so that note stays as written; C4's note records the number.
- PRs A, B and D need no ADR of their own:
  - A is a bug fix inside `ChaChaNotes_DB.py`;
  - B applies the September note's per-step WebSocket rule;
  - D changes only tests.

  ADR-NNN records the whole decision, including A's `close_all_connections()` rule and B's WebSocket steps.

## Findings (what the investigation measured)

### F1. `/audio/chat/stream` persistence on PostgreSQL (AC1, exercised first)

The existing persistence test (`tests/Audio/test_ws_audio_chat_stream.py:2922`) uses a dummy DB. B2's test runs the real handler `audio.websocket_audio_chat_stream` with `persist_history` over a real PostgreSQL `CharactersRAGDB`. It keeps the real `get_or_create_character_context`, `get_or_create_conversation` and `_persist_audio_chat_settings`.

1. **Which path it takes.** `ChaChaOperationMiddleware` only wraps `scope["type"] == "http"` (`operation_scope.py:223`). Persistence goes through `_shim_get_chacha_db_for_user_id` (`audio_streaming.py:2415`), outside the middleware. It reaches the database three ways, none with an operation:
   - `asyncio.to_thread` in `chat_helpers.py:182-262`;
   - `loop.run_in_executor(None, …)` for the settings upsert (`audio_streaming.py:2456`);
   - `loop.run_in_executor(None, _persist_sync)` for each turn (`audio_streaming.py:2528`).

   Every call takes the legacy thread-local checkout of the user's cached handle, on a default-executor thread.
2. **Concurrent sessions** (3 users x 3 sessions, v1 probe on `2b925562e5`):
   - With a 20-connection pool, 17 legacy checkouts on 9 live executor threads were still out after every session ended.
   - With a 10-connection pool the run ended in `PoolTimeout`.
   - Re-measured on `e521605198`, with and without PR A: B2's test raises `psycopg_pool.PoolTimeout: couldn't get a connection after 5.00 sec`.
3. **With an operation per persistence step** (B2): 0 legacy checkouts, and 0 outstanding after the sessions while the loop is still running.
4. **A pre-existing race this exposed.** Concurrent first sessions of a new user both try to create the default character. The loser's `add_character_card` raises `ConflictError`, and `_create_default` (`chat_helpers.py:254`) returns `(None, None)`. That session then stops persisting. In v1's 6 runs of 9 concurrent first sessions:
   - dev lost 6 of 54 sessions;
   - per-step operations alone lost 21 of 54;
   - with B1's read-back, none were lost.

   The review re-ran B2 without B1, and 5 of 5 runs failed.

### F2. Production callers that reach the legacy path on PostgreSQL (AC1)

**How the rows were found.**
- Search `app/` for `CharactersRAGDB(`, `get_chacha_db_for_user_id(`, `get_chacha_db_for_owner(` and `.close_all_connections()`.
- Check that against the September accessor audit (`author-nonhttp-accessor-audit.json`, 30 references in 20 files), then F11's Jobs audit.
- Callers that only run inside HTTP requests are covered by `main.py`'s middleware and are left out: `sharing.py`, `rag_unified.py`, `chat.py`, `slides.py`, `vn_play.py`, `flashcards.py`, `ingestion_sources.py` and `Sync/v2/factory.py`.

"Measured" means a test or probe ran that path against PostgreSQL; "code" means the row comes from reading the code. Every "measured" row has a test in this plan that fails on dev.

| Caller | Handle | Thread of the DB work | What it loses on dev | Evidence | Fix |
|---|---|---|---|---|---|
| `/audio/chat/stream` persistence, `audio_streaming.py:2402-2535` | cached | default executor | one checkout per (handle, executor thread), pinned while the thread lives and lost when it exits | measured (F1) | B2 |
| `/voice/assistant` WS, `voice_assistant.py:1102-1106` (init) and `:1409` (each command) | cached | event-loop thread | one checkout per handle on the loop thread, for the life of the process | measured: `['MainThread']` still held after the session | B3 |
| `/persona/stream` WS, `persona.py:8368` (per-session handle): 8 off-loop calls (`:8877`, `:8888`, `:8964`, `:10260`, `:10541`, `:10758`, `live_conversation.py:89`, `exemplar_runtime.py:113`) and 14 on-loop calls: 13 sync, plus the skill tool call's registry lookup (F12) | per session, shared backend | default executor, and the event-loop thread | off the loop: one checkout per executor thread the session used, returned only by the cyclic GC, so 4 sessions exhaust a 10-connection pool. On the loop: one checkout per live session, idle in a transaction, so 10 sessions exhaust it and the 11th blocks the event loop (F12) | measured | B4 (off-loop), B5 (on-loop, Decision 2). A wrapped on-loop call can still wait for a connection on the loop when other work holds the pool: TASK-<N> (F12) |
| Jobs owners on cached handles that close at job end on the loop thread: `workspace_file_inventory`, `writing_annotation_review`, `research_workspace_output`, `visual_identity`, `vn_asset` and `study_suggestions` workers, and `Sharing/shared_workspace_clone_jobs_worker.py` | cached | executor via `asyncio.to_thread` | the executor thread's checkout, pinned | measured for `workspace_file_inventory`: `['asyncio_0']` (C2 family a); the others from code | C2 |
| Chat_Macros cancel action (`Chat_Macros/jobs.py:281`, reached from the cancel check `:105`) | cached | executor; close on the loop | the same | measured: `['asyncio_0']` (C2 family b) | C2 (wrapped) |
| Chat_Macros run (`Chat_Macros/jobs.py:67`): repository calls from concurrent branches (`executor.py:322`, `:368-371`), the macro loader (`:87`), the executor build and the post-back on the loop thread (`jobs.py:81`, `:163`) | cached | executor, and the loop | one checkout per executor thread the run used (4 held after a 4-branch run) | measured: `['asyncio_0', …, 'asyncio_3']` (C2 family d) | C2 (each call owns its operation; the run is not wrapped, F13) |
| Worker-kept or per-job handles: Chatbooks `_handle_job` (`_DB_CACHE`, never closed), `Data_Tables/jobs_worker.py:716` (LRU, closed only on eviction), `Persona/visual_jobs_worker.py:606,665` (closed at job end on the loop), `Evaluations/recipe_runs_jobs_worker.py` (through the RAG pipeline) | worker cache, or per job | executor | the executor thread's checkout, pinned | measured for Chatbooks: `['asyncio_0']` (C2 family c); the others from code | C2 |
| notes-graph suggestions (`notes_graph_suggestions_worker.py:78`) | cached | executor; per-step `sync_cleanup` closes on the same thread | nothing today; fails inside an operation (F10) | measured (C1) | C1 (operation-aware cleanup) |
| study_pack worker | cached | already opens its own operation | nothing | `test_study_pack_worker_operation_lifecycle.py` | none |
| Per-call handles never closed: `core_jobs_worker.py:48` (chatbooks core Jobs worker), `chatbooks_cleanup_service.py:55`, `ingestion_sources_worker.py:186`, `Workflows/adapters/integration/messaging.py:671`, `Notifications/service.py:159` | per call, shared backend | loop or executor | the checkout, returned only by the cyclic GC (F4) | measured with GC disabled: `['MainThread']` for each (C3) | C3 |
| MCP `chats_module.py:262,424` (open), `:407,:527` (close); `rpg_module.py:572` (open), `:450,:618` (close) | per call, shared backend | one `asyncio.to_thread` callable opens and closes | nothing (the review listed them; each closes on the thread that used it) | code | A1 (they call `close_all_connections()`) |
| MCP modules `flashcards_module.py` (13 closes), `notes_module.py` (10), `quizzes_module.py` (14), `characters_module.py` (2), `slides_module.py` (2), `skills_module.py` (1), `persona_visuals_module.py` (1); `Persona/memory_integration.py:87`; `Notes/Notes_Library.py:137`; `ChaCha_Notes_DB_Deps.close_all_chacha_db_instances` (`:915`) | per call or cached, shared backend, then `close_all_connections()` | varies | **closes the shared content pool** (F5) | measured with the real `FlashcardsModule` | A1 |
| `notes_graph_suggestions_maintenance.py:38` | cached | executor; `maintain_owner` closes on the same thread | nothing | code (`:88-108`) | none |
| `Research/chat_handoff.py:39` | per call | caller's thread; closes it | nothing | code | none |
| `Chat/command_router.py:662` | cached | caller's | covered by its callers: HTTP chat, the voice step (B3), Chat_Macros (C2) | code | none |

### F3. A legacy checkout is lost when its thread exits (AC2)

Measured with a 3-connection pool. One read on a fresh thread, then the thread exits:
- **On dev** the checkout is never returned, for a plain `threading.Thread`, an AnyIO worker (`anyio.to_thread.run_sync`) and the asyncio default executor. Re-measured on `e521605198`: A2's thread-exit tests fail with `assert 1 == 0`.
- **With A2's guard** the checkout comes back, with no garbage collection. CPython clears a dead thread's thread-locals at once, so the guard's `__del__` runs and queues the return, and the next `close_connection()` makes it.

Why this happens:
- psycopg's `Connection.__del__` returns early for a pooled connection (`psycopg/_connection_base.py:150`).
- AnyIO worker threads exit after 10 s idle.
- `asyncio.run` shuts down its default executor.

### F4. A dropped per-call handle's checkout comes back only when the cyclic GC runs (AC2)

- **Every `CharactersRAGDB` is in a reference cycle**, because its stores hold `self` (`ChaChaNotes_DB.py:8219-8237`). Dropping the last reference does not free it; only the cyclic collector does.
- **In a server-sized heap that can take arbitrarily long.** The review's probe 4: with the app imported (1.65 M tracked objects), a dropped handle's checkout stayed out through 3 M rounds of short-lived allocations, and only an explicit `gc.collect()` returned it.
- **So PR A's guard is a backstop for dropped handles, not a fix.**
  - A2's dropped-handle test calls `gc.collect()` and says so.
  - The fix for the known sites is C3: each runs inside an operation, whose close returns its checkouts at once.
  - C3's test disables garbage collection. It fails on dev, and on dev with PRs A and B (`['MainThread']` for all five sites).
- **Cached handles are dropped without a close too.** The 20-entry LRU (`ChaCha_Notes_DB_Deps.py:360`) and the health check (`:646-649`) evict a handle without closing it. After PRs B and C no known caller leaves a legacy checkout on a cached handle, so an evicted handle's checkouts are left to the backstop (Review Focus 4).

### F5. `close_all_connections()` on PostgreSQL (AC3, plus a broader defect)

1. **It closes the shared content pool.**
   - In PostgreSQL content mode, every handle built without a backend resolves the one cached shared backend (`ChaChaNotes_DB.py:8301-8305`, `content_backend.py:308-321`).
   - `close_all_connections()` (`ChaChaNotes_DB.py:8755-8795`) calls `pool.close_all()` on whatever backend it has.
   - Measured with the real `FlashcardsModule.execute_tool("flashcards.decks.list", …)`: after one MCP call the shared pool was closed.
   - A cached handle then failed with `CharactersRAGDBError: Failed to acquire database connection: Connection pool is closed`, and so did new handles.
   - `get_content_backend()` keeps returning the same closed backend, so this lasts until restart.
2. **It closes a caller's pool before the returned checkout is reset.**
   - `close_connection()` returns the checkout through `putconn`, which queues psycopg_pool's `ReturnConnection`.
   - `close_all()` then closes the pool, and `MaintenanceTask.run` discards the queued task (`psycopg_pool/pool.py:946-950`).
   - Made deterministic by starting each queued return 0.5 s late: the checkout stayed open and `INTRANS`.
   - This is the product half of TASK-13841's teardown hang.
3. **The fix (A1).**
   - On the shared backend, return this thread's checkout and leave the pool open. Content-backend shutdown owns that pool.
   - On a caller's backend, close this thread's checkout (`discard_connection`) before closing the pool.

### F6. Test apps without the middleware (AC4)

- The task's grep finds 29 files.
- One more file has a bare app the grep cannot see. `DB_Management/test_character_shared_owner_contract.py:47` builds a bare `FastAPI()`, and the same file installs the middleware in a second app at `:388-389`, which hides the first.
- So the true count is **30 files and 39 apps** (re-measured on `e521605198`).

### F7. `pg_restricted_backend` teardown

- A test that leaves a restricted-role session in a transaction holding a lock makes the teardown's `DROP OWNED BY` (`tests/_plugins/postgres.py:409`) wait for it.
- In a `pytester` subprocess the run hung until the 90 s subprocess timeout (v1 and review).
- With `SET LOCAL lock_timeout = '10s'`, a report of the role's sessions, and a terminate-and-drop (D2), the run ends in about 12 s with `1 passed, 1 error` and the report.

### F8. The middleware in the test apps uncovers a production defect (M1)

- **The defect.** `get_workspace(..., for_update=True)` (`ChaChaNotes_DB.py:29042-29046`) and `get_persona_profile(..., for_update=True)` (`chacha/persona_state_store.py:1812-1816`) check `getattr(self._local, "tx_depth", 0)`, the legacy thread-local depth.
  - Inside an operation the depth lives on the operation's `ConnectionState`, so the check fails even inside `with db.transaction()`.
  - Every production HTTP request runs inside an operation (`main.py:2796`). So on PostgreSQL each request that takes one of these locking reads raises `InputError` and returns 400.
- **Requests affected:**
  - every non-Character workspace chat creation (`POST /api/v1/chats/` -> `create_workspace_persona_conversation`, `assistant_defaults.py:268-307`);
  - `POST /api/v1/chats/workspace-startup` (`chat_startup.py:156` -> `require_workspace_for_chat_creation(conn=...)`);
  - persona admission (`conversation_admission.py:64-65`);
  - chat creation in a workspace under a chat quota (`test_conversation_quota_count_backends.py[postgres-ordinary-workspace]` with the middleware: `{"detail":"Workspace locking reads require a transaction connection."}`, 400).
- **Why no test caught it.** The test apps had no middleware, so they used the legacy depth.
- `grep -rn 'getattr(self._local, "tx_depth"' tldw_Server_API/app` finds only these two sites.

### F9. Why not one operation per WebSocket connection

- A WebSocket session can last hours.
- An operation keeps its checkout from first use until it closes, and reads leave it `INTRANS`. So a connection-long operation would hold an idle-in-transaction session, with its locks, for the life of the socket.
- The September note says the same: "Per command/turn managed operation. A whole WebSocket lifetime can last hours and is not a read-transaction cleanup boundary."

### F10. A cleanup that closes and then reuses a handle fails inside an operation (Blocker 1)

- **The cleanup.** `handle_notes_graph_suggestions_job` passes `sync_cleanup=lambda: _close_database(db)` to `SuggestionWorker` (`notes_graph_suggestions_worker.py:87-91`).
  - `SuggestionWorker.handle` runs `get_run`, `start_run`, retrieve, prepare and the freshness check as separate `asyncio.to_thread` steps.
  - Each step calls `sync_cleanup()` on the same thread (`suggestion_service.py:225-243`).
- **Why it fails inside an operation.** Each close retires the operation's state, so the next step raises `ClosedChaChaOperationError`.
- **Measured with C1's test** (the real job, through `WorkerSDK.run`, on PostgreSQL):
  - plain, on dev: `completed`, 0 outstanding. This case passes on dev and pins today's behaviour.
  - inside `chacha_operation(independent=True)`, on dev and on dev with PRs A and B: the job fails with `The ChaCha operation has finished`.
  - with C1: `completed`, 0 outstanding.
- **The cost of holding an operation across the job** (C1's test, with the generate step probed):
  - the plain job holds **0** checkouts while the provider call runs;
  - the job inside an operation holds **1** (`['asyncio_0']`), idle in a transaction, for the whole provider call.

  That is why C2 wraps the job owners that already held a checkout until job end, and leaves notes-graph's per-step cleanup as it is.

### F11. Jobs audit: who reaches ChaCha, and who closes and then reuses a handle (Blocker 1, Important 3)

**Which modules reach ChaCha.** These `WorkerSDK` modules reach ChaCha, directly or through one level of imports (a scan of `app/` for `WorkerSDK` users that import a ChaCha accessor):
- `Chatbooks/services/jobs_worker.py`, `Data_Tables/jobs_worker.py` (also through RAG), `Evaluations/recipe_runs_jobs_worker.py` (through RAG), `Persona/visual_jobs_worker.py`;
- `chat_macros_jobs_worker.py` (through `Chat_Macros/jobs.py`), `notes_graph_suggestions_worker.py`, `research_workspace_output_jobs_worker.py`, `study_pack_jobs_worker.py`, `study_suggestions_jobs_worker.py`, `visual_identity_jobs_worker.py`, `vn_asset_jobs_worker.py`, `workspace_file_inventory_jobs_worker.py`, `writing_annotation_review_jobs_worker.py`;
- plus `Sharing/shared_workspace_clone_jobs_worker.py`, which gets its handles through a runtime callback.

**Each owner, checked for two things that break inside one operation:**
- **Close, then reuse:** closing a handle and then using it again in the same job (F10).
- **Concurrent transactions:** two ChaCha transactions open at once inside one owner (F13).

The second column comes from a scan of each owner and its callees for `asyncio.gather`, `TaskGroup`, `create_task`, `run_in_executor` and `run_coroutine_threadsafe`.

| Worker | Close sites | Reuse after close? | Concurrent ChaCha transactions inside one owner? | Shape |
|---|---|---|---|---|
| notes-graph suggestions | `sync_cleanup` after each step (`suggestion_service.py:239-240`), `_close_database` at `:135` | **yes** (F10) | no: the steps run one after another | not wrapped; C1 makes the cleanup operation-aware |
| study_suggestions, visual_identity, vn_asset, writing_annotation_review, workspace_file_inventory, research_workspace_output | one `_close_worker_database` in the handler's `finally` | no | no: sequential `await`s, and no fan-out in the owner or its callees | wrapped (C2) |
| Chat_Macros run | `_close_worker_database` in `handle_chat_macro_job`'s `finally` (`:102`) | no | **yes:** `_run_branches` gathers every step (`executor.py:322`), up to `max_concurrency` = 3 at once, and every repository call is a `db.transaction()` (`repository.py`, 17 sites) | **not wrapped**; each repository call, the macro loader, the executor build, the post-back and the failure-path write own their own operation (C2) |
| Chat_Macros cancel action | `_close_worker_database` in `_mark_macro_run_cancelled` (`:313`), on its own handle | no | no: three sequential `asyncio.to_thread` calls | wrapped (C2) |
| shared_workspace_clone | `_close_chacha_thread_connection` at the end of `_run_clone` (`:574-578`) and of `_run_target_resource_action._run` (`:771`), on the thread that used the handle | no | no. The clone thread's cancellation check (`_should_cancel`, `:538-551`) runs `resolve_clone` on the loop through `run_coroutine_threadsafe`, inside the same operation, and reads the owner's workspace with `get_workspace` (a plain read, no managed transaction). The clone thread waits for that result, so the two take turns; a check that times out after `authorization_timeout_seconds` (5 s) can overlap, and it is still only a read | wrapped (C2) |
| Data_Tables | `_close_chacha_db` only on LRU eviction or a lost cache race (`:726`, `:728`), for a different handle | no | no managed transactions. Its RAG retrieval gathers the notes and chat-history retrievers over one handle (`unified_pipeline.py:1699`, `:1736`), but they read through `execute_query`, which opens no `db.transaction()` | wrapped (C2) |
| recipe runs | none (RAG's own handles) | no | as Data_Tables: RAG reads only | wrapped (C2) |
| Persona visual (generation, portability) | `db.close_connection()` at job end (`:615`, `:675`) | no | no | wrapped (C2) |
| Chatbooks `_handle_job` | none (worker-cached handle) | no | no | wrapped (C2) |
| study_pack | its own operation already | no | no | unchanged |
| The five non-SDK runners (C3) | none | no | no: sequential | wrapped per pass, job or call (C3) |

**What stays outside an operation:**
- No wrapped owner starts a detached task that touches ChaCha. Every `asyncio.create_task` in these modules is a stop watcher or a worker start, outside the job owner.
- One wrapped owner can itself be reached from a request's detached task: `run_chatbooks_adapter`, through `engine.continue_run`, which `workflows.py:3497`, `:4228`, `:4350` and `:4566` start with `asyncio.create_task` inside a request. It is wrapped with `independent=True` (C3, Review Focus 3). The Notifications method joins its caller on purpose: `watchlists.py:5821` and `:7725` call it inside a request.
- `Data_Tables`' `run_in_executor(None, _call_llm)` (`:1106`) does not touch ChaCha.
- `WorkerSDK`'s `on_completed`, `on_failed` and `progress_cb` callbacks stay outside any operation. The notes-graph and shared-workspace callbacks close on the thread that used the handle (`_publish_completed`, `:108-135`; clone audit helpers), so they return their checkouts today.

**Runners that do not use `WorkerSDK`:** the chatbooks core Jobs worker (`core_jobs_worker.py`), the chatbooks cleanup loop (`chatbooks_cleanup_service.py`), the ingestion-sources worker (`ingestion_sources_worker.py`), the Workflows chatbooks adapter (`messaging.py:606`), and the chatbook notification (`Notifications/service.py:252`). C3 covers all five.

### F12. The persona live WebSocket (Important 2)

- **What is held.** `/persona/stream` opens a handle per session (`persona.py:8368`). It uses the handle off the loop through `asyncio.to_thread` at the six `persona.py` sites in F2, and through `live_conversation.py:89` and `exemplar_runtime.py:113`. Each executor thread the session used keeps one legacy checkout.
- **What happens at session end.** `close_all_connections()` (`:11450`) runs on the loop thread, so after PR A it returns only that thread's checkout. The executor threads' checkouts wait for the cyclic GC.
- **Measured** with B4's behavioural test (4 concurrent sessions, 12 off-loop reads each, 10-connection pool), on dev with PR A: `assert ['asyncio_0', …, 'asyncio_2', …] == []`, with 10 checkouts held, the whole pool. The review's probe 7 (PRs A and v1's B): 3 sessions held 9 checkouts, and 4 and 5 sessions hit `PoolTimeout`.
- **The per-call wrap (B4).** Each of the 8 off-loop calls runs `chacha_operation(...)` inside the `asyncio.to_thread` callable. With it, the same test ends with 0 checkouts out.
- **What the per-call wrap costs, in round trips.** Measured by counting psycopg `execute`, `commit` and `rollback` calls for 20 reads (`get_character_card_by_name`) on one handle:
  - legacy, on a warm thread-local checkout: 2 statements per read in the foreground;
  - per-call operation: 7 statements per read in the foreground. The 5 extra are the pool checkout's client-scope settings and their commit, and the transaction around the read. The pool's worker then resets the returned connection with 4 more, off the caller's path.

  So each wrapped call adds about 5 round trips to the database. On the local Docker server that measured 0.44 → 1.2-1.4 ms median (planning) and 0.61 → 2.69 ms at load ~30 (the re-review). Over a remote database it scales with the round-trip time. A persona turn makes an LLM call that takes hundreds of milliseconds or more.
- **The on-loop calls (B5).** The session also calls the handle directly on the event-loop thread, at 14 call sites an AST scan finds (dev line numbers):
  - `_load_persona_policy_rules_for_session`: `:9019`, `:9950`, `:10611`, `:10828`, `:11069`, `:11291`, `:11366`;
  - `persona_scope_db.upsert_persona_live_voice_session_summary`: `:9091`;
  - `record_persona_live_voice_event`: `:9128`;
  - `_load_persona_scope_metadata_for_session`: `:9417`;
  - `_persist_persona_session_preferences`: `:10069`, `:11114`;
  - `_load_persona_state_hints_for_runtime`: `:10138`;
  - `await handle_skill_tool_call(..., db=persona_scope_db)`, async: `:9561`.

  Two more calls receive the handle but own their work already: `resolve_persona_exemplar_runtime_context` and `live_conversation.complete_persona_turn` (B4).

  **The skill call.** `handle_skill_tool_call`'s only ChaCha work is its registry lookup: `SkillsService(...)` and `get_skill`. `SkillExecutor` gets no handle, and for a `context: fork` skill it makes an LLM call (`skill_executor.py:330`, `:436`). Two options:
  - wrap the whole call in an operation (revision 3): this holds the lookup's checkout through the forked skill's LLM call;
  - give the lookup its own independent operation inside `handle_skill_tool_call` (`context_integration.py:144-145`), a two-line change: no checkout spans the LLM call.

  B5 takes the narrower one, with `chacha_operation(independent=True)`. It never joins a caller's operation, so no caller's checkout can span the skill run, and a closed operation inherited by a detached task cannot make it raise. The cost is one extra checkout when a caller already holds an operation; `persona.py` is the only production caller, and it holds none.

  A PostgreSQL test pins it: on dev the lookup's checkout is still out while the skill runs (`[['asyncio_0']]`), and with B5 nothing is. So 13 sync sites are wrapped in `persona.py`, and the skill call owns its lookup.

  One on-loop use remains in the skill call, and it predates this plan. `SkillsService.__init__` runs 4 `CREATE ... IF NOT EXISTS` statements through `backend.execute` on the event-loop thread at every skill call (`ChaChaNotes_DB.py:27626-27654`). Each takes and returns its own pool connection, outside any operation. The re-review measured 4 `MainThread` checkouts per skill call, on dev and with B5. B5 leaves it alone; TASK-<N>'s third AC covers it.

  Measured after B4 on `2cae53929f`: 11 sessions, each one on-loop policy read as `:9019` does, on a 10-connection pool with `pool_timeout` 3 s:
  - 10 checkouts were held on `MainThread`, all `INTRANS`: idle in a transaction, with their locks, for the life of each session. This is what F9 rejects for a connection-long operation.
  - The 11th session's read raised `PoolTimeout`. Because it runs on the event-loop thread, the whole event loop stalled for 3.04 s, the full `pool_timeout`. The default is 30 s (`config.txt:400`), and production's pool (20 connections) is shared with HTTP requests and every other content database.
  - With each read wrapped in `chacha_operation(independent=True)`: 0 held. With no other load on the pool, nothing timed out, and the longest event-loop gap was 0.05 s.

  `close_all_connections()` returns each session's loop-thread checkout when that session ends (PR A). So without B5 the hold is bounded by the number of live sessions, but that bound is reached at pool size.

  **What B5 does not fix: a wrapped on-loop call still checks out on the event-loop thread.**
  - Every call runs `getconn` and about 5 round trips synchronously on the loop. Before B5 a session did that only on its first call.
  - When other work holds the whole pool, each such call blocks the whole server's event loop for up to `pool_timeout` (30 s by default), then raises `PoolTimeout`. Production's 20 connections are shared with HTTP requests and every other content database.
  - Measured by the re-review, pool of 10 with 9 held elsewhere and one warm session: the B5-wrapped call raised `PoolTimeout` after 2.0 s, with a 2.01 s event-loop gap. A warm legacy call took 0.0 s.
  - B5 removes persona's own contribution to exhaustion: no session keeps a slot between its calls. It does not stop an on-loop call from waiting when someone else exhausted the pool.
  - **The follow-up (filed in B6 Step 2).** 10 of the 13 sync sites sit directly in `async def`s (`:9417`, `:9950`, `:10069`, `:10138`, `:10611`, `:10828`, `:11069`, `:11114`, `:11291`, `:11366`), and can move off the loop as B4's do. The other three (`:9019`, `:9091`, `:9128`) sit in sync helpers. This plan does not move them.

  See Decision 2.

### F13. Concurrent ChaCha transactions inside one operation lose writes (Blocker 1 of the re-review)

- **The mechanism.** An operation gives every thread that joins it the same `ConnectionState`, and so one physical connection. `BackendManagedTransaction.__enter__` reads `tx_depth` from that state (`ChaChaNotes_DB.py:731`), and `__exit__` writes back the value it read (`:785`), with no lock around the pair. With two overlapping `db.transaction()` blocks on two threads:
  1. the first enters at depth 0 and is "managed": it will commit;
  2. the second enters at depth 1 and is not;
  3. the first commits and writes depth 0 back;
  4. the second leaves last and writes depth 1 back, without committing.

  From then on every transaction in the operation is unmanaged, so nothing more commits. The operation's close returns the checkout, and the pool's reset rolls the rest back.
- **Where revision 2 created it.** C2 wrapped `handle_chat_macro_job`, so the run's concurrent branches (F11) shared one state.
  - The re-review replayed `_run_branches` with the real `ChatMacroRepository` 50 times: dev's legacy path lost 0 runs; inside one operation, 7 of 50 runs lost writes (14 of 50 with 8 branches).
  - C2's new test forces the overlap deterministically: two branch writes, the second held open across the first's commit. On revision 2's PR C the run ended `('running', [])`: every branch row and the final status were lost. On dev (no operation) the data is intact, but `['asyncio_0', …, 'asyncio_3']` stay pinned. With option (a) both hold.
- **Option (a), chosen for Chat_Macros (C2).** The run is not wrapped. Each ChaCha call owns its own independent operation, so each branch gets its own connection and transaction, and returns it at once. This is the note's "borrowed repository keeps its external owner".
- **HTTP routes, audited for the same hazard.**
  - **How.** A scan of `app/` for `asyncio.gather`, `TaskGroup` and `create_task` in modules that touch a ChaCha handle or retriever found 20 fan-out sites. Each was read for whether two of its branches can hold `db.transaction()` blocks on one ChaCha handle at once inside a request's operation.
  - **Read-only fan-outs.** The RAG pipeline's are `unified_pipeline.py:1699`, `:1736` (slides sources), `:4819` (subqueries), `:5306` (follow-ups), `:9456`, `:9488` (batch), and `database_retrievers.py:2613`, `:5014`.
    - They run the notes and chat-history retrievers concurrently through `asyncio.to_thread` on one handle, so inside a request they share one connection.
    - Their ChaCha calls are `search_notes` and `search_messages_by_content`, which read through `execute_query`. With its default `read_only=False`, `execute_query` opens no transaction scope and never touches `tx_depth` (`ChaChaNotes_DB.py:8892-8904`).
    - So they cannot produce the depth corruption above. They do read inside whatever transaction another branch has open.
  - **Fan-outs that do not touch one ChaCha handle twice.**
    - `admin_ops.py:1478` probes ChaCha once among other databases.
    - `admin_data_subject_requests_service.py:302` counts through SQLite file paths.
    - `VN_Assets/service.py:467` uses the generated-files repository.
    - `Research_Workspace/capabilities.py:144` runs health probes.
    - `skills_module.py:419` lists a catalog.
    - `chat_service.py:6422`, `:6648` write audits.
    - `persona.py:11433` is session cleanup outside HTTP.
    - `Buddy/turns.py:303`, `:312` is shutdown.
  - **Result: no HTTP route was found that overlaps managed transactions on one ChaCha handle,** so nothing on dev loses writes this way today. That rests on reading the code, not on a measurement.
  - **The follow-up.** C4 files option (b), serializing `BackendManagedTransaction` per `ConnectionState` (or failing loudly on overlap), so that a future fan-out cannot lose writes silently. The read-only fan-outs above are listed there for re-checking.

## Fix shape

| Option | Measured cost | What breaks | What pins today's behaviour |
|---|---|---|---|
| **A. Operations at each non-HTTP caller.** | B: 7 files of WebSocket and persona steps. C: 1 helper, 16 decorated job owners, cancel actions and runners, per-call operations in the Chat_Macros run, 1 operation-aware cleanup, 2 runner loops. | Cannot reach callers nobody has found (Review Focus 4). A detached task that touches ChaCha after its step or job ends gets `ClosedChaChaOperationError` instead of the legacy path (none found, F11). An owner that fans out ChaCha transactions must not share one operation (F13). | `test_chacha_operation_scope.py` (12), `test_chacha_postgres_http_operation_lifecycle.py` (20), `test_study_pack_worker_operation_lifecycle.py` (4), C1's plain-job test |
| **B1. Return the legacy checkout after every command.** | Rejected without a prototype. | `get_connection()` hands the raw connection to callers (93 call sites). A per-command return would give the pool a connection that is still in use. `execute_query` defaults to `commit=False`, so a pending write would be rolled back. | None |
| **B2. Return the legacy checkout when its thread-local slot is dropped.** | About 60 lines in `ChaChaNotes_DB.py`. | Nothing measured. It is deterministic only for thread exit. Dropped handles wait for the cyclic GC (F4), and it does not help long-lived threads. | A2's `test_an_explicitly_returned_checkout_is_returned_once` |
| **B3. Raise outside an operation on PostgreSQL.** | Rejected without a prototype. | Every single-thread script or test that uses a handle directly: 107 test files call `close_all_connections()` over PostgreSQL ChaCha, mostly outside any operation. | The same 107 files |
| **C. B2 as a fallback, plus A at every known non-HTTP entry point (chosen).** | PR A: 2 files, +83/-3. PR B: 7 files, +116/-92, most of it the `audio_streaming.py` re-indent. PR C: 20 files, +103/-11. | As A, and Review Focus 1-4 and 8. | All of the above, plus the sweeps |

**Jobs: per worker at each job owner (recommended, Decision 1), not centrally in `WorkerSDK`.**

The September note's recommendation for Jobs reads:
> Wrap the actual job/pass owner in the new operation context, preserving injected externally supplied DB ownership. Existing cleanup can delegate to that owner. Do not globally wrap every Jobs SDK handler as an implicit DB transaction.

It also says: "Avoid hiding overlapping independently owned transaction scopes on one physical connection; shared joined work versus independent child operation must be explicit." For Chat_Macros it asks for "One managed operation per acquired run/cancellation action".

| | Per worker, at each job owner (C2-C3) | Central, in `WorkerSDK.run` (v1's B4) |
|---|---|---|
| notes-graph | Untouched. It returns its checkout after each step: 0 held during the provider call (F10). | Its per-step close retires the operation, and every job fails (Blocker 1). With C1 it works, but holds 1 checkout through the provider call (F10). |
| Owners that already held a checkout until job end | The checkout is held through the job, LLM calls included, and returns at job end instead of staying pinned (C2: `['asyncio_0']` → none) | The same |
| Owners that fan out ChaCha transactions (the Chat_Macros run) | Not wrapped; each call owns its operation, so concurrent branches keep separate connections (C2: all rows saved, nothing pinned) | Every branch shares one connection and one transaction depth: writes are lost silently (F13: 7 of 50 runs in the re-review's replay; `('running', [])` in C2's test). It would need option (b) first |
| Non-SDK runners (5) | Covered, each in its own way (C3) | Not covered; they need C3 anyway |
| Unknown or future workers | Not covered until wrapped; the thread-exit fix and GC backstop apply | Covered if they use `WorkerSDK`, but exposed to F13 if they fan out |
| Blast radius | 16 named functions, 2 loops and the Chat_Macros run's calls; each family has a PostgreSQL test | Every Jobs handler in the process; `WorkerSDK` imports `DB_Management.chacha` |
| What it matches | The note's proposal | The note says not to do this |

Per-worker comes out better on the measured costs. It is weaker only for workers nobody has found, which is why the audit (F11), the family tests (C2), the ADR's rule for new entry points, and Review Focus 4 exist.

**`close_all_connections()`: close the checkout (chosen), not reset it, and not wait for the pool.**
- **Rejected: reset before `putconn`.** A discarded connection would still stay open until freed, and every return, including HTTP operations', would gain a synchronous round trip.
- **Rejected: wait for the pool.** psycopg_pool has no public way to wait for queued `ReturnConnection` tasks.
- **Chosen.** `discard_connection` closes the checkout synchronously and keeps psycopg_pool's bookkeeping. The pool is closed right after, so reusing the connection would be pointless anyway.
- **The shared pool is not closed at all.** Content-backend shutdown owns it (`shutdown_post_runtime_cleanup.py:96-101`). `Collections_DB.close()` (`:828-842`) follows the same rule.
- **Caller-supplied pools are still closed.** The 107 test files rely on that for teardown.

**The finalizer: queue, do not return.**
- A cyclic collection can run `__del__` on any thread, at any bytecode boundary, including one that holds psycopg_pool's task-queue mutex. That mutex is a plain `threading.Lock`, and `putconn` needs it. The review's probe 2 deadlocked that way.
- So `__del__` only calls `queue.SimpleQueue.put`, which is documented as safe to call from `__del__`.
- `_open_new_connection` and `close_connection` drain the queue, outside any finalizer.

**Test apps (AC4): two lines in each app, not a shared fixture.**
- The 39 apps are built in 30 files by 39 local helpers, each with its own routers and overrides.
- A shared factory would replace the same 39 lines, add an import to each file, and hide the production wiring behind a helper.
- The codemod adds `app.add_middleware(ChaChaOperationMiddleware)` after each `app = FastAPI()`, as `main.py:2796` does. A guard test counts builders against installs in each file.

## Decisions for the owner

1. **Where Jobs own their PostgreSQL ChaCha checkouts.**
   - **Recommended: per worker, at each job owner and cancel action that runs its ChaCha work one call at a time (PR C), recorded in ADR-NNN.**
     - This is the September note's proposal.
     - It leaves notes-graph's per-step cleanup alone: 0 checkouts held through the provider call.
     - It leaves the Chat_Macros run unwrapped and gives each of its ChaCha calls its own operation, because the run's branches write concurrently (F13).
     - It covers the five runners that do not use `WorkerSDK`.
     - Each family has a PostgreSQL test.
   - **Cost:** a worker that nobody wraps keeps the legacy path, with the thread-exit fix and the GC backstop. ADR-NNN makes wrapping part of adding a worker that touches ChaCha, and forbids sharing one operation across concurrent ChaCha transactions.
   - **Alternative: v1's central `WorkerSDK` wrap,** around every handler and cancel check, plus C1 and C3.
     - It also covers unknown `WorkerSDK` workers.
     - **It shares one connection across everything a job does concurrently.** The Chat_Macros run's branches then lose writes silently: the re-review's replay lost 7 of 50 runs, and C2's test ends `('running', [])` (F13). This holds for any per-job or per-owner operation around a job that fans out, so the central wrap would need option (b), transactions serialized per `ConnectionState`, first.
     - It holds each touched handle's checkout, idle in a transaction, through every LLM call in the job. Notes-graph goes from 0 to 1 held during its provider call (F10).
     - It still needs C3 for the non-SDK runners.
     - It goes against the note's "Do not globally wrap every Jobs SDK handler".
2. **The persona live WebSocket.**
   - **Recommended: Alternative 1, per-call operations for every persona session call, off the loop (B4) and on it (B5), in PR B.**
     - B4 wraps the 8 off-loop calls. That fixes the measured exhaustion through executor threads: 4 sessions held a whole 10-connection pool (F12).
     - **B5 wraps the 13 sync on-loop call sites** as `chacha_operation(independent=True)(fn)(...)`, pinned by an AST guard over direct calls.
       - Without B5, each live session holds one checkout on the event-loop thread, idle in a transaction, until it ends. Measured: 10 sessions exhausted a 10-connection pool, and the 11th blocked the whole event loop for `pool_timeout` (3.04 s with 3 s; the default is 30 s) (F12).
       - With B5, persona sessions no longer cause that exhaustion: none keeps a slot between calls.
     - **The skill tool call owns only its registry lookup** (`context_integration.py`), so a forked skill's LLM call holds no checkout.
     - **What B5 does not fix.** Each wrapped on-loop call still checks out synchronously on the event-loop thread. When other work holds the whole pool, such a call stalls the whole server's loop for up to `pool_timeout` before it raises. Measured with 9 of 10 connections held elsewhere: `PoolTimeout` after 2.0 s, with a 2.01 s loop gap. This plan does not move the calls off the loop. B6 files a follow-up, TASK-<N>, to run the 10 calls in async handlers through `asyncio.to_thread`.
     - **Cost:** about 5 extra round trips per wrapped call, plus a 4-statement reset in the pool's worker (F12). That is small next to the LLM call in each turn.
     - **AC2** is checked in C4. The note names the residual and TASK-<N>: AC2 asks that no checkout pin a pool slot, and none does; the residual is a wait for a slot, not a held one.
   - **Alternative 2: only the off-loop calls (B4), with a follow-up task for the on-loop calls.** Skip B5; B6 files the follow-up, which then covers B5's work and the move off the loop. AC2 then stays unchecked, because a known slot-pinning hold remains: one idle-in-transaction checkout per live session, with the event-loop stall at pool size.
   - **Alternative 3: leave persona to a follow-up entirely.** Skip B4 and B5. AC2 stays unchecked, because the executor-thread checkouts come back only through the cyclic GC.

The owner's `## Change summary` (ADR-004) is needed on each of the four PRs. It is an input, not a decision. TASK-13861 is not covered by a recorded waiver.

## Review Focus

1. **A checkout escaping its thread.** A2 returns a thread's checkout when that thread's slot is dropped. A `BackendConnectionWrapper` passed to another thread and still used after its owning thread exits would then run on a returned connection. Nothing in `app/` does this, and no test covers it.
2. **The finalizer's queue (Important 5).**
   - `__del__` must stay lock-free. A2's `test_a_collected_guard_never_calls_the_pool_itself` fails if it calls the pool; the v1 guard failed it with "the finalizer called into the pool".
   - **A queued return waits for the next ChaCha call.** A thread exit queues its checkout, and the next ChaCha checkout or `close_connection()` on any handle in the process makes the return. In a server that is the next ChaCha access.
     - A caller already blocked in psycopg_pool's `getconn` gets no benefit. The re-review's probe (pool of 1): the waiter timed out after 3 s, although the holder's thread exited 0.5 s into the wait. v1's guard returned the checkout from the finalizer at once, and could deadlock.
     - This is rare, because threads exit when idle.
     - A drainer thread (blocking on `_DROPPED_CHECKOUTS.get()`) would close the gap without a lock in `__del__`. It is deliberately not added: one more thread to own and shut down, for a rare case.
   - `_return_dropped_checkouts` logs and continues on a failed return (for example, a pool that is already closed), so one bad entry cannot strand the rest.
   - A checkout dropped while `ACTIVE` is closed by psycopg_pool rather than reused (`psycopg_pool/pool.py:812-815`). No test covers the `ACTIVE` case.
3. **Detached tasks inside a step or job.** `asyncio.create_task` copies the context, including an open operation. A task that touches ChaCha after its step or job ends gets `ClosedChaChaOperationError` where it used to fall back.
   - `get_chacha_db_for_user_id`'s default-character task runs in its own independent operation (`ChaCha_Notes_DB_Deps.py:531-534`).
   - No other ChaCha-touching detached task was found in the WebSocket steps or the wrapped job owners (F11).
   - **The other direction: a wrapped owner reached from a request's detached task.**
     - `@owns_chacha_checkouts` joins the caller's operation, and joining an operation that has already closed raises.
     - `run_chatbooks_adapter` is reached that way: through `engine.continue_run`, which `workflows.py:3497`, `:4228`, `:4350` and `:4566` start with `asyncio.create_task` inside a request.
     - On dev that step already fails: the adapter returns `{'error': 'chatbooks_error'}`, because any ChaCha use resolves the closed operation.
     - C3 wraps it with `@owns_chacha_checkouts(independent=True)`, which always opens a new operation. Its test `[workflows-chatbooks-after-its-request]` fails on dev and on revision 2 (`ClosedChaChaOperationError`), and passes with C3.
     - No other wrapped owner is reachable from a detached task (F11). A new one that is uses `independent=True` (ADR-NNN).
4. **What the GC backstop still covers.** These checkouts come back only when the cyclic GC runs:
   - a cached handle evicted by the LRU or the health check, if an unknown non-HTTP caller used it outside an operation;
   - per-call handles in code nobody has found.

   The known sites are fixed in B and C. A new background entry point that opens a handle outside an operation reintroduces the leak; ADR-NNN's rule and the family tests are the defence.
5. **`close_all_connections()` on an explicit backend that is really the shared one.** `DB_Manager.create_chacha_database(backend=…)` passes the shared backend explicitly, so the handle does not know it is shared. There are 0 callers in `app/`. ADR-NNN records this.
6. **Non-tenant `client_id`s on PostgreSQL (seen in code, not measured).** The MCP modules and the persona WebSocket pass descriptive `client_id`s, but `_tenant_client_id` (`ChaCha_Notes_DB_Deps.py:791-817`) treats the `client_id` as the RLS tenant key. This is out of scope here.
7. **Joining versus independent operations.**
   - `live_conversation.py` and `exemplar_runtime.py` use `chacha_operation()`, which joins an HTTP request's operation when one is open and owns the call otherwise.
   - `persona.py`, the WebSocket steps and the skill lookup in `context_integration.py` use `independent=True`. The skill lookup never joins, so no caller's checkout spans a forked skill's LLM call.
   - `owns_chacha_checkouts` joins a caller's operation by default, so a job owner called from an HTTP route shares the request's checkouts. `owns_chacha_checkouts(independent=True)` does not.
8. **Concurrent transactions inside one operation (F13).**
   - The guard against F13 is a rule, not code: ADR-NNN, Global Constraints, and Chat_Macros' per-call operations.
   - A future fan-out of `db.transaction()` blocks on one handle inside one operation would lose writes silently, so C4 files option (b) as a follow-up.
   - The RAG fan-outs read concurrently inside a request's transaction. They see another branch's uncommitted writes on the same connection, which is harmless today because none of them writes.

---

## Setup (once, before PR A)

```bash
git fetch origin dev --quiet
export PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
export S=<scratchpad>                      # scratch logs and helper scripts, never committed
mkdir -p "$S/runs"
docker ps --format '{{.Names}} {{.Ports}}' | grep tldw_postgres_test
```
Expected: `tldw_postgres_test 0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp`. If it prints nothing, the first Postgres test starts it.

Create the sweep runner `$S/sweep.sh`. Do not use `pytest -n`: xdist workers collected different test ids (measured: `Different tests were collected between gw1 and gw0`). The runner shards by file instead.

```bash
#!/bin/zsh
# usage: sweep.sh <list-file> <log-prefix> [shards]   (run from the repository root)
k=${3:-4}
n=$(( ($(wc -l < "$1") + k - 1) / k ))
split -l $n "$1" "$2.shard."
for shard in "$2".shard.a?; do
  TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q -o timeout=600 $(cat "$shard") > "$shard.log" 2>&1 &
done
wait
for shard in "$2".shard.a?; do
  echo "== $shard: $(grep -E '^=+ .*(passed|failed|error).* =+$' $shard.log | tail -1)"
  grep -E "^(FAILED|ERROR) |\+\+\+ Timeout" $shard.log | cut -c1-220
done
```
Then run `chmod +x "$S/sweep.sh"`.

Create the ADR-number scan `$S/adr_next.sh`. It pages every open PR with more than 100 changed files (`files` stops at 100):

```bash
#!/bin/zsh
# Highest ADR number on origin/dev and in every open PR (run from the repository root).
{ git ls-tree --name-only origin/dev Docs/ADR/
  gh pr list --state open --limit 500 --json number,changedFiles,files \
    --jq '.[] | if .changedFiles > 100 then "PR#\(.number)" else .files[].path end' \
    | while IFS= read -r line; do
        if [[ $line == PR#* ]]; then
          gh api --paginate "repos/rmusser01/tldw_server/pulls/${line#PR#}/files" --jq '.[].filename'
        else
          printf '%s\n' "$line"
        fi
      done
} | sed -nE 's|^Docs/ADR/([0-9]{3})-.*|\1|p' | sort -n | tail -1
```
Then run `chmod +x "$S/adr_next.sh"`. It prints whatever is highest at execution. Measured on `2cae53929f` it prints `078`, because #3288 holds ADR-078; on `e521605198`, before #3288 opened, it printed `077`. The `[0-9]{3}` keeps it working past ADR-099.

Create the backlog-id scan `$S/task_next.sh`. AGENTS.md's recipe misses files in open PRs with more than 100 changed files, so this one pages them too:

```bash
#!/bin/zsh
# Highest backlog task id on origin/dev and in every open PR (run from the repository root).
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
Then run `chmod +x "$S/task_next.sh"`. Use the printed value + 1 as each new task's id.

---

# PR A — `fix/chacha-pg-legacy-checkouts`: the legacy path keeps its checkouts accounted for

PR A stands alone. With only its two product files on dev `e521605198`, its three new test files plus `test_chacha_connection_typing.py` give `14 passed`; on unchanged dev they give `9 failed, 5 passed`.

```bash
git worktree add <scratchpad>/pr-a -b fix/chacha-pg-legacy-checkouts origin/dev
cd <scratchpad>/pr-a
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13861 -s "In Progress"
```

### Task A1: `close_all_connections()` keeps the shared pool open and closes its own checkout before closing a caller's pool (AC3)

**Files:**
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_close_all_connections.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:8755-8761` (the head of `close_all_connections`), and add `_discard_legacy_checkout` directly above it.

**Interfaces:**
- Consumes:
  - `PostgreSQLConnectionPool.discard_connection(conn)` (`postgresql_backend.py:379-403`), which closes the connection and then hands it to psycopg_pool for bookkeeping;
  - `CharactersRAGDB._uses_shared_content_backend`, which `_resolve_backend` (`:8283-8318`) sets to True only for the cached content backend.
- Produces: `CharactersRAGDB._discard_legacy_checkout() -> None`. On PostgreSQL:
  - `close_all_connections()` never closes a shared content pool;
  - before closing a caller's pool, it closes this thread's legacy checkout instead of queuing its return.

  SQLite is unchanged. Task A2 adds one line to `_discard_legacy_checkout`.

- [ ] **Step 1: Write the failing tests**

Create `tldw_Server_API/tests/DB_Management/test_chacha_postgres_close_all_connections.py`:

```python
"""CharactersRAGDB.close_all_connections() on PostgreSQL (TASK-13861 AC3)."""

import time
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.flashcards_module import FlashcardsModule

pytestmark = pytest.mark.integration


async def test_an_mcp_tool_call_leaves_the_shared_content_pool_open(pg_database_config, tmp_path, monkeypatch):
    """Every handle built without a backend shares one cached content pool in PostgreSQL mode."""
    shared = DatabaseBackendFactory.create_backend(pg_database_config)
    monkeypatch.setattr(chacha_module, "get_content_backend", lambda _config: shared)
    cached = CharactersRAGDB(tmp_path / "cached.db", client_id="1")
    try:
        assert cached.backend is shared
        module = FlashcardsModule(ModuleConfig(name="flashcards"))
        context = SimpleNamespace(db_paths={"chacha": str(tmp_path / "mcp.db")}, metadata={})

        # The module opens a handle per call and closes it with close_all_connections().
        listed = await module.execute_tool("flashcards.decks.list", {"limit": 5}, context=context)

        assert listed["total"] == 0
        assert cached.get_character_card_by_name("nobody") is None
    finally:
        cached.close_connection()
        shared.get_pool().close_all()


def test_close_all_connections_closes_its_checkout_before_closing_the_callers_pool(
    pg_database_config, tmp_path, monkeypatch
):
    """A checkout handed to the pool's reset worker must not outlive the pool unreset and open."""
    from psycopg_pool import pool as psycopg_pool_module

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    if not backend.get_pool()._use_psycopg_pool:
        pytest.skip("the fallback pool rolls a returned checkout back synchronously")
    db = CharactersRAGDB(tmp_path / "own.db", client_id="1", backend=backend)
    db.execute_query("SELECT count(*) AS n FROM character_cards")
    raw = db._local.conn
    assert raw.info.transaction_status.name == "INTRANS"
    original_run = psycopg_pool_module.ReturnConnection.run

    def late_run(task):
        time.sleep(0.5)  # a busy pool worker: the pool closes before the queued reset starts
        original_run(task)

    monkeypatch.setattr(psycopg_pool_module.ReturnConnection, "run", late_run)

    db.close_all_connections()

    assert raw.closed, f"the returned checkout was left open ({raw.info.transaction_status.name})"
```

- [ ] **Step 2: Run them to verify they fail for the right reasons**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_close_all_connections.py 2>&1 | grep -E "^E   [a-zA-Z]|^FAILED|passed|failed"
```
Expected (measured on `e521605198`):
```
E   tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB.CharactersRAGDBError: Failed to acquire database connection: Connection pool is closed
E   AssertionError: the returned checkout was left open (INTRANS)
FAILED …::test_an_mcp_tool_call_leaves_the_shared_content_pool_open
FAILED …::test_close_all_connections_closes_its_checkout_before_closing_the_callers_pool
==== 2 failed, … ====
```

- [ ] **Step 3: Implement**

In `ChaChaNotes_DB.py`, add this method directly above `def close_all_connections` (dev `:8755`):

```python
    def _discard_legacy_checkout(self) -> None:
        """Close this thread's legacy PostgreSQL checkout instead of queuing its return."""
        conn = getattr(self._local, "conn", None) if self._connection_state() is self._local else None
        if conn is None:
            self.close_connection()
            return
        backend = self._get_pinned_backend() or self._backend
        self._local.conn = None
        self._local.backend_ref = None
        backend.get_pool().discard_connection(conn)
```

When there is no legacy checkout to close (an operation's checkout, or none), it still calls `close_connection()`. `test_chacha_connection_typing.py` (`:30-84`) pins that call order: `close_connection`, then `pool_close_all`, then the thread-local reset.

Replace the head of `close_all_connections` (dev `:8755-8761`: the docstring and the `self.close_connection()` line) with:

```python
    def close_all_connections(self) -> None:
        """
        Force-close all backend connections managed by this instance.

        Primarily used in tests/shutdown to ensure no background SQLite threads remain.
        On PostgreSQL the shared content pool is never closed here: every user's
        handle draws from it, and content-backend shutdown owns it.
        """
        if self.backend_type == BackendType.POSTGRESQL:
            if self._uses_shared_content_backend:
                self.close_connection()
                return
            # The caller's pool is closed below. putconn hands a checkout to a pool
            # worker for its reset, and a pool closed first discards that work, leaving
            # the connection open and unreset, so close this thread's checkout now.
            self._discard_legacy_checkout()
        else:
            self.close_connection()
```
Leave the rest of the method (dev `:8763-8795`) unchanged.

- [ ] **Step 4: Run the tests to verify they pass, with the unit pins on the same method**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_close_all_connections.py \
  tldw_Server_API/tests/DB_Management/test_chacha_connection_typing.py 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
```
Expected: `6 passed`, the 2 new tests plus the 4 in `test_chacha_connection_typing.py`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_close_all_connections.py
git commit -m "$(cat <<'EOF'
fix(chacha): close_all_connections keeps the shared PostgreSQL pool open (TASK-13861)

In PostgreSQL content mode every handle built without a backend shares
one cached content pool, and close_all_connections() closed it. One MCP
tool call (the modules open and close a handle per call) therefore made
every later ChaCha access fail with "Connection pool is closed" until
restart. The shared pool is now left to content-backend shutdown.

On a caller's own pool it returned this thread's checkout through putconn
and then closed the pool at once, so psycopg_pool discarded the queued
reset and left the connection open, possibly in a transaction. It now
closes that checkout first.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task A2: a legacy checkout's return is queued when its thread exits, and made outside any finalizer (AC2, part 1)

What A2 fixes, and what it only backs up:
- **Thread exit (fixed, at the next ChaCha call).** CPython clears a dead thread's thread-locals at once, so the guard queues the checkout at once. The return itself happens at the next ChaCha checkout or `close_connection()` in the process. A caller already blocked in psycopg_pool's `getconn` gets no benefit, which is rare because threads exit when idle (Review Focus 2).
- **A dropped handle (backstop only).** It is cyclic garbage, so its guard runs only when the cyclic GC does (F4). PRs B and C make the known callers own their checkouts instead.

**Files:**
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_legacy_checkouts.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`, at these points (dev line numbers; edit by content, because Task A1 shifted the lines below `:8755`):
  - the import block (`import queue` after `import math`, `:41`);
  - add the guard after `_CHACHA_NONCRITICAL_EXCEPTIONS` (which ends at `:467`);
  - `_open_new_connection`, `:8460`;
  - `_get_connection_for_state`, `:8646` and `:8652-8655`;
  - `close_connection`: a new first statement after its docstring (which ends at `:8681`), and `:8729`;
  - `_discard_legacy_checkout` (from Task A1).

**Interfaces:**
- Consumes: `_discard_legacy_checkout()` from Task A1.
- Produces:
  - `_LegacyCheckout(conn, backend)` with `.disarm()`;
  - `_disarm_legacy_checkout(state) -> None`;
  - `_return_dropped_checkouts() -> None`;
  - the module-level `_DROPPED_CHECKOUTS: queue.SimpleQueue`.

  On PostgreSQL the legacy thread-local state (`CharactersRAGDB._local`) gains a `checkout_guard` attribute next to `conn`. Operation states (`ConnectionState`) never get one.

- [ ] **Step 1: Write the failing tests**

Create `tldw_Server_API/tests/DB_Management/test_chacha_postgres_legacy_checkouts.py`:

```python
"""Legacy thread-local PostgreSQL checkouts go back to the pool (TASK-13861 AC2).

The thread-exit cases need no garbage collection: CPython clears a dead thread's
thread-locals at once. A dropped handle is different: its stores hold it, so only the
cyclic collector frees it, and that case is a backstop that the test drives with
gc.collect(). Returns are queued by the finalizer and made by the next checkout or
close_connection() on any handle, so each test ends with one of those.
"""

import asyncio
import gc
import threading
from collections import Counter

import anyio
import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture
def ledger_backend(pg_database_config, monkeypatch):
    """A three-connection backend whose pool records every checkout and return."""
    pg_database_config.pool_size = 3
    pg_database_config.pool_timeout = 3.0
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
    backend.taken, backend.returned = taken, returned
    try:
        yield backend
    finally:
        pool.close_all()


def _outstanding(backend):
    return sum((backend.taken - backend.returned).values())


def _run_in_plain_thread(work):
    worker = threading.Thread(target=work)
    worker.start()
    worker.join()


def _run_on_anyio_worker(work):
    async def main():
        await anyio.to_thread.run_sync(work)

    anyio.run(main)


def _run_on_default_executor(work):
    async def main():
        await asyncio.get_running_loop().run_in_executor(None, work)

    asyncio.run(main())


@pytest.mark.parametrize(
    "run", [_run_in_plain_thread, _run_on_anyio_worker, _run_on_default_executor], ids=["thread", "anyio", "executor"]
)
def test_a_legacy_checkout_returns_to_the_pool_when_its_thread_exits(ledger_backend, tmp_path, run):
    db = CharactersRAGDB(tmp_path / "exit.db", client_id="1", backend=ledger_backend)
    db.close_connection()
    before = _outstanding(ledger_backend)

    run(lambda: db.get_character_card_by_name("nobody"))
    db.close_connection()  # no checkout on this thread: it only makes the queued return

    assert _outstanding(ledger_backend) == before


def test_a_dropped_handle_returns_its_legacy_checkout_when_collected(ledger_backend, tmp_path, monkeypatch):
    """Backstop for per-call handles that are never closed; production must close or own them."""
    monkeypatch.setattr(chacha_module, "get_content_backend", lambda _config: ledger_backend)
    for call in range(5):  # more calls than the pool has connections
        handle = CharactersRAGDB(tmp_path / f"call{call}.db", client_id="1")
        handle.execute_query("SELECT count(*) AS n FROM character_cards")
        handle = None
        gc.collect()  # the handle is cyclic garbage; nothing but the collector frees it
    CharactersRAGDB(tmp_path / "last.db", client_id="1").close_connection()

    assert _outstanding(ledger_backend) == 0


def test_a_collected_guard_never_calls_the_pool_itself(ledger_backend, tmp_path):
    """A collection can run the finalizer on a thread that holds the pool's own lock.

    The finalizer must only queue the return; calling into the pool there can deadlock.
    """
    db = CharactersRAGDB(tmp_path / "queued.db", client_id="1", backend=ledger_backend)
    db.get_character_card_by_name("nobody")  # this thread's legacy checkout, now armed
    db = None
    returns_before = sum(ledger_backend.returned.values())

    gc.collect()

    assert sum(ledger_backend.returned.values()) == returns_before, "the finalizer called into the pool"
    CharactersRAGDB(tmp_path / "drain.db", client_id="1", backend=ledger_backend).close_connection()
    assert _outstanding(ledger_backend) == 0


def test_an_explicitly_returned_checkout_is_returned_once(ledger_backend, tmp_path):
    """Every explicit return disarms the guard, so a later thread exit returns nothing twice."""
    db = CharactersRAGDB(tmp_path / "once.db", client_id="1", backend=ledger_backend)
    db.close_connection()

    def work():
        db.get_character_card_by_name("nobody")
        db._local.conn.close()  # the closed-connection branch releases and replaces it
        db.get_character_card_by_name("nobody")
        db.close_connection()

    _run_in_plain_thread(work)
    db.close_connection()

    assert _outstanding(ledger_backend) == 0
    # The pool hands the same connection out again, so compare per checkout, not per object.
    assert not ledger_backend.returned - ledger_backend.taken, "a checkout was returned twice"
```

What each test pins:
- **The three thread-exit cases** call no `gc.collect()`. They are the deterministic half.
- **`test_a_dropped_handle_returns_its_legacy_checkout_when_collected`** drives the GC backstop explicitly. Production does not rely on it for the known callers (C3).
- **`test_a_collected_guard_never_calls_the_pool_itself`** pins Important 5: the finalizer only queues.
- **`test_an_explicitly_returned_checkout_is_returned_once`** pins the double-return hazard, through both explicit return paths: the closed-connection branch and `close_connection()`.

- [ ] **Step 2: Run them to verify they fail for the right reasons**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_legacy_checkouts.py 2>&1 | grep -E "^E   (assert|psycopg)|^FAILED|passed|failed"
```
Expected (measured on `e521605198`):
```
E   assert 1 == 0                      (x4: three thread kinds, and the collected-guard test's final check)
E   psycopg_pool.PoolTimeout: couldn't get a connection after 3.00 sec
FAILED …::test_a_legacy_checkout_returns_to_the_pool_when_its_thread_exits[thread]
FAILED …::test_a_legacy_checkout_returns_to_the_pool_when_its_thread_exits[anyio]
FAILED …::test_a_legacy_checkout_returns_to_the_pool_when_its_thread_exits[executor]
FAILED …::test_a_dropped_handle_returns_its_legacy_checkout_when_collected
FAILED …::test_a_collected_guard_never_calls_the_pool_itself
==== 5 failed, 1 passed, … ====
```
`test_an_explicitly_returned_checkout_is_returned_once` passes on dev. It pins the hazard that Step 3 introduces.

v1's guard returned the checkout from `__del__`. Against that guard, `test_a_collected_guard_never_calls_the_pool_itself` fails with `AssertionError: the finalizer called into the pool` (measured).

- [ ] **Step 3: Implement**

Add the import after `import math  # noqa: E402` (`:41`):

```python
import queue  # noqa: E402
```

Directly after the `_CHACHA_NONCRITICAL_EXCEPTIONS` tuple (dev `:444-467`) and before `class BackendCursorAdapter`, add:

```python
_DROPPED_CHECKOUTS: queue.SimpleQueue[tuple[Any, Any]] = queue.SimpleQueue()


class _LegacyCheckout:
    """Queue a legacy PostgreSQL checkout for return when its thread-local slot is dropped.

    Outside a ChaCha operation each thread keeps its checkout in the handle's
    ``threading.local``. The slot is dropped when the thread exits, which CPython
    does at once, or when the handle is collected, which needs the cyclic garbage
    collector because the handle's stores hold it. psycopg's ``Connection.__del__``
    skips pooled connections, so without this the checkout and its pool slot are lost.
    """

    __slots__ = ("backend", "conn")

    def __init__(self, conn: Any, backend: Any) -> None:
        self.conn = conn
        self.backend = backend

    def disarm(self) -> None:
        """The checkout was returned or closed explicitly."""
        self.conn = None

    def __del__(self) -> None:
        if self.conn is not None:
            # Lock-free: a collection can run this on any thread, even one holding the
            # pool's own lock, so the return itself waits for _return_dropped_checkouts().
            _DROPPED_CHECKOUTS.put((self.backend, self.conn))


def _disarm_legacy_checkout(state: Any) -> None:
    guard = getattr(state, "checkout_guard", None)
    if guard is not None:
        guard.disarm()
        state.checkout_guard = None


def _return_dropped_checkouts() -> None:
    """Return checkouts queued by _LegacyCheckout, outside any finalizer."""
    while True:
        try:
            backend, conn = _DROPPED_CHECKOUTS.get_nowait()
        except queue.Empty:
            return
        try:
            backend.get_pool().return_connection(conn)
        except Exception as exc:  # noqa: BLE001 - keep draining; one bad return must not strand the rest
            logger.debug("Returning a dropped legacy checkout failed ({})", type(exc).__name__)
```

Make `_return_dropped_checkouts()` the first statement of `_open_new_connection` (dev `:8460`):

```python
    def _open_new_connection(self, backend: DatabaseBackend | None = None) -> RawBackendConnection:
        _return_dropped_checkouts()
        active_backend = backend or self.backend
```

In `_get_connection_for_state`, in the closed-connection branch (dev `:8643-8650`), disarm right after the release:

```python
            self._release_connection(conn, backend=backend)
            _disarm_legacy_checkout(state)
            conn = None
```

In the same method's new-checkout branch (dev `:8652-8655`), arm the guard for legacy state only:

```python
        if conn is None:
            conn = self._open_new_connection(backend)
            state.conn = conn
            state.backend_ref = backend
            if state is self._local:
                state.checkout_guard = _LegacyCheckout(conn, backend)
```

In `close_connection`, make the drain the first statement after the docstring (which ends at dev `:8681`):

```python
        _return_dropped_checkouts()
        state = self._connection_state()
```

In the same method's non-SQLite branch (dev `:8728-8729`), disarm before returning:

```python
            else:
                _disarm_legacy_checkout(self._local)
                backend.get_pool().return_connection(conn)
```

In `_discard_legacy_checkout` (Task A1), disarm before discarding:

```python
        self._local.conn = None
        self._local.backend_ref = None
        _disarm_legacy_checkout(self._local)
        backend.get_pool().discard_connection(conn)
```

Why each part is there:
- **The queue.** A cyclic collection can run `__del__` on a thread that holds psycopg_pool's task-queue mutex, a plain `threading.Lock` that `putconn` needs. The review's probe 2 deadlocked that way. `SimpleQueue.put` is reentrant and documented as safe in `__del__`.
- **The drains** run at the next checkout or close on any handle, never inside a finalizer.
- **The disarms.** Every path that hands the checkout back explicitly disarms first, so a later thread exit or collection never queues it a second time. A second `putconn` could publish one connection twice.
- **`close_all_connections()` on a caller's pool** replaces `self._local` (`:8795`). Dropping the old local frees the other threads' slots by reference count, so their guards queue those checkouts at once. The next drain hands them to the closed pool, which closes them. On dev they stayed open for good.

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command. Expected: `6 passed` (measured on `e521605198` with PR A alone).

- [ ] **Step 5: Run Task A1's tests and the operation-scope pins**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_close_all_connections.py \
  tldw_Server_API/tests/DB_Management/test_chacha_connection_typing.py \
  tldw_Server_API/tests/DB_Management/test_chacha_operation_scope.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_http_operation_lifecycle.py \
  tldw_Server_API/tests/DB_Management/test_study_pack_worker_operation_lifecycle.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_legacy_checkouts.py 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
```
Expected: `109 passed` and no `FAILED` or `ERROR` lines (measured on `e521605198` with PR A alone, 448 s).

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_legacy_checkouts.py
git commit -m "$(cat <<'EOF'
fix(chacha): return a legacy PostgreSQL checkout when its thread exits (TASK-13861)

Outside a ChaCha operation a handle keeps one pooled checkout per thread
in a threading.local. psycopg's Connection.__del__ skips pooled
connections, so when the thread exited (an AnyIO worker idle for 10 s,
an executor shut down) the checkout was never returned and its pool
slot was lost.

A guard stored next to the checkout now queues it when the thread-local
slot is dropped, and the next checkout or close_connection() returns it.
Thread exit triggers that at once. A dropped handle triggers it only
when the cyclic GC collects the handle, so that path is a backstop; the
known callers that drop handles are fixed separately. The guard's
finalizer only puts onto a queue.SimpleQueue, because a collection can
run it on a thread that holds the pool's own lock. Every explicit return
disarms the guard first, so nothing is returned twice.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task A3: workspace and persona locking reads see an operation's transaction (found by AC4)

**Files:**
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_locking_reads_in_operation.py`
- Modify:
  - `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:29044` (`get_workspace`; dev line number, edit by content)
  - `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py:1814` (`get_persona_profile`)

**Interfaces:**
- Consumes: `CharactersRAGDB._connection_state()` (`:8337-8343`). It returns the operation's `ConnectionState` when one is open on PostgreSQL, and the legacy `threading.local` otherwise. Both carry `tx_depth`.
- Produces: `get_workspace(..., conn=…, for_update=True)` and `get_persona_profile(..., conn=…, for_update=True)` accept a transaction opened inside an operation. Their signatures are unchanged. PR D's quota test depends on this.

- [ ] **Step 1: Write the failing tests**

Create `tldw_Server_API/tests/DB_Management/test_chacha_postgres_locking_reads_in_operation.py`:

```python
"""Locking reads inside a ChaCha operation see the operation's transaction (TASK-13861)."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_db(pg_database_config, tmp_path):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "lock.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()


def test_a_workspace_locking_read_inside_an_operation_sees_its_transaction(pg_db):
    pg_db.upsert_workspace("ws-lock", "Locked workspace")

    with chacha_operation(independent=True):  # every production HTTP request runs inside one
        with pg_db.transaction() as conn:
            workspace = pg_db.get_workspace("ws-lock", conn=conn, for_update=True)

    assert workspace["id"] == "ws-lock"


def test_a_persona_locking_read_inside_an_operation_sees_its_transaction(pg_db):
    persona_id = pg_db.create_persona_profile({"user_id": "1", "name": "Locked persona"})

    with chacha_operation(independent=True):
        with pg_db.transaction() as conn:
            persona = pg_db.get_persona_profile(persona_id, user_id="1", conn=conn, for_update=True)

    assert persona["id"] == persona_id
```

- [ ] **Step 2: Run them to verify they fail**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_locking_reads_in_operation.py 2>&1 | grep -E "^E   [A-Za-z]|^FAILED|passed|failed"
```
Expected (measured on `e521605198`):
```
E   …ChaChaNotes_DB.InputError: Workspace locking reads require a transaction connection.
E   …ChaChaNotes_DB.InputError: Persona locking reads require a transaction connection.
==== 2 failed, … ====
```

- [ ] **Step 3: Implement**

In `get_workspace` (`ChaChaNotes_DB.py:29044`) and in `get_persona_profile` (`persona_state_store.py:1814`), replace the legacy depth with the current state's depth. The line becomes, in both files:

```python
            or (self.backend_type == BackendType.POSTGRESQL and not getattr(self._connection_state(), "tx_depth", 0))
```
Afterwards `grep -rn 'getattr(self._local, "tx_depth"' tldw_Server_API/app` prints nothing.

- [ ] **Step 4: Run them to verify they pass, with the quota tests**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_locking_reads_in_operation.py \
  tldw_Server_API/tests/DB_Management/test_conversation_quota_count_backends.py 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
```
Expected: `18 passed`. The quota file also passes before PR D, because its app has no middleware yet and uses the legacy depth.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_locking_reads_in_operation.py
git commit -m "$(cat <<'EOF'
fix(chacha): locking reads see a ChaCha operation's transaction (TASK-13861)

get_workspace(for_update=True) and get_persona_profile(for_update=True)
checked the legacy thread-local transaction depth. Inside an operation,
which every production HTTP request runs in, the depth lives on the
operation's state, so on PostgreSQL both raised "locking reads require a
transaction connection" even inside db.transaction(). Every non-Character
workspace chat creation, POST /chats/workspace-startup and persona
admission returned 400 on PostgreSQL, as did chat creation in a workspace
under a chat quota. Test apps without the operation middleware hid it.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task A4: Sweep, review and open PR A

**Files:**
- Create: `Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md`, a copy of this plan. It is not mirrored under `Docs/Published`.
- Backlog: `backlog/tasks/task-13861 - …md`.

**Interfaces:**
- Consumes: Tasks A1, A2 and A3.
- Produces: PR A into `dev`.

- [ ] **Step 1: Sweep every test file that reaches `close_all_connections()` over PostgreSQL ChaCha (the global-default rule)**

```bash
git grep -l 'close_all_connections' -- tldw_Server_API/tests | xargs grep -li postgres \
  | xargs grep -l 'CharactersRAGDB\|get_chacha_db' > "$S/closeall_pg_tests.txt"
wc -l < "$S/closeall_pg_tests.txt"
$S/sweep.sh "$S/closeall_pg_tests.txt" "$S/runs/a_sweep" | tee "$S/runs/a_sweep.summary"
```
Expected: `108` (measured on `2cae53929f` with PR A's tests in place, and by the re-review).
- Dev has 107, and `test_chacha_postgres_close_all_connections.py`, committed in A1, also matches.
- `test_chacha_postgres_locking_reads_in_operation.py` does not match. The word "postgres" appears only in its file name, so `grep -li postgres` drops it. A3 Step 4 runs it.

There must be no `FAILED` or `ERROR` lines beyond the dev baseline. Each shard runs for 10-40 minutes depending on load.

What was measured:
- **The review swept the 107 files with PR A (v1's guard)** in 4 shards and got `1 failed, 1223 passed, 1 skipped`, `807 passed`, `805 passed, 5 skipped` and `632 passed, 4 skipped`. The one failure is `ChaChaNotesDB/test_quizzes_basic.py::test_temporary_directory_retries_transient_cleanup_locks`. It patches the global `time.sleep` and recorded another thread's sleeps. It passes alone, on PR A and on dev.
- **Revision 3's sweep** ran all four PRs over the 325-file union of this list and B6's, C4's and D's lists (built as C4 Step 5 describes), in 8 shards on `2cae53929f`. The per-shard results are in C4 Step 5.
- **Dev itself:** in v1, dev's shard ab hung in `test_core_chat_owner_isolation.py`'s teardown (the TASK-13841 hang that A1 removes the mechanism for), and `test_workspace_persona_optout_v69.py::…[postgres]` failed once.

If a line fails, run the same test on unchanged `origin/dev` in a separate worktree before blaming the change, and record both results.

- [ ] **Step 2: Lint, Bandit and the ratchets**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_close_all_connections.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_legacy_checkouts.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_locking_reads_in_operation.py
uvx bandit -q -f json -o "$S/runs/bandit_a.json" tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_close_all_connections.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_legacy_checkouts.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_locking_reads_in_operation.py
$PY -c "import json,collections;d=json.load(open('$S/runs/bandit_a.json'));print(collections.Counter((r['filename'].rsplit('/',1)[-1],r['test_id']) for r in d['results']))"
$PY Helper_Scripts/ci/test_quality_triage.py --enforce 2>&1 | grep "test-triage\] OK"
git diff --check origin/dev...HEAD && echo "no trailing whitespace"
```
Expected:
- ruff reports exactly the 2 findings dev already has in `persona_state_store.py`: `:3:1: I001` and `:2115` `C420`. No other file has a finding.
- Bandit reports only `B101`: 5 in the close_all test, 6 in the legacy-checkout test and 2 in the locking-read test. It reports nothing in either product file, on dev or with the change (measured).
- `[test-triage] OK — no offenses beyond the baseline.`
- `no trailing whitespace`.

- [ ] **Step 3: Backlog notes, the new AC, and AC3**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13861 \
  --append-notes "PR A (fix/chacha-pg-legacy-checkouts). close_all_connections() on PostgreSQL: (1) it closed the shared content pool. Every handle built without a backend uses the one cached content backend, so one MCP tool call (flashcards.decks.list through FlashcardsModule.execute_tool) left every later ChaCha access failing with 'Connection pool is closed' until restart. Callers: MCP flashcards x13, notes x10, quizzes x14, characters x2, slides x2, skills x1, persona_visuals x1, chats x2, rpg x1; Persona/memory_integration x3; persona.py /stream x1; Notes_Library x2; close_all_chacha_db_instances. Fix: on the shared backend only this thread's checkout is returned; content-backend shutdown owns the pool (ADR-020). (2) On a caller's pool the checkout was returned through putconn and the pool closed at once, so psycopg_pool discarded the queued reset and left the connection open INTRANS. Fix: close this thread's checkout (discard_connection) before closing the pool." \
  --append-notes "PR A legacy-checkout guard: a thread-exit fix plus a GC-timed backstop. A legacy thread-local checkout was lost when its thread exited (plain thread, AnyIO worker, asyncio default executor). A guard stored next to the checkout queues it on a queue.SimpleQueue when the thread-local slot is dropped, and the next ChaCha checkout or close_connection() in the process returns it; a caller already blocked in getconn does not benefit. Thread exit is deterministic (CPython clears a dead thread's locals at once; the tests use no gc.collect()). A dropped handle is cyclic garbage, so its guard runs only when the cyclic GC does: in a server-sized heap that can take arbitrarily long (review probe 4). It is a backstop; PR C makes the known per-call sites own their checkouts. The finalizer never calls the pool (a collection can run on a thread holding psycopg_pool's task-queue lock); every explicit return disarms it first." \
  --append-notes "PR A locking reads: get_workspace(for_update=True) and get_persona_profile(for_update=True) read the legacy thread-local tx_depth, so inside an operation (every production HTTP request) they raised 'locking reads require a transaction connection' even inside db.transaction(). On PostgreSQL every non-Character workspace chat creation, POST /chats/workspace-startup and persona admission returned 400, as did chat creation in a workspace under a chat quota. Found when PR D's middleware went into test_conversation_quota_count_backends.py's app. Fix: read self._connection_state().tx_depth." \
  --append-notes "PR A RED/GREEN on e521605198: test_chacha_postgres_close_all_connections.py dev 2 failed, fix 2 passed. test_chacha_postgres_legacy_checkouts.py dev 5 failed 1 passed, fix 6 passed (v1's lock-taking guard fails the finalizer test). test_chacha_postgres_locking_reads_in_operation.py dev 2 failed, fix 2 passed. Operation-scope pins with the guard: 109 passed. Sweep of the 108 files that call close_all_connections over PostgreSQL ChaCha: [result] (dev baseline: [result]). Bandit: B101 only in the new tests; none in the product files. The ADR for the whole series (ADR-NNN, numbered in PR C) records the close_all_connections rule." \
  --ac "CharactersRAGDB.close_all_connections() never closes the shared PostgreSQL content pool" \
  --check-ac 3 --check-ac 5
```
Fill the bracketed results from Step 1. Expected: the task prints with AC #3 and the new AC #5 checked, and status `In Progress`.

- [ ] **Step 4: Commit the plan and the task**

```bash
cp <scratchpad>/plan_fu_chacha_pg_checkouts.md Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md
git add Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md
git add -A backlog/tasks
git commit -m "$(cat <<'EOF'
docs(plans): plan for PostgreSQL ChaCha checkouts outside HTTP; PR A notes (TASK-13861)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
git diff --check origin/dev...HEAD && echo "no trailing whitespace"
```
The plan copy keeps "ADR-NNN". PR C's Task C4 replaces it there too.

- [ ] **Step 5: Independent review**

Run `/superpowers:requesting-code-review` in a clean subagent (Qodo is out of credits until next month). Give it the branch diff, the task file and this plan's PR A section. Address the findings before opening the PR.

- [ ] **Step 6: Push and open PR A**

```bash
git push -u origin fix/chacha-pg-legacy-checkouts
gh pr create --base dev --title "fix(chacha): PostgreSQL close_all_connections keeps the shared pool; a dead thread's checkout comes back (TASK-13861, 1 of 4)" --body "$(cat <<'EOF'
## Summary
- `CharactersRAGDB.close_all_connections()` no longer closes the shared PostgreSQL content pool. One MCP tool call (the modules open and close a handle per call) made every later ChaCha access fail with "Connection pool is closed" until restart.
- On a caller's own pool it now closes this thread's checkout before closing the pool, instead of queuing a reset that the closing pool discards (the product half of TASK-13841's teardown hang).
- A legacy thread-local checkout whose thread exits is now queued and returned by the next ChaCha checkout or close in the process. Before, a dead AnyIO or executor thread lost its pool slot for good. A dropped handle's checkout takes the same path, but only when the cyclic GC collects the handle, so that is a backstop; PR C fixes the known callers. The finalizer only puts onto a `queue.SimpleQueue`, so it can never wait on the pool's own lock.
- `get_workspace(for_update=True)` and `get_persona_profile(for_update=True)` now see a transaction opened inside an operation. On PostgreSQL they failed inside every production HTTP request: non-Character workspace chat creation, workspace startup and persona admission returned 400.
- First of four PRs for TASK-13861. PR B: WebSocket and persona steps own their checkouts. PR C: Jobs and background passes, and the ADR. PR D: the test apps.

## Change summary
<!-- The owner writes this section (ADR-004). -->

## Test plan
- [ ] test_chacha_postgres_close_all_connections.py: dev 2 failed, branch 2 passed
- [ ] test_chacha_postgres_legacy_checkouts.py: dev 5 failed 1 passed, branch 6 passed
- [ ] test_chacha_postgres_locking_reads_in_operation.py: dev 2 failed, branch 2 passed
- [ ] operation-scope pins: 109 passed
- [ ] 108 PostgreSQL ChaCha test files that call close_all_connections: no new failures against the dev baseline
- [ ] ruff: no new findings; Bandit B101 only in new tests; test-quality triage OK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: the PR URL.

- [ ] **Step 7: Before merge**

- Ask the owner for the `## Change summary` (ADR-004).
- Check the merge-queue mode with `gh variable get MERGE_QUEUE`, and follow AGENTS.md "Merging into dev".
- After any PR-body edit, wait for the required statuses: an edit re-triggers the license gate.
- Confirm `MERGED` before removing `<scratchpad>/pr-a`.

---

# PR B — `fix/chacha-pg-websocket-operations`: WebSocket and persona steps own their checkouts

Cut PR B after PR A merges. The REDs of B1-B4 were measured on dev `e521605198` with PR A applied, and B5's on unchanged dev `3ad31cba25`. B4 changes none of B5's 13 sites or the skill lookup, so B5's RED on top of B4 is the same (lines one lower). The REDs on unchanged dev are the same, except where a step says otherwise.

```bash
git fetch origin dev --quiet
git worktree add <scratchpad>/pr-b -b fix/chacha-pg-websocket-operations origin/dev
cd <scratchpad>/pr-b
git log --oneline -1 --grep "close_all_connections keeps the shared PostgreSQL pool open"
```
Expected: one commit, PR A's A1. If it prints nothing, PR A has not merged yet: stop.

### Task B1: the request that loses the default-character insert race reads the winner's row

**Files:**
- Create: `tldw_Server_API/tests/Chat/unit/test_chat_helpers_default_character_race.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_helpers.py:254` (the `except` in `_create_default`)

**Interfaces:**
- Produces: `get_or_create_character_context(db, character_id, loop)` keeps its signature. When `add_character_card` raises `ConflictError`, it returns the existing default character instead of `(None, None)`. Task B2's concurrent test depends on this.

- [ ] **Step 1: Write the failing test**

Create `tldw_Server_API/tests/Chat/unit/test_chat_helpers_default_character_race.py`:

```python
"""Concurrent first requests both try to create the default character (TASK-13861)."""

import asyncio
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.Character_Chat.constants import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.core.Chat.chat_helpers import get_or_create_character_context
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError

pytestmark = pytest.mark.unit


async def test_the_request_that_loses_the_insert_race_reads_the_winners_default_character():
    winner = {"id": 7, "name": DEFAULT_CHARACTER_NAME}
    db = MagicMock()
    db.client_id = "1"
    # Fallback read, read inside the creator, then the read after the conflict.
    db.get_character_card_by_name.side_effect = [None, None, winner]
    db.add_character_card.side_effect = ConflictError("Character card already exists.", entity="character_cards")

    card, card_id = await get_or_create_character_context(db, None, asyncio.get_running_loop())

    assert (card, card_id) == (winner, 7)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Chat/unit/test_chat_helpers_default_character_race.py 2>&1 | grep -E "^E  |passed|failed"
```
Expected (measured): `E   AssertionError: assert (None, None) == ({'id': 7, 'n...ssistant'}, 7)` and `1 failed`.

- [ ] **Step 3: Implement**

In `chat_helpers.py`, `_create_default`, replace the single `except (CharactersRAGDBError, InputError, ConflictError) as db_error:` clause (`:254`) with two clauses. `ConflictError` subclasses `CharactersRAGDBError` (`ChaChaNotes_DB.py:352`), so its clause comes first:

```python
        except ConflictError:
            # A concurrent first request created it between our read and our insert.
            fetched = db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
            if fetched:
                return fetched, fetched.get("id")
            logger.error("Default character '{}' conflicted but could not be read back", DEFAULT_CHARACTER_NAME)
            return None, None
        except (CharactersRAGDBError, InputError) as db_error:
```
The body of the second clause is unchanged.

- [ ] **Step 4: Run it and the existing helper tests**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Chat/unit/test_chat_helpers_default_character_race.py \
  tldw_Server_API/tests/Chat/unit/test_chat_helpers.py 2>&1 | grep -E "^E  |passed|failed"
```
Expected (measured): `33 passed`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Chat/chat_helpers.py \
  tldw_Server_API/tests/Chat/unit/test_chat_helpers_default_character_race.py
git commit -m "$(cat <<'EOF'
fix(chat): read back the default character after losing its insert race (TASK-13861)

Concurrent first sessions of a new user both tried to create the default
character; the loser's ConflictError made get_or_create_character_context
return nothing, and /audio/chat/stream stopped persisting that session.
Measured: 6 of 54 concurrent first sessions on dev. It now reads back the
row the winner committed (0 of 54).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B2: `/audio/chat/stream` persistence steps own their checkouts (AC1, AC2)

**Files:**
- Create: `tldw_Server_API/tests/Audio/test_ws_audio_chat_stream_postgres.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`, at the import block (`:109`), `_ensure_persistence_context` (`:2402-2482`) and `_persist_sync` (`:2494`) (dev line numbers).

**Interfaces:**
- Consumes: `chacha_operation(*, independent=False, bindings=())` from `tldw_Server_API.app.core.DB_Management.chacha.operation_scope`, and Task B1's conflict read-back.
- Produces: each persistence step takes its checkouts inside its own independent operation and returns them when the step ends. There are two kinds of step: the one-time context set-up, and each turn's message writes. The handler's signature and messages are unchanged.

- [ ] **Step 1: Write the failing test**

Create `tldw_Server_API/tests/Audio/test_ws_audio_chat_stream_postgres.py`:

```python
"""/chat/stream persistence over a real PostgreSQL ChaCha database (TASK-13861 AC1, AC2)."""

import asyncio
import itertools
import threading
from collections import Counter

import pytest

from tldw_Server_API.app.api.v1.endpoints import audio
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.Audio.test_ws_audio_chat_stream import (  # noqa: F401 - autouse fixture
    _AUDIO_LABEL_TO_SAMPLE,
    DummyWebSocket,
    _DummyTTSService,
    _pcm16_audio,
    mock_audio_ws_dependencies,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_owners(pg_database_config, tmp_path, monkeypatch):
    """Three users' cached handles on one pool that records who holds each checkout."""
    pg_database_config.pool_timeout = 5.0  # a pinned pool fails in 5 s, not 30
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned = {}, Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    # New users: their concurrent first sessions race to create the default character.
    owners = {
        user_id: CharactersRAGDB(tmp_path / f"user{user_id}.db", client_id=str(user_id), backend=backend)
        for user_id in (1, 2, 3)
    }
    for db in owners.values():
        db.close_connection()

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield owners, outstanding
    finally:
        for db in owners.values():
            db.close_connection()
        pool.close_all()


def _persisting_session():
    payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    return DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
                "metadata": {"persist_history": True},
            },
            {"type": "audio", "data": payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )


async def test_chat_stream_sessions_return_every_checkout_while_the_server_runs(pg_owners, monkeypatch):
    owners, outstanding = pg_owners
    sessions_per_user = 3
    order = itertools.cycle([user_id for user_id in owners for _ in range(sessions_per_user)])

    async def _tts():
        return _DummyTTSService([b"tts"])

    async def _db_for_user(_user_id, client_id=None):  # noqa: ARG001 - one call per session
        return owners[next(order)]

    monkeypatch.setattr(audio, "get_tts_service", _tts)
    monkeypatch.setattr(audio, "get_chacha_db_for_user_id", _db_for_user, raising=False)
    sockets = [_persisting_session() for _ in range(len(owners) * sessions_per_user)]

    await asyncio.gather(*(audio.websocket_audio_chat_stream(ws, token=None) for ws in sockets))

    # The event loop and its default executor threads are still alive here, as in a running server.
    assert outstanding() == []
    warnings = [m for ws in sockets for m in ws.sent_json if "persist" in str(m.get("message", "")).lower()]
    assert warnings == []
    with chacha_operation(independent=True):
        rows = owners[1].execute_query("SELECT sender FROM messages").fetchall()  # one database for all three
    assert Counter(row["sender"] for row in rows) == {"user": len(sockets), "assistant": len(sockets)}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Audio/test_ws_audio_chat_stream_postgres.py 2>&1 | grep -E "^E  |passed|failed"
```
Expected (measured on `e521605198`, with PR A and without): `E   psycopg_pool.PoolTimeout: couldn't get a connection after 5.00 sec`, raised from the `gather`, then `1 failed`. The sessions' pinned legacy checkouts fill the 10-connection pool. If fewer executor threads were used, the same run fails instead at `assert outstanding() == []`, which lists `asyncio_N` holders.

- [ ] **Step 3: Implement**

Add the import directly above the `ChaChaNotes_DB` import (`:109`). Ruff's isort places `chacha.operation_scope` first:

```python
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
```

Replace `_ensure_persistence_context` (dev `:2402-2482`) with the version below. Everything inside the `try` moves one level in, under the operation. The settings upsert switches to `asyncio.to_thread`, which carries the operation (`run_in_executor` does not):

```python
        async def _ensure_persistence_context() -> Optional[tuple[CharactersRAGDB, str]]:
            nonlocal persistence_db
            nonlocal persistence_session_id
            nonlocal persistence_ready
            nonlocal persistence_enabled
            nonlocal session_id
            nonlocal persistence_announced
            if not persistence_enabled:
                return None
            if persistence_ready and persistence_db is not None and persistence_session_id:
                return persistence_db, persistence_session_id
            try:
                # A WebSocket runs outside the HTTP operation middleware. Own this step's
                # checkouts so they go back to the pool when it ends.
                with chacha_operation(independent=True):
                    if persistence_db is None:
                        persistence_db = await _shim_get_chacha_db_for_user_id(
                            int(user_id_for_usage),
                            client_id=str(user_id_for_usage),
                        )
                    loop = asyncio.get_running_loop()
                    character_card, character_db_id = await _shim_get_or_create_character_context(
                        persistence_db,
                        None,
                        loop,
                    )
                    if not character_db_id:
                        raise ValueError("Unable to resolve character context for WS persistence")
                    character_name = "Helpful AI Assistant"
                    if isinstance(character_card, dict) and character_card.get("name"):
                        character_name = str(character_card.get("name"))

                    persistence_session_id, conversation_created = await _shim_get_or_create_conversation(
                        persistence_db,
                        persistence_session_id,
                        int(character_db_id),
                        character_name,
                        str(user_id_for_usage),
                        loop,
                    )
                    session_id = persistence_session_id

                    if session_id and not any(
                        msg.get("role") == "system" and msg.get("content") == f"session:{session_id}"
                        for msg in chat_history
                    ):
                        chat_history.insert(0, {"role": "system", "content": f"session:{session_id}"})

                    action_hint = _action_hint()
                    settings_payload = {
                        "audio_chat_ws": {
                            "session_id": persistence_session_id,
                            "action_hint": action_hint,
                            "metadata": metadata or {},
                        }
                    }
                    try:
                        await asyncio.to_thread(
                            partial(
                                _persist_audio_chat_settings,
                                persistence_db,
                                persistence_session_id,
                                settings_payload,
                                conversation_created=conversation_created,
                            ),
                        )
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as settings_exc:
                        logger.debug(f"audio.chat.stream conversation_settings upsert failed: {settings_exc}")

                    if _outer_stream and not persistence_announced:
                        await _outer_stream.send_json({"type": "session", "session_id": persistence_session_id})
                        persistence_announced = True

                    persistence_ready = True
                    return persistence_db, persistence_session_id
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"audio.chat.stream persistence unavailable: {exc}")
                persistence_enabled = False
                await _send_persistence_warning(
                    "Session persistence unavailable; continuing without persistence",
                    _maybe_debug_details(exc),
                )
                return None
```

In `_persist_turn`, decorate the per-turn writer (dev `:2494`). It still runs through `loop.run_in_executor(None, _persist_sync)` (`:2528`), so the operation must open inside it:

```python
            @chacha_operation(independent=True)
            def _persist_sync() -> None:
```

- [ ] **Step 4: Run it to verify it passes, five times**

```bash
for i in 1 2 3 4 5; do TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Audio/test_ws_audio_chat_stream_postgres.py 2>&1 | grep -E "^=+ .*(passed|failed)"; done
```
Expected: five lines of `1 passed` (v1 and the review: 6.6-9.5 s each). Without Task B1 the review saw 5 of 5 runs fail on `assert warnings == []`.

- [ ] **Step 5: Run the existing chat-stream tests**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
```
Expected: `80 passed`, with no `FAILED` or `ERROR` lines. The existing tests use a dummy DB, so the operations change nothing they observe.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  tldw_Server_API/tests/Audio/test_ws_audio_chat_stream_postgres.py
git commit -m "$(cat <<'EOF'
fix(audio): /chat/stream persistence steps own their PostgreSQL checkouts (TASK-13861)

ChaChaOperationMiddleware only wraps HTTP, so /audio/chat/stream
persisted through the user's cached handle's legacy thread-local
checkouts on default-executor threads. Nine concurrent sessions left 17
checkouts pinned on 9 live threads; a 10-connection pool timed out.

The context set-up and each turn's writes now run in their own
operation, which returns its checkout when the step ends. The settings
upsert moves to asyncio.to_thread because run_in_executor does not carry
the operation into the worker thread.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B3: voice-assistant WebSocket steps own their checkouts (AC2)

**Files:**
- Create: `tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/voice_assistant.py`, at the import (`:64`), the session-start block (`:1100-1107`) and `_process_text_command` (`:1407-1414`).

**Interfaces:**
- Consumes: `chacha_operation`.
- Produces: the session-start step and each command's `router_instance.process_command(...)` run in their own operations. TTS and the WebSocket sends after `process_command` stay outside, so no checkout is held while audio streams.

- [ ] **Step 1: Write the failing tests**

Create `tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py`:

```python
"""Voice assistant WebSocket over a real PostgreSQL ChaCha database (TASK-13861 AC1, AC2)."""

import threading
from collections import Counter

import pytest

from tldw_Server_API.app.api.v1.endpoints import voice_assistant
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.VoiceAssistant import test_ws_integration as ws_integration

pytestmark = pytest.mark.integration
DummyWebSocket = ws_integration.DummyWebSocket
mock_ws_dependencies = ws_integration.mock_ws_dependencies  # the shared fixture, registered in this module


@pytest.fixture
def pg_voice_db(pg_database_config, tmp_path, monkeypatch):
    """The user's cached handle, served to the WebSocket, on a pool that records its holders."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned = {}, Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    db = CharactersRAGDB(tmp_path / "voice.db", client_id="1", backend=backend)

    async def _db_for_user(_user_id, client_id=None):  # noqa: ARG001
        return db

    monkeypatch.setattr(voice_assistant, "get_chacha_db_for_user_id", _db_for_user)

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield db, outstanding
    finally:
        db.close_connection()
        pool.close_all()


async def test_session_start_returns_its_checkout(mock_ws_dependencies, pg_voice_db):
    _db, outstanding = pg_voice_db
    ws = DummyWebSocket([{"type": "auth", "token": "valid-token"}, {"type": "config"}])

    await voice_assistant.websocket_voice_assistant(ws, token=None)

    assert [m for m in ws.sent_json if m.get("type") == "auth_ok"]
    assert outstanding() == []


async def test_a_text_command_returns_its_checkout(mock_ws_dependencies, pg_voice_db, monkeypatch):
    _db, outstanding = pg_voice_db
    _sessions, command_router = mock_ws_dependencies
    original = command_router.process_command

    async def process_with_db(text, user_id, session_id, db=None, persona_id=None):
        # The real router reads and saves through the handle on the event-loop thread.
        db.get_character_card_by_name("nobody")
        return await original(text, user_id, session_id, db=db, persona_id=persona_id)

    monkeypatch.setattr(command_router, "process_command", process_with_db)
    ws = DummyWebSocket(
        [{"type": "auth", "token": "valid-token"}, {"type": "config"}, {"type": "text", "text": "hello there"}]
    )

    await voice_assistant.websocket_voice_assistant(ws, token=None)

    assert command_router.processed_commands == ["hello there"]
    assert outstanding() == []
```

How the test borrows its fixtures:
- **The shared fixture.** It reaches `mock_ws_dependencies` and `DummyWebSocket` through a module import, not `from … import mock_ws_dependencies`. A direct import of a fixture that is then used as a parameter is ruff F811 x2 (the review's Minor 4); the module form is clean (measured).
- **A logged failure that is expected.** The fixture's `DummyRegistry` has no `refresh_user_commands`, so the session-start block logs "Voice assistant DB init failed". By then `save_voice_session` has already used the handle. Dev does the same, and the checkout under test is taken before it.

- [ ] **Step 2: Run them to verify they fail**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q -rf \
  tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py 2>&1 | grep -E "^E   Assert|^FAILED|passed|failed"
```
Expected (measured on `e521605198` with PR A, and without):
```
E   AssertionError: assert ['MainThread'] == []      (x2)
FAILED …::test_session_start_returns_its_checkout
FAILED …::test_a_text_command_returns_its_checkout
==== 2 failed, … ====
```

- [ ] **Step 3: Implement**

Add the import (`:64`). `voice_assistant.py` already fails ruff I001 on dev, so do not re-sort the rest of the block:

```python
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
```

Session start (dev `:1101-1107`):

```python
        try:
            # A WebSocket runs outside the HTTP operation middleware: own this step's checkouts.
            with chacha_operation(independent=True):
                db = await get_chacha_db_for_user_id(user_id, client_id="voice_assistant")
                save_voice_session(db, session)
                registry = get_voice_command_registry()
                registry.load_defaults()
                registry.refresh_user_commands(db, user_id, include_disabled=True)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
```

`_process_text_command` (dev `:1407-1414`):

```python
    """Process text through Intent -> Action -> TTS pipeline."""
    # A WebSocket runs outside the HTTP operation middleware: this command owns its checkouts.
    with chacha_operation(independent=True):
        result, _ = await router_instance.process_command(
            text=text,
            user_id=user_id,
            session_id=session_id,
            db=db,
        )
```
`_process_audio_command` ends by calling `_process_text_command` (`:1356-1364`), so audio commands are covered too.

- [ ] **Step 4: Run them, and the existing WebSocket tests**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py \
  tldw_Server_API/tests/VoiceAssistant/test_ws_integration.py 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
$PY -m ruff check --output-format concise tldw_Server_API/app/api/v1/endpoints/voice_assistant.py | grep -c ": [A-Z]"
$PY -m ruff check tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py
```
Expected:
- `15 passed` with no `FAILED` or `ERROR` lines: the 2 new tests plus `test_ws_integration.py`'s 13;
- a ruff count of `1`, the I001 that dev already has;
- `All checks passed!` for the new test.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/voice_assistant.py \
  tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py
git commit -m "$(cat <<'EOF'
fix(voice): voice-assistant WebSocket steps own their PostgreSQL checkouts (TASK-13861)

The voice-assistant WebSocket saved its session and ran each command
through the user's cached handle on the event-loop thread, outside any
operation, so the loop thread kept that checkout for the life of the
process. Session start and each command's routing now run in their own
operation.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B4: each off-loop persona call owns its checkout (AC2, Decision 2)

Skip this task if the owner chose Decision 2's Alternative 3. In that case B6 files the whole persona session as a follow-up instead.

**Files:**
- Create: `tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py`
- Modify:
  - `tldw_Server_API/app/api/v1/endpoints/persona.py`: the import (`:166`) and the `asyncio.to_thread` callables at `:8878`, `:8889`, `:8965`, `:10261`, `:10542` and `:10759`;
  - `tldw_Server_API/app/core/Persona/live_conversation.py`: the import (`:13`) and `:90`;
  - `tldw_Server_API/app/core/Persona/exemplar_runtime.py`: the import (`:11`) and `:114`.

**Interfaces:**
- Consumes: `chacha_operation`. `chacha_operation(...)(func)` returns `func` wrapped so that each call runs inside a new operation, which closes, returning its checkout, before the call returns. Inside `asyncio.to_thread` that happens on the worker thread.
- Produces: every off-loop use of a persona session's handle returns its checkout when the call returns.
  - `persona.py` opens an independent operation.
  - `live_conversation.py` and `exemplar_runtime.py` use `chacha_operation()`, which joins the operation of an HTTP request that calls them (they are not WebSocket-only), and owns the call otherwise.

  Signatures are unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py`:

```python
"""The persona live WebSocket's per-session handle returns each checkout per call (TASK-13861)."""

import ast
import asyncio
import threading
from collections import Counter
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Persona.exemplar_runtime import resolve_persona_exemplar_runtime_context
from tldw_Server_API.app.core.Persona.exemplar_turn_classifier import PersonaTurnClassification

pytestmark = pytest.mark.integration

_APP = Path(__file__).resolve().parents[2] / "app"
# Off-loop uses of a persona session's handle: the WebSocket and the two helpers its turns call.
_HANDLE_SITES = {
    "api/v1/endpoints/persona.py": "persona_scope_db",
    "core/Persona/live_conversation.py": "profile_store",
    "core/Persona/exemplar_runtime.py": "persona_scope_db",
}


def test_every_off_loop_persona_handle_call_runs_in_its_own_operation():
    unwrapped, wrapped = [], 0
    for rel, handle in _HANDLE_SITES.items():
        for node in ast.walk(ast.parse((_APP / rel).read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "asyncio.to_thread"):
                continue
            if handle not in ast.unparse(node):
                continue
            callable_text = ast.unparse(node.args[0])
            if callable_text.startswith("chacha_operation("):
                wrapped += 1
            else:
                unwrapped.append(f"{rel}:{node.lineno} {callable_text}")
    assert unwrapped == [], f"wrap these in chacha_operation(...)(callable): {unwrapped}"
    assert wrapped == 8


@pytest.fixture
def session_handles(pg_database_config, tmp_path, monkeypatch):
    """Four persona sessions' own handles on one ten-connection pool that records its holders."""
    pg_database_config.pool_timeout = 5.0
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned = {}, Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    handles = [CharactersRAGDB(tmp_path / f"session{n}.db", client_id="1", backend=backend) for n in range(4)]
    for handle in handles:
        handle.close_connection()

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield handles, outstanding
    finally:
        for handle in handles:
            handle.close_connection()
        pool.close_all()


async def test_concurrent_persona_sessions_hold_no_checkout_between_calls(session_handles):
    handles, outstanding = session_handles

    async def session(handle):
        for _turn in range(12):
            await resolve_persona_exemplar_runtime_context(
                persona_scope_db=handle,
                user_id="1",
                persona_id="persona-1",
                classifier=PersonaTurnClassification(),
                current_turn_text="hello",
            )

    await asyncio.gather(*(session(handle) for handle in handles))

    # The loop and its default executor threads are alive, as during live sessions.
    assert outstanding() == []
```

The two tests:
- **The AST guard** pins the 8 call sites, so a new off-loop call on the persona handle cannot go unwrapped.
- **The behavioural test** reproduces F12 with the real `resolve_persona_exemplar_runtime_context`. It runs 4 concurrent sessions, each with its own handle, on a 10-connection pool.

- [ ] **Step 2: Run them to verify they fail**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py 2>&1 | grep -E "^E   (Assert|  Left)|^FAILED|passed|failed" | cut -c1-200
```
Expected (measured on `e521605198` with PR A):
```
E   AssertionError: wrap these in chacha_operation(...)(callable): ['api/v1/endpoints/persona.py:8877 _load_persona_policy_rules_for_session', 'api/v1/endpoi…
E     Left contains 8 more items, first extra item: 'api/v1/endpoints/persona.py:8877 _load_persona_policy_rules_for_session'
E   AssertionError: assert ['asyncio_0',...yncio_2', ...] == []
E     Left contains 10 more items, first extra item: 'asyncio_0'
==== 2 failed, … ====
```
The second failure is the whole 10-connection pool, held by executor threads after the sessions ended.

- [ ] **Step 3: Implement**

In `persona.py`, add the import directly above `from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (` (dev `:166`):

```python
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
```

Then wrap the callable passed to `asyncio.to_thread` at each of the six sites (dev line of the callable):

| Line | Before | After |
|---|---|---|
| `:8878` | `_load_persona_policy_rules_for_session,` | `chacha_operation(independent=True)(_load_persona_policy_rules_for_session),` |
| `:8889` | `persona_scope_db.get_persona_profile,` | `chacha_operation(independent=True)(persona_scope_db.get_persona_profile),` |
| `:8965` | `_load_persona_policy_rules_for_session, persona_scope_db,` | `chacha_operation(independent=True)(_load_persona_policy_rules_for_session), persona_scope_db,` |
| `:10261` | `_load_persona_policy_rules_for_session, persona_scope_db,` | `chacha_operation(independent=True)(_load_persona_policy_rules_for_session), persona_scope_db,` |
| `:10542` | `_load_persona_policy_rules_for_session, persona_scope_db,` | `chacha_operation(independent=True)(_load_persona_policy_rules_for_session), persona_scope_db,` |
| `:10759` | `_load_persona_policy_rules_for_session, persona_scope_db,` | `chacha_operation(independent=True)(_load_persona_policy_rules_for_session), persona_scope_db,` |

Keep each line's indentation. For example, `:8877-8882` becomes:

```python
            runtime_context = await asyncio.to_thread(
                chacha_operation(independent=True)(_load_persona_policy_rules_for_session),
                persona_scope_db,
                session_id=session_id,
                user_id=authenticated_user_id,
            )
```

In `live_conversation.py`, add the import before `from tldw_Server_API.app.core.exceptions import PersonaConversationError` (dev `:13`), and replace `:90`:

```python
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
```
```python
    profile = await asyncio.to_thread(
        # Join the caller's ChaCha operation, or own this read outside one (a WebSocket turn).
        chacha_operation()(profile_store.get_persona_profile),
        persona_id,
```

In `exemplar_runtime.py`, add the same import before `from tldw_Server_API.app.core.Persona.exemplar_prompt_assembly import (` (dev `:11`), and replace `:114`:

```python
            persona_exemplars = await asyncio.to_thread(
                # Join the caller's ChaCha operation, or own this read outside one (a WebSocket turn).
                chacha_operation()(persona_scope_db.list_persona_exemplars),
                user_id=user_id,
```

- [ ] **Step 4: Run them, and the existing persona tests**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
git add -N tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py  # git grep reads tracked files only
git grep -lE 'endpoints(\.| import )persona\b|Persona(\.| import )(live_conversation|exemplar_runtime)|persona/stream' -- 'tldw_Server_API/tests/*test_*.py' > "$S/persona_tests.txt"
wc -l < "$S/persona_tests.txt"
$S/sweep.sh "$S/persona_tests.txt" "$S/runs/b_persona" | tee "$S/runs/b_persona.summary"
```
Expected:
- `2 passed`;
- `16` files: the 8 on dev that import the persona endpoint or its helpers, the 7 that drive `/api/v1/persona/stream` without importing it, and B4's new test (measured on `3ad31cba25`; the re-review ran the same 16 with B4 and B5 applied: `314 passed`);
- no `FAILED` or `ERROR` lines in the persona sweep.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/app/core/Persona/live_conversation.py \
  tldw_Server_API/app/core/Persona/exemplar_runtime.py \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py
git commit -m "$(cat <<'EOF'
fix(persona): each off-loop persona call owns its PostgreSQL checkout (TASK-13861)

/persona/stream used its per-session handle through asyncio.to_thread
outside any operation, so every executor thread a session used kept a
legacy checkout. At session end close_all_connections() returned only
the loop thread's, and the rest waited for the cyclic GC. Four concurrent
sessions held a whole 10-connection pool.

Each of the 8 off-loop calls now runs its callable inside an operation
that returns the checkout before the call returns (about 5 more round
trips per call). The two core helpers join an HTTP request's operation when one is
open. An AST guard pins the 8 sites.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B5: each on-loop persona call owns its checkout (AC2, Decision 2's recommendation)

Do this task only if the owner took Decision 2's recommendation (Alternative 1). Under Alternative 2 or 3, skip it; B6 Step 2 files the follow-up.

**What B5 fixes, and what it does not.**
- **Fixed:** persona sessions no longer *cause* pool exhaustion. No session keeps a checkout, idle in a transaction, between its calls.
- **Not fixed: a wrapped on-loop call still checks out synchronously on the event-loop thread.** It runs `getconn` and about 5 round trips at every call. Before B5, a session did that only on its first call.
  - When other work holds the whole pool, such a call blocks the whole server's event loop for up to `pool_timeout` (30 s by default) and then raises `PoolTimeout`. That other work includes HTTP requests and every other content database, which share production's 20 connections.
  - Measured by the re-review, pool of 10 with 9 held elsewhere and one warm session: the B5-wrapped call raised `PoolTimeout` after 2.0 s, with a 2.01 s event-loop gap. A warm legacy call took 0.0 s.
  - B6 Step 2 files a follow-up to move the 10 call sites that sit directly in `async def`s off the event loop.

**Files:**
- Create: `tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py`
- Modify:
  - `tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py` (from B4): add the on-loop guard.
  - `tldw_Server_API/app/api/v1/endpoints/persona.py`, at the 13 on-loop call sites listed in Step 3. These are dev line numbers; on this branch each is one line lower, because of B4's import.
  - `tldw_Server_API/app/core/Skills/context_integration.py`: the import (after `:20`) and `handle_skill_tool_call`'s lookup (`:144-145`).

**Interfaces:**
- Consumes: `chacha_operation`, imported in `persona.py` by B4. `chacha_operation(independent=True)(fn)(...)` runs one synchronous call inside a new operation, on the calling thread, and returns its checkout before it returns.
- Produces:
  - No use of a persona session's handle keeps a checkout after the call that made it.
  - `handle_skill_tool_call` returns its lookup's ChaCha checkout before `SkillExecutor.execute` runs, so a forked skill's LLM call holds none. This holds for every caller: the lookup's operation is independent, so it never joins an operation the caller holds.
  - Signatures are unchanged.

- [ ] **Step 1: Write the failing tests**

In `tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py`, add this directly after `test_every_off_loop_persona_handle_call_runs_in_its_own_operation`:

```python
# Calls that receive the session's handle but own their ChaCha work themselves, or end the session.
_SELF_OWNING = {
    "asyncio.to_thread",
    "resolve_persona_exemplar_runtime_context",
    "live_conversation.complete_persona_turn",
    "handle_skill_tool_call",
    "persona_scope_db.close_all_connections",
    "persona_scope_db.close_connection",
}


def test_every_on_loop_persona_handle_call_runs_in_its_own_operation():
    """A direct call on the event-loop thread would keep a checkout, idle in a transaction, for the session."""
    tree = ast.parse((_APP / "api/v1/endpoints/persona.py").read_text(encoding="utf-8"))
    unwrapped, wrapped = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        per_call = isinstance(target, ast.Call) and ast.unparse(target.func).startswith("chacha_operation(")
        callee = target.args[0] if per_call else target
        passes_handle = "persona_scope_db" in {ast.unparse(v) for v in [*node.args, *(k.value for k in node.keywords)]}
        calls_handle = isinstance(callee, ast.Attribute) and ast.unparse(callee.value) == "persona_scope_db"
        if not (passes_handle or calls_handle) or ast.unparse(callee) in _SELF_OWNING:
            continue
        if per_call:
            wrapped += 1
        else:
            unwrapped.append(f"persona.py:{node.lineno} {ast.unparse(callee)}")
    assert unwrapped == [], f"wrap these as chacha_operation(independent=True)(fn)(...): {unwrapped}"
    assert wrapped == 13
```

Create `tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py`:

```python
"""A skill tool call returns its PostgreSQL checkout before the skill runs (TASK-13861).

A forked skill makes an LLM call inside SkillExecutor.execute; no ChaCha checkout may
be held across it. Only the registry lookup before it touches ChaCha.
"""

import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Skills import context_integration
from tldw_Server_API.app.core.Skills.skills_service import SkillsService

pytestmark = pytest.mark.integration


@pytest.fixture
async def skills_owner(pg_database_config, tmp_path, monkeypatch):
    """Owner 1's handle, with one skill, on a pool that records which thread holds each checkout."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned = {}, Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    db = CharactersRAGDB(tmp_path / "skills.db", client_id="1", backend=backend)
    with chacha_operation(independent=True):
        await SkillsService(user_id=1, base_path=tmp_path, db=db).create_skill(
            "summarize", "---\ndescription: Summarize text\ncontext: fork\n---\nSummarize: $ARGUMENTS"
        )

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield SimpleNamespace(db=db, base_path=tmp_path, outstanding=outstanding)
    finally:
        db.close_connection()
        pool.close_all()


async def test_no_checkout_is_held_while_the_skill_runs(skills_owner, monkeypatch):
    held_while_running = []

    class RecordingExecutor:
        """Stands in for SkillExecutor, whose forked skills make an LLM call."""

        async def execute(self, *, skill_data, arguments, context):  # noqa: ARG002
            held_while_running.append(skills_owner.outstanding())
            return SimpleNamespace(
                skill_name=skill_data["name"], rendered_prompt=arguments, allowed_tools=None,
                model_override=None, execution_mode="fork", fork_output="done",
            )

    monkeypatch.setattr(context_integration, "SkillExecutor", RecordingExecutor)

    result = await context_integration.handle_skill_tool_call(
        skill_name="summarize", args="text", user_id=1, base_path=skills_owner.base_path, db=skills_owner.db
    )

    assert result["success"] is True, result
    assert held_while_running == [[]]
    assert skills_owner.outstanding() == []
```

The guard is the regression pin for the 13 call sites. The behaviour it protects was measured with a probe (F12), not a committed test. A behavioural test would have to drive `persona_stream` over PostgreSQL, because the sites live inside its nested functions. 11 concurrent sessions on a 10-connection pool showed:
- unwrapped: 10 checkouts `INTRANS` on `MainThread`, and `PoolTimeout` on the 11th after the event loop stalled for 3.04 s;
- wrapped: 0 held.

The skill test pins the narrower operation around the skill lookup (ruling 2 of the third review; F12 explains the choice). Its executor stand-in records the checkouts that are out while the skill "runs".

- [ ] **Step 2: Run them to verify they fail**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  "tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py::test_every_on_loop_persona_handle_call_runs_in_its_own_operation" \
  tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py 2>&1 | grep -E "^E   (Assert|  Left)|passed|failed" | cut -c1-200
```
Expected (measured on `3ad31cba25`):
```
E   AssertionError: wrap these as chacha_operation(independent=True)(fn)(...): ['persona.py:9417 _load_persona_scope_metadata_for_session', …
E     Left contains 13 more items, first extra item: 'persona.py:9417 _load_persona_scope_metadata_for_session'
E   AssertionError: assert [['asyncio_0']] == [[]]
==== 2 failed, … ====
```
The skill test's failure is the lookup's checkout, still out while the skill runs. Revision 3's `with chacha_operation(independent=True):` around the whole call in `persona.py` would hold it the same way, through a forked skill's LLM call.

- [ ] **Step 3: Implement**

In `persona.py`, wrap each of the 13 synchronous calls by replacing the callee at the start of the call, keeping the arguments and the indentation. Line numbers are dev's; add 1 on this branch.

| Dev line | Before | After |
|---|---|---|
| `:9019`, `:9950`, `:10611`, `:10828`, `:11069`, `:11291`, `:11366` | `runtime_context = _load_persona_policy_rules_for_session(` | `runtime_context = chacha_operation(independent=True)(_load_persona_policy_rules_for_session)(` |
| `:9091` | `persona_scope_db.upsert_persona_live_voice_session_summary(` | `chacha_operation(independent=True)(persona_scope_db.upsert_persona_live_voice_session_summary)(` |
| `:9128` | `record_persona_live_voice_event(` | `chacha_operation(independent=True)(record_persona_live_voice_event)(` |
| `:9417` | `scope_metadata = _load_persona_scope_metadata_for_session(` | `scope_metadata = chacha_operation(independent=True)(_load_persona_scope_metadata_for_session)(` |
| `:10069`, `:11114` | `_ = _persist_persona_session_preferences(` | `_ = chacha_operation(independent=True)(_persist_persona_session_preferences)(` |
| `:10138` | `persona_state_hints = _load_persona_state_hints_for_runtime(` | `persona_state_hints = chacha_operation(independent=True)(_load_persona_state_hints_for_runtime)(` |

For example, `:9019-9023` becomes:

```python
                runtime_context = chacha_operation(independent=True)(_load_persona_policy_rules_for_session)(
                    persona_scope_db,
                    session_id=session_id,
                    user_id=authenticated_user_id,
                )
```

**The async call, `await handle_skill_tool_call(..., db=persona_scope_db)` (dev `:9561`), stays as it is in `persona.py`.** Its only ChaCha work is the skill registry lookup, so the operation goes around that lookup, inside `handle_skill_tool_call`.

In `context_integration.py`, add the import after the `ContextIntegrityBlocked` import (`:20`):

```python
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
```
and replace the two lookup lines (`:144-145`):

```python
        # The registry lookup gets its own operation, so its checkout is returned before a
        # forked skill's LLM call, whatever operation the caller holds.
        with chacha_operation(independent=True):
            service = SkillsService(user_id=user_id, base_path=base_path, db=db)
            skill_data = await service.get_skill(skill_name)
```
Why the lookup and not the whole call:
- `SkillExecutor()` gets no handle.
- For a `context: fork` skill, `SkillExecutor.execute` makes an LLM call (`skill_executor.py:330`, `:436`) and may run tools. A block around the whole call would hold the lookup's checkout through all of that, which is the cost this plan rejects for notes-graph (F10).
- Why `independent=True`, not a joining `chacha_operation()`:
  - it never joins a caller's operation, so no caller's checkout can span its skill run;
  - a closed operation inherited by a detached task cannot make it raise `ClosedChaChaOperationError`, which the function's `except` clauses would not catch;
  - the cost is one extra checkout when a caller already holds an operation, and no such caller exists today: `persona.py` is the only production caller (`git grep handle_skill_tool_call`).
- `SkillsService(...)` also runs 4 `CREATE ... IF NOT EXISTS` statements through `backend.execute` on the event-loop thread at every skill call (`_ensure_skill_registry_table`, `ChaChaNotes_DB.py:27626-27654`). Each takes and returns its own pool connection, outside any operation. This predates the plan, and B5 leaves it as it is. TASK-<N>, filed in B6 Step 2, covers it (its AC 3).

Leave these alone:
- `resolve_persona_exemplar_runtime_context`, `live_conversation.complete_persona_turn` and `handle_skill_tool_call` own their ChaCha work already (B4, and the lookup above);
- the session-end `close_all_connections()` and `close_connection()` (`:11450`, `:11452`).

- [ ] **Step 4: Run the persona and skills tests**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py \
  tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
git add -N tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py  # git grep reads tracked files only
git grep -lE 'endpoints(\.| import )persona\b|Persona(\.| import )(live_conversation|exemplar_runtime)|persona/stream|Skills(\.| import )context_integration' \
  -- 'tldw_Server_API/tests/*test_*.py' > "$S/persona_skills_tests.txt"
wc -l < "$S/persona_skills_tests.txt"
$S/sweep.sh "$S/persona_skills_tests.txt" "$S/runs/b_persona_onloop" 2 | tee "$S/runs/b_persona_onloop.summary"
$PY -m ruff check --output-format concise tldw_Server_API/app/api/v1/endpoints/persona.py | grep -c ": [A-Z]"
$PY -m ruff check tldw_Server_API/app/core/Skills/context_integration.py tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py
```
Expected:
- `4 passed` (measured; with `independent=True` the skill test also passes, together with `test_skill_mcp_integration.py`: `14 passed` on `3c4d734981`);
- `18` files: B4's persona list, plus the Skills tests that import `context_integration` (`test_skill_mcp_integration.py` and this task's test). Measured with all four PRs: `154 passed` and `174 passed` in 2 shards, with no `FAILED` or `ERROR` lines;
- a ruff count of `2`, the I001 and SIM114 that dev already has, and `All checks passed!` for the other two files.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/core/Skills/context_integration.py \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py \
  tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py
git commit -m "$(cat <<'EOF'
fix(persona): each on-loop persona call owns its PostgreSQL checkout (TASK-13861)

/persona/stream also used its per-session handle directly on the
event-loop thread, outside any operation. Each live session therefore
kept one checkout, idle in a transaction, until it ended. Measured on a
10-connection pool: 10 sessions held all ten, and the 11th session's
read blocked the whole event loop for pool_timeout before raising
PoolTimeout.

Each of the 13 synchronous calls now runs in its own operation, which
returns the checkout when the call returns: about 5 more round trips per
call. The skill tool call owns only its registry lookup, so a forked
skill's LLM call holds no checkout. An AST guard pins the 13 sites.

Not fixed here: a wrapped call still checks out on the event-loop
thread, so when other work holds the whole pool it can stall the loop
for up to pool_timeout. A follow-up moves the calls in async handlers
off the loop.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task B6: Sweep, the persona follow-up, notes, and open PR B

**Files:** `backlog/tasks/task-13861 - …md`, and one persona follow-up task (its content depends on Decision 2).

**Interfaces:**
- Consumes: Tasks B1-B5, and the owner's answer to Decision 2.
- Produces: PR B into `dev`.

- [ ] **Step 1: Sweep the test files that import or drive what PR B changed**

```bash
git grep -lE 'core\.Chat\.chat_helpers|Chat import chat_helpers|endpoints\.audio|endpoints import audio|endpoints\.voice_assistant|endpoints import voice_assistant|endpoints\.persona\b|endpoints import persona\b|Persona\.live_conversation|Persona import live_conversation|Persona\.exemplar_runtime|Persona import exemplar_runtime|persona/stream|Skills(\.| import )context_integration' \
  -- 'tldw_Server_API/tests/*test_*.py' > "$S/b_callers.txt"
wc -l < "$S/b_callers.txt"
$S/sweep.sh "$S/b_callers.txt" "$S/runs/b_sweep" | tee "$S/runs/b_sweep.summary"
```
Expected: `120` (measured on `3ad31cba25` with PR B's tests in place):
- the 107 files on dev that import PR B's modules;
- the 7 that drive `/api/v1/persona/stream` without importing it (the re-review's Minor 5);
- `Skills/integration/test_skill_mcp_integration.py`, which imports `context_integration` (B5);
- PR B's 5 new test files.

The re-review's `118` predates B5's skill-lookup change, which adds the `context_integration` alternative and B5's new test.

There must be no `FAILED` or `ERROR` lines beyond the dev baseline. The one dev failure planning saw in these files is `TTS/test_tts_module.py::TestStreamingAudioWriter::test_init_valid_formats`: `ModuleNotFoundError: No module named 'av'`. It fails on dev and with the change alike, because the venv has no PyAV.

- [ ] **Step 2: File the persona follow-up**

Compute a free id with `$S/task_next.sh` and use the printed value + 1 as `<N>`. It printed `13893` on `2cae53929f`.

Under the recommendation (B4 and B5 done), file the residual from B5:
```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create \
  "Persona: run the 10 on-loop ChaCha calls inside async handlers off the event loop" \
  --id TASK-<N> -l persona,postgres,bug \
  -d "Found by TASK-13861's third review. After its PR B (plan Task B5) every persona_stream call on the session's handle runs in its own chacha_operation, so no session keeps a checkout idle in a transaction. But each of the 13 synchronous calls now checks out synchronously on the event-loop thread at every call (getconn plus about 5 round trips). When other work holds the whole content pool (HTTP requests and every other content database share pg_pool_size = 20 in production), each such call blocks the whole server's event loop for up to pool_timeout (30 s by default) and then raises PoolTimeout. Measured by the re-review (pool of 10, 9 connections held elsewhere, one warm session): the B5-wrapped call raised PoolTimeout after 2.0 s, with a 2.01 s event-loop gap; a warm legacy call took 0.0 s. The 10 sites that sit directly in async handlers can move off the loop as plan Task B4 does, as await asyncio.to_thread(chacha_operation(independent=True)(fn), ...): persona.py :9417, :9950, :10069, :10138, :10611, :10828, :11069, :11114, :11291 and :11366 (origin/dev 3ad31cba25 line numbers; one lower after PR B's import). The other three (:9019, :9091, :9128) sit in the sync helpers _resolve_persona_live_event_persona_id, _upsert_persona_live_session_summary_safe and _record_persona_live_voice_event_safe, whose callers would have to move them. Separately, and already so on dev before TASK-13861: in Skills/context_integration.handle_skill_tool_call, SkillsService.__init__ runs 4 table-setup statements (CREATE ... IF NOT EXISTS in _ensure_skill_registry_table, ChaChaNotes_DB.py:27626-27654) through backend.execute on the event-loop thread at every skill call, each with its own pool checkout and outside any operation; the re-review measured 4 MainThread checkouts per skill call on dev and with PR B. Under pool exhaustion those stall the loop the same way. (The lookup's own operation, added by PR B, takes its checkout on an executor thread.)" \
  --ac "The 10 calls in persona_stream's async handlers run through asyncio.to_thread with their own chacha_operation(independent=True), and the persona AST guard requires that form for them" \
  --ac "A PostgreSQL test with the pool held by other work shows a persona call waiting for a connection without stalling the event loop" \
  --ac "The skill tool call's SkillsService setup and lookup make no ChaCha checkout on the event-loop thread" \
  --dep TASK-13861
```
Expected: `TASK-<N> - Persona: run the 10 on-loop ChaCha calls inside async handlers off the event loop (To Do)`.

Under Alternative 2 (B4 done, B5 skipped), file instead:
```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create \
  "Persona live WebSocket: own PostgreSQL ChaCha checkouts for the on-loop calls" \
  --id TASK-<N> -l persona,postgres,bug \
  -d "Found by TASK-13861. After its PR B every off-loop use of /persona/stream's per-session CharactersRAGDB runs in its own chacha_operation (8 asyncio.to_thread sites). The session still calls the handle directly on the event-loop thread at 13 sync call sites (persona.py on origin/dev 3ad31cba25: _load_persona_policy_rules_for_session :9019, :9950, :10611, :10828, :11069, :11291, :11366; upsert_persona_live_voice_session_summary :9091; record_persona_live_voice_event :9128; _load_persona_scope_metadata_for_session :9417; _persist_persona_session_preferences :10069, :11114; _load_persona_state_hints_for_runtime :10138). The skill tool call (handle_skill_tool_call, :9561) holds its registry lookup's checkout, on an executor thread, through the whole skill run, including a forked skill's LLM call; and SkillsService.__init__ runs 4 table-setup statements through backend.execute on the loop at every skill call (pre-existing, outside any operation). Measured: each live session holds one legacy checkout on the loop thread, idle in a transaction, until close_all_connections() returns it at session end (:11450); 10 live sessions exhausted a 10-connection pool, and the 11th session's read blocked the whole event loop for pool_timeout (3.04 s with 3 s; 30 s by default) before raising PoolTimeout. Fix: plan Task B5 (chacha_operation(independent=True)(fn)(...) at each sync site, about 5 extra round trips per call, and an independent operation around the skill lookup inside handle_skill_tool_call); then move the 10 sites in async handlers off the loop, because a wrapped call still checks out on the loop and stalls it when other work holds the pool (2.0 s measured with 9 of 10 connections held elsewhere). The handle also uses client_id persona_ws_<uid>, not the tenant id ChaCha_Notes_DB_Deps._tenant_client_id requires on PostgreSQL." \
  --ac "Every on-loop use of the persona session's handle runs inside its own chacha_operation, and the calls in async handlers run off the event loop" \
  --ac "No live persona session holds a PostgreSQL checkout between calls, and an AST guard pins the call sites" \
  --ac "The skill tool call's SkillsService setup and lookup make no ChaCha checkout on the event-loop thread" \
  --dep TASK-13861
```
Expected: `TASK-<N> - Persona live WebSocket: own PostgreSQL ChaCha checkouts for the on-loop calls (To Do)`.

Under Alternative 3 (B4 and B5 skipped), file the Alternative 2 task titled "Persona live WebSocket: own PostgreSQL ChaCha checkouts per call". Its description also covers the 8 off-loop sites of F2, where 4 sessions exhausted a 10-connection pool through executor threads.

- [ ] **Step 3: Notes**

Fill the bracketed results from Step 1, and `<N>` from Step 2.
```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13861 \
  --append-notes "AC1 evidence (part 1): /audio/chat/stream persistence exercised on PostgreSQL first (tests/Audio/test_ws_audio_chat_stream_postgres.py, 9 concurrent sessions over 3 cached handles). On dev it ran through legacy thread-local checkouts on default-executor threads: 17 checkouts pinned on 9 live threads with a 20-connection pool; PoolTimeout with 10 (re-measured on e521605198). The full caller list, with evidence and fix per row, is Findings F2 of Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md." \
  --append-notes "PR B (fix/chacha-pg-websocket-operations): one operation per /chat/stream persistence step (asyncio.to_thread for the settings upsert; run_in_executor does not carry the operation); one per voice-assistant session start and command; one per /persona/stream call: 8 off-loop sites (chacha_operation inside each asyncio.to_thread callable; the two core helpers join an HTTP request's operation) and 13 on-loop sites (chacha_operation(independent=True)(fn)(...)), and the skill tool call owns only its registry lookup, so a forked skill's LLM call holds no checkout. Residual: a wrapped on-loop call still checks out on the event-loop thread, so when other work holds the whole pool it stalls the loop for up to pool_timeout (2.0 s measured with 9 of 10 connections held elsewhere); TASK-<N> moves the 10 calls in async handlers off the loop. Persona before: off the loop, 4 concurrent sessions held a whole 10-connection pool on executor threads, returned only by the cyclic GC; on the loop, each live session held one checkout idle in a transaction, 10 sessions exhausted the pool, and the 11th blocked the event loop for pool_timeout. Per-call cost: about 5 extra round trips plus a 4-statement reset in the pool's worker. Also fixed a pre-existing race the per-step operations made more frequent: concurrent first sessions of a new user lost persistence when their default-character insert conflicted (dev 6/54 sessions, operations alone 21/54, with the read-back 0/54)." \
  --append-notes "PR B RED/GREEN (RED on dev with PR A): default-character race unit test 1 failed -> 1 passed (33 with test_chat_helpers.py). chat-stream PostgreSQL test 1 failed (PoolTimeout) -> 1 passed x5. voice-assistant PostgreSQL tests 2 failed (['MainThread'] pinned) -> 2 passed. persona off-loop tests 2 failed (8 unwrapped sites; 10 checkouts held) -> 2 passed. persona on-loop guard 1 failed (13 unwrapped sites) -> 1 passed. skill tool call test 1 failed ([['asyncio_0']] held while the skill ran) -> 1 passed. Sweep of the 120 files that import or drive PR B's modules:  [result] (dev baseline: [result])."
```
Expected: the task prints with status `In Progress`.

- [ ] **Step 4: Lint, Bandit, triage, whitespace**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  tldw_Server_API/app/core/Persona/live_conversation.py tldw_Server_API/app/core/Persona/exemplar_runtime.py \
  tldw_Server_API/tests/Chat/unit/test_chat_helpers_default_character_race.py \
  tldw_Server_API/tests/Audio/test_ws_audio_chat_stream_postgres.py \
  tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py \
  tldw_Server_API/app/core/Skills/context_integration.py \
  tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py
$PY -m ruff check --output-format concise tldw_Server_API/app/api/v1/endpoints/persona.py
uvx bandit -q -f json -o "$S/runs/bandit_b.json" tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  tldw_Server_API/app/api/v1/endpoints/voice_assistant.py tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/app/core/Persona/live_conversation.py tldw_Server_API/app/core/Persona/exemplar_runtime.py \
  tldw_Server_API/app/core/Chat/chat_helpers.py tldw_Server_API/app/core/Skills/context_integration.py \
  tldw_Server_API/tests/Chat/unit/test_chat_helpers_default_character_race.py \
  tldw_Server_API/tests/Audio/test_ws_audio_chat_stream_postgres.py \
  tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py \
  tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py
$PY -c "import json,collections;d=json.load(open('$S/runs/bandit_b.json'));print(sorted(collections.Counter((r['filename'].rsplit('/',1)[-1],r['test_id']) for r in d['results']).items()))"
$PY Helper_Scripts/ci/test_quality_triage.py --enforce 2>&1 | grep "test-triage\] OK"
git diff --check origin/dev...HEAD && echo "no trailing whitespace"
```
Expected:
- **ruff:** the first command prints `All checks passed!`. The second prints exactly dev's two findings in `persona.py`, `:4:1: I001` and `:2495:5: SIM114` (dev's `:2494`, one line lower after B4's import), and nothing else (measured). `voice_assistant.py` and `chat_helpers.py` are left out because each has one I001 on dev; B3 Step 4 checked the voice count.
- **Bandit:** `chat_helpers.py` `B311` x1, the same as dev. In the tests: `B101` x1, x3, x4, x5 and x3 (the skill test), plus `B105` x2 in the voice test for its `"valid-token"` strings. `context_integration.py`: nothing (measured).
- **triage:** `OK`.
- `no trailing whitespace`.

- [ ] **Step 5: Commit, review, push, open PR B**

```bash
git add -A backlog/tasks
git commit -m "$(cat <<'EOF'
docs(backlog): TASK-13861 PR B notes; file the persona off-the-loop follow-up

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```
Run `/superpowers:requesting-code-review` in a clean subagent with the branch diff and this plan's PR B section, and address its findings. Then:
```bash
git push -u origin fix/chacha-pg-websocket-operations
gh pr create --base dev --title "fix: WebSocket and persona steps own their PostgreSQL ChaCha checkouts (TASK-13861, 2 of 4)" --body "$(cat <<'EOF'
## Summary
- `/audio/chat/stream` persistence and the voice-assistant WebSocket run each step in its own `chacha_operation`. Before, they held the user's cached handle's thread-local checkouts on default-executor threads or the event-loop thread: 9 concurrent chat-stream sessions pinned 17 checkouts, and a 10-connection pool timed out.
- Every `/persona/stream` call on the session's handle runs inside an operation that returns its checkout when the call returns: 8 off the loop, 13 on it, and the skill tool call owns only its registry lookup, so a forked skill's LLM call holds no checkout. Before, 4 concurrent sessions held a whole 10-connection pool through executor threads, and on the loop each live session held a checkout idle in a transaction, so 10 sessions exhausted the pool and the 11th blocked the event loop for `pool_timeout`. Cost: about 5 extra round trips per call.
- Not fixed here: a wrapped on-loop call still checks out on the event-loop thread, so when other work holds the whole pool it stalls the loop for up to `pool_timeout` (2.0 s measured with 9 of 10 connections held elsewhere). TASK-<N> moves the 10 calls in async handlers off the loop.
- Concurrent first sessions of a new user no longer lose persistence when their default-character inserts collide (dev 6 of 54 sessions; 0 of 54 now).
- PR C covers Jobs and background passes and adds the ADR; PR D covers the test apps.

## Change summary
<!-- The owner writes this section (ADR-004). -->

## Test plan
- [ ] chat-stream PostgreSQL test: dev PoolTimeout, branch passes (5 runs)
- [ ] voice-assistant PostgreSQL tests: dev 2 failed (MainThread pinned), branch 2 passed
- [ ] persona tests: dev 3 failed (8 unwrapped off-loop sites; 10 checkouts held; 13 unwrapped on-loop sites), branch 3 passed
- [ ] skill tool call test: dev fails (the lookup's checkout held while the skill runs), branch passes
- [ ] default-character race unit test; existing chat-stream, voice, persona and chat-helper tests pass
- [ ] the 120 files that import or drive PR B's modules (including `/api/v1/persona/stream`): no new failures against dev
- [ ] ruff no new findings; Bandit unchanged in product files; triage OK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Under Alternative 2 or 3, edit the persona bullets to what was done.

Before merge: the owner's Change summary, `gh variable get MERGE_QUEUE`, and `MERGED` before removing `<scratchpad>/pr-b`.

---

# PR C — `fix/chacha-pg-background-operations`: Jobs and background passes own their checkouts

This PR implements Decision 1's recommendation. If the owner chose the central `WorkerSDK` wrap instead:
- keep C1 and C3;
- replace C2's decorators with v1's Task B4: an independent `chacha_operation` beside `llm_call_root` at `worker_sdk.py:720`, `:934` and `:957`;
- implement option (b) first, transactions serialized per `ConnectionState`. Without it, C2's macro-run test fails under the central wrap, as it did on revision 2 (`('running', [])`);
- rewrite ADR-NNN's Decision and Alternatives to match.

PR C stands alone: with only its product files on dev `2cae53929f`, its three new test files give `12 passed`. Cut it after PR B merges anyway, so that its sweep runs on top of A and B.

Where the REDs below were measured:
- on unchanged dev `2cae53929f`;
- against revision 2's PR C (the plan as written), for the two tests the re-review added;
- C1's, on dev with PRs A and B (revision 2).

None of PR A's or B's changes touches these paths.

```bash
git fetch origin dev --quiet
git worktree add <scratchpad>/pr-c -b fix/chacha-pg-background-operations origin/dev
cd <scratchpad>/pr-c
git log --oneline -1 --grep "each off-loop persona call owns its PostgreSQL checkout"
```
Expected: one commit, PR B's B4. If it prints nothing, PR B has not merged yet (or B4 was skipped under Decision 2's Alternative 3; then grep for B3's subject instead).

### Task C1: the notes-graph cleanup leaves an operation's checkout to the operation (Blocker 1)

**Files:**
- Create: `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_job_postgres_checkouts.py`
- Modify:
  - `tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py`: add `current_operation_is_open` after `chacha_operation` (which ends at `:213`) and before `class ChaChaOperationMiddleware` (`:216`).
  - `tldw_Server_API/app/services/notes_graph_suggestions_worker.py`: the import block (after `:15`) and `_close_database` (`:67`).

**Interfaces:**
- Consumes: the operation release contract, unchanged. Inside an operation, `close_connection()` retires the handle's state, and the next use raises `ClosedChaChaOperationError("The ChaCha operation has finished")`.
- Produces: `current_operation_is_open() -> bool`, which is true when an open `chacha_operation` owns this context's checkouts. `_close_database(db)` returns without closing in that case. Outside an operation it closes, as today.

- [ ] **Step 1: Write the failing test**

Create `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_job_postgres_checkouts.py`:

```python
"""The notes-graph suggestions job on PostgreSQL, run by WorkerSDK (TASK-13861).

Each pipeline step runs on a worker thread and closes its checkout on that thread.
That must keep working when an operation owns the job's checkouts, which retires
the handle's state on an explicit close.
"""

import asyncio
import functools
import threading
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerSDK
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint
from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import JOB_QUEUE, SuggestionAdmissionService
from tldw_Server_API.app.core.Notes_Graph.suggestion_service import SuggestionWorker
from tldw_Server_API.app.services import notes_graph_suggestions_worker as worker

pytestmark = pytest.mark.integration

DATASET_ID = "dataset-1"
SOURCE_ID = "10000000-0000-4000-8000-000000000001"
REVISION = f"sha256:{'a' * 64}"


@pytest.fixture
def admitted_job(pg_database_config, tmp_path, monkeypatch):
    """One admitted suggestion job for owner 1, whose handle draws from a recording pool."""
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned = {}, Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    notes = CharactersRAGDB(tmp_path / "notes.db", client_id="1", backend=backend)
    notes.add_note("Source", "source body", note_id=SOURCE_ID)
    with notes.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (notes.client_id, DATASET_ID),
        )
    jobs = JobManager(tmp_path / "jobs.db")
    note = notes.get_note_by_id(SOURCE_ID, include_deleted=True)
    SuggestionAdmissionService(store=notes.note_graph_suggestion_store, jobs=jobs, owner_user_id="1").admit(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=content_fingerprint(note["title"], note["content"]),
        provider="openai",
        model="model-a",
        capability_revision=REVISION,
        prompt_contract_version="notes-graph-suggestions-v1",
        idempotency_key="request-1",
        now=datetime.now(timezone.utc),
    )
    notes.close_connection()

    async def owner_database(_user_id, client_id=None):  # noqa: ARG001
        return notes

    def no_suggestions(**_kwargs):  # the provider call, the only part not under test
        return SimpleNamespace(relationships=(), tags=(), validation_counts={}, input_tokens=0, output_tokens=0)

    monkeypatch.setattr(worker, "get_chacha_db_for_user_id", owner_database)
    monkeypatch.setattr(
        worker, "resolve_generation_capability",
        lambda **_kwargs: (SimpleNamespace(revision=REVISION, generation_available=True), object()),
    )
    monkeypatch.setattr(worker, "SuggestionWorker", functools.partial(SuggestionWorker, generate=no_suggestions))

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield jobs, outstanding
    finally:
        notes.close_connection()
        pool.close_all()


async def _run_one_job(jobs, handler):
    sdk = WorkerSDK(jobs, worker.build_worker_config(worker_id="notes-graph-test"))
    finished = []

    async def run_once(job):
        try:
            return await handler(job, sdk=sdk)
        finally:
            finished.append(job["id"])
            sdk.stop()

    await asyncio.wait_for(sdk.run(handler=run_once), timeout=60)
    return jobs.get_job(int(finished[0]))


async def test_the_suggestions_job_completes_and_returns_every_checkout(admitted_job):
    jobs, outstanding = admitted_job

    async def handler(job, *, sdk):
        return await worker.handle_notes_graph_suggestions_job(job, jobs=jobs, worker_id=sdk.cfg.worker_id)

    stored = await _run_one_job(jobs, handler)

    assert stored["status"] == "completed", stored.get("error_message")
    assert outstanding() == []


async def test_the_suggestions_job_completes_when_an_operation_owns_its_checkouts(admitted_job):
    jobs, outstanding = admitted_job

    async def handler(job, *, sdk):
        with chacha_operation(independent=True):  # as an owner around the whole job would
            return await worker.handle_notes_graph_suggestions_job(job, jobs=jobs, worker_id=sdk.cfg.worker_id)

    stored = await _run_one_job(jobs, handler)

    assert stored["status"] == "completed", stored.get("error_message")
    assert outstanding() == []
```

The two tests:
- **The plain job** passes on dev. It pins today's per-step cleanup: a completed job, and no checkout left.
- **The job inside an operation** is the case a job owner's operation, or v1's central wrap, creates.

- [ ] **Step 2: Run it to verify it fails**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_job_postgres_checkouts.py 2>&1 | grep -E "^E   |^FAILED|passed|failed"
```
Expected (measured on `e521605198` with PRs A and B, and on unchanged dev):
```
E   AssertionError: The ChaCha operation has finished
E   assert 'failed' == 'completed'
FAILED …::test_the_suggestions_job_completes_when_an_operation_owns_its_checkouts
==== 1 failed, 1 passed, … ====
```

- [ ] **Step 3: Implement**

In `operation_scope.py`, between `chacha_operation` and `class ChaChaOperationMiddleware`, add:

```python
def current_operation_is_open() -> bool:
    """Whether an open ChaCha operation owns this context's PostgreSQL checkouts."""
    owner = _current_operation.get()
    return owner is not None and not owner._closed
```

Why it reads `_closed`, and why not reuse an existing helper:
- It reads `_closed` without the operation's lock. `ChaChaOperation.close()` sets that flag once, and a stale `False` only means the cleanup closes as it does today.
- `current_operation_holds_connection(db)` is not enough here. A step that did not touch the handle would still close it, and the close would retire the handle's state.

In `notes_graph_suggestions_worker.py`, add the import after the `ChaCha_Notes_DB_Deps` import block (`:13-15`):

```python
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import current_operation_is_open
```

and make this the start of `_close_database` (dev `:67`):

```python
def _close_database(db: Any) -> None:
    if current_operation_is_open():
        # The operation returns the checkout when it closes; closing here would retire
        # its state, and the job's next step on this handle would fail.
        return
```
The rest of the function is unchanged.

- [ ] **Step 4: Run it, and the existing notes-graph tests**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_job_postgres_checkouts.py 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Notes_Graph \
  tldw_Server_API/tests/Services/test_notes_graph_suggestions_workers.py 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
```
Expected: `2 passed`. The second command must show no `FAILED` or `ERROR` lines. Planning ran `Notes_Graph/unit/test_suggestion_api.py` and `Services/test_notes_graph_suggestions_workers.py` with all four PRs (`50 passed`), and all of them in the 325-file union sweep (C4 Step 5).

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py \
  tldw_Server_API/app/services/notes_graph_suggestions_worker.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_job_postgres_checkouts.py
git commit -m "$(cat <<'EOF'
fix(notes-graph): leave an operation's checkout to the operation (TASK-13861)

The suggestions job closes its handle after every pipeline step and then
uses it again. Inside a ChaCha operation the close retires the handle's
state, so the next step failed with "The ChaCha operation has finished"
and the job failed. The cleanup now returns early while an operation is
open; that operation returns the checkout when it closes. Outside an
operation nothing changes, and the per-step return keeps the job from
holding a checkout through its provider call.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task C2: each ChaCha-touching job owner owns its checkouts, and the Chat_Macros run owns each call (AC2, Decision 1)

**Files:**
- Create: `tldw_Server_API/tests/Jobs/test_chacha_jobs_workers_postgres.py`
- Modify `tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py`: the imports (`:9`, `:14`) and the helper `owns_chacha_checkouts`.
- Modify `tldw_Server_API/app/core/Chat_Macros/executor.py`: the imports (after `:10`), the macro loader call (`:87`) and `_repo_call` (`:368-371`).
- Modify `tldw_Server_API/app/core/Chat_Macros/jobs.py`:
  - the import (after `:19`);
  - the executor build (`:81`) and the failure-path status write (`:88`) in `handle_chat_macro_job`;
  - `_post_back` (`:163`);
  - `_mark_macro_run_cancelled` (`:281`).
- Modify each module below (dev line numbers):
  - add `from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import owns_chacha_checkouts` directly after the line given, which is its isort position. Where that line is the blank line after `from loguru import logger`, the import becomes the first first-party import.
  - put `@owns_chacha_checkouts` on the line above each function.

| Module (`tldw_Server_API/app/…`) | Import after | Decorated |
|---|---|---|
| `core/Chatbooks/services/jobs_worker.py` | `:51` | `_handle_job` (`:518`) |
| `core/Data_Tables/jobs_worker.py` | `:45` | `_handle_job` (`:895`) |
| `core/Evaluations/recipe_runs_jobs_worker.py` | `:21` | `handle_recipe_run_job_async` (`:1229`) |
| `core/Persona/visual_jobs_worker.py` | `:11` | both nested `_handler`s (`:601`, `:660`), indented to match |
| `core/Sharing/shared_workspace_clone_jobs_worker.py` | `:25` | `handle_shared_workspace_clone_job` (`:487`) |
| `services/research_workspace_output_jobs_worker.py` | `:13` | `process_research_workspace_output_job` (`:51`) |
| `services/study_suggestions_jobs_worker.py` | `:12` | `handle_study_suggestions_job` (`:42`) |
| `services/visual_identity_jobs_worker.py` | `:13` | `handle_visual_identity_import_zip_job` (`:38`) |
| `services/vn_asset_jobs_worker.py` | `:13` | `handle_vn_asset_job` (`:44`) |
| `services/workspace_file_inventory_jobs_worker.py` | `:12` | `handle_workspace_file_inventory_job` (`:47`) |
| `services/writing_annotation_review_jobs_worker.py` | `:21` | `handle_writing_annotation_review_job` (`:50`) |

That is 12 decorated functions in 11 modules, plus `_mark_macro_run_cancelled`.

Not wrapped:
- `handle_chat_macro_job`: its branches write concurrently, so one operation would share one connection and one transaction depth between them (F13). Each of its ChaCha calls owns its own operation instead.
- the notes-graph worker: it returns its checkout after every step (F10);
- `study_pack_jobs_worker.py`: it already opens its own operation;
- `notes_graph_suggestions_maintenance.py`: it closes on the thread that used the handle.

**Interfaces:**
- Consumes: `chacha_operation`.
- Produces: `owns_chacha_checkouts(func=None, *, independent=False)`. It returns `func` wrapped so that each call runs inside a ChaCha operation, which closes when the call returns.
  - By default the operation is the caller's, if one is open, and otherwise its own. With `independent=True` it is always its own. Both `@owns_chacha_checkouts` and `@owns_chacha_checkouts(independent=True)` work.
  - For an `async def` it opens the operation inside the coroutine, so `await` runs inside it.
  - For a plain function it is `chacha_operation(independent=independent)(func)`.
  - Signatures, `__name__` and `__wrapped__` are kept (`functools.wraps`).
- In the Chat_Macros run, each repository call, the macro loader, the executor build, the post-back and the failure-path status write run in their own independent operation. Each returns its checkout when it returns. The executor's and `handle_chat_macro_job`'s signatures are unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tldw_Server_API/tests/Jobs/test_chacha_jobs_workers_postgres.py`:

```python
"""Jobs workers that touch ChaCha return their PostgreSQL checkouts, one family per test (TASK-13861).

Each test runs a real worker's job owner through WorkerSDK.run and stubs only the
work that does not decide where the handle's checkouts go.
"""

import asyncio
import itertools
import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chat_Macros import jobs as chat_macro_jobs
from tldw_Server_API.app.core.Chat_Macros.branch_runner import BranchPromptResult
from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.Chatbooks.services import jobs_worker as chatbooks_worker
from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.services import workspace_file_inventory_jobs_worker as inventory_worker

pytestmark = pytest.mark.integration

WRAPUP_BRANCHES = ["action_items", "decisions", "open_questions", "summary"]


@pytest.fixture
def pg_owner(pg_database_config, tmp_path, monkeypatch):
    """Owner 1's long-lived handle on a pool that records which thread holds each checkout."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned = {}, Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    monkeypatch.setattr(chacha_module, "get_content_backend", lambda _config: backend)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    db = CharactersRAGDB(tmp_path / "owner.db", client_id="1", backend=backend)
    db.close_connection()

    async def owner_database(_user_id, client_id=None):  # noqa: ARG001 - the cached per-user handle
        return db

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield SimpleNamespace(db=db, owner_database=owner_database, outstanding=outstanding, jobs=JobManager(tmp_path / "q.db"))
    finally:
        db.close_connection()
        pool.close_all()


@pytest.fixture
def chatbooks_cache(tmp_path, monkeypatch):
    """The Chatbooks worker's own per-user handle cache, empty and pointed at tmp_path."""
    monkeypatch.setattr(chatbooks_worker, "_DB_CACHE", {})
    monkeypatch.setattr(chatbooks_worker.DatabasePaths, "get_chacha_db_path", staticmethod(lambda _uid: tmp_path / "cb.db"))


@pytest.fixture
def pending_wrapup_run(pg_owner):
    """A pending run of the built-in wrapup macro (four branch steps) for owner 1."""
    with chacha_operation(independent=True):
        repository = ChatMacroRepository(pg_owner.db)
        repository.ensure_ready()
        run = repository.create_run(
            user_id="1",
            macro_name="wrapup",
            macro_command="wrapup",
            macro_source="builtin",
            macro_version=1,
            normalized_args={},
            status="pending",
            surface="chat",
            output_profile="default",
            context_snapshot={
                "conversation_id": "conv-1",
                "messages": [{"id": "m1", "role": "user", "excerpt": "Please wrap this up."}],
                "selected_message_ids": ["m1"],
                "model_selection": {"api_provider": "openai", "model": "gpt-test"},
                "output_profile": "default",
                "acp": {},
            },
            model_selection={"api_provider": "openai", "model": "gpt-test"},
        )
    return run.run_id


class _AnsweringBranchRunner:
    """The LLM call each branch makes, the only part not under test.

    It records which checkouts are out while the branch "calls the LLM": the event-loop
    thread must hold none, or the executor build's checkout would span every branch.
    """

    def __init__(self, outstanding, seen):
        self.outstanding, self.seen = outstanding, seen

    async def run_branch(self, *, prompt, snapshot, model_selection):  # noqa: ARG002
        self.seen.append(self.outstanding())
        return BranchPromptResult(text=f"answer to {prompt[:20]}", citations=[], usage={})


def _overlapping_branch_writes(original):
    """Hold the first branch write's transaction open across the second's, as concurrent branches can.

    The second write enters while the first is open and leaves after the first has
    committed. Branches that share one connection's transaction depth lose later writes.
    """
    first_in, second_in, first_out = threading.Event(), threading.Event(), threading.Event()
    arrivals = itertools.count()

    def upsert_branch(self, *args, **kwargs):
        arrival = next(arrivals)
        if arrival > 1:
            return original(self, *args, **kwargs)
        if arrival == 1:
            first_in.wait(10)
        with self.db.transaction():
            if arrival == 0:
                first_in.set()
                second_in.wait(10)
            else:
                second_in.set()
                first_out.wait(10)
            result = original(self, *args, **kwargs)
        if arrival == 0:
            first_out.set()
        return result

    return upsert_branch


async def _run(jobs, *, domain, handler, cancel_check=None, job_type="job", payload=None):
    jobs.create_job(domain=domain, queue="default", job_type=job_type, payload=payload or {}, owner_user_id="1")
    sdk = WorkerSDK(jobs, WorkerConfig(domain=domain, queue="default", worker_id="family-test", retry_on_exception=False))
    seen = []

    async def handle(job):
        seen.append(job["id"])
        try:
            return await handler(job)
        finally:
            sdk.stop()

    async def check(job):
        try:
            return await cancel_check(job)
        finally:
            seen.append(job["id"])
            sdk.stop()

    await asyncio.wait_for(sdk.run(handler=handle, cancel_check=check if cancel_check else None), timeout=60)
    return jobs.get_job(int(seen[0]))


async def test_a_cached_handle_used_off_the_loop_and_closed_at_job_end(pg_owner, monkeypatch):
    """Family: workspace file inventory, writing review, research output, visual identity, VN assets, study suggestions, clone."""
    monkeypatch.setattr(inventory_worker, "get_chacha_db_for_user_id", pg_owner.owner_database)
    monkeypatch.setattr(inventory_worker, "_validate_job_payload", lambda job: dict(job["payload"]))

    def scan(_payload, db, _allowed_roots, _resolver):  # the scan itself; it reads through the handle
        db.get_character_card_by_name("nobody")
        return {"status": "scanned"}

    monkeypatch.setattr(inventory_worker, "_handle_workspace_file_inventory_job_sync", scan)

    stored = await _run(pg_owner.jobs, domain="workspaces", handler=inventory_worker.handle_workspace_file_inventory_job)

    assert stored["status"] == "completed", stored.get("error_message")
    assert pg_owner.outstanding() == []


async def test_a_cancel_check_that_reads_the_owners_handle(pg_owner, monkeypatch):
    """Family: Chat_Macros' cancel check mirrors the cancellation into the owner's database."""
    monkeypatch.setattr(chat_macro_jobs, "get_chacha_db_for_user_id", pg_owner.owner_database)
    cancelled = SimpleNamespace(
        get_job=lambda _job_id: {"status": "processing", "cancel_requested_at": "now", "payload": {}},
        finalize_cancelled=lambda *_args, **_kwargs: None,
    )

    async def cancel_check(job):
        return await chat_macro_jobs.should_cancel_chat_macro_job(job, job_manager=cancelled)

    async def never_runs(_job):
        raise AssertionError("a cancelled job must not be handled")

    stored = await _run(
        pg_owner.jobs, domain="chat_macros", handler=never_runs, cancel_check=cancel_check,
        payload={"macro_run_id": "run-1", "user_id": "1"},
    )

    assert stored["status"] == "cancelled"
    assert pg_owner.outstanding() == []


async def test_a_handle_the_worker_keeps_and_never_closes(pg_owner, chatbooks_cache, monkeypatch):
    """Family: Chatbooks and Data Tables keep per-user handles in a worker cache; Persona visuals opens one per job."""
    # The worker's own per-user handle cache (_get_db) is under test; ChatbookService is not.
    monkeypatch.setattr(chatbooks_worker, "_get_service", lambda user_id: SimpleNamespace(db=chatbooks_worker._get_db(user_id)))

    async def export(service, _payload, _job_id):  # the export itself; it reads through the handle
        await asyncio.to_thread(service.db.get_character_card_by_name, "nobody")
        return {"status": "exported"}

    monkeypatch.setattr(chatbooks_worker, "_handle_export", export)

    stored = await _run(
        pg_owner.jobs, domain="chatbooks", handler=chatbooks_worker._handle_job,
        payload={"action": "export", "chatbooks_job_id": "export-1"},
    )

    assert stored["status"] == "completed", stored.get("error_message")
    assert pg_owner.outstanding() == []


async def test_a_macro_run_whose_branches_write_concurrently(pg_owner, pending_wrapup_run, monkeypatch):
    """Family: Chat_Macros runs its branches with asyncio.gather; each branch writes its own row."""
    monkeypatch.setattr(chat_macro_jobs, "get_chacha_db_for_user_id", pg_owner.owner_database)
    during_branches = []
    monkeypatch.setattr(
        chat_macro_jobs, "ChatMacroLLMBranchRunner", lambda: _AnsweringBranchRunner(pg_owner.outstanding, during_branches)
    )
    monkeypatch.setattr(ChatMacroRepository, "upsert_branch", _overlapping_branch_writes(ChatMacroRepository.upsert_branch))

    stored = await _run(
        pg_owner.jobs, domain=chat_macro_jobs.CHAT_MACROS_DOMAIN, handler=chat_macro_jobs.handle_chat_macro_job,
        job_type=chat_macro_jobs.CHAT_MACROS_JOB_TYPE, payload={"macro_run_id": pending_wrapup_run, "user_id": "1"},
    )

    assert stored["status"] == "completed", stored.get("error_message")
    with chacha_operation(independent=True):  # read what the job committed, on a fresh checkout
        repository = ChatMacroRepository(pg_owner.db)
        saved = (repository.get_run(pending_wrapup_run).status, sorted(b.step_id for b in repository.list_branches(pending_wrapup_run)))
    assert saved == ("completed", WRAPUP_BRANCHES)
    assert [held for held in during_branches if "MainThread" in held] == []
    assert pg_owner.outstanding() == []
```

There is one test per worker family (F11):
- (a) a cached handle used off the loop and closed at job end;
- (b) a cancel check that reads the owner's handle;
- (c) a handle that the worker keeps and never closes;
- (d) a run whose branches write concurrently: the Chat_Macros run.

Each test runs the real job owner through `WorkerSDK.run`, and stubs only the work that does not decide where the checkouts go.

Test (d) runs the built-in wrapup macro, 4 branch steps, with the real `ChatMacroRepository`, executor and post-back. Its branch runner only answers, standing in for the LLM call. To make the hazard deterministic, `_overlapping_branch_writes` holds the first branch write's transaction open while a second enters, and lets the second leave last. That is the interleaving that loses writes when the two share a connection (F13). The test then checks the run's saved status and branch rows on a fresh checkout.

Test (d)'s branch runner also records which checkouts are out while each branch "calls the LLM". The test asserts that the event-loop thread (`MainThread`) holds none. That pins the executor-build operation: without it, `ensure_ready` and the output-profile read leave the loop thread's checkout out through every branch (measured, below). The post-back's operation is not pinned. The fixture's run has no `conversation_id`, so the post-back returns early, and the job's `finally` returns the loop thread's checkout at job end anyway.

Notes-graph is C1, and study_pack has `test_study_pack_worker_operation_lifecycle.py`. Fixtures hold each test's set-up, so no test makes more than three patches (the test-quality ratchet flags four).

- [ ] **Step 2: Run them to verify they fail**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Jobs/test_chacha_jobs_workers_postgres.py 2>&1 | grep -E "^E   |^FAILED|passed|failed"
```
Expected on unchanged dev (measured on `3ad31cba25`):
```
E   AssertionError: assert ['asyncio_0'] == []      (x3)
E   AssertionError: assert [['MainThread... 'asyncio_3']] == []
FAILED …::test_a_cached_handle_used_off_the_loop_and_closed_at_job_end
FAILED …::test_a_cancel_check_that_reads_the_owners_handle
FAILED …::test_a_handle_the_worker_keeps_and_never_closes
FAILED …::test_a_macro_run_whose_branches_write_concurrently
==== 4 failed, … ====
```
PR A's guard does not help here: the executor threads are alive, as they are in a running server.

In (d) on dev, the saved status and branch rows are intact, and the loop thread holds a checkout during each of the 4 branches (`Left contains 4 more items`).

Against revision 2's PR C, which wrapped `handle_chat_macro_job` in one operation (the plan as written), tests (a)-(c) pass. Test (d) fails with `E   AssertionError: assert ('running', []) == ('completed',...', 'summary'])`: every branch row and the final status were rolled back (measured).

- [ ] **Step 3: Implement**

In `operation_scope.py`, change the imports (`:8-14`) to:

```python
import asyncio
import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock, get_ident
from typing import Any, TypeVar
```

Directly above `current_operation_is_open` (from C1), add the type variable, and directly below it add the helper:

```python
_F = TypeVar("_F", bound=Callable[..., Any])
```
```python
def owns_chacha_checkouts(func: _F | None = None, *, independent: bool = False) -> Any:
    """Run ``func`` inside a ChaCha operation: the caller's, or its own outside one.

    For a unit of work outside HTTP (a job handler, a cancel check, a call that opens
    a handle of its own), so its PostgreSQL checkouts go back to the pool when it
    returns, on whichever thread they were taken. Decorate the ``async def`` itself:
    a lambda that returns a coroutine would leave the operation before the coroutine
    runs, and so would ``@chacha_operation()`` on an ``async def``.

    ``independent=True`` always opens a new operation. Use it for work that a detached
    task can reach after the request that started it has closed its operation.
    """
    if func is None:
        return functools.partial(owns_chacha_checkouts, independent=independent)
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def owned(*args: Any, **kwargs: Any) -> Any:
            with chacha_operation(independent=independent):
                return await func(*args, **kwargs)

        return owned
    return chacha_operation(independent=independent)(func)
```

Then add the import and the decorator in each module in the table. For example, `services/workspace_file_inventory_jobs_worker.py` becomes:

```python
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import owns_chacha_checkouts
```
```python
@owns_chacha_checkouts
async def handle_workspace_file_inventory_job(
```
For `core/Persona/visual_jobs_worker.py`, the decorator goes on each nested `async def _handler(job: dict[str, Any]) -> dict[str, Any]:` at its indentation (four spaces). `_handler` is what `WorkerSDK.run` calls.

**The Chat_Macros run: each call owns its operation.**

In `Chat_Macros/executor.py`, add the import between the standard-library block and the relative imports (after `from typing import Any`, `:10`):

```python
from typing import Any

from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation

from .acp_adapter import resolve_acp_branch_capability, select_branch_strategy
```
Wrap the macro loader call (`:87`):

```python
        macro = await asyncio.to_thread(chacha_operation(independent=True)(self.macro_loader), run)
```
And replace `_repo_call` (`:368-371`):

```python
    async def _repo_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Run one synchronous repository operation outside the event-loop thread.

        Branches call this concurrently, so each call owns its ChaCha operation, and with
        it its own connection and transaction, and returns the checkout when it returns.
        """
        method = getattr(self.repository, method_name)
        return await asyncio.to_thread(chacha_operation(independent=True)(method), *args, **kwargs)
```

In `Chat_Macros/jobs.py`, import both names after the `ChatMacroStorage` import (`:19`):

```python
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation, owns_chacha_checkouts
```
In `handle_chat_macro_job`, which stays undecorated, build the executor inside its own operation (`:81`). The build runs `repository.ensure_ready()` and reads the output profiles on the event-loop thread:

```python
        # The run is not wrapped in one operation: its branches write concurrently, and one
        # operation would give them one connection and one transaction depth. Each ChaCha
        # step owns its own operation instead (here, the executor and post-back).
        executor = chacha_operation(independent=True)(build_chat_macro_executor)(chat_db=chat_db, user_id=user_id)
```
and own the failure-path status write (`:88`):

```python
                await asyncio.to_thread(
                    chacha_operation(independent=True)(executor.repository.update_run_status),
```
In `build_chat_macro_executor`, decorate the post-back (`:163`). The executor calls it synchronously on the event-loop thread, and it opens `chat_db.transaction()`:

```python
    @chacha_operation(independent=True)
    def _post_back(*, run_id: str, final_output: str, post_idempotency_key: str) -> str:
```
Finally, decorate the cancel action (`:281`). Its three `asyncio.to_thread` calls run one after another, so one operation is safe there:

```python
@owns_chacha_checkouts
async def _mark_macro_run_cancelled(
```

Then check that no product file gained a ruff finding, against dev in a scratch worktree:
```bash
git diff --name-only origin/dev -- tldw_Server_API/app > "$S/c_app_files.txt"
$PY -m ruff check --output-format concise $(cat "$S/c_app_files.txt") | grep -c ": [A-Z]"
git worktree add --detach "$S/c_base" origin/dev
(cd "$S/c_base" && $PY -m ruff check --output-format concise $(cat "$S/c_app_files.txt") | grep -c ": [A-Z]")
git worktree remove --force "$S/c_base"
```
Expected: the two counts are equal. Planning compared each of the 29 product files separately, and no file gains a finding.

- [ ] **Step 4: Run them, and the Chat_Macros suite**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Jobs/test_chacha_jobs_workers_postgres.py tldw_Server_API/tests/Chat_Macros 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
$PY Helper_Scripts/ci/test_quality_triage.py --enforce 2>&1 | grep "test-triage\]"
```
Expected:
- `182 passed`: the 4 new tests plus `tests/Chat_Macros`' 178 (measured on `2cae53929f` with all four PRs).
- `[test-triage] OK — no offenses beyond the baseline.` Revision 2's version of this file failed it with `mock_density=1 (new)`. Never raise the baseline to pass.

Check that test (d) pins the build operation: remove the `chacha_operation(independent=True)` around `build_chat_macro_executor` and the decorator on `_post_back`, and rerun (d). It must fail with `E   AssertionError: assert [['MainThread...'MainThread']] == []`, the loop thread holding a checkout through every branch. Then restore both. Measured on `3ad31cba25`: `1 failed` without them, and `188 passed` for this file, the per-call file and `tests/Chat_Macros` with them.

C4's sweep runs every test file that imports a wrapped module.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app tldw_Server_API/tests/Jobs/test_chacha_jobs_workers_postgres.py
git commit -m "$(cat <<'EOF'
fix(jobs): ChaCha-touching job owners own their PostgreSQL checkouts (TASK-13861)

Jobs handlers used per-user ChaCha handles through asyncio.to_thread
outside any operation and closed them, if at all, on the event-loop
thread, so each executor thread they used kept a checkout for the life
of the process. Measured for a cached handle closed at job end, the
Chat_Macros cancel check, the Chatbooks worker's own handle cache and a
4-branch macro run: ['asyncio_0'] (and asyncio_1-3) still held after the job.

A new owns_chacha_checkouts decorator runs each job owner and cancel
action in a ChaCha operation (the caller's, or its own), as the
September design note proposed. It is applied per worker, not centrally
in WorkerSDK: the notes-graph job returns its checkout after every step
and would otherwise hold one through its provider call.

The Chat_Macros run is not wrapped: its branches write concurrently,
and one operation would give them one connection and one transaction
depth, so later writes were silently rolled back (a 4-branch run saved
neither its branches nor its final status). Each of its ChaCha calls
owns an independent operation instead.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task C3: background runners and per-call handles own their checkouts (AC2, Blocker 2)

**Files:**
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_per_call_handles_postgres.py`
- Modify (dev line numbers):
  - `tldw_Server_API/app/services/chatbooks_cleanup_service.py`: the import (after `:9`) and the per-user pass (`:75-78`);
  - `tldw_Server_API/app/services/core_jobs_worker.py`: the import (after `:19`), the loop body (`:90`, `:104`, `:509`);
  - `tldw_Server_API/app/services/ingestion_sources_worker.py`: the import (after `:9`), `_process_sync_job` (`:535`);
  - `tldw_Server_API/app/core/Workflows/adapters/integration/messaging.py`: the import (after `:21`), `run_chatbooks_adapter` (`:606`);
  - `tldw_Server_API/app/core/Notifications/service.py`: the import (after `:9`), `NotificationsService.deliver_chatbook` (`:252`).

**Interfaces:**
- Consumes: `chacha_operation`, and `owns_chacha_checkouts` from C2.
- Produces:
  - each chatbooks cleanup pass per user, and each chatbooks core job, runs in an independent operation;
  - `_process_sync_job` and `deliver_chatbook` run in the caller's operation, or their own. `deliver_chatbook` joins on purpose: `watchlists.py:5821` and `:7725` call it inside a request;
  - `run_chatbooks_adapter` always runs in its own operation (`independent=True`). A request's detached task can run it after the request's operation has closed (F11, Review Focus 3).

  Signatures are unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tldw_Server_API/tests/DB_Management/test_chacha_per_call_handles_postgres.py`:

```python
"""Background code that opens a ChaCha handle per call returns its checkouts (TASK-13861 AC2).

Each case runs the real code that opens the handle, on the shared PostgreSQL content
backend, with garbage collection off: a dropped handle is cyclic garbage, so a
return that waited for the collector would not count. Services that only decide
what to read are stubbed with one read through the handle.
"""

import asyncio
import contextvars
import gc
import threading
from collections import Counter
from contextlib import asynccontextmanager
from types import SimpleNamespace

import aiosqlite
import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.integration


class _ReadingChatbookService:
    """Stands in for ChatbookService: every call reads once through the handle it was given."""

    def __init__(self, user_id, db, user_id_int=None):  # noqa: ARG002 - ChatbookService's signature
        self.db = db
        db.get_character_card_by_name("nobody")

    def _read(self, *_args, **_kwargs):
        self.db.get_character_card_by_name("nobody")
        return 0

    cleanup_expired_exports = cleanup_import_orphans = _read

    def list_export_jobs(self, **_kwargs):
        self._read()
        return []


@pytest.fixture
def shared_backend(pg_database_config, tmp_path, monkeypatch):
    """The shared content backend every per-call handle resolves, on a recording pool."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned = {}, Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    monkeypatch.setattr(chacha_module, "get_content_backend", lambda _config: backend)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    (tmp_path / "user_dbs" / "1").mkdir(parents=True)

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    gc.disable()
    try:
        yield SimpleNamespace(outstanding=outstanding, tmp_path=tmp_path)
    finally:
        gc.enable()
        gc.collect()
        pool.close_all()


async def _chatbooks_cleanup_pass(env, monkeypatch):
    from tldw_Server_API.app.services import chatbooks_cleanup_service as cleanup

    stop = asyncio.Event()

    class Service(_ReadingChatbookService):
        def cleanup_import_orphans(self, *_args, **_kwargs):
            stop.set()  # one pass is enough
            return self._read()

    monkeypatch.setattr(cleanup, "ChatbookService", Service)
    monkeypatch.setattr(cleanup, "_enumerate_user_ids", lambda: [1])
    monkeypatch.setenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "1")
    await asyncio.wait_for(cleanup.run_chatbooks_cleanup_loop(stop), timeout=30)


async def _chatbooks_core_job(env, monkeypatch):
    from tldw_Server_API.app.services import core_jobs_worker as core_worker

    jobs = JobManager(env.tmp_path / "core-jobs.db")
    jobs.create_job(domain="chatbooks", queue="default", job_type="chatbooks", payload={"action": "noop"}, owner_user_id="1")
    stop = asyncio.Event()
    original_fail = jobs.fail_job

    def fail_and_stop(*args, **kwargs):  # the unknown action ends the only job
        stop.set()
        return original_fail(*args, **kwargs)

    monkeypatch.setattr(jobs, "fail_job", fail_and_stop)
    monkeypatch.setattr(core_worker, "JobManager", lambda: jobs)
    monkeypatch.setattr(core_worker, "ChatbookService", _ReadingChatbookService)
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")
    await asyncio.wait_for(core_worker.run_chatbooks_core_jobs_worker(stop), timeout=30)


async def _ingestion_notes_sync(env, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Sources.service import create_source, ensure_ingestion_sources_schema
    from tldw_Server_API.app.services import ingestion_sources_worker as ingestion

    root = env.tmp_path / "allowed" / "docs"
    root.mkdir(parents=True)
    (root / "alpha.md").write_text("# Alpha\n\nbody\n", encoding="utf-8")
    monkeypatch.setenv("INGESTION_SOURCE_ALLOWED_ROOTS", str(env.tmp_path / "allowed"))
    async with aiosqlite.connect(str(env.tmp_path / "sources.sqlite3")) as meta:
        meta.row_factory = aiosqlite.Row
        await ensure_ingestion_sources_schema(meta)
        source = await create_source(
            meta, user_id=1,
            payload={"source_type": "local_directory", "sink_type": "notes", "policy": "canonical", "config": {"path": str(root)}},
        )

        class Pool:
            @asynccontextmanager
            async def transaction(self):
                yield meta

        async def get_pool():
            return Pool()

        jobs = SimpleNamespace(renew_job_lease=lambda *a, **k: None, complete_job=lambda *a, **k: None, fail_job=lambda *a, **k: None)
        monkeypatch.setattr(ingestion, "get_db_pool", get_pool, raising=False)
        await ingestion._process_sync_job(jobs, jid=1, lease_id="lease-1", worker_id="w", source_id=int(source["id"]), user_id=1)


async def _workflow_chatbooks_step(env, monkeypatch, *, context=None):
    from tldw_Server_API.app.core.Chatbooks import chatbook_service
    from tldw_Server_API.app.core.Workflows.adapters.integration import messaging

    monkeypatch.setattr(chatbook_service, "ChatbookService", _ReadingChatbookService)
    monkeypatch.setattr(messaging, "is_test_mode", lambda: False)  # test mode simulates the step
    adapter = messaging.registry.get_adapter("chatbooks")  # what the engine runs (engine.py:1906)
    step = adapter({"action": "list_jobs"}, {"user_id": "1"})
    result = await asyncio.create_task(step, context=context)
    assert "error" not in result, result


async def _workflow_chatbooks_step_after_its_request(env, monkeypatch):
    """A request starts engine.continue_run with asyncio.create_task (workflows.py:3497), which outlives it."""
    with chacha_operation(independent=True):
        inherited = contextvars.copy_context()  # what that task copies; the operation closes with the response
    await _workflow_chatbooks_step(env, monkeypatch, context=inherited)


async def _notification_into_chatbook(env, monkeypatch):
    from tldw_Server_API.app.core.Notifications import service as notifications

    class Documents:
        def __init__(self, db, user_id=None):  # noqa: ARG002
            self.db = db

        def create_manual_document(self, **_kwargs):
            self.db.get_character_card_by_name("nobody")
            return 7

    monkeypatch.setattr(notifications, "DocumentGeneratorService", Documents)
    result = notifications.NotificationsService(user_id=1).deliver_chatbook(title="t", content="c")
    assert result.status == "stored", result


@pytest.mark.parametrize(
    "site",
    [
        _chatbooks_cleanup_pass, _chatbooks_core_job, _ingestion_notes_sync, _workflow_chatbooks_step,
        _workflow_chatbooks_step_after_its_request, _notification_into_chatbook,
    ],
    ids=[
        "chatbooks-cleanup", "chatbooks-core-jobs", "ingestion-notes-sink", "workflows-chatbooks",
        "workflows-chatbooks-after-its-request", "notifications-chatbook",
    ],
)
async def test_a_per_call_handle_returns_its_checkouts_without_waiting_for_gc(shared_backend, monkeypatch, site):
    await site(shared_backend, monkeypatch)

    assert shared_backend.outstanding() == []
```

The fixture disables garbage collection, so this is the test that does not call `gc.collect()` (the B2 ruling). A dropped handle is cyclic garbage, so only an operation, or a close on the thread that used the handle, can pass it. PR A's backstop cannot.

Both Workflows cases run the adapter that `registry.get_adapter("chatbooks")` returns, which is what the engine runs (`engine.py:1906`), not the module attribute. With the decorator placed above `@registry.register`, both fail: `[workflows-chatbooks]` with `['MainThread']` and the after-request case with `chatbooks_error` (measured). Placed as Step 3 shows, both pass.

The `[workflows-chatbooks-after-its-request]` case runs the adapter in a context copied inside an operation that has since closed. That is what `asyncio.create_task(engine.continue_run(...))` copies inside a request (`workflows.py:3497`, `:4228`, `:4350`, `:4566`). It pins M6 of the re-review: the adapter must open its own operation there, not join the closed one.

The MCP `chats_module` and `rpg_module` sites that the review listed are not in it. Each opens and closes its handle inside one `asyncio.to_thread` callable (F2), and `close_all_connections()` on the shared backend returns that thread's checkout (A1's MCP test).

- [ ] **Step 2: Run them to verify they fail**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_per_call_handles_postgres.py 2>&1 | grep -E "^E   |^FAILED|passed|failed"
```
Expected on unchanged dev `2cae53929f`:
```
E   AssertionError: assert ['MainThread'] == []      (x5)
E   AssertionError: {'error': 'chatbooks_error'}
FAILED …[chatbooks-cleanup]
FAILED …[chatbooks-core-jobs]
FAILED …[ingestion-notes-sink]
FAILED …[workflows-chatbooks]
FAILED …[workflows-chatbooks-after-its-request]
FAILED …[notifications-chatbook]
==== 6 failed, … ====
```
On dev the adapter, running after its request, catches the `ClosedChaChaOperationError` and returns `chatbooks_error`. Against revision 2's PR C (the plan as written, a joining decorator), the five others pass, and the after-request case fails with `E   tldw_Server_API.app.core.DB_Management.chacha.operation_scope.ClosedChaChaOperationError: The ChaCha operation has finished` (measured).

- [ ] **Step 3: Implement**

`chatbooks_cleanup_service.py`: add the import after the `ChatbookService` import (`:9`):

```python
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
```
and put each user's pass in its own operation (dev `:75-78`):

```python
            for user_id in _enumerate_user_ids():
                with chacha_operation(independent=True):  # this user's pass owns its checkouts
                    db = _build_chacha_db_for_user(user_id)
                    svc = ChatbookService(str(user_id), db, user_id_int=user_id)
                    deleted_exports += svc.cleanup_expired_exports()
                    deleted_imports += svc.cleanup_import_orphans()
```

`core_jobs_worker.py`: add the same import after the `ChatbookService` import (`:19`). Then give each loop iteration a scope (`contextlib` is already imported at `:4`). Before the iteration's `try:` (dev `:90`):

```python
        job_scope = contextlib.ExitStack()  # closed after every iteration, however it ends
        try:
```
Before `db = _build_chacha_db_for_user(...)`'s `try:` (dev `:104`, after the `# Build a per-user service` comment):

```python
            # Build a per-user service
            # This job owns the ChaCha checkouts it takes, on whichever thread.
            job_scope.enter_context(chacha_operation(independent=True))
            try:
                db = _build_chacha_db_for_user(str(owner))
```
At the end of that `try`'s handlers, after the last `except` block's `await asyncio.sleep(poll_sleep)` (dev `:509`), add:

```python
        finally:
            job_scope.close()
```
Why an `ExitStack` and not a `with` block:
- the operation opens only once a job is acquired, so idle polls take no checkout;
- every exit from the iteration (`continue`, an exception, normal flow) closes it;
- the 400-line body is not re-indented.

`ingestion_sources_worker.py`: add the `owns_chacha_checkouts` import as the first first-party import, directly after the blank line that follows `from loguru import logger` (dev `:9`), and decorate:

```python
@owns_chacha_checkouts
async def _process_sync_job(
```

`messaging.py`: add the `owns_chacha_checkouts` import after the `prompt_template_manager` import (`:21`). Then decorate with an independent operation, **between `@registry.register(...)` and `async def`** (dev `:598-606`). The registry stores the function it is handed (`_registry.py:120-131`), and the Workflows engine runs `registry.get_adapter("chatbooks")` (`engine.py:1906`). Above `@registry.register`, the decorator would wrap only the module attribute, and the engine would still run the unwrapped function:

```python
@registry.register(
    "chatbooks",
    category="integration",
    description="Manage chatbooks",
    parallelizable=False,
    tags=["integration", "chatbooks"],
    config_model=ChatbooksConfig,
)
# Independent: a request's detached task (engine.continue_run) can run this after the
# request has closed its operation.
@owns_chacha_checkouts(independent=True)
async def run_chatbooks_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
```

`Notifications/service.py`: add the `owns_chacha_checkouts` import after the `document_generator` import (`:9`), and decorate the method (a plain function, so the operation opens and closes around each call):

```python
    @owns_chacha_checkouts
    def deliver_chatbook(
```

- [ ] **Step 4: Run them to verify they pass, and the runners' own tests**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_per_call_handles_postgres.py \
  tldw_Server_API/tests/Jobs/test_chacha_jobs_workers_postgres.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_job_postgres_checkouts.py 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
```
Expected: `12 passed`: 6 + 4 + 2, measured with all four PRs (`3ad31cba25`) and with PR C alone (`2cae53929f`).

Check that the Workflows cases see what the engine runs: move the comment and `@owns_chacha_checkouts(independent=True)` above `@registry.register(`, and rerun the per-call file. It must give `2 failed, 4 passed`: `[workflows-chatbooks]` with `assert ['MainThread'] == []`, and `[workflows-chatbooks-after-its-request]` with `{'error': 'chatbooks_error'}`. Then restore the placement (measured on `3ad31cba25`).

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app tldw_Server_API/tests/DB_Management/test_chacha_per_call_handles_postgres.py
git commit -m "$(cat <<'EOF'
fix(chacha): background runners and per-call handles own their checkouts (TASK-13861)

The chatbooks core Jobs worker, the chatbooks cleanup loop, the
ingestion-sources worker, the Workflows chatbooks adapter and the
chatbook notification each opened a ChaCha handle per call on the
shared PostgreSQL backend and never closed it. The handle is cyclic
garbage, so its checkout came back only when the cyclic GC ran, which in
a running server can take arbitrarily long. With garbage collection
disabled, each left ['MainThread'] holding a checkout.

Each now runs in a ChaCha operation (one per user pass, per job, or per
call), which returns its checkouts when it closes. The Workflows adapter
always opens its own: a request's detached task can run it after the
request's operation has closed, and joining that would raise.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task C4: ADR-NNN, docs, sweep, the option (b) follow-up, notes, and open PR C

**Files:**
- Create: `Docs/ADR/NNN-chacha-postgres-checkouts-owned-by-entry-points.md`
- Modify:
  - `Docs/ADR/README.md`: an index row after the last ADR row;
  - `Docs/Code_Documentation/Databases/ChaChaNotes_DB.md`: the "Thread Safety" section (`:193-197`);
  - `Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md`: replace ADR-NNN.
- Modify (generated): `Docs/Published/ADR/…` and `Docs/Published/Code_Documentation/Databases/ChaChaNotes_DB.md`.
- Backlog: `TASK-13861`, and one new follow-up task (option (b)).

**Interfaces:**
- Consumes: Tasks C1-C3, and the owner's answers to Decisions 1 and 2.
- Produces: PR C into `dev`, with ADR-NNN numbered.

- [ ] **Step 1: Number the ADR**

```bash
git fetch origin dev --quiet
$S/adr_next.sh
```
Use whatever the scan prints. Measured on `2cae53929f` it printed `078`: the sibling plan's #3288 holds ADR-078. If #3288 has merged, its ADR is on dev and the scan still prints 078 or higher. Set the next number:

```bash
export ADR=$(printf '%03d' $(( 10#$($S/adr_next.sh) + 1 )))   # the printed value + 1, three digits
echo "$ADR"
ls Docs/ADR | grep -c "^$ADR-" ; gh pr list --state open --search "ADR-$ADR" --json number --jq length
```
Expected: the number (`079` while #3288 holds 078), then `0` and `0`.

How the number gets everywhere:
- The ADR, the index row and the guide are written below with the literal `NNN`, and Step 3 replaces it in those files and in the plan copy.
- The notes and the commit message expand `$ADR` themselves.
- The PR body is piped through `sed`.
- PR A's backlog note keeps "ADR-NNN, numbered in PR C". Backlog notes are append-only, so it stays as written; Step 7's note records the number.

- [ ] **Step 2: Write ADR-NNN** (Status `Accepted` only after the owner adopts Decision 1; otherwise `Proposed`)

Create `Docs/ADR/$ADR-chacha-postgres-checkouts-owned-by-entry-points.md` (the file name uses the number; the content below still says NNN until Step 3):

```markdown
# ADR-NNN: Every non-HTTP entry point owns its PostgreSQL ChaCha checkouts

**Status:** Accepted
**Date:** 2026-10-20
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13861
**Related spec/plan:** `Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md`, `Docs/superpowers/plans/2026-10-18-test-reliability-pg-import-and-admin-health.md`

## Decision

On PostgreSQL, ChaCha work outside HTTP runs inside a `chacha_operation` opened by the unit of work that owns it:
- one per WebSocket step or persona session call, never one per connection;
- one per Jobs job owner and cancel action, through `@owns_chacha_checkouts` on that function, not centrally in `WorkerSDK`;
- one per pass, job or call in other background runners.

HTTP keeps `ChaChaOperationMiddleware`. Inside one operation:
- **Never run ChaCha transactions concurrently.** All its threads share one connection and one transaction depth, so overlapping `db.transaction()` blocks lose later writes. Code that fans out ChaCha work (`asyncio.gather` over `asyncio.to_thread`) gives each call its own `chacha_operation(independent=True)`, as the Chat_Macros executor does.
- **Never close a handle and then use it again.** A cleanup helper that may run inside an operation checks `current_operation_is_open()`.

Work that a request's detached task can reach after the request ends opens its own operation: `@owns_chacha_checkouts(independent=True)`.

The thread-local checkout outside an operation is a fallback. Its return is queued, lock-free, when its thread exits (at once) or when its handle is collected (when the cyclic GC runs). The next ChaCha checkout or `close_connection()` in the process makes the return. `CharactersRAGDB.close_all_connections()` never closes the shared content pool.

## Context

`chacha_operation` and `ChaChaOperationMiddleware` (September 2026) gave HTTP requests ownership of their checkouts. Everything else fell back to one pooled checkout per thread per handle, which nothing returned. TASK-13861 measured what that cost on PostgreSQL:
- A thread's exit lost its pool slot for good, because psycopg's `Connection.__del__` skips pooled connections.
- Long-lived threads (the event loop, default-executor threads) kept checkouts after `/audio/chat/stream` and voice-assistant sessions, persona calls and jobs. Nine chat-stream sessions pinned 17 checkouts. Four persona sessions held a whole 10-connection pool through executor threads; on the event-loop thread, ten held it and the eleventh stalled the event loop.
- Per-call handles that were never closed returned their checkouts only when the cyclic GC ran, because every handle is in a reference cycle.
- One MCP tool call closed the shared content pool through `close_all_connections()`.
- A first design put the Chat_Macros run in one operation. Its concurrent branches then shared one connection, and later writes were rolled back.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| One operation around every `WorkerSDK` handler and cancel check | It shares one connection across everything a job does concurrently, so the Chat_Macros run's branches lost writes (7 of 50 runs in a replay). The notes-graph job closes and reuses its handle after every step, so every job failed inside the operation; with that cleanup made operation-aware, it holds a checkout, idle in a transaction, through its provider call instead of none. It also misses the runners that do not use `WorkerSDK`. The September design note: "Do not globally wrap every Jobs SDK handler as an implicit DB transaction". |
| Return the thread-local checkout after every command | `get_connection()` hands the raw connection to callers, and `execute_query` defaults to `commit=False`. A per-command return would give the pool a connection still in use, or roll back a caller's pending write. |
| Raise when PostgreSQL work runs outside an operation | Breaks every single-thread script and test that uses a handle directly, where the thread-local path is correct. |
| One operation per WebSocket connection | A session lasts hours, and its checkout would sit idle in a transaction, holding locks, for that long. |
| Return the checkout from the finalizer itself | A collection can run the finalizer on a thread that holds psycopg_pool's task-queue lock, which `putconn` needs: a self-deadlock. |

## Consequences

- **New entry points.** A new non-HTTP entry point that touches ChaCha (a worker, a WebSocket, a background loop, or a service that opens a handle per call) does two things:
  - it wraps its unit of work in `chacha_operation(independent=True)` or decorates it with `@owns_chacha_checkouts`. If it fans out ChaCha transactions, it gives each call its own operation instead;
  - it adds a PostgreSQL test that checks no checkout is left.
- **Threads.** `run_in_executor` does not carry the operation into the worker thread. Use `asyncio.to_thread`, or open the operation inside the callable. `@chacha_operation(...)` decorates plain functions only; async code uses `with` or `@owns_chacha_checkouts`.
- **Detached tasks.** A detached task that outlives the step or job that started it must open its own operation; otherwise it gets `ClosedChaChaOperationError`.
- **Cost.** Each per-call operation costs about 5 extra database round trips, plus a reset in the pool's worker. One that runs on the event-loop thread checks out synchronously there, so when other work holds the whole pool it can stall the loop for up to `pool_timeout`. Calls in async code therefore go through `asyncio.to_thread` where they can; the persona session's remaining on-loop calls are TASK-<N>.
- **The fallback.** Checkouts from code nobody has wrapped still come back on thread exit, at the next ChaCha call, and only through the cyclic GC when a handle is dropped.
- **Concurrent transactions are a rule, not a guard.** `BackendManagedTransaction` does not serialize transactions per connection state, so a future fan-out inside one operation would lose writes silently. TASK-<P> tracks that guard.
- **An explicit shared backend.** A handle that is given the shared backend explicitly (`DB_Manager.create_chacha_database(backend=…)`, no callers today) does not know the backend is shared, so `close_all_connections()` on it would still close the shared pool.
```

Edit the ADR to match the owner's answers:
- If the owner chose the central wrap for Decision 1, swap the first Alternatives row into the Decision, with option (b) as its prerequisite, and record per-worker operations as the rejected alternative with Decision 1's costs.
- Under Decision 2's Alternative 2 or 3, add to Consequences: "The persona live WebSocket still holds one loop-thread checkout per live session until the session ends (TASK-<N>)."
- Fill `TASK-<P>` with Step 7's follow-up id, and `TASK-<N>` with B6 Step 2's.

Add the index row to `Docs/ADR/README.md` after the last ADR row:

```markdown
| [ADR-NNN](NNN-chacha-postgres-checkouts-owned-by-entry-points.md) | Accepted | On PostgreSQL each non-HTTP unit of work (a WebSocket step or persona call, a job owner or cancel action, a background pass) owns its ChaCha checkouts through `chacha_operation` or `@owns_chacha_checkouts`, not centrally in `WorkerSDK`; ChaCha transactions never run concurrently inside one operation; the thread-local fallback is returned on thread exit, a dropped handle only at cyclic GC, and `close_all_connections()` never closes the shared content pool. |
```
#3288 (the sibling plan's ADR-078) also appends a row after the last row. Whichever of the two PRs merges second rebases and keeps both rows, in number order.

- [ ] **Step 3: Update the ChaCha DB guide, and number everything**

In `Docs/Code_Documentation/Databases/ChaChaNotes_DB.md`, append this bullet to the end of the "Thread Safety" section, after the WAL bullet at `:197`:

```markdown
*   **PostgreSQL:** a pooled connection is owned by a `chacha_operation` (`DB_Management/chacha/operation_scope.py`). `ChaChaOperationMiddleware` opens one per HTTP request; outside HTTP each unit of work opens its own: a WebSocket step or persona call, a job owner or cancel action (`@owns_chacha_checkouts`), or a background pass. All the threads of one operation share one connection, so never run ChaCha transactions concurrently inside it: code that fans out gives each call its own `chacha_operation(independent=True)`. Outside an operation a handle falls back to one checkout per thread. It goes back to the pool at the next ChaCha call after the thread exits, but a dropped handle's only when the cyclic GC collects it, so new background entry points must own their work ([ADR-NNN](../../ADR/NNN-chacha-postgres-checkouts-owned-by-entry-points.md)). `close_all_connections()` never closes the shared content pool.
```

Then replace `NNN` everywhere it was written, including the plan copy committed in PR A:

```bash
sed -i '' -e "s/ADR-NNN/ADR-$ADR/g" -e "s/NNN-chacha-postgres/$ADR-chacha-postgres/g" \
  "Docs/ADR/$ADR-chacha-postgres-checkouts-owned-by-entry-points.md" Docs/ADR/README.md \
  Docs/Code_Documentation/Databases/ChaChaNotes_DB.md \
  Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md
grep -n "NNN" "Docs/ADR/$ADR-chacha-postgres-checkouts-owned-by-entry-points.md" Docs/ADR/README.md \
  Docs/Code_Documentation/Databases/ChaChaNotes_DB.md
grep -c "ADR-NNN" Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md
ls Docs/Code_Documentation/Databases/../../ADR/$ADR-chacha-postgres-checkouts-owned-by-entry-points.md
```
Expected:
- the first `grep` prints nothing;
- the second prints `0`. Bare `NNN` in the plan copy's numbering instructions stays, as history;
- the `ls` resolves the guide's relative link.

- [ ] **Step 4: Refresh the published mirror, stage, then run the docs tests (Minor 3)**

```bash
bash Helper_Scripts/refresh_docs_published.sh
git add "Docs/ADR/$ADR-chacha-postgres-checkouts-owned-by-entry-points.md" Docs/ADR/README.md \
  Docs/Code_Documentation/Databases/ChaChaNotes_DB.md Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md Docs/Published
git status --short Docs/Published
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed)"
```
Expected:
- `A  Docs/Published/ADR/$ADR-…md`, `M  Docs/Published/ADR/README.md` and `M  Docs/Published/Code_Documentation/Databases/ChaChaNotes_DB.md`;
- `212 passed` with no `FAILED` lines. Measured on `2cae53929f`, and again on `3ad31cba25` with this revision's ADR text, numbered 078, after staging. Before staging, `test_refresh_output_matches_tracked_published_files` fails on the new file (the review).

- [ ] **Step 5: Sweep every test file that imports what PR C changed (the global-default rule)**

```bash
git grep -lE 'WorkerSDK|worker_sdk|notes_graph_suggestions_worker|chatbooks_cleanup_service|core_jobs_worker|ingestion_sources_worker|integration(\.| import )messaging|Notifications(\.| import )service|study_suggestions_jobs_worker|research_workspace_output_jobs_worker|visual_identity_jobs_worker|vn_asset_jobs_worker|workspace_file_inventory_jobs_worker|writing_annotation_review_jobs_worker|shared_workspace_clone_jobs_worker|Data_Tables(\.| import )jobs_worker|Chatbooks\.services(\.| import )jobs_worker|recipe_runs_jobs_worker|Persona(\.| import )visual_jobs_worker|Chat_Macros(\.| import )(jobs|executor)|chacha\.operation_scope' \
  -- 'tldw_Server_API/tests/*test_*.py' > "$S/c_callers.txt"
wc -l < "$S/c_callers.txt"
$S/sweep.sh "$S/c_callers.txt" "$S/runs/c_sweep" 6 | tee "$S/runs/c_sweep.summary"
```
Expected: `117`, measured on `3ad31cba25` with PRs A to C's tests in place:
- dev's 111;
- A's locking-read test, and B's chat-stream and skill tests, which import `chacha.operation_scope`;
- C's three tests.

There must be no `FAILED` or `ERROR` lines beyond the dev baseline.

**What planning measured, and how the union was built (M7).**
- **The lists.** Each list was built with this plan's own commands, with all four PRs applied and their 13 new test files present. The lists are A4 Step 1's, B6 Step 1's, this step's, D2 Step 4's and D1's `testapps.txt`.
- **The union.** The union of those lists, plus the 13 new test files, minus `RAG/test_restricted_postgres_media_retrieval.py`, is 334 files (`3ad31cba25`). The RAG file hangs on dev; D3 files it.

| List | Files |
|---|---|
| A4 (`close_all_connections` over PostgreSQL ChaCha) | 108 |
| B6 (PR B's modules, `persona/stream`, Skills' `context_integration`) | 120 |
| C4 (PR C's modules; after PR D's codemod, the 30 test apps also import `chacha.operation_scope`) | 141 |
| D2 (`pg_restricted_backend` users) | 13 |
| D1 (`testapps.txt`) | 30 |
| Union, plus the 13 new tests, minus the RAG file | 334 |

**How the count got here.**
- Revision 3's union was 325. The re-review counted 324, because its C pattern lacked `Chat_Macros(\.| import )executor`. Revision 2's "349" came from a broader scratch pattern, not from these commands.
- The third review's rulings added 9 files:
  - the 7 that drive `/api/v1/persona/stream` (B4, B6);
  - `Skills/integration/test_skill_mcp_integration.py`;
  - B5's new skill test.

**What ran.**
- The 325 ran in 8 shards with all four PRs applied (revision 3, results below).
- The 9 added files ran with all four PRs and the third review's changes, inside B5 Step 4's 18-file persona and skills list: `154 passed` and `174 passed`, with no failure (`3ad31cba25`).

| Shard | All four PRs (revision 3, `2cae53929f`, 325 files) |
|---|---|
| aa | `505 passed` (it includes `Admin_Webhooks/test_recovery_backend_matrix.py`, which stalled in the re-review's run) |
| ab | `1339 passed, 2 skipped` |
| ac | `1114 passed` |
| ad | `1505 passed, 5 skipped` |
| ae | `716 passed` |
| af | `781 passed` |
| ag | `1045 passed, 10 skipped` |
| ah | `1 failed, 792 passed, 6 skipped`: `TTS/test_tts_module.py::TestStreamingAudioWriter::test_init_valid_formats`, `ModuleNotFoundError: No module named 'av'`, which fails the same way on dev (env-only: no PyAV) |

In all: 7797 passed, 23 skipped, and one env-only failure; no failure from the change.

- [ ] **Step 6: Lint, Bandit, triage, whitespace**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_job_postgres_checkouts.py \
  tldw_Server_API/tests/Jobs/test_chacha_jobs_workers_postgres.py \
  tldw_Server_API/tests/DB_Management/test_chacha_per_call_handles_postgres.py
uvx bandit -q -f json -o "$S/runs/bandit_c.json" $(git diff --name-only origin/dev -- tldw_Server_API/app) \
  tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_job_postgres_checkouts.py \
  tldw_Server_API/tests/Jobs/test_chacha_jobs_workers_postgres.py \
  tldw_Server_API/tests/DB_Management/test_chacha_per_call_handles_postgres.py
$PY -c "import json,collections;d=json.load(open('$S/runs/bandit_c.json'));print(sorted(collections.Counter((r['filename'].rsplit('/',1)[-1],r['test_id']) for r in d['results']).items()))"
$PY Helper_Scripts/ci/test_quality_triage.py --enforce 2>&1 | grep "test-triage\]"
git diff --check origin/dev...HEAD && echo "no trailing whitespace"
```
Expected:
- ruff `All checks passed!` for the new tests (C2 Step 3 compared the product files);
- Bandit reports only `B101`, in the three new tests: x4 (notes-graph), x10 (families) and x3 (per-call). It reports nothing in the 20 product files (measured, the same as dev);
- `[test-triage] OK — no offenses beyond the baseline.` (measured with all 13 new test files; the baseline untouched);
- `no trailing whitespace`.

- [ ] **Step 7: File the option (b) follow-up, then notes and ACs**

Compute a free id `<P>` with `$S/task_next.sh` (it is above `<N>`, filed in B6; `<P>` is the printed value + 1), then:
```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create \
  "ChaCha operations: serialize managed transactions per connection state, or fail loudly on overlap" \
  --id TASK-<P> -l chacha,postgres,bug \
  -d "Found by TASK-13861's review. Inside one chacha_operation every thread shares one ConnectionState and one physical connection, and BackendManagedTransaction keeps tx_depth on that state without a lock: __enter__ reads it (ChaChaNotes_DB.py:731 on origin/dev 2cae53929f) and __exit__ writes back the value it read (:785). Two overlapping db.transaction() blocks on two threads leave the depth at 1, so every later transaction in the operation is unmanaged, never commits, and is rolled back when the operation returns its checkout. Measured: a Chat_Macros run wrapped in one operation lost every branch row and its final status ('running', []); a replay of its branch fan-out lost writes in 7 of 50 runs. TASK-13861 avoids it by rule (ADR-$ADR; the Chat_Macros executor gives each call an independent operation). No HTTP route was found that overlaps managed transactions on one handle (plan Findings F13, static audit). Request-path fan-outs that read one ChaCha handle concurrently, to re-check if they ever write: core/RAG/rag_service/unified_pipeline.py:1699, :1736, :4819, :5306, :9456, :9488 and database_retrievers.py:2613, :5014 (notes and chat-history retrievers through execute_query, no managed transaction). Fix: a per-state lock taken in __enter__ and released in __exit__, reentrant per thread, or a loud error on overlap; it changes a pinned transaction contract, so it needs its own tests." \
  --ac "Two overlapping db.transaction() blocks on one ConnectionState from two threads either serialize or raise; neither loses a write (PostgreSQL test)" \
  --ac "test_chacha_operation_scope.py and the HTTP operation lifecycle tests still pass" \
  --dep TASK-13861
```
Expected: `TASK-<P> - ChaCha operations: serialize managed transactions per connection state, or fail loudly on overlap (To Do)`.

Then the notes. Fill the bracketed results from Steps 4 and 5:
```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13861 \
  --append-notes "PR C (fix/chacha-pg-background-operations): Jobs own their checkouts per worker, at each job owner and cancel action, through a new @owns_chacha_checkouts (12 modules, 13 functions; plus 3 runners in C3), not centrally in WorkerSDK. Jobs audit (plan F11): of the 14 ChaCha-reaching worker modules, only the notes-graph suggestions job closes and then reuses its handle inside a job (its cleanup now returns early while an operation is open, and the job owner is left unwrapped: 0 checkouts held during the provider call, 1 if wrapped), and only the Chat_Macros run runs ChaCha transactions concurrently (its branches; it is left unwrapped and each repository call, the macro loader, the executor build, the post-back and the failure-path write own an independent operation, because one shared operation lost every branch row and the final status). Non-WorkerSDK runners: the chatbooks core Jobs worker (one operation per job), the chatbooks cleanup loop (one per user pass), the ingestion-sources worker, the Workflows chatbooks adapter (independent: a request's detached task can run it after the request) and NotificationsService.deliver_chatbook (one per call). MCP chats_module and rpg_module already close on the thread that used the handle. HTTP routes audited for concurrent managed transactions on one handle: none found (F13); option (b) filed as TASK-<P>." \
  --append-notes "PR C RED/GREEN (RED on unchanged dev 2cae53929f; the two re-review tests also against revision 2's PR C): notes-graph job through WorkerSDK.run on PostgreSQL: plain 1 passed (pin), inside an operation 1 failed ('The ChaCha operation has finished') -> 2 passed. Worker families through WorkerSDK.run: 4 failed (['asyncio_0'] x3; in the macro run the event-loop thread held a checkout during every branch) -> 4 passed; the macro-run test failed against revision 2 with ('running', []), and fails with the executor-build and post-back operations removed. Per-call handles with garbage collection disabled: 6 failed (['MainThread'] x5; the Workflows step after its request returned chatbooks_error, and ClosedChaChaOperationError against revision 2) -> 6 passed; the Workflows cases run the adapter the engine resolves (registry.get_adapter), so a decorator placed above @registry.register fails them. Sweep of the 117 files that import PR C's modules: [result] (dev baseline: [result]). Docs tests after refresh and staging: [result]. Test-quality triage: OK." \
  --append-notes "ADR: required and added, Docs/ADR/$ADR-chacha-postgres-checkouts-owned-by-entry-points.md (worker ownership and persistence rule; ADR-$ADR, numbered by a paged scan of dev and open PRs; PR A's note says ADR-NNN and stays as written, notes being append-only). Governing: ADR-020 (shared content backend), ADR-006 (Bandit), ADR-059 (backlog-py), ADR-004 (owner's Change summary)." \
  --append-notes "AC1: /chat/stream exercised on PostgreSQL first (PR B), and every production caller that reaches the legacy path is listed with evidence and fix in the plan's Findings F2 and F11. AC2: every listed caller now runs inside chacha_operation or returns its checkout on the thread that took it; each has a PostgreSQL test that fails on dev. Accepted bound: code nobody has wrapped falls back to the thread-exit return (at the next ChaCha call) and, for dropped handles, the GC-timed backstop (ADR-$ADR)." \
  --check-ac 1
```
Then check AC2 only if the owner took Decision 2's recommendation, so that B5 is done, and record the residual B5 leaves:
```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13861 \
  --append-notes "AC2 under Decision 2's recommendation (B5 done): a wrapped on-loop persona call holds no slot between calls, but still checks out on the event-loop thread, so when other work holds the whole pool it can stall the loop for up to pool_timeout; TASK-<N> moves those calls off the loop." \
  --check-ac 2
```
Under Alternative 2 or 3, leave AC2 unchecked instead, and append the bound that stops it:
```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13861 \
  --append-notes "AC2 left unchecked by the owner's Decision 2: the persona live WebSocket still holds one loop-thread checkout per live session, idle in a transaction (under Alternative 3 also its executor-thread checkouts); 10 sessions exhaust a 10-connection pool and the 11th blocks the event loop for pool_timeout. Tracked as TASK-<N>."
```
Expected: TASK-13861 prints with ACs #1, #3 and #5 checked, and #2 checked or not as above. Its status is `In Progress`.

- [ ] **Step 8: Commit, review, push, open PR C**

```bash
git add -A backlog/tasks
git commit -m "$(cat <<EOF
docs(adr): ADR-$ADR, non-HTTP entry points own their PostgreSQL ChaCha checkouts (TASK-13861)

Records the rule PRs B and C implement, updates the ChaCha DB guide,
refreshes Docs/Published, numbers the ADR in the committed plan, files
the transaction-serialization follow-up, and records AC1 and AC2
evidence on TASK-13861.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
git diff --check origin/dev...HEAD && echo "no trailing whitespace"
```
This heredoc is unquoted so that `$ADR` expands; it contains no backticks.

Run `/superpowers:requesting-code-review` in a clean subagent with the branch diff, ADR-$ADR and this plan's PR C section, and address its findings. Then:
```bash
git push -u origin fix/chacha-pg-background-operations
gh pr create --base dev --title "fix: Jobs and background passes own their PostgreSQL ChaCha checkouts; ADR-$ADR (TASK-13861, 3 of 4)" --body "$(sed "s/ADR-NNN/ADR-$ADR/g" <<'EOF'
## Summary
- A new `@owns_chacha_checkouts` runs each ChaCha-touching Jobs owner and cancel action in a `chacha_operation` (12 modules, 13 functions), per worker as the September design note proposed, not centrally in `WorkerSDK`. Before, each executor thread a job used kept its checkout (`['asyncio_0']` after the job).
- The Chat_Macros run is not wrapped: its branches write concurrently, and one shared operation would give them one connection and one transaction depth, which rolled back every branch row and the final status. Each of its ChaCha calls owns an independent operation instead.
- The notes-graph suggestions job, which returns its checkout after every step, is left unwrapped, and its cleanup now leaves an operation's checkout to the operation. Inside an operation every such job failed with "The ChaCha operation has finished".
- The chatbooks core Jobs worker, the chatbooks cleanup loop, the ingestion-sources worker, the Workflows chatbooks adapter and the chatbook notification opened a handle per call and never closed it; its checkout came back only at cyclic GC. Each now runs in an operation; the Workflows adapter always opens its own, because a request's detached task can run it after the request.
- ADR-NNN records the rule for every non-HTTP entry point, including "no concurrent ChaCha transactions inside one operation"; the ChaCha DB guide and Docs/Published are updated. A guard for that rule is TASK-<P>.

## Change summary
<!-- The owner writes this section (ADR-004). -->

## Test plan
- [ ] notes-graph job through WorkerSDK.run on PostgreSQL: dev fails inside an operation, branch passes; the plain job passes on both
- [ ] one PostgreSQL WorkerSDK test per worker family: dev 4 failed (asyncio threads held), branch 4 passed; the macro-run test also fails when the run shares one operation
- [ ] per-call handles with GC disabled: dev 6 failed (MainThread held; the Workflows step after its request errored), branch 6 passed
- [ ] the 117 files that import PR C's modules: no new failures against dev
- [ ] Docs tests pass after refresh and staging; ruff no new findings; Bandit unchanged in product files; triage OK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Before merge: the owner's Change summary, `gh variable get MERGE_QUEUE`, and `MERGED` before removing `<scratchpad>/pr-c`. If #3288 merged first, rebase and keep both `Docs/ADR/README.md` rows (Step 2).

---

# PR D — `test/chacha-pg-test-app-middleware`: test apps and the restricted fixture

PR D changes tests only, and it closes TASK-13861.

It depends on PR A. With the middleware in its app, `test_conversation_quota_count_backends.py::test_create_chat_uses_real_quota_and_rejects_at_limit[postgres-ordinary-workspace]` needs Task A3's fix. On dev with PR D alone, that test fails with `{"detail":"Workspace locking reads require a transaction connection."}` (v1 and the review).

Cut PR D after PR C merges.

```bash
git fetch origin dev --quiet
git worktree add <scratchpad>/pr-d -b test/chacha-pg-test-app-middleware origin/dev
cd <scratchpad>/pr-d
export ADR=$(ls Docs/ADR | sed -nE 's/^([0-9]{3})-chacha-postgres-checkouts-owned-by-entry-points\.md$/\1/p')
echo "$ADR"
```
Expected: the number C4 Step 1 assigned. If it prints nothing, PR C has not merged yet: stop.

### Task D1: every PostgreSQL ChaCha test app installs the operation middleware (AC4)

**Files:**
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_test_apps_install_operation_middleware.py`
- Modify: the 30 files listed in Step 3 (39 apps)

**Interfaces:**
- Consumes: `ChaChaOperationMiddleware` from `tldw_Server_API.app.core.DB_Management.chacha.operation_scope`. `main.py:2796` installs it in production.
- Produces: every test file that builds `app = FastAPI()`, mentions `postgres`, and uses `CharactersRAGDB` or `get_chacha_db_for_user` adds `app.add_middleware(ChaChaOperationMiddleware)` after each such builder. A guard test checks that for the literal `app = FastAPI()`, which is how every such app is built today (checked). It does not check apps built another way, and its docstring says so (the review's Minor 7).

- [ ] **Step 1: Write the failing guard**

Create `tldw_Server_API/tests/DB_Management/test_chacha_test_apps_install_operation_middleware.py`:

```python
"""Test apps over PostgreSQL ChaCha install the middleware main.py installs (TASK-13861 AC4).

Without ChaChaOperationMiddleware every request falls back to a thread-local
PostgreSQL checkout; TASK-13841's 5-connection pool ran out that way.

It recognises the literal ``app = FastAPI()`` in files that mention postgres and
CharactersRAGDB or get_chacha_db_for_user, which is how every such test app is built
today. An app built another way is not checked.
"""

import re
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]
_BUILDER = re.compile(r"^[ \t]*app = FastAPI\(\)[ \t]*$", re.MULTILINE)
_INSTALLED = re.compile(r"^[ \t]*app\.add_middleware\(ChaChaOperationMiddleware\)[ \t]*$", re.MULTILINE)


def test_every_postgres_chacha_test_app_installs_the_operation_middleware():
    missing = []
    for path in sorted(_TESTS.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        builders = len(_BUILDER.findall(text))
        if not builders or "postgres" not in text:
            continue
        if not re.search(r"CharactersRAGDB|get_chacha_db_for_user", text):
            continue
        if len(_INSTALLED.findall(text)) < builders:
            missing.append(str(path.relative_to(_TESTS)))

    assert missing == [], f"Add app.add_middleware(ChaChaOperationMiddleware) after app = FastAPI() in: {missing}"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_test_apps_install_operation_middleware.py 2>&1 | grep -E "^E   (Assert|  Left)|passed|failed" | cut -c1-200
```
Expected (measured on `e521605198`): `E   AssertionError: Add app.add_middleware(ChaChaOperationMiddleware) after app = FastAPI() in: ['ChaChaNotesDB/test_note_attachment_postgres_tenancy.py', 'Character_Chat_NEW/integration/test_chat_session_client_id_create.py', …`, then `Left contains 30 more items` and `1 failed`.

- [ ] **Step 3: Record the dev baseline for the 30 files**

```bash
$PY - <<'EOF' > "$S/testapps.txt"
import re
from pathlib import Path
builder = re.compile(r"^[ \t]*app = FastAPI\(\)[ \t]*$", re.M)
installed = re.compile(r"^[ \t]*app\.add_middleware\(ChaChaOperationMiddleware\)[ \t]*$", re.M)
for path in sorted(Path("tldw_Server_API/tests").rglob("*.py")):
    text = path.read_text(encoding="utf-8")
    n = len(builder.findall(text))
    if n and "postgres" in text and re.search(r"CharactersRAGDB|get_chacha_db_for_user", text) and len(installed.findall(text)) < n:
        print(path)
EOF
wc -l < "$S/testapps.txt"
grep -v "RAG/test_restricted_postgres_media_retrieval.py" "$S/testapps.txt" > "$S/testapps_sweep.txt"
$S/sweep.sh "$S/testapps_sweep.txt" "$S/runs/d_testapps_dev" | tee "$S/runs/d_testapps_dev.summary"
```
Expected: `30`. Record the dev results for the other 29.

`RAG/test_restricted_postgres_media_retrieval.py` stays in `testapps.txt`, so the codemod and the guard cover it, but it is left out of the sweep. It hangs the same way on dev and with all the PRs applied (v1, measured with `-o timeout=150`):
- its 19th test in collection order, `test_permission_first_auth_stream_without_preseeded_scope[sqlite-Rowan-True]`, fails;
- the file's own role teardown, `DROP OWNED BY` at `:74`, then waits on a Media DB session left idle in a transaction (`SELECT * FROM MediaFiles …`, role `rag_retrieval_*`, seen in `pg_stat_activity`);
- pytest-timeout did not end it within 13 minutes.

This is a Media DB leak, not ChaCha, so D3 files it as a follow-up.

- [ ] **Step 4: Install the middleware with a codemod**

Save this as `$S/add_middleware.py` (a scratch tool, not committed) and run it from the repository root:

```python
"""Install ChaChaOperationMiddleware in every PostgreSQL ChaCha test app that lacks it (TASK-13861 AC4).

Run from the repository root. Uses the same predicate as
tldw_Server_API/tests/DB_Management/test_chacha_test_apps_install_operation_middleware.py.
"""

import re
from pathlib import Path

TESTS = Path("tldw_Server_API/tests")
IMPORT = "from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ChaChaOperationMiddleware\n"
BUILDER = re.compile(
    r"^(?P<indent>[ \t]*)app = FastAPI\(\)[ \t]*\n(?!(?:[ \t]*#.*\n)*[ \t]*app\.add_middleware\(ChaChaOperationMiddleware\))",
    re.MULTILINE,
)

files = apps = 0
for path in sorted(TESTS.rglob("*.py")):
    text = path.read_text(encoding="utf-8")
    if "postgres" not in text or not re.search(r"CharactersRAGDB|get_chacha_db_for_user", text):
        continue
    text, count = BUILDER.subn(
        lambda m: f"{m.group(0)}{m.group('indent')}app.add_middleware(ChaChaOperationMiddleware)\n", text
    )
    if not count:
        continue
    if not re.search("^" + re.escape(IMPORT), text, re.MULTILINE):
        first = re.search(r"^from tldw_Server_API\.", text, re.MULTILINE)
        text = text[: first.start()] + IMPORT + text[first.start() :]  # ruff --select I --fix sorts it
    path.write_text(text, encoding="utf-8")
    files += 1
    apps += count
    print(f"{path}: {count}")
print(f"{files} files, {apps} apps")
```
```bash
$PY "$S/add_middleware.py" | tail -1
$PY -m ruff check --select I --fix $(cat "$S/testapps.txt") | tail -1
$PY -m ruff check $(cat "$S/testapps.txt") | tail -1
git diff --numstat -- tldw_Server_API/tests | awk '{a+=$1; d+=$2} END {print NR" files +"a" -"d}'
```
Expected (measured on `e521605198`): `30 files, 39 apps`; `Found 29 errors (29 fixed, 0 remaining).`; `All checks passed!`; `30 files +78 -8`. The `-8` is `ruff --select I --fix` re-sorting two import blocks that were already unsorted on dev.

Two details of the codemod:
- The negative lookahead skips a builder already followed by the middleware, even with a comment line between them (`test_chat_import_local_chats.py:139-141`).
- The `^`-anchored import check adds a module-level import when the only existing one is local (`test_character_shared_owner_contract.py:367`).

- [ ] **Step 5: Run the guard and the 30 files**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_test_apps_install_operation_middleware.py 2>&1 | grep -E "passed|failed"
$S/sweep.sh "$S/testapps_sweep.txt" "$S/runs/d_testapps" | tee "$S/runs/d_testapps.summary"
```
Expected: `1 passed`, and no `FAILED` or `ERROR` lines beyond the dev baseline from Step 3. Planning ran the 29 files with all four PRs in the 325-file union sweep (C4 Step 5).

If `test_create_chat_uses_real_quota_and_rejects_at_limit[postgres-ordinary-workspace]` fails with `{"detail":"Workspace locking reads require a transaction connection."}`, the branch does not contain PR A's Task A3: rebase onto a `dev` that has PR A.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/tests
git commit -m "$(cat <<'EOF'
test: PostgreSQL ChaCha test apps install the operation middleware (TASK-13861)

Thirty test files built 39 bare FastAPI() apps over PostgreSQL ChaCha
without the ChaChaOperationMiddleware main.py installs, so each request
took a thread-local checkout (TASK-13841's PoolTimeout). The task's grep
found 29; test_character_shared_owner_contract.py hid a bare app beside
one that has the middleware. A guard test now counts builders against
middleware installs per file.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task D2: `pg_restricted_backend` fails fast on a leaked session (the task's optional item)

**Files:**
- Create: `tldw_Server_API/tests/DB_Management/test_pg_restricted_backend_teardown.py`
- Modify: `tldw_Server_API/tests/_plugins/postgres.py:404-411` (the fixture's `finally`), and add `_drop_restricted_role` after the fixture.

**Interfaces:**
- Produces: `_drop_restricted_role(admin, role: str) -> None`. It drops the role's objects and the role under `SET LOCAL lock_timeout = '10s'`. If a session still holds a lock, it:
  1. records the role's sessions;
  2. terminates them (`pg_terminate_backend(pid, 5000)`);
  3. drops the role;
  4. raises `AssertionError` naming the sessions.

  `pg_restricted_backend`'s signature and yield value are unchanged.

- [ ] **Step 1: Write the failing test**

Create `tldw_Server_API/tests/DB_Management/test_pg_restricted_backend_teardown.py`:

```python
"""pg_restricted_backend's teardown fails fast on a leaked session (TASK-13861, optional item)."""

import os
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.integration

_LEAKING_TEST = '''
_kept = []


def test_leaks_a_transaction(pg_restricted_backend):
    conn = pg_restricted_backend.get_pool().get_connection()
    conn.execute("CREATE TABLE leaked (id int)")
    conn.commit()
    conn.execute("SELECT * FROM leaked")  # an open transaction holding a lock on the role's table
    _kept.append(conn)  # never returned, as a leaked checkout is not
'''


def test_teardown_names_the_leaked_session_instead_of_hanging(pytester, monkeypatch):
    root = str(Path(__file__).resolve().parents[3])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [root, os.getenv("PYTHONPATH")])))
    monkeypatch.setenv("TLDW_TEST_POSTGRES_REQUIRED", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    pytester.makeconftest('pytest_plugins = ["tldw_Server_API.tests._plugins.postgres"]\n')
    pytester.makepyfile(test_leak=_LEAKING_TEST)

    result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider", timeout=90)

    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*left 1 session(s) of tldw_rls_*idle in transaction*"])
```
The `PYTHONPATH` set-up is the one `tests/Jobs/test_shared_retry_admission_pg_fixtures.py:97-99` uses.

- [ ] **Step 2: Run it to verify it fails**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_pg_restricted_backend_teardown.py 2>&1 | grep -E "^E   _pytest|passed|failed"
```
Expected (v1 and the review, 92-93 s): `E   _pytest.pytester.Pytester.TimeoutExpired: 90 second timeout expired running: …` and `1 failed`. The subprocess's teardown waits on `DROP OWNED BY`.

- [ ] **Step 3: Implement**

Replace the fixture's `finally` block (`postgres.py:404-411`) with:

```python
    finally:
        if backend is not None:
            backend.get_pool().close_all()
        try:
            _drop_restricted_role(admin, role)
        finally:
            admin.get_pool().close_all()


_RESTRICTED_ROLE_LOCK_TIMEOUT = "10s"


def _drop_restricted_role(admin, role: str) -> None:
    """Drop the role, failing fast and naming its sessions when one still holds a lock.

    A session the test leaked in a transaction keeps its locks, and ``DROP OWNED BY``
    would wait for them until pytest-timeout ends the whole run.
    """
    from psycopg import errors, sql

    drop = (
        sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)),
        sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)),
    )
    try:
        with admin.transaction() as conn:
            conn.execute(sql.SQL("SET LOCAL lock_timeout = {}").format(sql.Literal(_RESTRICTED_ROLE_LOCK_TIMEOUT)))
            for statement in drop:
                conn.execute(statement)
        return
    except errors.LockNotAvailable:
        pass
    with admin.transaction() as conn:
        sessions = conn.execute(
            "SELECT pid, state, left(query, 120) AS query FROM pg_stat_activity WHERE usename = %s",
            (role,),
        ).fetchall()
        conn.execute(
            "SELECT pg_terminate_backend(pid, 5000) FROM pg_stat_activity WHERE usename = %s",
            (role,),
        )
    with admin.transaction() as conn:
        for statement in drop:
            conn.execute(statement)
    raise AssertionError(
        f"The test left {len(sessions)} session(s) of {role} holding locks (a leaked checkout?): {sessions}"
    )
```

What each part does:
- `PostgreSQLBackend.transaction()` re-raises the body's exception after its rollback (`postgresql_backend.py:990-996`), so `LockNotAvailable` (SQLSTATE 55P03) reaches the `except`.
- Terminating the sessions before the second drop keeps the role from being left behind in the shared cluster.
- The test still fails, with the session report, so the leak is not hidden.

- [ ] **Step 4: Run it, and the fixture's users**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_pg_restricted_backend_teardown.py 2>&1 | grep -E "passed|failed"
git grep -l pg_restricted_backend -- tldw_Server_API/tests ':!tldw_Server_API/tests/_plugins' > "$S/restricted.txt"
wc -l < "$S/restricted.txt"
$S/sweep.sh "$S/restricted.txt" "$S/runs/d_restricted" | tee "$S/runs/d_restricted.summary"
$PY -m ruff check --output-format concise tldw_Server_API/tests/_plugins/postgres.py | grep -c ": [A-Z]"
```
Expected:
- `1 passed`, measured on `e521605198` with all four PRs. The subprocess shows `1 passed, 1 error` and the session report.
- `13`: dev's 12 users plus D2's own test.
- no `FAILED` or `ERROR` lines in the sweep beyond dev. Of dev's 12 users:
  - **six are outside A4's list:** `test_web_clipper_postgres_tenancy.py`, `test_claims_jobs_handlers.py`, `test_chacha_postgres_sequence_ownership.py`, `test_media_db_connection_cleanup.py`, `test_media_postgres_runtime_validation.py` and `MediaDB2/test_claims_review_metrics_postgres.py`. The review measured them at `291 passed` with all the PRs.
  - **six are inside A4's list,** and A4 swept them.
  - Planning swept all 13 in the 325-file union sweep (C4 Step 5).
- a ruff count of `10`, unchanged from dev.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/tests/_plugins/postgres.py tldw_Server_API/tests/DB_Management/test_pg_restricted_backend_teardown.py
git commit -m "$(cat <<'EOF'
test(postgres): pg_restricted_backend fails fast on a leaked session (TASK-13861)

A test that left a restricted-role session in a transaction made the
teardown's DROP OWNED BY wait until pytest-timeout os._exit-ed the whole
run (TASK-13841's 300 s hang). The drop now runs under a 10 s
lock_timeout; on timeout it reports the role's sessions, terminates
them, drops the role, and fails the test.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task D3: Close TASK-13861 and open PR D

**Files:** `backlog/tasks/task-13861 - …md`, and one new follow-up task.

**Interfaces:**
- Consumes: PRs A, B and C merged, and Tasks D1-D2.
- Produces: PR D into `dev`, with TASK-13861 marked Done.

- [ ] **Step 1: File the Media DB teardown hang, then close the task**

Compute a free id `<M>` with `$S/task_next.sh` (it is above `<P>`, filed in C4), then:
```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create \
  "RAG restricted Media DB test leaves a session idle in transaction and hangs its own teardown" \
  --id TASK-<M> -l rag,postgres,tests \
  -d "Found by TASK-13861 (measured on origin/dev 2b925562e5 and with TASK-13861's changes alike). In tests/RAG/test_restricted_postgres_media_retrieval.py, the 19th test in collection order (test_permission_first_auth_stream_without_preseeded_scope[sqlite-Rowan-True]) fails; the file's own teardown then runs DROP OWNED BY on its rag_retrieval_* role (:74), which waits on a Media DB session of that role left idle in transaction (SELECT * FROM MediaFiles WHERE media_id = ... in pg_stat_activity). pytest-timeout (-o timeout=150) did not end the run within 13 minutes. Not ChaCha." \
  --ac "The failing test is fixed or its cause recorded, and no Media DB session of the test role is left in a transaction at teardown" \
  --ac "The file's teardown fails fast on a leaked session instead of hanging (as pg_restricted_backend does after TASK-13861)"
```
Expected: `TASK-<M> - RAG restricted Media DB test leaves a session idle in transaction and hangs its own teardown (To Do)`.

Then close TASK-13861, filling the bracketed results from D1 Steps 3 and 5 and D2 Step 4:
```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13861 \
  --append-notes "PR D (test/chacha-pg-test-app-middleware): 30 test files / 39 bare FastAPI() apps over PostgreSQL ChaCha now install ChaChaOperationMiddleware (the task's grep found 29; test_character_shared_owner_contract.py hid one beside an app that has it). Guard test test_chacha_test_apps_install_operation_middleware.py counts builders against installs per file, for the literal app = FastAPI(): dev 1 failed (30 files listed), fix 1 passed. The other 29 files on PostgreSQL (the RAG file hangs on dev too, TASK-<M>): dev [result], with the middleware [result]." \
  --append-notes "Optional item done: pg_restricted_backend drops the role under lock_timeout 10s; on a leaked session it reports the role's sessions, terminates them, drops the role and fails. pytester probe: dev TimeoutExpired after 90 s; fix 1 passed (subprocess 1 passed, 1 error with the report). The fixture's 12 users (6 outside PR A's sweep, 6 inside): [result]." \
  --append-notes "Bandit: PR A B101 only in new tests; PR B unchanged from dev in product files (chat_helpers B311 x1, pre-existing), B101 and B105 (the voice test's 'valid-token') in new tests; PR C nothing in product files, B101 only in new tests; PR D B101 x1 in the guard, postgres.py unchanged. Test-quality triage OK with all 13 new test files, baseline untouched. Docs: ADR-$ADR and the ChaCha DB guide (PR C), Docs/Published refreshed." \
  --check-ac 4 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Outside HTTP, PostgreSQL ChaCha work used one thread-local checkout per thread per handle that nothing returned. PR A: a dead thread's checkout now comes back (a dropped handle's only at cyclic GC, a backstop), the finalizer never takes the pool's lock, close_all_connections() neither closes the shared content pool (one MCP call used to break ChaCha until restart) nor leaves a returned checkout open without its reset, and two locking reads that returned 400 inside every HTTP request on PostgreSQL see the operation's transaction. PR B: /chat/stream persistence, the voice assistant and every persona session call (8 off the loop, 13 on it, and the skill tool call's registry lookup) run in their own chacha_operation, and a default-character insert race is fixed. PR C: each ChaCha-touching Jobs owner and cancel action, and the five background runners that opened per-call handles, own their checkouts (ADR-$ADR, per worker rather than in WorkerSDK); the Chat_Macros run, whose branches write concurrently, gives each ChaCha call its own operation, because one shared operation rolled back its writes; the notes-graph cleanup is operation-aware. PR D: the 30 PostgreSQL ChaCha test apps install the middleware, a guard keeps them, and pg_restricted_backend fails fast on a leaked session. Follow-ups: TASK-<N> (persona: run the 10 on-loop ChaCha calls inside async handlers off the event loop, so a call waiting for a connection cannot stall the loop), TASK-<P> (serialize managed transactions per connection state, or fail loudly on overlap), TASK-<M> (RAG restricted Media DB test hang, not ChaCha)." \
  -s Done
```
Expected: `TASK-13861 - Legacy thread-local PostgreSQL ChaCha checkouts leak (…) (Done)`, with ACs #1-#5 checked. Under Decision 2's Alternative 2 or 3, AC2 stays unchecked (C4 Step 7); then describe `TASK-<N>` in the final summary as B6 Step 2 filed it.

- [ ] **Step 2: Lint, Bandit, triage, task format, whitespace**

```bash
$PY -m ruff check tldw_Server_API/tests/DB_Management/test_chacha_test_apps_install_operation_middleware.py \
  tldw_Server_API/tests/DB_Management/test_pg_restricted_backend_teardown.py
uvx bandit -q -f json -o "$S/runs/bandit_d.json" tldw_Server_API/tests/DB_Management/test_chacha_test_apps_install_operation_middleware.py \
  tldw_Server_API/tests/DB_Management/test_pg_restricted_backend_teardown.py tldw_Server_API/tests/_plugins/postgres.py
$PY -c "import json,collections;d=json.load(open('$S/runs/bandit_d.json'));print(sorted(collections.Counter((r['filename'].rsplit('/',1)[-1],r['test_id']) for r in d['results']).items()))"
$PY Helper_Scripts/ci/test_quality_triage.py --enforce 2>&1 | grep "test-triage\] OK"
git add -A backlog/tasks && git commit -m "$(cat <<'EOF'
docs(backlog): close TASK-13861; file the RAG restricted-test teardown hang

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
BACKLOG_TASK_FORMAT_BASE=$(git rev-parse origin/dev) PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
  $PY -m pytest -q --disable-warnings -p pytest_asyncio.plugin -p no:cacheprovider \
  tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py 2>&1 | grep -E "^=+ .*(passed|failed)"
git diff --check origin/dev...HEAD && echo "no trailing whitespace"
```
Expected:
- ruff `All checks passed!`;
- Bandit: `test_chacha_test_apps_install_operation_middleware.py` `B101` x1, nothing in the teardown test, and `postgres.py`'s `B101` x2, `B404` x1 and `B603` x2, the same as dev (measured);
- triage `OK`;
- the task-format ratchet passes;
- `no trailing whitespace`.

- [ ] **Step 3: Review, push, open PR D**

Run `/superpowers:requesting-code-review` in a clean subagent with the branch diff and this plan's PR D section, and address its findings. Then:
```bash
git push -u origin test/chacha-pg-test-app-middleware
gh pr create --base dev --title "test: PostgreSQL ChaCha test apps install the operation middleware; restricted fixture fails fast (TASK-13861, 4 of 4)" --body "$(cat <<'EOF'
## Summary
- 30 test files (39 apps) over PostgreSQL ChaCha now install `ChaChaOperationMiddleware`, as `main.py` does; a guard test counts `app = FastAPI()` builders against installs per file. The task's grep found 29; one more file hid a bare app beside one that has the middleware.
- `pg_restricted_backend` drops its role under a 10 s `lock_timeout`; a leaked session now fails the test with a report of the role's sessions instead of hanging until pytest-timeout ends the run.
- Closes TASK-13861 (PRs A-C fixed the product side). Follow-ups: TASK-<N> (persona's on-loop calls off the event loop), TASK-<P> (a guard against concurrent ChaCha transactions inside one operation), TASK-<M> (a RAG Media DB test that hangs its own teardown on dev).

## Change summary
<!-- The owner writes this section (ADR-004). -->

## Test plan
- [ ] guard test: dev fails listing 30 files; branch passes
- [ ] the 30 files on PostgreSQL (29 swept; the RAG file hangs on dev too, TASK-<M>): no new failures against dev
- [ ] restricted teardown probe: dev times out at 90 s; branch reports the session in about 14 s; the fixture's 12 users pass
- [ ] ruff clean on new files; Bandit B101 only; triage OK; task-format ratchet passes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Before merge: the owner's Change summary, `gh variable get MERGE_QUEUE`, and `MERGED` before removing `<scratchpad>/pr-d`.
