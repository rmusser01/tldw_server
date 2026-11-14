# ChaCha Calls Off the Event Loop, and One Transaction per Operation State: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Re-review folded in (r3, 2026-10-23).** The scoped re-review is `<scratchpad>/rereview_plan_fu2_chacha_runtime.md`: Ready with fixes, 0 Blocker, 0 Important, 3 Minor, all text-only. r2 is kept as `<scratchpad>/plan_fu2_chacha_runtime.r2.md`. No code or measurement changed.
> - **m1.** The audit is now described as it is: name-based within one module, skipping modules with no ChaCha marker.
>   - The reviews spot-read the gather/wait sites, a gap scan of direct-transaction fan-outs (13 sites) and the MCP and Workflows fan-outs, and found no second overlap. Cross-module chains are not exhaustively covered.
>   - This is corrected in Design choice 1, Review Focus 4, Decision 1, the Task 1 commit message, the TASK-13899 note and the PR body.
>   - Decision 1 now says the "7 of 50 Chat_Macros replays" figure measured a configuration that never shipped.
>   - Decision 1 adds a staged "log only now, raise later" variant of (B).
> - **m2.** Review Focus 6 states the shielded writes' teardown bound: about `pool_timeout` for the drain, plus up to 2 × `pool_timeout` per seen session for the finalize. Measured: 1.91 s with a 2 s timeout.
> - **m3.** Review Focus 6 notes that under an anyio scope cancel `run_quiescent_operation` spins the loop's CPU until the write finishes (0.133 s over a 0.45 s write). Native cancels do not.
> - **Also recorded:** the re-review's `Buddy/turns.py` observation, under "Not filed".

> **Review folded in (r2, 2026-10-22).** The review is `<scratchpad>/review_plan_fu2_chacha_runtime.md`: Not ready, 2 Blocker, 2 Important, 5 Minor. Its probes are in `<scratchpad>/rvcr_probes/`. r1 is kept as `<scratchpad>/plan_fu2_chacha_runtime.r1.md`. Every changed step was re-measured on origin/dev `e06eeec68c` (unchanged since r1).
> - **B1, fixed. A current caller overlaps.**
>   - `SemanticProjector.validate_conversion` (`semantic_projector.py:1206`) gathered two `_visible_ids` lookups. Each is a `to_thread` into `NoteSemanticStore.list_visible_vector_ids`, which opens `db.transaction()`.
>   - Inside the request's operation the guard refused it on every call: the review measured 40 of 40.
>   - New **Task 1 Step 6** awaits the two lookups one after the other. A PostgreSQL test drives `validate_conversion` inside an operation:
>     - dev: `(None, 2)`: both lookups ran at once;
>     - the guard without the fix: `('ConcurrentChaChaTransactionError', 2)`, 3 of 3;
>     - with the fix: `(None, 1)`.
>   - **The audit is widened to fan-outs** (Design choice 1, F5). Method: an index of the 486 ChaCha methods that open a managed transaction, plus 135 fan-out sites in the 268 app files that name ChaCha, followed up to 3 hops in their module. 17 sites were flagged, each read by hand. Only `validate_conversion` overlaps.
>   - Design choice 1, the Task 1 commit message, the ADR bullet and the TASK-13899 notes are corrected.
> - **B2, fixed. Write first, then notify.**
>   - The commit's and the manual-mode notice's analytics writes now run before their notice. The telemetry test went from 3 of 3 failing on r1 to 6 of 6 passing.
>   - **Every moved write was checked** for a session-end cancel that lands before its executor job starts (Design choice 7):
>     - the 4 writes, the summary upsert, the event record and both preference saves, are now shielded once queued: `run_quiescent_operation(asyncio.to_thread(...))`;
>     - the reads are harmless to lose.
>   - **New pins:** an AST guard, and a PostgreSQL test of a queued preferences write whose task is cancelled. Shielded, it is saved; as plain `to_thread`, it is lost.
>   - Review Focus 6 is rewritten.
>   - **Loop sweep under load (up to 11):** `328 passed` and `317 passed`.
> - **I1.** N1 is retitled "Voice commands are unusable on PostgreSQL …", lists the failing helpers and `upsert_voice_command`, and gains the save/list/read/delete AC.
> - **I2, an owner decision.** Raise on overlap (recommended, implemented) or log only. See "Decisions for the owner".
> - **M1.** The session's finalize upserts now run after the owned turn tasks are cancelled and drained. The native-cancel and first-insert race are in Review Focus 6.
> - **M2.** N3 lists the two chat-path async helpers and the two sync ones.
> - **M3.** The workflows plan edits only ADR-080's "Detached tasks" bullet. A merge simulation found `operation_scope.py` clean, and one ADR-080 conflict at lines 51-52, plus the mirror.
> - **M4.** New follow-up N4: a voice command's action modules (the notes workflow step, MCP modules) still check out on the loop.
> - **M5, folded in:**
>   - Task 3's snippet indentation;
>   - the union list (now 263 files, including the semantic projector's tests);
>   - the union sweep, now measured: `7318 passed, 12 failed, 20 skipped`. All 12 failures fail the same way on dev; the operation-focused files with the semantic test give `163 passed`;
>   - group 1's dropped command cache on a failed session save (Review Focus 10);
>   - the 7 untracked dev failures, now listed and filed as N5.
> - **The split and task count are unchanged:** 1 PR, 5 plan tasks.
>   - **New in r2:** 1 test file (semantic conversion on PostgreSQL), and in the persona test file a shield guard and a parametrized queued-write test.
>   - **Product:** `semantic_projector.py` +4/-4, `persona.py` +64/-47 (r1 had +38/-25); 8 product files in all.

**Goal:**
- **TASK-13896.** No ChaCha checkout is taken on the event-loop thread by:
  - the persona live WebSocket's calls on its session handle;
  - the skill tool call's `SkillsService` setup and lookup;
  - the voice-assistant WebSocket's session start and its router's call groups.

  So when other work holds the whole content pool, such a call waits on a worker thread instead of stalling the whole server for up to `pool_timeout`. A voice command also holds no checkout through its LLM call, MCP action or SYNC workflow, and a full pool at voice session start degrades the session instead of killing the socket.
- **TASK-13899.** Inside one `chacha_operation`, a `db.transaction()` that enters while another thread's or task's transaction is open on the same `ConnectionState` raises `ConcurrentChaChaTransactionError`, instead of joining it and silently losing every later write in the operation. The one current caller that overlapped, semantic conversion's two lookups, runs them one after the other.

**Architecture:**
- **TASK-13899 (fail loudly, not serialize).** `ConnectionState` gains `tx_owner`, the `(thread ident, asyncio task)` key that `use()` already uses.
  - `BackendManagedTransaction.__enter__` claims it through `ConnectionState.enter_transaction()`. The same key nests and joins, as before. Another key raises.
  - `__exit__`, and a failed `__enter__`, release the claim.
  - The legacy thread-local path (`_local`, outside any operation) is unchanged.
  - `SemanticProjector.validate_conversion` awaits its two `_visible_ids` lookups in turn instead of gathering them.
- **TASK-13896 (persona).** All 13 on-loop calls on `persona_scope_db` become `await asyncio.to_thread(chacha_operation(independent=True)(fn), ...)`.
  - That is the 10 in async handlers, plus the 3 inside the sync analytics helpers.
  - The three helpers become `async def`, so their in-memory bookkeeping stays on the loop. Their 9 call sites gain `await`.
  - **The 4 writes are shielded once queued:** `await run_quiescent_operation(asyncio.to_thread(...))`, the existing helper in `Notes_Graph/semantic_publication.py`. A cancel of the awaiting task lets a queued or running write finish, then propagates. The writes are the summary upsert, the event record and both preference saves.
  - **Write first, then notify.** The commit's and the manual-mode notice's analytics writes run before their notice.
  - **Finalize last.** The session-end finalize upserts run after the owned turn tasks are cancelled and drained.
  - **The AST guards** now require the off-loop form everywhere in `persona.py`, an `await` on every helper call, and the shield on every write.
- **TASK-13896 (skill).** `handle_skill_tool_call` builds `SkillsService` through `asyncio.to_thread`, inside the lookup's existing independent operation.
- **TASK-13896 (voice).**
  - **Session start.** `save_voice_session` and the command read run in one independent operation on a worker thread. The registry cache is applied on the loop. `except Exception` degrades the session with the existing sanitized warning.
  - **Commands.** `_process_text_command` no longer wraps the command in one operation. The router runs its three ChaCha call groups, each through `asyncio.to_thread(chacha_operation()(fn), ...)`:
    - the command refresh, plus the new session's save;
    - the analytics row, plus the session save;
    - the external-connection lookup.

    So it joins an HTTP request's operation (`/voice/command`) and owns each group in the WebSocket.
  - **The registry.** `VoiceCommandRegistry.refresh_user_commands` is split into `load_user_commands` (database only) and `apply_user_commands` (cache only), and refresh keeps its behaviour.
- **Docs.**
  - ADR-080's two Consequences bullets that name these tasks as open become status facts.
  - The ChaChaNotes_DB guide states the new error.
  - `Docs/Published` is refreshed.

**Tech Stack:** Python 3.12.13, pytest 9.1.1, pytest-asyncio 1.4.0, pytest-timeout 2.4.0, psycopg 3.3.6, psycopg-pool 3.3.2, FastAPI 0.142.1, Starlette 1.6.0, PostgreSQL 18 (Docker container `tldw_postgres_test`), backlog-py.

**Spec:** on origin/dev:
- `backlog/tasks/task-13896 - Persona-and-voice-assistant-run-on-loop-ChaCha-calls-off-the-event-loop.md`
- `backlog/tasks/task-13899 - ChaCha-operations-serialize-managed-transactions-per-connection-state-or-fail-loudly-on-overlap.md`

**Background:**
- `Docs/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md`;
- `Docs/superpowers/plans/2026-10-20-chacha-pg-legacy-checkouts.md`: F12, persona; F13, concurrent transactions; B3, B4 and B5;
- the TASK-13861 reports in `<scratchpad>`: `impl_chacha_prA.md` … `prD.md`, `clean3293_review.md`, `rereview3293.md`, `rereview3293_r34.md`, `clean3295_review.md` and `clean3297_review.md`.

**Where this was measured.**
- **The checkout.** origin/dev `e06eeec68c` (the merge of #3300) in my isolated worktree, detached.
- **The database.** PostgreSQL through the repo's `pg_database_config` fixture on the `tldw_postgres_test` container (postgres:18, port 5432).
- **Load.** The 1-minute load was 4-28 while other sessions ran.
- **REDs.** Each RED ran with the product files at `e06eeec68c` and the new tests in place, unless the step says otherwise (B1's RED with the guard, B2's on r1's persona edit).
- **GREENs.** Each GREEN ran with the prototype applied.
- **Artifacts.**
  - **Prototype diffs.** r2's is `<scratchpad>/fu2cr_prototype_r2.diff`, which embeds every file in this plan; r1's is `fu2cr_prototype_final.diff`.
  - **Logs.** `<scratchpad>/fu2cr_*.log` and `<scratchpad>/runs/fu2cr_*`; r2's sweeps are `runs/fu2cr_loop_r2*` and `runs/fu2cr_union_r2*`.
  - **Audit.** `fu2cr_overlap_audit.py`, with its output in `fu2cr_overlap_audit.txt`.
- **Cleanup.** The worktree was restored to `e06eeec68c` afterwards.

## Global Constraints

- **TASK-13896 ACs:**
  1. "The 10 calls in persona_stream's async handlers run through asyncio.to_thread with their own chacha_operation(independent=True), and the persona AST guard requires that form for them".
  2. "A PostgreSQL test with the pool held by other work shows a persona call waiting for a connection without stalling the event loop".
  3. "The skill tool call's SkillsService setup and lookup make no ChaCha checkout on the event-loop thread".
  4. "The voice-assistant WebSocket's session start and command handling make no ChaCha checkout on the event-loop thread, and a command holds no checkout through its LLM call, MCP action or SYNC workflow". Task 5 rewords it; see Design choice 6.
- **TASK-13899 ACs:**
  1. "Two overlapping db.transaction() blocks on one ConnectionState from two threads either serialize or raise; neither loses a write (PostgreSQL test)".
  2. "test_chacha_operation_scope.py and the HTTP operation lifecycle tests still pass".
- **Fix defects test-first.** Each task's test fails on `e06eeec68c` for the stated reason, then passes with the fix.
- **ADR-080 is binding.**
  - One operation per WebSocket step or persona call, never one per connection.
  - Never run ChaCha transactions concurrently inside one operation.
  - `@chacha_operation(...)` decorates plain functions only. Async code uses `with`, or `chacha_operation(...)(fn)` inside `asyncio.to_thread`.
  - `run_in_executor` does not carry the operation: use `asyncio.to_thread`.
- **Hazards measured in TASK-13861 that no step may reintroduce:**
  - **Cross-thread close.** No step closes or returns a connection from a thread other than the one using it.
    - Every operation here opens and closes on the worker thread inside the `to_thread` callable, or (the skill lookup, the router inside an HTTP request) is released by `ConnectionState.release()`, which defers the return until the last `use()` ends on its own thread.
    - The guard in Task 1 touches only `tx_owner` under `state.lock`, never the connection.
  - **Weakrefs in cyclic garbage.** No step adds a finalizer or a weak reference.
  - **Detached tasks.** The only new task is the one `run_quiescent_operation` creates around each persona write's `to_thread`.
    - It is awaited until it finishes; it never outlives its caller's cancellation.
    - It starts in the persona session's context, which holds no operation, and its callable opens its own independent operation.
    - The persona and voice WebSockets hold no operation, so nothing they spawn inherits one.
    - The router's joining `chacha_operation()` would raise `ClosedChaChaOperationError` only under an inherited closed operation, and no caller has one (Review Focus 5).
  - **On-loop checkouts.** Every ChaCha call these steps touch moves into `asyncio.to_thread`.
- **Never raise `pool_size` or `pool_timeout` to hide a leak.** Tests may lower them.
- **Postgres steps** run with `TLDW_TEST_POSTGRES_REQUIRED=1` and without `TLDW_TEST_NO_DOCKER`. Steps without a database may set `TLDW_TEST_NO_DOCKER=1`.
- **Backlog** changes only through backlog-py: `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`.
  - Use `--append-notes`, never `--notes`. Pass explicit ids. Stage with `git add -A backlog/tasks`.
  - New ids are allocated under the id lock (Task 5 Step 4).
- **The PR** targets `dev`, from a branch cut from `origin/dev`.
  - Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - Never `--no-verify`, never force-push, never `git stash`.
- **Mirrored docs.** `Docs/ADR/080-…md` and `Docs/Code_Documentation/Databases/ChaChaNotes_DB.md` are mirrored under `Docs/Published/`. Run `bash Helper_Scripts/refresh_docs_published.sh`, stage everything, then run `tests/Docs`.
- **Line baselines.** None of the touched files is in `Helper_Scripts/ci/scope_predicate_baseline.txt` (39 lines; grep on `e06eeec68c`). Shifted lines need no regeneration.
- **Ruff is non-blocking in CI; the rule is no new findings.** These touched files already have findings on dev (measured on `e06eeec68c`):

  | File | Findings on dev, and on the prototype |
  |---|---|
  | `app/api/v1/endpoints/persona.py` | I001, SIM114 |
  | `app/api/v1/endpoints/voice_assistant.py` | I001 |
  | `app/core/VoiceAssistant/router.py` | F401, F821 ×5 |

  Leave their import blocks as they are; do not run `ruff --fix` on them. The other touched files, including `semantic_projector.py` and all six test files, give `All checks passed!` (measured on r2).
- **Global defaults.** Task 1 changes every `BackendManagedTransaction` inside an operation, on PostgreSQL. Task 5 Step 2 sweeps 263 files, without TASK-13900's hanging RAG file:
  - every test file that names a ChaCha handle or operation and PostgreSQL;
  - every test that imports or drives the persona, voice or skill code;
  - the semantic projector's tests.
- **A parallel plan touches two of the same files:** `<scratchpad>/plan_fu2_workflows.md` (TASK-13897 and others).
  - **`operation_scope.py`.** It adds `outside_chacha_operation()` after `chacha_operation`; Task 1 edits the top of that file.
  - **ADR-080.** It edits only the "Detached tasks" bullet (line 51); Task 5 edits the "Cost" and "Concurrent transactions" bullets (lines 52 and 56).
  - **The review's three-way merge** with dev as the base: `operation_scope.py` merges cleanly and the merged file parses. ADR-080 has one conflict, at lines 51-52, and so does its `Docs/Published` copy.
  - **Whichever PR merges second** rebases, keeps both bullets, and re-runs the refresh.
  - That plan says a voice command's SYNC run "still joins that operation". After Task 4 a voice command has no operation, so its SYNC run starts outside one. That is the dev-before-TASK-13861 shape, which that plan already handles.

## Design choices settled by the code

1. **TASK-13899: raise on overlap, do not serialize.** The task allows either. Raise versus log-only is put to the owner as Decision 1.
   - **Who overlaps today: one caller, fixed in Task 1 Step 6.** r1 said no current caller overlaps. That was wrong, because r1 scanned only the bodies of `with x.transaction()` blocks (the first two scans below). The audit is now three scans, all on `e06eeec68c`:
     - **The `with` bodies.** `app/` has 1228 sync `with <x>.transaction(...)` blocks. None awaits or hands work to a thread inside its body. The only hit is `Notes_Tasks/reconciler.py:112`, where `db.create_task` is a DB method (`<scratchpad>/fu2cr_tx_await_scan.py`).
     - **Transactions held across a `yield`.** 17 blocks do this (`fu2cr_tx_yield_scan.py`). Two are on a ChaCha handle:
       - `transaction_utils.db_transaction`: its callers (`chat.py:7125` and the module's own) do not await inside;
       - `chat_dictionary._connection`, a sync context manager.

       `@transactional` has no users.
     - **Fan-outs** (`<scratchpad>/fu2cr_overlap_audit.py`, output in `fu2cr_overlap_audit.txt`).
       - **The index.** Every method of `ChaChaNotes_DB.py` and `DB_Management/chacha/*.py` that opens a managed transaction is indexed: 486 methods. That covers `self.transaction()`, `self._db.transaction()` and `db.transaction()`, and same-file helpers that reach one by name.
         - `backend.transaction(connection=…)` is not counted: it is `execute_query(read_only=True)`'s own read scope, which the guard does not claim.
         - Twelve names that mean something else everywhere are dropped, such as `get`, `create_task` and `close_connection`.
       - **The sites.** Every fan-out in the 268 app files that name ChaCha: 135 sites. That is `gather`, `create_task` and `ensure_future`, `TaskGroup.create_task`, `as_completed`, `asyncio.wait`, anyio `start_soon`, `run_in_executor`, executor `submit`/`map`, `run_coroutine_threadsafe`, `threading.Thread` and `copy_context().run`.
       - **The match.** A site is flagged when what it hands off reaches an indexed method by name, within 3 hops through its own module. 17 sites were flagged, and each was read:
         - **`Notes_Graph/semantic_projector.py:1206`**: `validate_conversion` gathers two `to_thread(store.list_visible_vector_ids)`, each opening `db.transaction()`, inside the request's operation. This is the review's B1, and the only overlap. Task 1 Step 6 makes the lookups sequential.
         - **Own operation each**:
           - the default-character task and its executor (`ChaCha_Notes_DB_Deps.py:504`, `:784`, `:837`, `:884`: `@chacha_operation(independent=True)`);
           - the conversation enrichment threads (`conversation_enrichment.py:386`, `:406`: `chacha_operation(independent=True)` in each runner).
         - **No operation to share**:
           - `run_in_executor` does not carry the context: `chat.py:3349`, `audio_streaming.py:2533`, `async_db_wrapper.py:60`, `:92`, `:203`;
           - tasks a WebSocket starts outside any operation: `audio_streaming.py:3400`, `persona.py:10523`;
           - worker start-up tasks: `recipe_runs_jobs_worker.py:1280` (where `get_run` is a name collision), and `shared_workspace_clone_jobs_worker.py:1567`, whose tasks start before any job's operation.
         - **One hand-off at a time**: `chat.py:5445`, an OAuth refresh awaited by its own LLM thread.
         - **SQLite only**: `VNAssetPacks_DB.py:2606` (`_require_sqlite_chacha_db`).
       - **Read by hand because its chain crosses modules:** the clone job's cancellation check (`shared_workspace_clone_jobs_worker.py:541`, `run_coroutine_threadsafe` into the job's operation). It reaches `get_workspace` and the Jobs DB. `get_workspace` reads through `execute_query(read_only=True)` and opens no managed transaction, so the guard cannot refuse it (Review Focus 4).
       - **Limits.** The audit is name-based within one module, which leaves three gaps:
         - it misses chains through other modules, service objects and dynamic dispatch;
         - it misses code outside the store files that opens `x.transaction()` directly on a ChaCha handle;
         - it skips modules with no ChaCha marker that still receive a handle.
       - **What the reviews read on top of it.** None of these found a second overlap.
         - The r1 review spot-read the `gather`/`wait`/`TaskGroup` sites (about 35) and some `create_task` sites among its 131.
         - The r2 re-review ran a gap scan of direct `.transaction(` openers in the 198 other app modules. It found 13 fan-out sites and triaged each.
         - It also read MCP `knowledge_module`'s gather and the Workflows `parallel`/`map` sub-steps.
       - **Cross-module chains are not exhaustively covered.**
     - **The HTTP routes.** The F13 audit (2026-10-20) found no HTTP route that overlaps managed transactions on one handle, apart from the one above, which it missed.
   - **What raising costs (the review's I2, Decision 1).** Refusal also covers overlaps that happened to commit everything on dev. The review measured three:
     - nested in time (B enters after A and leaves before it);
     - a hand-off to a thread that opens a transaction while the caller holds one;
     - `asyncio.run` inside a sync transaction.

     Each committed every row on dev and raises with the guard. A caller this audit missed becomes a 500 (HTTP) or a failed, possibly retried, job instead of usually working. The crossing interleaving it also refuses was silently losing every later write in the operation.
   - **A lock fails worse:**
     - a transaction that waits, inside itself, on another thread's transaction in the same operation would deadlock instead of failing;
     - an event-loop thread waiting for a worker's transaction would stall the server: the hazard Tasks 2-4 remove;
     - a per-thread `RLock` does not see two tasks on the loop thread at all. Measured: with a thread-only key, the two-task case loses writes exactly as dev does (Task 1 Step 5).
   - **Raising enforces ADR-080's existing rule.** A fan-out gives each call its own operation, as Chat_Macros does. It also logs at ERROR, so a caller that catches `RuntimeError` broadly still leaves a trace.
2. **Persona helpers become `async def`; they are not handed whole to a thread.**
   - `_upsert_persona_live_session_summary_safe` adds to `persona_live_summary_sessions_seen`, a set that the session's `finally` copies with `list(...)` on the loop (`persona.py:11422`), while owned turn tasks can still be running.
   - Running the whole helper in a thread would race that copy. As `async def`, only the ChaCha call leaves the loop.
3. **The voice registry cache is applied on the loop.**
   - `refresh_user_commands` iterates and mutates the process-wide `_user_commands` dict (`registry.py:359-366`) that every session's parser reads.
   - Two worker threads refreshing at once could raise "dictionary changed size during iteration". So the database read moves to a thread (`load_user_commands`), and the cache update (`apply_user_commands`) stays on the loop.
   - The session that a worker saves is a `model_copy(deep=True)` taken on the loop, so no worker reads an object the loop is mutating.
4. **Joining in the router, independent at WebSocket session start.**
   - The router runs both in HTTP `/voice/command`, inside the request's operation, and in the WebSocket, with none. Joining `chacha_operation()` shares the request's checkout in HTTP, as B4's core helpers do, and owns each group in the WebSocket.
   - Session start is a WebSocket step, so it uses `independent=True` (ADR-080).
5. **One operation per call group, not per statement.** The refresh and the new session's save share one operation, and so do the analytics row and the session save. Each operation costs about 5 round trips (TASK-13861 F12), so two groups cost about 10 round trips, not 20.
6. **AC4's premise is narrower than its words.**
   - "Command handling" could be read to include the action's own module. A SYNC workflow's notes step runs `NotesInteropService` on the loop with a per-call handle that it closes; MCP modules use per-call handles closed on the same thread.
   - The description's fix shape covers the WebSocket and the router's call groups only.
   - The workflow adapters are TASK-13897's area, and `plan_fu2_workflows.md` keeps the notes adapter on the loop.
   - So Task 5 rewords AC4 through backlog-py to name the router's call groups, and records why.
   - **The residual is filed, not dropped:** follow-up N4 covers the action modules' own on-loop checkouts during a voice command (the review's M4).
7. **The persona writes: write first, then notify; shield once queued; finalize last.** This is the review's B2 and M1.
   - **Why it changed.** On dev each persona write was synchronous: it either ran fully or was never reached. As `await asyncio.to_thread(...)` it has two new ways to be lost.
     - **A client that closes on a notice cancels the session task while the writes behind that notice are still pending.** The commit's `VOICE_TURN_COMMITTED` and the manual-mode notice were emitted before their analytics writes.
       - Measured: `test_persona_ws.py::test_persona_audio_chunk_records_live_voice_commit_telemetry` failed 3 of 3 on r1 at load 6. The review saw 6 of 6.
       - The writes now run before the notice. r2: 6 of 6 passed.
     - **A cancel while the write's executor job is still queued drops it**: `asyncio.to_thread` cancels its `run_in_executor` future, and a queued job never starts. This happens when a session ends while the default executor is saturated.
       - The fix uses the existing `run_quiescent_operation` (`Notes_Graph/semantic_publication.py:308`). It runs the `to_thread` as its own shielded task, waits for it to finish, and then re-raises the caller's cancellation (`committed_result_wins=False`).
       - Measured with a one-worker executor held busy: shielded, the queued preferences write is saved; as plain `to_thread`, it is lost.
       - `run_reconciled_transaction`, the review's pointer, is the same module's thread variant. It is not used, because it swallows the cancellation when the write returns a value, and `_persist_persona_session_preferences` returns a dict. A cancelled turn task would then run on past session cleanup.
   - **Each moved call, checked:**

     | Call | On a cancel before its job starts | Shielded |
     |---|---|---|
     | summary upsert (`_upsert_persona_live_session_summary_safe`: commit, manual mode, voice config, TTS text-only, finalize) | a missing count in the voice analytics a user sees | yes |
     | event record (`_record_persona_live_voice_event_safe`: commit, manual mode) | a missing analytics event | yes |
     | `_persist_persona_session_preferences` (turn `:10072`, receive loop `:11117`) | the user's preferences not saved | yes |
     | policy rules, scope metadata, state hints, the persona-id resolve | a read the cancelled task would not use | no: harmless |

   - **What remains.** A cancel during the persona-id read that comes before an analytics write means that write is never reached. On dev a cancel a moment earlier did the same, so it is a narrower window, not a new kind of loss (Review Focus 6).
   - **Finalize last.** The session's `finally` now finalizes the summaries after cancelling and draining the owned turn tasks, which also write summaries. That restores "finalize is the last summary write".

## ADR assessment

**No new ADR. ADR-080 governs, and two of its Consequences bullets are updated in place as status facts.**
- **The guard (TASK-13899) enforces a rule ADR-080 already decides:** "Never run ChaCha transactions concurrently" inside one operation. The decision is unchanged; only its enforcement is new.
  - The new Consequences text states what enforcement costs: an overlap that happened to work on dev also raises (Decision 1).
  - If the owner chooses log-only instead, the bullet says so, and AC1 changes; no ADR changes.
- **Moving calls off the loop (TASK-13896) applies ADR-080's Cost consequence:** "Calls in async code therefore go through `asyncio.to_thread`".
- **`Docs/ADR/README.md` rule 8 forbids rewriting accepted rationale.** So the Decision, Context and Alternatives sections are not touched. Two Consequences bullets name TASK-13896 and TASK-13899 as open, and become false when this PR merges:
  - "Cost": "… the persona and voice-assistant calls that still run on the loop are TASK-13896";
  - "Concurrent transactions are a rule, not a guard. … TASK-13899 tracks that guard."
- **Precedent.** `b3c1a1fa56` added two Consequences bullets to the accepted ADR-080. The parallel workflows plan updates its "Detached tasks" bullet the same way.
- **The README row** ("ChaCha transactions never run concurrently inside one operation") stays true, and is unchanged.
- **Other ADRs.**
  - ADR-006 (Bandit on touched scope), ADR-059 (backlog-py) and ADR-004 (the owner's Change summary) apply.
  - ADR-020 (the shared content backend) is unaffected.

## Findings (measured on `e06eeec68c`)

### F1. Overlapping transactions in one operation (TASK-13899)

`BackendManagedTransaction.__enter__` reads `tx_depth` from the operation's shared `ConnectionState` (`ChaChaNotes_DB.py:835`), and `__exit__` writes back the value it read (`:889`).

Task 1's test runs two interleavings inside one `chacha_operation(independent=True)`:
1. The first transaction writes `r1` and stays open.
2. The second enters, waits for the first to commit, then writes `r2` and leaves last.
3. A later transaction in the same operation writes `r3`.

| Case | dev | with the guard |
|---|---|---|
| two threads | `([], ['r1'])`: no error; `r2` and `r3` were rolled back when the operation returned its checkout | `(['ConcurrentChaChaTransactionError'], ['r1', 'r3'])` |
| two tasks on the event-loop thread | `([], ['r1'])` | the same as above |
| two tasks, with a thread-only key (what an `RLock` would give) | — | `([], ['r1'])`: still lost |

**The review's probe A** (`<scratchpad>/rvcr_probes/test_rvcr_probe_guard.py`). Each case runs inside one `chacha_operation(independent=True)` on PostgreSQL:

| Interleaving | dev (errors, committed rows) | with the guard |
|---|---|---|
| nested in time: B enters after A and leaves before it | `[]`, `['a','b','c']` | `['ConcurrentChaChaTransactionError']`, `['a','c']` |
| a hand-off: an async caller holds a transaction and awaits `to_thread` work that opens one | `[]`, `['inner','outer']` | `['ConcurrentChaChaTransactionError']`, `['outer']` |
| `asyncio.run()` inside a sync transaction | `[]`, `['coro','sync']` | `['ConcurrentChaChaTransactionError']`, `['sync']` |
| sequential, on the loop then a worker then the loop; nesting on one task (plain and `preserve_existing=True`); entered on one thread and exited on another | all committed | all committed: no false positive |

**The one current caller, semantic conversion** (Task 1 Step 6).
- `validate_conversion` gathers two `to_thread(store.list_visible_vector_ids)`, and each lookup opens `db.transaction()`. The review's probe B ran that pair with the real store, 40 rounds:
  - dev: 0 errors, 40 of 40 later writes saved;
  - the guard: 40 of 40 raise. A `RuntimeError` reaches `create_manual_link`'s `_link_error`, so `POST /notes/{id}/links` with `semantic_conversion` answers HTTP 500.
- Task 1 Step 6's test drives `validate_conversion` itself, with both lookups held inside a real PostgreSQL transaction:

  | Code | Result |
  |---|---|
  | dev | `(None, 2)`: both lookups ran at once |
  | the guard without the fix (3 runs) | `('ConcurrentChaChaTransactionError', 2)` |
  | the guard and the fix | `(None, 1)`; with the existing unit and integration semantic-conversion tests, `38 passed` |

### F2. The persona session (TASK-13896 AC1, AC2)

- **The 13 sites on `e06eeec68c`**, found by AST:

  | Site | Lines |
  |---|---|
  | In async handlers | `persona.py:9420` (`_call_mcp_tool`); `:9953`, `:10072`, `:10141` (`_handle_persona_live_turn`); `:10614`, `:10831`, `:11072`, `:11117`, `:11294`, `:11369` (`persona_stream`'s receive loop) |
  | In the sync helpers | `:9022` (`_resolve_persona_live_event_persona_id`), `:9094` (`_upsert_persona_live_session_summary_safe`), `:9131` (`_record_persona_live_voice_event_safe`) |
  | Helper call sites | `:8747`, `:9164`, `:9168`, `:9245`, `:9250`, `:10644`, `:11423`, plus `:9084` and `:9127` inside the helpers. All are inside `async def`s |

  TASK-13896 cites origin/dev `ad946dd687` line numbers. On `e06eeec68c` each is 3 lines further down; the sites themselves are the same, as the AST scan shows.
- **AC2, measured with Task 2's test.** A live `/persona/stream` runs on a PostgreSQL handle whose 2-connection pool the test holds for 1.5 s. Then it sends `voice_config`.

  | | Each checkout: (on the loop?, seconds waited) | The server loop's longest gap |
  |---|---|---|
  | dev | `[(True, 1.5), (True, 0.0) ×4]` | 1.529 s |
  | prototype, 3 runs | `[(False, 1.5), (False, 0.0) ×4]` | 0.021-0.030 s |
- **The writes, after r1 (B2, Design choice 7).**

  | Test | dev | r1's Task 2 | r2 |
  |---|---|---|---|
  | `test_persona_ws.py::test_persona_audio_chunk_records_live_voice_commit_telemetry` | passes | 3 of 3 failed: `assert [] == [{'event_type...` | 6 of 6 passed |
  | the shield guard (new) | `([], 0) == ([], 4)` | 4 unshielded: `persona.py:9095`, `:9133`, `:10077`, `:11127` | passes |
  | a queued preferences write, its task cancelled (new; a one-worker executor held busy) | shielded: saved; plain `to_thread`: lost. This is a pin: it tests the form, not `persona.py` | the same | the same |

  The loop sweep under load (up to 11), with r2: `328 passed` and `317 passed`.

### F3. The skill tool call (TASK-13896 AC3)

`SkillsService.__init__` runs `_ensure_skill_registry_table` (`ChaChaNotes_DB.py:28413-28501`): 4 `backend.execute` statements, each with its own pool checkout. Measured with Task 3's test:
- dev: `4` checkouts on the event loop per skill call;
- prototype: `0`.

### F4. The voice assistant (TASK-13896 AC4)

| Test | dev | prototype |
|---|---|---|
| session start | `1` checkout on the loop (`save_voice_session`) | `0` |
| a full pool at session start (pool of 3, all held, `pool_timeout` 1 s) | `('PoolTimeout', [], 1.017)`: `PoolTimeout` escaped `websocket_voice_assistant`, nothing was sent, and the loop stalled 1.02 s | passes: `auth_ok` and `config_ack` sent, longest loop gap under 0.5 s |
| a command through the real router (the action stubbed) | `([['MainThread']], 2, [], 0)`: the command's checkout was held through the action, 2 checkouts were taken on the loop, and the analytics row was lost | `([[]], 0, [], 1)` |
| the external-connection lookup | `1` checkout on the loop | `0` |

**Two pre-existing PostgreSQL defects found on the way. Both are filed in Task 5.**
- **Voice commands are unusable on PostgreSQL.** The review's I1 widened this from "never load".
  - `voice_commands.deleted` and `.enabled` are `INTEGER` on PostgreSQL (`information_schema`: `('deleted', 'integer')`), but the SQLite-to-PostgreSQL transform rewrites `= 0` and `= 1` to booleans. Measured on dev:
    - `get_user_voice_commands`, `get_voice_command` (`db_helpers.py:77`), `get_voice_command_counts` (`:925`) and `delete_voice_command`: `UndefinedFunction: operator does not exist: integer = boolean`;
    - `save_voice_command` → `upsert_voice_command` (`ChaChaNotes_DB.py:9586`): `DatatypeMismatch: column "deleted" is of type integer but expression is of type boolean`, from its `ON CONFLICT … SET deleted = 0`.
  - So create, list, get, count and delete all fail, over HTTP and the WebSocket.
  - On dev, with the real registry, every PostgreSQL voice session start also raised `CharactersRAGDBError` out of `websocket_voice_assistant`. It was outside both caught tuples, so the socket died.
  - Task 4's `except Exception` degrades that session instead; the commands still do not work. This is N1, and it stays a follow-up: the fix needs a choice between a column migration and per-query casts.
- **A failed statement poisons the operation's connection.**
  - `PostgreSQLBackend.execute` rolls back a failed statement only on a connection it took itself (`if not external_conn: conn.rollback()`, `postgresql_backend.py` near `:1141`).
  - Inside an operation, the handle passes its own connection. So after a failed read the connection stays in an aborted transaction, and every later statement in the operation fails.
  - Measured on dev: the command's refresh failed (the bug above), and then its analytics insert failed: `Failed to record voice analytics: Query execution failed`, `events == 0`.
  - With per-group operations, each group has its own connection, so the analytics row is saved.

### F5. Compatibility of the guard with current callers (TASK-13899 AC2)

- **The overlap audit** (Design choice 1): one current caller overlaps, semantic conversion, which Task 1 Step 6 fixes.
- **The 13 operation-focused PostgreSQL files**, with the guard: `162 passed` with `RUN_JOBS=1`. The review reproduced it. They are:
  - `test_chacha_operation_scope.py` and `test_chacha_postgres_http_operation_lifecycle.py`;
  - the lifecycle, legacy-checkout, close_all, locking-read and per-call files;
  - the notes-graph and character_chat checkout tests;
  - `Jobs/test_chacha_jobs_workers_postgres.py`, which includes the Chat_Macros concurrent-branch run.
- **The 35 middleware test-app files**, with the full r1 prototype applied (Tasks 1-4; the review read them as guard-only), in 2 shards with `RUN_JOBS=1`:
  - shard aa: `988 passed`;
  - shard ab: everything before `RAG/test_restricted_postgres_media_retrieval.py` passed;
  - that file passed 18 tests, then failed its 19th and hung in teardown. This is TASK-13900's defect, and dev fails the same way (PR D's report): `MediaFilesRepository.get_for_media` leaves a session idle in a transaction, and `DROP OWNED BY` waits forever; pytest-timeout did not fire. Its fix is planned in `<scratchpad>/plan_fu2_pg_tests.md`. The file is excluded from every sweep since;
  - the 6 files after it, run on their own: `164 passed`.
- **The review's union sweep on r1** (258 files): 7259 passed and 13 failed.
  - 12 of the failures fail the same way on dev. 5 are TASK-13901's; the other 7 are untracked anywhere, so they are now N5.
  - The 13th was B2.
- **The union sweep on r2** (263 files, 6 shards, `RUN_JOBS=1`): `7318 passed, 12 failed, 20 skipped`, 6 shards of 20-48 min at load 4-26, no `+++ Timeout`. The 12 failures are exactly the known dev failures: TASK-13901's 5 and N5's 7. On dev their 4 files gave the same `12 failed, 200 passed, 1 skipped` (re-run here; the review measured the same). The telemetry test from B2 passed. So the branch causes no failure.

## Fix shape

| Option | Cost | What breaks | What pins it |
|---|---|---|---|
| **Raise on overlap (chosen, TASK-13899; Decision 1)** | `operation_scope.py` +32 (an error class, a field and 2 methods); `ChaChaNotes_DB.py` +11; `semantic_projector.py` +4/-4 for its one overlapping caller | Every overlap fails loudly, including one that happened to work on dev, if the audit missed its caller | Task 1's PostgreSQL tests (2 interleavings, plus semantic conversion) |
| Log only (Decision 1's alternative) | Similar size | The crossing interleaving still loses every later write; AC1 must change | — |
| Per-state `RLock` (serialize) | Similar size | A transaction that waits, inside itself, on another thread's in the same operation deadlocks; the loop thread can block; tasks on one thread are not covered (F1) | None |
| **Persona: `await asyncio.to_thread(chacha_operation(independent=True)(fn), ...)` at all 13 sites, helpers async, writes shielded and before their notices (chosen)** | `persona.py` +64/-47 | New await points: a cancel can land at more places. Started writes finish; a write not yet reached is skipped (Review Focus 6) | 3 AST guards, Task 2's two PostgreSQL tests, and `test_persona_ws.py`'s telemetry test |
| Persona: only the 10 sites the AC names | Smaller | The 3 helper sites (every voice commit, manual-mode notice, voice config and session end) still check out on the loop | — |
| **Voice: per-group operations in the router, off the loop (chosen)** | `router.py` +37/-12, `registry.py` +21/-2, `voice_assistant.py` +28/-16 | — | Task 4's 4 PostgreSQL tests |
| Voice: keep the command-wide operation and move its body to a thread | — | Holds a checkout through the LLM call, MCP action and workflow (AC4) | — |

## PR split

**One PR**, for three reasons:
- Both tasks edit ADR-080's Consequences and its `Docs/Published` mirror. Two PRs would conflict on the same file pair and serialize anyway.
- Task 1's guard is itself the check that Tasks 2-4 introduce no overlapping transactions. The union sweep runs them together.
- The product diff is +201/-83 lines in 8 files (measured on the r2 prototype). The tests are 2 new files and 3 changed ones.

Branch: `fix/chacha-off-loop-and-transaction-guard`.

**Commit plan:**
1. `fix(chacha): an overlapping transaction in one operation raises instead of losing writes (TASK-13899)`. Task 1, including the semantic-conversion fix.
2. `fix(persona): persona session calls run off the event loop, and their writes survive a cancel (TASK-13896)`. Task 2.
3. `fix(skills): the skill tool call builds SkillsService off the event loop (TASK-13896)`. Task 3.
4. `fix(voice): voice-assistant ChaCha calls run off the event loop, one operation per call group (TASK-13896)`. Task 4.
5. `docs(adr): ADR-080 records the transaction guard and the off-loop calls (TASK-13896, TASK-13899)`. Task 5 Step 1.
6. `docs(backlog): close TASK-13896 and TASK-13899; file the follow-ups; add the plan`. Task 5 Steps 4-6.

## Review Focus

1. **Ownership is keyed by `(thread ident, asyncio task)`.**
   - A transaction entered on one thread and exited on another still releases its claim (`_claimed` is per instance). A nested `db.transaction()` on the second thread while it is open would raise, though.
   - FastAPI's sync yield-dependencies enter and exit on different threadpool threads. No dependency holds a ChaCha transaction across its `yield` (17 yield-in-transaction blocks; the 2 on ChaCha handles do not cross threads). No test covers a cross-thread enter and exit.
2. **`ConcurrentChaChaTransactionError` is a `RuntimeError`, like `ClosedChaChaOperationError`.**
   - Callers that catch `RuntimeError` broadly (the persona helpers catch `(OSError, RuntimeError, ValueError, CharactersRAGDBError)`) would turn it into their fallback. The guard logs at ERROR first.
   - Making it a `CharactersRAGDBError` instead would be caught even more widely.
3. **The legacy thread-local path is not guarded.**
   - Outside an operation, two tasks on the event-loop thread that each hold a transaction across an `await` still share the loop thread's `_local` depth.
   - No such block exists on a ChaCha handle: `db_transaction`'s two callers do not await inside it.
4. **Other `tx_depth` readers still read the shared depth, and `execute_query(read_only=True)`'s own scope is not claimed.**
   - **The readers.** The locking reads (`ChaChaNotes_DB.py:29875`, `persona_state_store.py:1814`), `execute_query(read_only=True)` (`:9493`), the workspace delete check (`:30130`) and the chacha stores. A non-transactional call on thread B sees thread A's open transaction. This is unchanged and harmless today: F13's read-only fan-outs.
   - **The read scope.**
     - `execute_query(read_only=True)` opens the backend's own read scope (`backend.transaction(connection=…)`), not `db.transaction()`, and only when the connection is idle at depth 0. The guard does not claim it.
     - So a managed transaction that enters on another thread while that read scope is open on the same operation's connection is not refused.
     - The one cross-thread read found inside an operation is the clone job's cancellation check (`get_workspace`). The clone thread waits for it, except after a 5 s timeout.
     - This was not measured, and it is not in this PR.
   - **The audit's limits** (Design choice 1).
     - It is name matching within one module, and it skips modules with no ChaCha marker.
     - The reviews' spot reads are the cross-check: the gather/wait sites, a gap scan of direct-transaction fan-outs, and the MCP and Workflows fan-outs.
     - Cross-module chains are not exhaustively covered.
5. **The router joins (`chacha_operation()`).**
   - In `/voice/command` (`voice_assistant.py:310`, inside the request's operation) the groups share the request's checkout, held to request end as for any HTTP route, now taken on a worker thread.
   - Under an inherited closed operation, a group would raise `ClosedChaChaOperationError`. The router's `except Exception` (groups 1, 2) or `process_command`'s outer `except` (group 3) would catch it. No caller has one.
6. **New await points, and what a cancel can now do (B2, M1; Design choice 7).**
   - **Interleaving.** In `persona_stream` the helpers' calls, and in `_commit_persona_live_turn`, `_ensure_persona_live_manual_mode_notice` and the receive loop the 13 sites, now yield to the loop.
     - Other tasks of the same session (owned turn and control tasks) can run there. They already could at the `_emit_notice` awaits beside them.
     - The commit's double-commit flag (`current_utterance_committed`) is set before the first of them.
   - **Cancellation semantics changed.**
     - On dev a persona write ran fully or was never reached. Now a cancel can also land:
       - before a write is reached, at the preceding persona-id read;
       - while a write's executor job is queued or running.
     - The writes are shielded, so the second case finishes the write and then cancels.
     - The first case skips the write, as a cancel a moment earlier did on dev.
     - The commit's and the manual-mode notice's writes now come before their notice, so a client that closes on the notice cannot cancel them. That was the CI failure the review found.
   - **The session's `finally`.**
     - It now finalizes the summaries after cancelling and draining the owned turn tasks (M1), inside the existing `CancelScope(shield=True)`.
     - The review measured that the anyio shield does not hold against a native second `task.cancel()`. Such a cancel during finalize would skip the rest of the cleanup: the remaining finalizes, the registry clears, the STT cleanup and the handle close.
     - On dev the first await in that block was already the task `gather`, so the exposure is not new. Its window is now longer by the finalize writes.
     - In production the handle is on the shared backend, and every persona call returns its own checkout, so a skipped `close_all_connections()` leaks no checkout.
   - **A first-insert race, analytics only.**
     - `upsert_persona_live_voice_session_summary` is SELECT-then-INSERT with `UNIQUE(user_id, persona_id, session_id)`.
     - Two first upserts for one session on separate connections can collide: a turn task's and the receive loop's. The loser's `IntegrityError` is swallowed at DEBUG, and its increment is lost.
     - Updates to an existing row use atomic increments.
     - Finalize-last removes the turn-task-versus-finalize case.
   - **How long a session's teardown can now wait (the re-review's m2).** A cancelled session waits for each shielded write in flight.
     - **The drain.** The `finally`'s `gather` waits for the turn tasks' in-flight writes. They run in parallel, each bounded by about `pool_timeout` for its checkout (30 s by default), plus any default-executor queue.
     - **The finalize.** It then runs 2 sequential calls per seen session, the persona-id resolve and the upsert, each up to `pool_timeout`. That is up to about 2 × `pool_timeout` per seen session.
     - **Measured** on a held 2-connection pool with `pool_timeout` 2 s: a cancelled shielded preference write let its task finish the cancel 1.91 s later, after the write gave up on `PoolTimeout`, with nothing saved.
     - **Against dev.** Dev ran the finalize on the loop, stalling the whole server for the same time. Here only the session waits, a worker holds the wait, and nothing is held on the loop.
     - **A statement itself has no timeout** (no `statement_timeout` on the content backend). That is the same exposure dev had on the loop.
   - **CPU while shielding under an anyio scope cancel (the re-review's m3).**
     - When the awaiting task is cancelled through an anyio cancel scope, anyio re-delivers the cancel on every loop iteration. `run_quiescent_operation` catches it, uncancels and re-awaits each time, so it spins the loop's CPU until the write finishes.
     - **Measured** on a 0.45 s write: 0.133 s of loop CPU under an anyio scope cancel; 0.000-0.001 s under one or three native `task.cancel()`s.
     - **Where it applies.** Only the receive-loop task's own writes (voice_config, commit, manual mode, the receive-loop preference save), and only when the handler is cancelled through an anyio scope: TestClient, or a server that cancels WebSocket handlers through task groups.
     - **Where it does not.** The owned turn tasks and uvicorn's cancel are native, and do not spin.
     - The loop still yields between iterations, so this costs CPU and does not stall the loop. No code change.
7. **Voice session start catches `Exception`.**
   - Any failure now degrades the session, with the same sanitized warning that `test_auth_sanitizes_db_init_failure_log` pins. That covers a pool timeout, a DB error (F4's `CharactersRAGDBError`) or a programming error.
   - Dev killed the socket for anything outside `(AttributeError, OSError, RuntimeError, TypeError, ValueError)`. `CancelledError` is a `BaseException`, so it still propagates.
8. **Out of scope: the action modules' own ChaCha work in a voice command (Design choice 6; follow-up N4).**
   - A SYNC workflow's notes step still checks out on the loop, with a handle it closes. MCP modules' per-call handles were not measured on or off the loop.
   - A `PoolTimeout` inside a persona call still ends that persona session through `persona_stream`'s catch-all, as on dev, but without stalling the loop.
9. **The tests depend on measured behaviour.**
   - The full-pool tests rely on `psycopg_pool` raising `PoolTimeout` after `pool_timeout`.
   - The persona test also relies on `TestClient`'s portal running the app on its own loop, where the heartbeat task is started from the patched authenticator.
   - The queued-write test relies on `asyncio.to_thread` using the loop's default executor, which the test replaces with one worker.
   - The semantic test relies on a barrier with a 1 s timeout to hold both lookups inside their transactions at once.
10. **Voice group 1 on a failed new-session save.**
    - Group 1 loads the commands and saves a new session in one worker call. If the save fails, the freshly loaded commands are not applied: the exception leaves the thread before `apply_user_commands`.
    - Dev applied the cache before saving.
    - This is harmless: the cache is refreshed again on the next command. It is stated so it is not mistaken for a regression.

---

## Setup (once)

In your own isolated worktree:
```bash
git fetch origin dev --quiet
git switch -c fix/chacha-off-loop-and-transaction-guard origin/dev
export PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
export S=<scratchpad>
mkdir -p "$S/runs"
docker ps --format '{{.Names}} {{.Ports}}' | grep tldw_postgres_test
git diff --stat e06eeec68c origin/dev -- tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/app/api/v1/endpoints/voice_assistant.py tldw_Server_API/app/core/VoiceAssistant/router.py \
  tldw_Server_API/app/core/VoiceAssistant/registry.py tldw_Server_API/app/core/Skills/context_integration.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py \
  Docs/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md Docs/Code_Documentation/Databases/ChaChaNotes_DB.md
```
Expected:
- `tldw_postgres_test 0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp`;
- an empty stat.

If a file moved, re-anchor the line numbers below before editing. The persona script in Task 2 asserts each anchor and stops on a mismatch. If the parallel workflows PR has merged, ADR-080 and `operation_scope.py` will show changes; see Global Constraints.

Create the sweep runner `$S/fu2cr_sweep.sh` (`RUN_JOBS=1` so the Jobs-marked tests run too; it shards by file, not `-n`):
```bash
#!/bin/zsh
# usage: fu2cr_sweep.sh <list-file> <log-prefix> [shards]   (run from the repository root)
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
k=${3:-4}
n=$(( ($(wc -l < "$1") + k - 1) / k ))
split -l $n "$1" "$2.shard."
for shard in "$2".shard.a?; do
  RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q -o timeout=600 $(cat "$shard") > "$shard.log" 2>&1 &
done
wait
for shard in "$2".shard.a?; do
  echo "== $shard: $(grep -E '^=+ .*(passed|failed|error).* =+$' $shard.log | tail -1)"
  grep -E "^(FAILED|ERROR) |\+\+\+ Timeout" $shard.log | cut -c1-220
done
```
Then run `chmod +x "$S/fu2cr_sweep.sh"`.

---

### Task 1: an overlapping transaction in one operation raises (TASK-13899)

**Files:**
- Create:
  - `tldw_Server_API/tests/DB_Management/test_chacha_postgres_concurrent_transactions.py`
  - `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_conversion_postgres_operation.py` (Step 6)
- Modify:
  - `tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py`: after `ClosedChaChaOperationError` (`:22-23`), and in `ConnectionState` (`:26-92`).
  - `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`: `BackendManagedTransaction.__init__` (`:828-829`), `__enter__` (`:831-853`) and `__exit__`'s `finally` (`:888-889`).
  - `tldw_Server_API/app/core/Notes_Graph/semantic_projector.py`: `validate_conversion`'s two lookups (`:1206-1209`), in Step 6.

**Interfaces:**
- Produces:
  - `operation_scope.ConcurrentChaChaTransactionError(RuntimeError)`;
  - `ConnectionState.tx_owner: tuple[int, Any] | None`;
  - `ConnectionState.enter_transaction() -> bool`: True when this call claimed it; it raises on another key;
  - `ConnectionState.leave_transaction() -> None`.
- `BackendManagedTransaction` claims and releases the state's transaction around its existing depth logic. Its signatures and the SQLite and legacy paths are unchanged.

- [ ] **Step 1: Write the failing test**

Create `tldw_Server_API/tests/DB_Management/test_chacha_postgres_concurrent_transactions.py`:

```python
"""Two managed transactions open at once on one operation's connection state (TASK-13899).

Inside one chacha_operation every thread and task shares one ConnectionState, so one
connection and one transaction depth. A second db.transaction() that enters while
another thread's or task's is open must raise, not join it: joined, it leaves the
depth wrong, and every later transaction in the operation is silently rolled back.
"""

import asyncio
import threading
from contextvars import copy_context

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration
_INSERT = "INSERT INTO tx_probe (name) VALUES (?)"


@pytest.fixture
def tx_db(pg_database_config, tmp_path):
    """A handle on its own pool, and a reader of what was committed."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    backend.execute("CREATE TABLE tx_probe (name TEXT NOT NULL)")
    db = CharactersRAGDB(tmp_path / "tx.db", client_id="1", backend=backend)
    db.close_connection()

    def committed():
        return [row["name"] for row in backend.execute("SELECT name FROM tx_probe ORDER BY name").rows]

    try:
        yield db, committed
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def _overlap_on_two_threads(db):
    """The second transaction enters while the first is open, and leaves after it."""
    first_open, second_entered, first_left = threading.Event(), threading.Event(), threading.Event()
    outcome = []

    def first():
        with db.transaction() as conn:
            conn.execute(_INSERT, ("r1",))
            first_open.set()
            second_entered.wait(5)
        first_left.set()

    def second():
        first_open.wait(5)
        try:
            with db.transaction() as conn:
                second_entered.set()
                first_left.wait(5)
                conn.execute(_INSERT, ("r2",))
        except Exception as exc:  # noqa: BLE001 - the guard's error class does not exist before the fix
            outcome.append(type(exc).__name__)
        finally:
            second_entered.set()

    threads = [threading.Thread(target=copy_context().run, args=(fn,)) for fn in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    return outcome


async def _overlap_on_two_tasks(db):
    """The same interleaving between two tasks on the event-loop thread."""
    first_open, second_entered, first_left = asyncio.Event(), asyncio.Event(), asyncio.Event()
    outcome = []

    async def first():
        with db.transaction() as conn:
            conn.execute(_INSERT, ("r1",))
            first_open.set()
            await second_entered.wait()
        first_left.set()

    async def second():
        await first_open.wait()
        try:
            with db.transaction() as conn:
                second_entered.set()
                await first_left.wait()
                conn.execute(_INSERT, ("r2",))
        except Exception as exc:  # noqa: BLE001 - the guard's error class does not exist before the fix
            outcome.append(type(exc).__name__)
        finally:
            second_entered.set()

    await asyncio.wait_for(asyncio.gather(first(), second()), timeout=10)
    return outcome


@pytest.mark.parametrize("overlap", ["two-threads", "two-tasks-on-one-thread"])
def test_an_overlapping_transaction_raises_and_no_write_is_lost(tx_db, overlap):
    db, committed = tx_db
    with chacha_operation(independent=True):
        if overlap == "two-threads":
            outcome = _overlap_on_two_threads(db)
        else:
            outcome = asyncio.run(_overlap_on_two_tasks(db))
        # A later transaction in the same operation must still commit.
        with db.transaction() as conn:
            conn.execute(_INSERT, ("r3",))

    assert (outcome, committed()) == (["ConcurrentChaChaTransactionError"], ["r1", "r3"])
```

How the test works:
- **No import of the new error.** It catches `Exception` and records the class name, so it collects on dev and fails for the right reason.
- **What the interleaving shows.** The second transaction writes after the first commits and leaves last. On dev it joined at depth 1, wrote `r2` in a new implicit transaction, and wrote the depth back as 1. So `r3`'s transaction was unmanaged too, and the operation's close rolled both back.

- [ ] **Step 2: Run it to verify it fails**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_concurrent_transactions.py 2>&1 | grep -E "^E   AssertionError|passed|failed"
```
Expected (measured on `e06eeec68c`):
```
E   AssertionError: assert ([], ['r1']) == (['Concurrent... ['r1', 'r3'])      (x2)
======================= 2 failed, 31 warnings in 13.37s ========================
```

- [ ] **Step 3: Implement**

In `operation_scope.py`, add the error after `ClosedChaChaOperationError`:

```python
class ConcurrentChaChaTransactionError(RuntimeError):
    """Another thread or task already holds this operation's managed transaction (ADR-080)."""
```

In `ConnectionState`, add the field directly before `_uses: dict[...]`:

```python
    tx_owner: tuple[int, Any] | None = None
```

and these two methods directly after `_execution_key`:

```python
    def enter_transaction(self) -> bool:
        """Claim the managed transaction for this thread and task; True if this call claimed it.

        Every thread and task of an operation shares this one connection and one
        ``tx_depth``. A transaction entered while another thread's or task's is open
        would join it, and the depth each writes back on exit would leave every later
        transaction in the operation uncommitted. So that raises instead (ADR-080).
        Nesting on the owning thread and task joins as before.
        """
        key = self._execution_key()
        with self.lock:
            if self.tx_owner is None:
                self.tx_owner = key
                return True
            if self.tx_owner == key:
                return False
        logger.error("A ChaCha transaction overlapped another thread's or task's in one operation")
        raise ConcurrentChaChaTransactionError(
            "Another thread or task holds this ChaCha operation's transaction; "
            "give concurrent work its own chacha_operation(independent=True)"
        )

    def leave_transaction(self) -> None:
        """Release the claim taken by the outermost ``enter_transaction()``."""
        with self.lock:
            self.tx_owner = None
```

In `ChaChaNotes_DB.py`, `BackendManagedTransaction`:

```diff
         self._managed = False
         self._depth = 0
+        self._claimed = False

     def __enter__(self):
         self._state = self._db._connection_state()
         self._use = self._state.use() if isinstance(self._state, ConnectionState) else contextlib.nullcontext()
         self._use.__enter__()
+        try:
+            # An operation's threads share this state: one transaction at a time (TASK-13899).
+            self._claimed = isinstance(self._state, ConnectionState) and self._state.enter_transaction()
+        except BaseException:
+            self._use.__exit__(*sys.exc_info())
+            raise
         self._depth = getattr(self._state, "tx_depth", 0)
         try:
```
```diff
         except BaseException:
             self._state.tx_depth = self._depth
+            if self._claimed:
+                self._state.leave_transaction()
             self._use.__exit__(*sys.exc_info())
             raise
```
```diff
         finally:
             self._state.tx_depth = self._depth
+            if self._claimed:
+                self._state.leave_transaction()
             self._wrapper = None
             self._raw_conn = None
```

The claim comes before the depth is read. A refused enter therefore never writes a depth back, which could undo the owner's.

- [ ] **Step 4: Run it, and the operation-scope pins**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_concurrent_transactions.py \
  tldw_Server_API/tests/DB_Management/test_chacha_operation_scope.py 2>&1 | grep -E "^(FAILED|ERROR) |passed|failed"
```
Expected: `21 passed` (2 new + 19). `test_failed_transaction_entry_does_not_leave_an_active_checkout_use` covers the claim's release on a failed entry.

- [ ] **Step 5: Mutation check (do not commit)**

In `enter_transaction`, replace `key = self._execution_key()` with `key = (get_ident(), None)`, which is what a per-thread lock would see. Run Step 2's command, then restore the line and check `git diff --stat` shows only the intended files.

Expected (measured):
```
E   AssertionError: assert ([], ['r1']) == (['Concurrent... ['r1', 'r3'])
FAILED …::test_an_overlapping_transaction_raises_and_no_write_is_lost[two-tasks-on-one-thread]
=================== 1 failed, 1 passed …
```

- [ ] **Step 6: Semantic conversion's two lookups run one after the other (the one current overlap; the review's B1)**

`SemanticProjector.validate_conversion` gathers two `_visible_ids` lookups. Each is a `to_thread` into `NoteSemanticStore.list_visible_vector_ids`, which opens `db.transaction()` (`note_semantic_store.py:4554`). Inside the request's operation (`POST /notes/{id}/links` with `semantic_conversion`), the guard refuses the second, and the route answers 500.

Create `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_conversion_postgres_operation.py`:
```python
"""Semantic conversion's two visibility lookups inside one ChaCha operation on PostgreSQL (TASK-13899).

NoteSemanticStore.list_visible_vector_ids opens db.transaction(). Inside an HTTP request's
operation, two lookups at once would hold two managed transactions on one connection state,
which ChaCha refuses (ConcurrentChaChaTransactionError). validate_conversion therefore runs
them one after the other.
"""

import asyncio
import contextlib
import threading

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.Notes_Graph.unit import test_semantic_projector as fakes

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_db(pg_database_config, tmp_path):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "semantic.db", client_id="1", backend=backend)
    db.close_connection()
    try:
        yield db
    finally:
        db.close_connection()
        backend.get_pool().close_all()


async def test_conversion_lookups_inside_one_operation_do_not_overlap(pg_db):
    source = fakes._projection_chunk(
        generation_id="generation-a", note_id="focus-note", title="Focus", content="alpha beta gamma"
    )
    target = fakes._projection_chunk(
        generation_id="generation-a", note_id="target-note", title="Target", content="alpha beta delta"
    )
    store = fakes._Store(source, target)
    vectors = fakes._VectorStore(source.vector_id, target.vector_id)
    vectors.distance = 0.4
    projector, _service, _factory_calls = fakes._projector(
        store=store, vectors=vectors, ordinary=fakes._ordinary_response()
    )
    fake_lookup, guard = store.list_visible_vector_ids, threading.Lock()
    in_flight, peak, both_inside = [0], [0], threading.Barrier(2)

    def lookup_in_a_postgres_transaction(dataset_id, generation_id, note_id=None):
        with guard:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        try:
            with pg_db.transaction():  # what NoteSemanticStore.list_visible_vector_ids opens
                pg_db.note_semantic_store.list_visible_vector_ids(dataset_id, generation_id, note_id)
                with contextlib.suppress(threading.BrokenBarrierError):
                    both_inside.wait(timeout=1.0)  # two concurrent lookups are surely inside together
                return fake_lookup(dataset_id, generation_id, note_id)
        finally:
            with guard:
                in_flight[0] -= 1

    store.list_visible_vector_ids = lookup_in_a_postgres_transaction
    outcome = None
    with chacha_operation(independent=True):  # the request's operation
        try:
            await projector.validate_conversion(
                source_note_id="focus-note", target_note_id="target-note", generation_id="generation-a"
            )
        except Exception as exc:  # noqa: BLE001 - the RED records the guard's error by name
            outcome = type(exc).__name__
        for _ in range(60):  # a lookup that gather left behind finishes before the operation closes
            if not in_flight[0]:
                break
            await asyncio.sleep(0.05)

    assert (outcome, peak[0]) == (None, 1)
```

How the test is built:
- **The fakes.** It reuses the unit tests' projector fakes (`tests/Notes_Graph/unit/test_semantic_projector.py`) for configuration, chunks and vectors.
- **The real transaction.** Each lookup runs inside a real PostgreSQL `db.transaction()` on the operation's state, and calls the real `NoteSemanticStore.list_visible_vector_ids`, which nests and joins.
- **The barrier.** It makes two concurrent lookups certainly overlap. With sequential lookups it times out once (1 s) and then passes through.
- **The wait at the end.** Without it, a lookup that `gather` left behind would outlive the test.

Run it with the guard in place:
```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Notes_Graph/integration/test_semantic_conversion_postgres_operation.py 2>&1 | grep -E "^E   AssertionError|passed|failed"
```
Expected (measured, 3 of 3): `E   AssertionError: assert ('ConcurrentC...tionError', 2) == (None, 1)`, then `1 failed`. On dev, without the guard, it gives `(None, 2)`: both lookups ran at once on one connection state.

Then in `semantic_projector.py`, replace the gather (dev `:1206-1209`):
```diff
-        source_ids, target_ids = await asyncio.gather(
-            self._visible_ids(generation_id, source_note_id),
-            self._visible_ids(generation_id, target_note_id),
-        )
+        # One after the other: each lookup opens a ChaCha transaction, and inside a request's
+        # operation two at once would share one connection state, which ChaCha refuses (ADR-080).
+        source_ids = await self._visible_ids(generation_id, source_note_id)
+        target_ids = await self._visible_ids(generation_id, target_note_id)
```
Each lookup is one indexed read, a few ms, so running them in turn costs about one lookup's latency. It opens no extra checkout; per-lookup independent operations would cost one each. `asyncio` is still used in the module.

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Notes_Graph/integration/test_semantic_conversion_postgres_operation.py \
  tldw_Server_API/tests/Notes_Graph/unit/test_semantic_projector.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_semantic_manual_conversion.py 2>&1 | grep -E "^(FAILED|ERROR) |passed|failed"
$PY -m ruff check tldw_Server_API/app/core/Notes_Graph/semantic_projector.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_semantic_conversion_postgres_operation.py
```
Expected (measured): `38 passed` and `All checks passed!`.

- [ ] **Step 7: The operation-focused PostgreSQL files (TASK-13899 AC2)**

```bash
RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q -o timeout=600 \
  tldw_Server_API/tests/DB_Management/test_chacha_operation_scope.py \
  tldw_Server_API/tests/DB_Management/test_chacha_per_call_handles_postgres.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_close_all_connections.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_http_operation_lifecycle.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_legacy_checkouts.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_locking_reads_in_operation.py \
  tldw_Server_API/tests/DB_Management/test_chat_retry_postgres_operation_lifecycle.py \
  tldw_Server_API/tests/DB_Management/test_conversation_enrichment_operation_lifecycle.py \
  tldw_Server_API/tests/DB_Management/test_study_pack_worker_operation_lifecycle.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_job_postgres_checkouts.py \
  tldw_Server_API/tests/Workflows/adapters/test_character_chat_adapter_postgres_operation.py \
  tldw_Server_API/tests/Jobs/test_chacha_jobs_workers_postgres.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_concurrent_transactions.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_semantic_conversion_postgres_operation.py 2>&1 | grep -E "^(FAILED|ERROR) |passed|failed"
```
Expected (measured on r2, 270 s at load ~7): `163 passed`, with no `FAILED` or `ERROR` lines. That is r1's `162 passed` (measured twice: 484 s and the review's 339 s) plus Step 6's test.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/Notes_Graph/semantic_projector.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_concurrent_transactions.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_semantic_conversion_postgres_operation.py
git commit -m "$(cat <<'EOF'
fix(chacha): an overlapping transaction in one operation raises instead of losing writes (TASK-13899)

Inside one chacha_operation every thread and task shares one
ConnectionState, and BackendManagedTransaction kept tx_depth on it
without an owner. A transaction that entered while another thread's or
task's was open joined it and wrote its stale depth back on exit, so
every later transaction in the operation was unmanaged and was rolled
back when the operation returned its checkout. Measured: ([], ['r1'])
for both a two-thread and a two-task overlap; r2 and r3 were lost.

The outermost transaction now claims the state for its thread and task.
Nesting on the same thread and task joins as before; any other thread
or task gets ConcurrentChaChaTransactionError, logged at ERROR. It
raises rather than waits: a lock would deadlock a transaction that
waits on another thread's, would block the event loop, and, per thread,
would not see two tasks on the loop. Overlaps that happened to commit
on dev (nested in time, a hand-off to a thread) are refused too.

One current caller overlapped: semantic conversion gathered two
visibility lookups, each opening db.transaction(), inside the request's
operation, so POST /notes/{id}/links with semantic_conversion would
answer 500. It now runs them one after the other. A name-based audit
of the 135 fan-outs in the 268 app files that name ChaCha, and the
reviews' spot reads, found no other; cross-module chains are not
exhaustively covered.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: persona session calls run off the event loop, and their writes survive a cancel (TASK-13896 AC1, AC2)

**Files:**
- Modify:
  - `tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py`: the docstring, imports, the off-loop guard's count, the on-loop guard (replaced), a shield guard, and two new PostgreSQL tests.
  - `tldw_Server_API/app/api/v1/endpoints/persona.py`:
    - the 13 sites, the 3 helper definitions and the 9 helper calls of F2;
    - then r2's edits: the 4 shielded writes, the two writes-before-notice moves, the finalize move and one import.

**Interfaces:**
- Consumes:
  - `chacha_operation`, already imported at `persona.py:167`; `asyncio`, at `:6`;
  - `run_quiescent_operation` from `tldw_Server_API.app.core.Notes_Graph.semantic_publication` (exported in its `__all__`; that module imports only Notes_Graph internals and chacha models).
- Produces:
  - every use of `persona_scope_db` in `persona.py` is either self-owning (`_SELF_OWNING`) or `await asyncio.to_thread(chacha_operation(independent=True)(fn), ...)`;
  - the 4 writes run inside `run_quiescent_operation(...)`;
  - `_resolve_persona_live_event_persona_id`, `_upsert_persona_live_session_summary_safe` and `_record_persona_live_voice_event_safe` are `async def`, and every call is awaited.

  Signatures are otherwise unchanged.

- [ ] **Step 1: Write the failing tests**

In `test_persona_postgres_per_call_operations.py`:

Replace the module docstring with:
```python
"""The persona live WebSocket's per-session handle returns each checkout per call (TASK-13861), off the event loop (TASK-13896)."""
```

Add `import time` after `import threading`, and `from types import SimpleNamespace` after `from pathlib import Path`.

In `test_every_off_loop_persona_handle_call_runs_in_its_own_operation`, replace `assert wrapped == 8` with:
```python
    assert wrapped == 21  # persona.py's 19 (TASK-13896 moved its 13 on-loop calls here) and the helpers' 2
```

Replace everything from `_SELF_OWNING = {` through the end of `test_every_on_loop_persona_handle_call_runs_in_its_own_operation` (the `assert wrapped == 13` line) with:
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
# The session's analytics helpers close over the handle; they are async, so every call is awaited.
_HANDLE_HELPERS = {
    "_resolve_persona_live_event_persona_id",
    "_upsert_persona_live_session_summary_safe",
    "_record_persona_live_voice_event_safe",
}


def test_no_persona_handle_call_runs_on_the_event_loop():
    """On the loop, a call's checkout would stall the whole server for pool_timeout when the pool is full."""
    tree = ast.parse((_APP / "api/v1/endpoints/persona.py").read_text(encoding="utf-8"))
    awaited = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Await)}
    on_loop, unawaited = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if ast.unparse(target) in _HANDLE_HELPERS and id(node) not in awaited:
            unawaited.append(f"persona.py:{node.lineno} {ast.unparse(target)}")
        per_call = isinstance(target, ast.Call) and ast.unparse(target.func).startswith("chacha_operation(")
        callee = target.args[0] if per_call else target
        passes_handle = "persona_scope_db" in {ast.unparse(v) for v in [*node.args, *(k.value for k in node.keywords)]}
        calls_handle = isinstance(callee, ast.Attribute) and ast.unparse(callee.value) == "persona_scope_db"
        if (passes_handle or calls_handle) and ast.unparse(callee) not in _SELF_OWNING:
            on_loop.append(f"persona.py:{node.lineno} {ast.unparse(target)}")
    assert on_loop == [], f"run these as await asyncio.to_thread(chacha_operation(independent=True)(fn), ...): {on_loop}"
    assert unawaited == [], f"await these: {unawaited}"
```

The direct form `chacha_operation(independent=True)(fn)(...)` that B5's guard required is now a violation itself. The off-loop guard's count of 21 keeps both guards from passing vacuously.

Directly after `test_no_persona_handle_call_runs_on_the_event_loop`, add the shield guard:
```python
# The session's writes. Each must finish once queued, even if the session's task is cancelled meanwhile.
_WRITES = {
    "persona_scope_db.upsert_persona_live_voice_session_summary",
    "record_persona_live_voice_event",
    "_persist_persona_session_preferences",
}


def test_every_persona_write_is_shielded_once_queued():
    """A cancel that lands while a write's executor job is still queued would drop the write."""
    tree = ast.parse((_APP / "api/v1/endpoints/persona.py").read_text(encoding="utf-8"))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    unshielded, shielded = [], 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "asyncio.to_thread" and node.args):
            continue
        wrapped = node.args[0]
        inner = ast.unparse(wrapped.args[0]) if isinstance(wrapped, ast.Call) and wrapped.args else ""
        if inner not in _WRITES:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and ast.unparse(parent.func) == "run_quiescent_operation":
            shielded += 1
        else:
            unshielded.append(f"persona.py:{node.lineno} {inner}")
    assert (unshielded, shielded) == ([], 4), "wrap these as run_quiescent_operation(asyncio.to_thread(...))"
```

Append at the end of the file:
```python
def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@pytest.fixture
def live_session_on_a_small_pool(pg_database_config, tmp_path, monkeypatch):
    """A live persona WebSocket over a PostgreSQL handle whose two-connection pool the test can fill.

    It records, for each checkout, whether it was taken on an event-loop thread and how long
    it waited, and the server loop's gaps between 20 ms heartbeat ticks.
    """
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
    from tldw_Server_API.tests.Persona.test_persona_ws import fastapi_app

    pg_database_config.pool_size = 2
    pg_database_config.pool_timeout = 10.0
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    original_get = pool.get_connection
    takes, gaps, stop = [], [], threading.Event()

    def get_connection():
        started = time.monotonic()
        raw = original_get()
        takes.append((_on_event_loop(), round(time.monotonic() - started, 1)))
        return raw

    monkeypatch.setattr(pool, "get_connection", get_connection)
    db = CharactersRAGDB(tmp_path / "persona.db", client_id="1", backend=backend)
    persona_id = db.create_persona_profile(
        {"id": "research_assistant", "user_id": "1", "name": "Research Assistant", "mode": "session_scoped",
         "system_prompt": "Helper", "is_active": True}
    )
    db.create_persona_session(
        {"id": "pg-live", "persona_id": persona_id, "user_id": "1", "mode": "session_scoped", "status": "active"}
    )
    db.close_connection()

    async def heartbeat():
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    beats = []

    async def authenticate(*_args, **_kwargs):
        beats.append(asyncio.get_running_loop().create_task(heartbeat()))  # on the server's loop
        return "1", True, True

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", authenticate)
    monkeypatch.setattr(persona_ep, "_open_persona_ws_db", lambda _user_id: db)
    try:
        with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
            ws.receive_json()
            takes.clear()
            gaps.clear()
            yield SimpleNamespace(ws=ws, pool=pool, hold=original_get, takes=takes, gaps=gaps)
            stop.set()
    finally:
        stop.set()
        db.close_connection()
        pool.close_all()


def test_a_persona_call_waits_for_a_full_pool_without_stalling_the_event_loop(live_session_on_a_small_pool):
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    live = live_session_on_a_small_pool
    held = [live.hold() for _ in range(2)]  # other work holds the whole pool for 1.5 s
    release = threading.Timer(1.5, lambda: [live.pool.return_connection(conn) for conn in held])
    release.start()
    try:
        live.ws.send_json({"type": "voice_config", "session_id": "pg-live", "stt": {"model": "tiny"}})
        _recv_until(live.ws, lambda event: event.get("reason_code") == "VOICE_CONFIG_UPDATED", timeout=20)
    finally:
        release.join()

    on_loop = [waited for loop, waited in live.takes if loop]
    longest_wait = max((waited for loop, waited in live.takes if not loop), default=0.0)
    assert (on_loop, longest_wait >= 1.0, max(live.gaps) < 0.75) == ([], True, True), (live.takes, max(live.gaps))


@pytest.mark.parametrize("shielded", [True, False], ids=["shielded", "plain-to-thread"])
async def test_a_queued_preferences_write_survives_its_task_being_cancelled(pg_database_config, tmp_path, shielded):
    """The shield persona.py puts around its writes: a cancel while the write is still queued must not drop it.

    The default executor gets one worker, held busy, so the write's job is queued when its task is cancelled,
    as when a session ends while the executor is saturated. A plain to_thread drops the queued job.
    """
    from concurrent.futures import ThreadPoolExecutor

    from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
    from tldw_Server_API.app.core.Notes_Graph.semantic_publication import run_quiescent_operation

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "persona.db", client_id="1", backend=backend)
    persona_id = db.create_persona_profile(
        {"id": "research_assistant", "user_id": "1", "name": "Research Assistant", "mode": "session_scoped",
         "system_prompt": "Helper", "is_active": True}
    )
    db.create_persona_session(
        {"id": "pg-live", "persona_id": persona_id, "user_id": "1", "mode": "session_scoped", "status": "active"}
    )
    db.close_connection()
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    release = threading.Event()
    busy = loop.run_in_executor(None, release.wait, 5)  # the only worker is taken
    write = asyncio.to_thread(
        chacha_operation(independent=True)(persona_ep._persist_persona_session_preferences),
        db, session_id="pg-live", user_id="1", base_preferences={}, patch_preferences={"use_memory_context": False},
    )
    task = asyncio.create_task(run_quiescent_operation(write) if shielded else write)
    try:
        await asyncio.sleep(0.1)  # the write's job is queued behind the busy worker
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await busy
        await asyncio.to_thread(time.sleep, 0)  # anything still queued has run
        with chacha_operation(independent=True):
            saved = db.get_persona_session("pg-live", user_id="1")["preferences"]
        assert saved.get("use_memory_context") is (False if shielded else None)
    finally:
        release.set()
        db.close_connection()
        backend.get_pool().close_all()
```

The queued-write test pins the form that `persona.py` uses, with the real `_persist_persona_session_preferences` on PostgreSQL. Its `plain-to-thread` case shows the loss deterministically. It is a pin, not a RED: it passes on any `persona.py`. The shield guard pins that `persona.py` uses the form.

What the behavioural test drives:
- **The real handler.** It drives the real `persona_stream` through the test app the other persona WebSocket tests use. `voice_config` reaches the policy-rules site at dev `:10614` and the summary helper at `:10644`.
- **The server's loop.** The heartbeat runs on the server's loop: `TestClient`'s portal thread. It is started from the patched authenticator, which runs there.
- **Teardown.** At session end `persona_stream` closes the handle (`close_all_connections()`). On this caller-owned pool that closes the pool, which the fixture tolerates.

- [ ] **Step 2: Run them to verify they fail**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q -o timeout=300 \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py 2>&1 \
  | grep -v " | " | grep -E "^E   (Assert|assert|  Left)|^FAILED|passed|failed" | cut -c1-200
```
Expected (measured on `e06eeec68c`):
```
E   assert 8 == 21
E   AssertionError: run these as await asyncio.to_thread(chacha_operation(independent=True)(fn), ...): ['persona.py:9420 chacha_operation(independent=True)(_load_persona_scope_metadata_for_session)', …
E     Left contains 13 more items, first extra item: 'persona.py:9420 chacha_operation(independent=True)(_load_persona_scope_metadata_for_session)'
E   AssertionError: wrap these as run_quiescent_operation(asyncio.to_thread(...))
E   assert ([], 0) == ([], 4)
E   AssertionError: ([(True, 1.5), (True, 0.0), (True, 0.0), (True, 0.0), (True, 0.0)], 1.533…)
E   assert ([1.5, 0.0, 0... False, False) == ([], True, True)
==== 4 failed, 3 passed … ====
```
Three tests pass on dev and after:
- the pin `test_concurrent_persona_sessions_hold_no_checkout_between_calls`;
- the two queued-write cases, which pin the form itself.

The shield guard also fails on r1's edit alone (`fu2cr_persona_edit.py` without r2's), naming the four writes: `(['persona.py:10077 _persist_persona_session_preferences', 'persona.py:9095 …upsert_persona_live_voice_session_summary', 'persona.py:9133 record_persona_live_voice_event', 'persona.py:11127 …'], 0)`.

- [ ] **Step 3: Implement**

Create `$S/fu2cr_persona_edit.py`. It asserts every anchor on `e06eeec68c` before it rewrites anything:
```python
"""TASK-13896's persona.py change; asserts every anchor first (origin/dev e06eeec68c line numbers).

- the 10 direct calls in async handlers become `await asyncio.to_thread(chacha_operation(independent=True)(fn), ...)`;
- the 3 sync helpers become `async def`, with the same form for their ChaCha call;
- their 7 call sites (all inside `async def`s) and the 2 internal resolver calls gain `await`.
"""
import pathlib
import re

path = pathlib.Path("tldw_Server_API/app/api/v1/endpoints/persona.py")
lines = path.read_text(encoding="utf-8").split("\n")
DIRECT = [9420, 9953, 10072, 10141, 10614, 10831, 11072, 11117, 11294, 11369]  # in async defs
HELPER_SITES = [9022, 9094, 9131]  # inside the three sync helpers
HELPER_DEFS = {9018: "_resolve_persona_live_event_persona_id", 9069: "_upsert_persona_live_session_summary_safe",
               9115: "_record_persona_live_voice_event_safe"}
HELPER_CALLS = [8747, 9084, 9127, 9164, 9168, 9245, 9250, 10644, 11423]
pattern = re.compile(r"^(?P<indent>\s*)(?P<lhs>(?:\w+ = )?)chacha_operation\(independent=True\)\((?P<fn>[\w.]+)\)\($")

for no in DIRECT + HELPER_SITES:
    m = pattern.match(lines[no - 1])
    assert m, (no, lines[no - 1])
    inner = m["indent"] + "    "
    lines[no - 1] = f"{m['indent']}{m['lhs']}await asyncio.to_thread(\n{inner}chacha_operation(independent=True)({m['fn']}),"
for no, name in HELPER_DEFS.items():
    assert lines[no - 1].strip().startswith(f"def {name}("), (no, lines[no - 1])
    lines[no - 1] = lines[no - 1].replace(f"def {name}(", f"async def {name}(", 1)
for no in HELPER_CALLS:
    text = lines[no - 1]
    name = next(n for n in HELPER_DEFS.values() if f"{n}(" in text)
    assert text.strip().startswith(name) or f"= {name}(" in text, (no, text)
    lines[no - 1] = text.replace(f"{name}(", f"await {name}(", 1)
path.write_text("\n".join(lines), encoding="utf-8")
print("rewrote", len(DIRECT) + len(HELPER_SITES), "calls,", len(HELPER_DEFS), "defs,", len(HELPER_CALLS), "helper calls")
```
Run it: `$PY $S/fu2cr_persona_edit.py`. Expected: `rewrote 13 calls, 3 defs, 9 helper calls`.

The rewrite at each of the 13 sites keeps the argument lines and their indentation:

| Dev line | Before | After |
|---|---|---|
| `:9953`, `:10614`, `:10831`, `:11072`, `:11294`, `:11369`, `:9022` | `runtime_context = chacha_operation(independent=True)(_load_persona_policy_rules_for_session)(` | `runtime_context = await asyncio.to_thread(` / `    chacha_operation(independent=True)(_load_persona_policy_rules_for_session),` |
| `:9420` | `scope_metadata = chacha_operation(independent=True)(_load_persona_scope_metadata_for_session)(` | `scope_metadata = await asyncio.to_thread(` / `    chacha_operation(independent=True)(_load_persona_scope_metadata_for_session),` |
| `:10072`, `:11117` | `_ = chacha_operation(independent=True)(_persist_persona_session_preferences)(` | `_ = await asyncio.to_thread(` / `    chacha_operation(independent=True)(_persist_persona_session_preferences),` |
| `:10141` | `persona_state_hints = chacha_operation(independent=True)(_load_persona_state_hints_for_runtime)(` | `persona_state_hints = await asyncio.to_thread(` / `    chacha_operation(independent=True)(_load_persona_state_hints_for_runtime),` |
| `:9094` | `chacha_operation(independent=True)(persona_scope_db.upsert_persona_live_voice_session_summary)(` | `await asyncio.to_thread(` / `    chacha_operation(independent=True)(persona_scope_db.upsert_persona_live_voice_session_summary),` |
| `:9131` | `chacha_operation(independent=True)(record_persona_live_voice_event)(` | `await asyncio.to_thread(` / `    chacha_operation(independent=True)(record_persona_live_voice_event),` |

For example, dev `:10614-10618` becomes:
```python
                runtime_context = await asyncio.to_thread(
                    chacha_operation(independent=True)(_load_persona_policy_rules_for_session),
                    persona_scope_db,
                    session_id=session_id,
                    user_id=authenticated_user_id,
                )
```
and the helper at dev `:9018` becomes:
```python
        async def _resolve_persona_live_event_persona_id(session_id: str) -> str | None:
            if persona_scope_db is None or not authenticated_user_id:
                return None
            try:
                runtime_context = await asyncio.to_thread(
                    chacha_operation(independent=True)(_load_persona_policy_rules_for_session),
                    persona_scope_db,
                    session_id=session_id,
                    user_id=authenticated_user_id,
                )
            except Exception as exc:
```
The helpers' `except Exception` still catches a ChaCha failure, including `PoolTimeout`, and `CancelledError` still propagates. `persona_live_summary_sessions_seen.add(session_id)` stays on the loop (Design choice 2).

Then the r2 edits: shield the writes, write before notifying, and finalize last (Design choice 7). Create `$S/fu2cr_persona_edit_r2.py`. It runs on the output of `fu2cr_persona_edit.py` and asserts every block it moves:
```python
"""TASK-13896 r2 persona.py edits, run after fu2cr_persona_edit.py; asserts every block first.

1. Started writes are shielded: the 4 write calls run as
   `await run_quiescent_operation(asyncio.to_thread(chacha_operation(independent=True)(fn), ...))`,
   so a cancel of the awaiting task lets a started write finish, then propagates.
2. Write first, then notify: the commit's and the manual-mode notice's analytics writes run
   before their notice, so a client that closes on the notice cannot cancel them.
3. The session's finalize upserts run after the owned tasks are cancelled and drained.
"""
import ast
import pathlib

path = pathlib.Path("tldw_Server_API/app/api/v1/endpoints/persona.py")
src = path.read_text(encoding="utf-8")


def replace_once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:120])
    return text.replace(old, new, 1)


# 1. Shield the 4 writes. Locate each asyncio.to_thread call by its callable, then wrap it.
WRITES = {
    "chacha_operation(independent=True)(persona_scope_db.upsert_persona_live_voice_session_summary)",
    "chacha_operation(independent=True)(record_persona_live_voice_event)",
    "chacha_operation(independent=True)(_persist_persona_session_preferences)",
}
lines = src.split("\n")
calls = [
    node for node in ast.walk(ast.parse(src))
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "asyncio.to_thread"
    and node.args and ast.unparse(node.args[0]) in WRITES
]
assert len(calls) == 4, len(calls)
for node in sorted(calls, key=lambda n: n.lineno, reverse=True):  # bottom-up keeps earlier positions valid
    end = lines[node.end_lineno - 1]
    lines[node.end_lineno - 1] = end[: node.end_col_offset] + ")" + end[node.end_col_offset:]
    start = lines[node.lineno - 1]
    assert start[node.col_offset:].startswith("asyncio.to_thread("), start
    lines[node.lineno - 1] = start[: node.col_offset] + "run_quiescent_operation(" + start[node.col_offset:]
src = "\n".join(lines)

src = replace_once(
    src,
    "from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation\n",
    "from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation\n"
    "from tldw_Server_API.app.core.Notes_Graph.semantic_publication import run_quiescent_operation\n",
)

# 2. Write first, then notify.
manual_notice = '''            await _emit_notice(
                session_id=session_id,
                client_message_id=state.get("client_message_id"),
                level="warning",
                reason_code="VOICE_MANUAL_MODE_REQUIRED",
                message="Server VAD unavailable for this live session. Use Send now to commit heard speech manually.",
                details=str(state.get("manual_mode_reason") or "vad_unavailable"),
            )
'''
manual_writes = '''            await _record_persona_live_voice_event_safe(
                session_id=session_id,
                event_type="manual_mode_required",
            )
            await _upsert_persona_live_session_summary_safe(
                session_id=session_id,
                voice_runtime=state.get("voice_runtime") if isinstance(state, dict) else None,
                manual_mode_required_increment=1,
            )
'''
src = replace_once(
    src,
    manual_notice + manual_writes,
    "            # Write first, then notify: a client may close on the notice (TASK-13896).\n"
    + manual_writes + manual_notice,
)
commit_notice = '''            await _emit_notice(
                session_id=session_id,
                level="info",
                reason_code="VOICE_TURN_COMMITTED",
                client_message_id=client_message_id,
                message="Voice turn committed.",
                commit_source=commit_source,
                transcript=cleaned_transcript,
            )
'''
commit_writes = '''            await _record_persona_live_voice_event_safe(
                session_id=session_id,
                event_type="commit",
                commit_source=commit_source,
            )
            await _upsert_persona_live_session_summary_safe(
                session_id=session_id,
                voice_runtime=voice_runtime if isinstance(voice_runtime, dict) else None,
                commit_source=commit_source,
            )
'''
src = replace_once(
    src,
    commit_notice + commit_writes,
    "            # Write first, then notify: a client may close on the notice (TASK-13896).\n"
    + commit_writes + commit_notice,
)

# 3. Finalize after the owned tasks are cancelled and drained.
finalize = '''            for session_id in list(persona_live_summary_sessions_seen):
                await _upsert_persona_live_session_summary_safe(
                    session_id=session_id, voice_runtime=None, finalize=True,
                )
'''
src = replace_once(src, finalize, "")
drained = '''            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
'''
src = replace_once(
    src,
    drained,
    drained + "            # Finalize last: the owned turn tasks, which also write summaries, are done.\n" + finalize,
)
path.write_text(src, encoding="utf-8")
print("shielded 4 writes; moved 2 write blocks before their notices; moved finalize after the drain")
```
Run it: `$PY $S/fu2cr_persona_edit_r2.py`. Expected: `shielded 4 writes; moved 2 write blocks before their notices; moved finalize after the drain`.

For example, the summary upsert inside its helper becomes:
```python
                await run_quiescent_operation(asyncio.to_thread(
                    chacha_operation(independent=True)(persona_scope_db.upsert_persona_live_voice_session_summary),
                    user_id=user_id_int,
                    …
                    finalize=finalize,
                ))
```
The helpers' `except Exception` catches a failed write as before. `run_quiescent_operation` re-raises the caller's cancellation once the write has finished. The cancellation is a `BaseException`, so it propagates.

Then confirm the file parses and the diff is the expected size:
```bash
$PY -c "import ast,pathlib;ast.parse(pathlib.Path('tldw_Server_API/app/api/v1/endpoints/persona.py').read_text())" && echo parsed-ok
git diff --numstat tldw_Server_API/app/api/v1/endpoints/persona.py
```
Expected (measured): `parsed-ok`, and `64	47	tldw_Server_API/app/api/v1/endpoints/persona.py`.

- [ ] **Step 4: Run the tests, and the guard's mutation check**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q -o timeout=300 \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py 2>&1 | grep -E "^(FAILED|ERROR) |passed|failed"
```
Expected (measured): `7 passed`.

Mutation (do not commit): remove `await ` from the `await _upsert_persona_live_session_summary_safe(` that follows `preferences={"voice_runtime": voice_runtime},` in the `voice_config` branch. Run only `test_no_persona_handle_call_runs_on_the_event_loop`, then restore the file and check it with `git diff --stat`.

Expected (measured; branch line numbers):
```
E   AssertionError: await these: ['persona.py:10655 _upsert_persona_live_session_summary_safe']
==== 1 failed …
```

The existing telemetry test that r1 broke (B2), run six times:
```bash
for i in 1 2 3 4 5 6; do TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  "tldw_Server_API/tests/Persona/test_persona_ws.py::test_persona_audio_chunk_records_live_voice_commit_telemetry" 2>&1 | grep -E "passed|failed"; done
```
Expected (measured): six lines of `1 passed`. With r1's edit alone it failed 3 of 3 here (`assert [] == [{'event_type...`), and 6 of 6 for the review.

- [ ] **Step 5: Ruff**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/api/v1/endpoints/persona.py | grep -oE ": [A-Z]+[0-9]+" | sort | uniq -c
$PY -m ruff check tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py
```
Expected: `1 : I001` and `1 : SIM114` (dev's), and `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py
git commit -m "$(cat <<'EOF'
fix(persona): persona session calls run off the event loop, and their writes survive a cancel (TASK-13896)

TASK-13861 gave each persona session call on the event loop its own
ChaCha operation, but the operation still checked out on the loop
thread. When other work held the whole content pool, each such call
blocked the whole server for up to pool_timeout. Measured: with a
2-connection pool held for 1.5 s, voice_config's checkouts were all
taken on the loop and the server loop stalled 1.53 s.

All 13 calls now run as await asyncio.to_thread(chacha_operation(
independent=True)(fn), ...): the 10 in async handlers and the 3 in the
analytics helpers, which become async so their bookkeeping stays on
the loop. The same test: every checkout on a worker thread, the first
waiting 1.5 s, and loop gaps of 0.02-0.03 s.

An awaited write can now be cancelled before its executor job runs. So
the commit's and the manual-mode notice's analytics writes run before
their notice (a client that closed on the notice lost the commit's
telemetry), the four writes run under run_quiescent_operation so a
queued or running write finishes before the cancel propagates, and the
session finalizes its summaries after its turn tasks are drained. The
AST guards require the off-loop form, an await on every helper call
and the shield on every write.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: the skill tool call builds `SkillsService` off the event loop (TASK-13896 AC3)

**Files:**
- Modify:
  - `tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py`: the docstring, imports, the fixture records each checkout's thread kind, and a new test.
  - `tldw_Server_API/app/core/Skills/context_integration.py`: the imports (`:15`) and the lookup block (`:144-149`).

**Interfaces:**
- `handle_skill_tool_call` is unchanged in signature and results.
- `SkillsService(...)` now runs in `asyncio.to_thread` inside the existing `with chacha_operation(independent=True):`. Its 4 table statements, which take their own pool connections through `backend.execute`, run on a worker thread.

- [ ] **Step 1: Write the failing test**

In the test file:
- Replace the docstring's last sentence, "Only the registry lookup before it touches ChaCha.", with:
  ```
  Only the registry lookup before it touches ChaCha, and neither it nor
  the service's setup checks out on the event loop (TASK-13896).
  ```
- Add `import asyncio` above `import threading`.
- In the fixture:
  - change `holders, taken, returned = {}, Counter(), Counter()` to `holders, taken, returned, on_loop = {}, Counter(), Counter(), []`;
  - make `get_connection` start with the lines in the block below, at the function body's 8-space indentation;
  - add `on_loop.clear()` before `try:`, and pass `on_loop=on_loop` in the yielded `SimpleNamespace`.

The fixture's `get_connection` then begins:
```python
    def get_connection():
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        raw = original_get()
```
- Append:
```python
async def test_the_skill_setup_and_lookup_take_no_checkout_on_the_event_loop(skills_owner, monkeypatch):
    class InlineExecutor:
        async def execute(self, *, skill_data, arguments, context):  # noqa: ARG002
            return SimpleNamespace(
                skill_name=skill_data["name"], rendered_prompt=arguments, allowed_tools=None,
                model_override=None, execution_mode="inline", fork_output=None,
            )

    monkeypatch.setattr(context_integration, "SkillExecutor", InlineExecutor)

    result = await context_integration.handle_skill_tool_call(
        skill_name="summarize", args="text", user_id=1, base_path=skills_owner.base_path, db=skills_owner.db
    )

    assert (result["success"], len(skills_owner.on_loop) > 0, sum(skills_owner.on_loop)) == (True, True, 0)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py 2>&1 | grep -E "^E   assert|passed|failed"
```
Expected (measured on `e06eeec68c`): `E   assert (True, True, 4) == (True, True, 0)`, then `1 failed, 1 passed`. The 4 are `_ensure_skill_registry_table`'s statements.

- [ ] **Step 3: Implement**

In `context_integration.py`, add `import asyncio` above `from pathlib import Path`, and replace the lookup block (dev `:144-149`):
```python
        # The registry lookup gets its own operation, so its checkout is returned before a
        # forked skill's LLM call, whatever operation the caller holds. The service's setup
        # (4 table statements) runs off the event loop, like the lookup itself.
        with chacha_operation(independent=True):
            service = await asyncio.to_thread(SkillsService, user_id=user_id, base_path=base_path, db=db)
            skill_data = await service.get_skill(skill_name)
```

How the setup's errors and checkouts behave now:
- `SkillsService.__init__` raises `SkillsError` or `SkillStorageError`. They now come through the `await` to the same `except` clauses.
- `get_skill` already offloads (`_sync_registry_async`, then `to_thread(_get_skill_sync)`).
- The operation is opened on the loop, but it checks out only at first use, which happens on a worker thread.

- [ ] **Step 4: Run them, and the skills MCP tests**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py \
  tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py 2>&1 | grep -E "^(FAILED|ERROR) |passed|failed"
$PY -m ruff check tldw_Server_API/app/core/Skills/context_integration.py \
  tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py
```
Expected (measured): `15 passed` (2 here and 13 in the MCP file), and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Skills/context_integration.py \
  tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py
git commit -m "$(cat <<'EOF'
fix(skills): the skill tool call builds SkillsService off the event loop (TASK-13896)

SkillsService.__init__ runs 4 CREATE ... IF NOT EXISTS statements
through backend.execute, each with its own pool checkout. The skill
tool call built the service on the event loop, so every skill call took
4 checkouts there; under a full pool each could stall the server for
pool_timeout. The service is now built through asyncio.to_thread inside
the lookup's own operation: 4 on-loop checkouts per call -> 0.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: voice-assistant ChaCha calls run off the event loop, one operation per call group (TASK-13896 AC4)

**Files:**
- Replace: `tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py`, which keeps the fixture's name and the shared-fixture import, as below.
- Modify:
  - `tldw_Server_API/app/core/VoiceAssistant/registry.py`: `refresh_user_commands` (`:329-367`).
  - `tldw_Server_API/app/core/VoiceAssistant/router.py`: imports (`:7-12`), two helpers after `_close_response` (`:84-91`), group 1 (`:188-203`), group 2 (`:280-314`), and the external lookup (`:822-827`).
  - `tldw_Server_API/app/api/v1/endpoints/voice_assistant.py`: a helper above `@ws_router.websocket("/assistant")` (`:1023`), session start (`:1101-1111`), and `_process_text_command` (`:1411-1418`).

**Interfaces:**
- Produces:
  - `VoiceCommandRegistry.load_user_commands(db, user_id, include_disabled=True, persona_id=None) -> list[VoiceCommand]`, database only;
  - `VoiceCommandRegistry.apply_user_commands(user_id, commands, persona_id=None) -> None`, cache only;
  - `refresh_user_commands`, unchanged in signature and result (it is `apply(load(...))`). Its HTTP callers, `voice_assistant.py:382`, `:1629` and `persona.py:7309`, are untouched.
- `process_command` and `websocket_voice_assistant` keep their signatures.

- [ ] **Step 1: Write the failing tests**

Replace `tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py` with:

```python
"""Voice assistant WebSocket over a real PostgreSQL ChaCha database (TASK-13861, TASK-13896).

Every ChaCha checkout the WebSocket makes is taken off the event loop: on the loop, a
full pool would stall the whole server for up to pool_timeout.
"""

import asyncio
import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import voice_assistant
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.VoiceAssistant.registry import VoiceCommandRegistry
from tldw_Server_API.app.core.VoiceAssistant.router import VoiceCommandRouter
from tldw_Server_API.app.core.VoiceAssistant.schemas import ActionResult, ActionType, ParsedIntent, VoiceIntent
from tldw_Server_API.app.core.VoiceAssistant.session import VoiceSessionManager
from tldw_Server_API.tests.VoiceAssistant import test_ws_integration as ws_integration

pytestmark = pytest.mark.integration
DummyWebSocket = ws_integration.DummyWebSocket
mock_ws_dependencies = ws_integration.mock_ws_dependencies  # the shared fixture, registered in this module
_POOL_SIZE = 3


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


async def _heartbeat(gaps: list[float]) -> None:
    """Record how long the event loop goes between 20 ms ticks."""
    last = time.monotonic()
    while True:
        await asyncio.sleep(0.02)
        now = time.monotonic()
        gaps.append(now - last)
        last = now


@pytest.fixture
def pg_voice_db(pg_database_config, tmp_path, monkeypatch):
    """The user's cached handle, served to the WebSocket, on a small pool that records each checkout."""
    pg_database_config.pool_size = _POOL_SIZE
    pg_database_config.pool_timeout = 1.0
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned, on_loop = {}, Counter(), Counter(), []
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        on_loop.append(_on_event_loop())
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
    db.close_connection()
    on_loop.clear()

    async def _db_for_user(_user_id, client_id=None):  # noqa: ARG001
        return db

    monkeypatch.setattr(voice_assistant, "get_chacha_db_for_user_id", _db_for_user)

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield SimpleNamespace(
            db=db, outstanding=outstanding, on_loop=on_loop, hold=original_get, give_back=original_return
        )
    finally:
        db.close_connection()
        pool.close_all()


async def test_session_start_takes_no_checkout_on_the_event_loop(mock_ws_dependencies, pg_voice_db):
    # The shared fixture's registry loads no commands, so the session's save is the ChaCha work here.
    ws = DummyWebSocket([{"type": "auth", "token": "valid-token"}, {"type": "config"}])

    await voice_assistant.websocket_voice_assistant(ws, token=None)

    sent = [m.get("type") for m in ws.sent_json]
    assert ("auth_ok" in sent, len(pg_voice_db.on_loop) > 0, sum(pg_voice_db.on_loop)) == (True, True, 0)
    assert pg_voice_db.outstanding() == []


async def test_a_full_pool_at_session_start_degrades_the_session(mock_ws_dependencies, pg_voice_db):
    held = [pg_voice_db.hold() for _ in range(_POOL_SIZE)]  # other work holds the whole pool
    gaps: list[float] = []
    beat = asyncio.create_task(_heartbeat(gaps))
    await asyncio.sleep(0.05)  # the heartbeat is ticking before the session starts
    ws = DummyWebSocket([{"type": "auth", "token": "valid-token"}, {"type": "config"}])
    escaped = None
    try:
        await voice_assistant.websocket_voice_assistant(ws, token=None)
    except Exception as exc:  # noqa: BLE001 - the RED shape: PoolTimeout used to escape the socket
        escaped = type(exc).__name__
    finally:
        await asyncio.sleep(0.05)  # one more tick records any stall the session caused
        beat.cancel()
        for conn in held:
            pg_voice_db.give_back(conn)

    sent = [m.get("type") for m in ws.sent_json]
    assert (escaped, "auth_ok" in sent, "config_ack" in sent, max(gaps) < 0.5) == (None, True, True, True), (
        escaped,
        sent,
        max(gaps),
    )


class _StubParser:
    """Answers every utterance with an LLM chat intent, as the real parser's fallback does."""

    async def parse(self, text, user_id, context, persona_id=None):  # noqa: ARG002
        return ParsedIntent(intent=VoiceIntent(action_type=ActionType.LLM_CHAT, raw_text=text), match_method="stub")


async def test_a_command_takes_no_checkout_on_the_event_loop_and_holds_none_while_it_acts(
    mock_ws_dependencies, pg_voice_db, monkeypatch
):
    sessions = VoiceSessionManager()
    router = VoiceCommandRouter(
        registry=VoiceCommandRegistry(), parser=_StubParser(), session_manager=sessions, workflow_handler=object()
    )
    held_while_acting = []

    async def act(intent, session, *, db=None, persona_id=None):  # noqa: ARG001
        """Stands in for the LLM call, MCP action or SYNC workflow the command runs."""
        held_while_acting.append(pg_voice_db.outstanding())
        return ActionResult(success=True, action_type=ActionType.LLM_CHAT, response_text="done")

    monkeypatch.setattr(router, "_execute_intent", act)
    monkeypatch.setattr(voice_assistant, "get_voice_session_manager", lambda: sessions)
    monkeypatch.setattr(voice_assistant, "get_voice_command_router", lambda: router)
    ws = DummyWebSocket(
        [{"type": "auth", "token": "valid-token"}, {"type": "config"}, {"type": "text", "text": "hello there"}]
    )
    try:
        await voice_assistant.websocket_voice_assistant(ws, token=None)
    finally:
        await sessions.stop()

    loop_takes, outstanding = sum(pg_voice_db.on_loop), pg_voice_db.outstanding()
    with chacha_operation(independent=True):  # the command's analytics row was committed
        events = pg_voice_db.db.execute_query("SELECT count(*) AS n FROM voice_command_events").fetchone()["n"]
    assert (held_while_acting, loop_takes, outstanding, events) == ([[]], 0, [], 1)


async def test_an_external_action_reads_its_connection_off_the_event_loop(pg_voice_db):
    router = VoiceCommandRouter(
        registry=VoiceCommandRegistry(), parser=_StubParser(), session_manager=VoiceSessionManager(), workflow_handler=object()
    )

    result = await router._execute_external_connection_action(
        intent=VoiceIntent(action_type=ActionType.CUSTOM, raw_text="open the door"),
        session=SimpleNamespace(user_id=1, session_id="voice-1"),
        db=pg_voice_db.db,
        persona_id="persona-1",
        command=SimpleNamespace(connection_id="conn-1", persona_id="persona-1"),
    )

    assert (result.error_message, len(pg_voice_db.on_loop) > 0, sum(pg_voice_db.on_loop)) == (
        "Connection 'conn-1' was not found for persona 'persona-1'.",
        True,
        0,
    )
    assert pg_voice_db.outstanding() == []
```

How the four tests are built:
- **The registries.**
  - The WebSocket tests keep `mock_ws_dependencies`'s `DummyRegistry` for session start. Its missing methods raise `AttributeError` after the session's save, on dev and on the branch alike, so the save is the ChaCha work under test.
  - With the real registry, dev's session start dies on F4's pre-existing query bug before the checkout assertion can speak.
  - The router keeps a real `VoiceCommandRegistry`. Its refresh fails on PostgreSQL (F4), which the router catches.
- **The replaced test.** PR B's `test_a_text_command_returns_its_checkout` stood in for the router with `db.get_character_card_by_name(...)` called directly on the loop. That is the shape this task removes, so the stand-in now models the old router. The command test exercises the real router instead, with stronger assertions.
- **The heartbeat.** It ticks before and after the handler. Dev's handler blocks before its first real yield, so without that the heartbeat would record nothing.

- [ ] **Step 2: Run them to verify they fail**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q -o timeout=300 \
  tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py 2>&1 \
  | grep -v " | " | grep -E "^E   (Assert|assert)|^FAILED|passed|failed" | cut -c1-200
```
Expected (measured on `e06eeec68c`):
```
E   assert (True, True, 1) == (True, True, 0)
E   AssertionError: ('PoolTimeout', [], 1.0166…)
E   AssertionError: assert ([['MainThread']], 2, [], 0) == ([[]], 0, [], 1)
E   assert ("Connection ...1'.", True, 1) == ("Connection ...1'.", True, 0)
======================= 4 failed …
```

- [ ] **Step 3: Implement**

**`registry.py`.** Replace `refresh_user_commands`'s body after its docstring (dev `:347-367`) so that it delegates, and add the two methods after it:
```python
        commands = self.load_user_commands(db, user_id, include_disabled=include_disabled, persona_id=persona_id)
        self.apply_user_commands(user_id, commands, persona_id=persona_id)
        return commands

    def load_user_commands(
        self,
        db,
        user_id: int,
        include_disabled: bool = True,
        persona_id: str | None = None,
    ) -> list[VoiceCommand]:
        """Read a user's commands from the database without touching the cache (safe off the event loop)."""
        from .db_helpers import get_user_voice_commands

        return get_user_voice_commands(
            db,
            user_id=user_id,
            include_system=False,
            enabled_only=not include_disabled,
            persona_id=persona_id,
        )

    def apply_user_commands(
        self,
        user_id: int,
        commands: list[VoiceCommand],
        persona_id: str | None = None,
    ) -> None:
        """Replace the cached user commands; call it on the event loop, which owns the cache."""
        if persona_id is not None:
            self._user_commands[self._persona_bucket_key(user_id, persona_id)] = {cmd.id: cmd for cmd in commands}
        else:
            keys_to_remove = [key for key in self._user_commands if key[0] == user_id]
            for key in keys_to_remove:
                del self._user_commands[key]
            for command in commands:
                bucket_key = self._persona_bucket_key(user_id, command.persona_id)
                if bucket_key not in self._user_commands:
                    self._user_commands[bucket_key] = {}
                self._user_commands[bucket_key][command.id] = command
```
The cache code is dev's `:356-366`, moved unchanged.

**`router.py`.**
- Add `import asyncio` above `import time`, and `from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation` above the `http_client` import.
- After `_close_response`, add:
```python
# A command's ChaCha work runs in these groups, each off the event loop in its own
# operation (or an HTTP request's), so no checkout waits on the loop or spans the action.
def _load_commands_and_save_new_session(
    registry: VoiceCommandRegistry,
    db: Any,
    user_id: int,
    persona_id: str | None,
    new_session: VoiceSessionContext | None,
) -> list[Any]:
    commands = registry.load_user_commands(db, user_id, include_disabled=True, persona_id=persona_id)
    if new_session is not None:
        from .db_helpers import save_voice_session

        save_voice_session(db, new_session)
    return commands


def _record_command_and_save_session(db: Any, session: VoiceSessionContext, **event: Any) -> None:
    from .db_helpers import record_voice_command_event, save_voice_session

    record_voice_command_event(db, **event)
    save_voice_session(db, session)
```
- Group 1 (dev `:189-203`):
```diff
         if db:
             try:
                 self.registry.load_defaults()
-                self.registry.refresh_user_commands(
+                commands = await asyncio.to_thread(
+                    chacha_operation()(_load_commands_and_save_new_session),
+                    self.registry,
                     db,
                     user_id,
-                    include_disabled=True,
-                    persona_id=persona_id,
+                    persona_id,
+                    session.model_copy(deep=True) if created else None,
                 )
-                if created:
-                    from .db_helpers import save_voice_session
-
-                    save_voice_session(db, session)
+                self.registry.apply_user_commands(user_id, commands, persona_id=persona_id)
             except Exception as e:
                 logger.warning(f"Failed to refresh voice commands from DB: {e}")
```
- Group 2 (dev `:281-314`): delete the lazy import line (`from .db_helpers import record_voice_command_event, save_voice_session`) and the blank line after it; then:
```diff
-                    record_voice_command_event(
+                    await asyncio.to_thread(
+                        chacha_operation()(_record_command_and_save_session),
                         db,
+                        session.model_copy(deep=True),
                         command_id=parsed.intent.command_id,
```
```diff
                         ),
                     )
-                    save_voice_session(db, session)
                 except Exception as e:
                     logger.warning(f"Failed to record voice analytics: {e}")
```
- Group 3 (dev `:822`):
```diff
-        connection = self._load_persona_connection(
+        connection = await asyncio.to_thread(
+            chacha_operation()(self._load_persona_connection),
             db=db,
```

**`voice_assistant.py`.** Directly above `@ws_router.websocket("/assistant")`, not between the decorator and the function:
```python
def _save_session_and_load_commands(db: CharactersRAGDB, session: Any, registry: Any, user_id: int) -> list[Any]:
    """Session start's ChaCha work; it runs off the event loop, and the loop applies the commands."""
    save_voice_session(db, session)
    return registry.load_user_commands(db, user_id, include_disabled=True)
```
Session start (dev `:1102-1111`):
```python
        try:
            db = await get_chacha_db_for_user_id(user_id, client_id="voice_assistant")
            registry = get_voice_command_registry()
            registry.load_defaults()
            # A WebSocket runs outside the HTTP operation middleware: this step owns its
            # checkout, off the event loop, so a full pool cannot stall the server.
            commands = await asyncio.to_thread(
                chacha_operation(independent=True)(_save_session_and_load_commands),
                db,
                session.model_copy(deep=True),
                registry,
                user_id,
            )
            registry.apply_user_commands(user_id, commands)
        except Exception:  # noqa: BLE001 - a full pool (PoolTimeout) or a DB error degrades the session, not the socket
            logger.warning("Voice assistant DB init failed")
```
`get_chacha_db_for_user_id` already does its own DB work off the loop in independent operations (`ChaCha_Notes_DB_Deps.py:452-531`, `:604-607`, `:715`), so it leaves the operation.

`_process_text_command` (dev `:1411-1418`):
```python
    # No operation around the command: the router runs each of its ChaCha call groups off
    # the event loop in its own, so no checkout spans the LLM call, MCP action or workflow.
    result, _ = await router_instance.process_command(
        text=text,
        user_id=user_id,
        session_id=session_id,
        db=db,
    )
```

- [ ] **Step 4: Run them, and the existing voice WebSocket tests**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q -o timeout=300 \
  tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py \
  tldw_Server_API/tests/VoiceAssistant/test_ws_integration.py 2>&1 | grep -E "^(FAILED|ERROR) |passed|failed"
$PY -m ruff check tldw_Server_API/app/core/VoiceAssistant/registry.py \
  tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py
$PY -m ruff check --output-format concise tldw_Server_API/app/api/v1/endpoints/voice_assistant.py \
  tldw_Server_API/app/core/VoiceAssistant/router.py | grep -oE ": [A-Z]+[0-9]+" | sort | uniq -c
```
Expected:
- `17 passed` (4 + 13), measured. `test_auth_sanitizes_db_init_failure_log` still sees exactly `[("Voice assistant DB init failed", (), {})]`.
- `All checks passed!`.
- `5 : F821`, `1 : F401` and `1 : I001`, which are dev's.

- [ ] **Step 5: The persona, voice and skill callers**

```bash
git grep -lE 'endpoints(\.| import )persona\b|Persona(\.| import )(live_conversation|exemplar_runtime)|persona/stream|Skills(\.| import )context_integration|endpoints(\.| import )voice_assistant|VoiceAssistant(\.| import )(router|registry)|voice/assistant|VoiceCommandRouter|VoiceCommandRegistry|refresh_user_commands' \
  -- 'tldw_Server_API/tests/*test_*.py' > "$S/fu2cr_loop_sweep.txt"
wc -l < "$S/fu2cr_loop_sweep.txt"
$S/fu2cr_sweep.sh "$S/fu2cr_loop_sweep.txt" "$S/runs/fu2cr_loop" 2
```
Expected (measured on the r2 prototype, while the union sweep ran: load 4 → 11):
- `32` files;
- `328 passed` and `317 passed`, with no `FAILED` or `ERROR` lines. That includes `test_persona_ws.py`'s telemetry test (B2) and Task 2's 3 new tests.

r1 gave `325` here at load ~10, and the review measured `1 failed, 324 passed`, the telemetry test that r2 fixes.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/VoiceAssistant/registry.py tldw_Server_API/app/core/VoiceAssistant/router.py \
  tldw_Server_API/app/api/v1/endpoints/voice_assistant.py \
  tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py
git commit -m "$(cat <<'EOF'
fix(voice): voice-assistant ChaCha calls run off the event loop, one operation per call group (TASK-13896)

TASK-13861 gave the voice WebSocket's session start and each command
an operation, but both checked out on the event-loop thread, and the
command's operation held its checkout through the LLM call, MCP action
or SYNC workflow. Measured on a full 3-connection pool: session start
stalled the loop 1.02 s, then PoolTimeout escaped and killed the socket;
a command took 2 checkouts on the loop and held one through its action.

Session start now saves and loads in one operation on a worker thread
and degrades the session on any failure. The command has no operation of
its own: the router runs its three ChaCha call groups (refresh and new
session save, analytics and session save, external-connection lookup)
through asyncio.to_thread, each in its own operation or an HTTP
request's. The registry's database read and its cache update are split
so the shared cache is only changed on the loop. Measured: 0 checkouts
on the loop, none held during the action, auth_ok sent on a full pool.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Docs, sweep, backlog close-out, plan copy, PR

**Files:**
- `Docs/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md`
- `Docs/Code_Documentation/Databases/ChaChaNotes_DB.md`
- their two `Docs/Published` copies
- `backlog/tasks/task-13896 …md`, `backlog/tasks/task-13899 …md`, and three new follow-up tasks
- `Docs/superpowers/plans/2026-10-22-chacha-off-loop-and-transaction-guard.md`

- [ ] **Step 1: Docs**

In `Docs/ADR/080-…md`, Consequences:
- in the "Cost" bullet, replace `Calls in async code therefore go through `asyncio.to_thread` where they can; the persona and voice-assistant calls that still run on the loop are TASK-13896.` with:
  ```
  Calls in async code therefore go through `asyncio.to_thread`, as the persona and voice-assistant WebSockets and the skill tool call do since TASK-13896.
  ```
- replace the whole bullet `- **Concurrent transactions are a rule, not a guard.** …TASK-13899 tracks that guard.` with:
  ```
  - **Concurrent transactions are refused.** Since TASK-13899, a `db.transaction()` that enters while another thread or task holds the operation's transaction raises `ConcurrentChaChaTransactionError` instead of joining it, so a fan-out inside one operation fails loudly instead of losing writes. Nesting on the same thread and task still joins. The refusal also covers overlaps that happened to commit (one nested in time, or a hand-off to a thread inside a transaction), so a caller that overlaps gets an error, not usually-correct writes; semantic conversion's two lookups were the one such caller, and now run in turn. It raises rather than waits: a lock would deadlock a transaction that waits on another thread's, would block the event loop while it waited, and, being per thread, would not see two tasks on the loop.
  ```

In `Docs/Code_Documentation/Databases/ChaChaNotes_DB.md` (`:198`), after `code that fans out gives each call its own `chacha_operation(independent=True)`.` insert:
```
A `db.transaction()` entered while another thread or task holds the operation's transaction raises `ConcurrentChaChaTransactionError`.
```

Then:
```bash
bash Helper_Scripts/refresh_docs_published.sh
git add Docs/ADR/080-chacha-postgres-checkouts-owned-by-entry-points.md Docs/Code_Documentation/Databases/ChaChaNotes_DB.md Docs/Published
git status --short Docs
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs tldw_Server_API/tests/CI/test_test_quality_ratchet.py 2>&1 | grep -E "^(FAILED|ERROR) |passed|failed"
```
Expected (measured):
- `M` for the two docs and their two `Docs/Published` copies, and nothing else;
- `213 passed`.

Commit:
```bash
git commit -m "$(cat <<'EOF'
docs(adr): ADR-080 records the transaction guard and the off-loop calls (TASK-13896, TASK-13899)

Two Consequences bullets named TASK-13896 and TASK-13899 as open. They
now state what is true: async callers go through asyncio.to_thread, and
an overlapping transaction in one operation raises. The Decision is
unchanged. The ChaChaNotes guide names the new error.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Sweep the union**

```bash
git grep -lE 'CharactersRAGDB|get_chacha_db|chacha_operation|ChaChaOperationMiddleware|owns_chacha_checkouts' -- 'tldw_Server_API/tests/*test_*.py' \
  | xargs grep -liE 'postgres|pg_database_config|pg_restricted_backend' > "$S/fu2cr_tx_sweep.txt"
git grep -lE 'semantic_projector' -- 'tldw_Server_API/tests/*test_*.py' > "$S/fu2cr_semantic_tests.txt"
sort -u "$S/fu2cr_tx_sweep.txt" "$S/fu2cr_loop_sweep.txt" "$S/fu2cr_semantic_tests.txt" \
  | grep -v "RAG/test_restricted_postgres_media_retrieval.py" > "$S/fu2cr_union.txt"
wc -l < "$S/fu2cr_tx_sweep.txt"; wc -l < "$S/fu2cr_union.txt"
$S/fu2cr_sweep.sh "$S/fu2cr_union.txt" "$S/runs/fu2cr_union" 6 | tee "$S/runs/fu2cr_union.summary"
```
Expected:
- `231` and `263` (measured on the r2 tree). The union includes the 6 new or changed test files and the semantic projector's 5 test files, and excludes the RAG file below.
- No `FAILED`, `ERROR` or `+++ Timeout` beyond dev's baseline. Measured on r2: `7318 passed, 12 failed, 20 skipped`, and the 12 are the known dev failures listed below (on dev their 4 files gave `12 failed, 200 passed, 1 skipped`).

**Why the RAG file is excluded.** `RAG/test_restricted_postgres_media_retrieval.py` is TASK-13900's defect, and its fix is planned separately (`plan_fu2_pg_tests.md`).
- On dev and on this branch alike, its 19th test fails.
- `MediaFilesRepository.get_for_media` leaves a session idle in a transaction, and the teardown's `DROP OWNED BY` then waits forever. pytest-timeout does not fire (F5).
- A hung run leaves a `rag_retrieval_*` role and a `tldw_test_*` database behind. If that happens anyway:
  - end the process;
  - drop only those two, by name: no connections first, then `DROP DATABASE`, then `DROP OWNED BY` and `DROP ROLE`.

For any other failure, re-run that node id on a detached baseline worktree at `origin/dev` before blaming the change, then remove the worktree.

Known failures on dev in this list, each re-run on dev by the review (12 failed, 200 passed, 1 skipped on dev for their 4 files) or by PR D:
- **TASK-13901's five:** `test_chacha_postgres_sequence_ownership.py::test_chacha_sync_continues_after_table_error_without_settling_caller`, and `test_history_selection_transactions.py` ×4 (`assert 2 == 1`, `assert 3 == 1`).
- **Seven untracked until this PR, filed as N5:**
  - `Character_Chat_NEW/unit/test_postgres_backends.py` ×4: `StubDB` has no `_verify_dictionary_schema_postgres`;
  - `DB_Management/test_character_behavior_snapshot_migration.py` ×3: `'CharactersRAGDB' object has no attribute '_owner_user_id'`.
- **Three Admin `pg_dump not found on PATH` tests**, if the environment lacks `pg_dump`.

- [ ] **Step 3: Lint, Bandit, triage, whitespace**

```bash
uvx bandit -q -f json -o "$S/fu2cr_bandit.json" tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/app/api/v1/endpoints/voice_assistant.py tldw_Server_API/app/core/VoiceAssistant/router.py \
  tldw_Server_API/app/core/VoiceAssistant/registry.py tldw_Server_API/app/core/Skills/context_integration.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py \
  tldw_Server_API/app/core/Notes_Graph/semantic_projector.py \
  tldw_Server_API/tests/Persona/test_persona_postgres_per_call_operations.py \
  tldw_Server_API/tests/VoiceAssistant/test_ws_voice_assistant_postgres.py \
  tldw_Server_API/tests/Skills/integration/test_skill_tool_call_postgres_checkouts.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_concurrent_transactions.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_semantic_conversion_postgres_operation.py 2>/dev/null
$PY -c "import json,collections;d=json.load(open('$S/fu2cr_bandit.json'));print(sorted(collections.Counter((r['filename'].rsplit('/',1)[-1],r['test_id'],r['issue_severity']) for r in d['results']).items()))"
$PY Helper_Scripts/ci/test_quality_triage.py --enforce 2>&1 | grep "test-triage\] OK"
git diff --check origin/dev...HEAD && echo "no trailing whitespace"
```
Expected (measured on the r2 prototype):
- **Bandit:** nothing in the 8 product files. The tests have B101 LOW only (concurrent 1, persona 8, semantic 1, skill 4, voice 6), plus B105 LOW ×3 in the voice test for its `"valid-token"` strings.
- **Triage:** `[test-triage] OK — no offenses beyond the baseline.`
- `no trailing whitespace`.

- [ ] **Step 4: Backlog, under the id lock**

Take the lock with `mkdir $S/id_lock`. If it exists, wait; never break another session's lock.

Create `$S/task_next_fu2cr.sh`, the TASK-13861 plan's `task_next.sh` verbatim:
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
Run `git fetch origin dev --quiet && zsh $S/task_next_fu2cr.sh`. Take `<N1>` … `<N5>` as the printed value + 1 … + 5. Check that no `task-<N1>` … `task-<N5>` file exists locally.

Then set `B="env PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd ."`, and file the five follow-ups:
```bash
$B task create "Voice commands are unusable on PostgreSQL: INTEGER deleted/enabled against the boolean rewrite" \
  --id TASK-<N1> -l voice,postgres,bug \
  -d "Found by TASK-13896's planning and its review. voice_commands.deleted and .enabled are INTEGER on PostgreSQL (information_schema on a fresh ChaCha schema: ('deleted', 'integer'), ('enabled', 'integer')), but the SQLite-to-PostgreSQL transform rewrites '= 0' and '= 1' to booleans. Measured on origin/dev e06eeec68c, raw psycopg errors: get_user_voice_commands, get_voice_command (VoiceAssistant/db_helpers.py:77), get_voice_command_counts (:925) and delete_voice_command raise UndefinedFunction: operator does not exist: integer = boolean; save_voice_command -> CharactersRAGDB.upsert_voice_command (ChaChaNotes_DB.py:9586) raises DatatypeMismatch: column \"deleted\" is of type integer but expression is of type boolean (its ON CONFLICT ... SET deleted = 0). So on a PostgreSQL deployment the whole user voice-command feature fails: create, list, get, count and delete, over HTTP and the voice-assistant WebSocket. On dev, VoiceCommandRegistry.refresh_user_commands' CharactersRAGDBError also escaped websocket_voice_assistant at every session start and killed the socket; since TASK-13896 the session degrades with 'Voice assistant DB init failed', and the router logs 'Failed to refresh voice commands from DB' per command. No data loss, no security impact. The fix needs a choice: a PostgreSQL migration of the two columns to BOOLEAN, or per-query casts; check the other voice_* tables." \
  --ac "A voice command can be saved, listed, read, counted and deleted on PostgreSQL, pinned by a PostgreSQL test" \
  --ac "A voice-assistant session start on PostgreSQL with the real registry loads the user's commands"
$B task create "ChaCha on PostgreSQL: a failed statement outside a managed transaction leaves the operation's connection aborted" \
  --id TASK-<N2> -l chacha,postgres,bug \
  -d "Found by TASK-13896's planning. PostgreSQLBackend.execute rolls back a failed statement only on a connection it took itself ('if not external_conn: conn.rollback()', postgresql_backend.py near :1141). A CharactersRAGDB handle passes its own connection (the operation's, or the thread-local one), so a failed statement outside BackendManagedTransaction leaves that connection in an aborted transaction and every later statement on it fails with 'current transaction is aborted' until the operation ends or the thread closes it. Measured on origin/dev e06eeec68c (TASK-13896 plan F4): inside one voice command operation, the command refresh failed (TASK-<N1>), and the same command's analytics insert then failed ('Failed to record voice analytics: Query execution failed'; voice_command_events count 0). An HTTP request has the same shape: one failed read poisons the rest of its writes. Direction: roll back after a failure when no managed transaction is open on the state (tx_depth == 0), taking care of execute_query(commit=False) writes a caller means to commit later." \
  --ac "A failed statement outside a managed transaction inside a chacha_operation does not make the next statement in the operation fail (PostgreSQL test)" \
  --ac "A failure inside a caller's managed transaction still leaves the rollback to that transaction"
$B task create "SkillsService re-runs its registry DDL at every construction" \
  --id TASK-<N3> -l skills,postgres,performance \
  -d "Found by TASK-13896's planning and its review. SkillsService.__init__ calls CharactersRAGDB._ensure_skill_registry_table (ChaChaNotes_DB.py:28413), which on PostgreSQL runs 4 CREATE TABLE/INDEX IF NOT EXISTS statements through backend.execute, each with its own pool checkout, at every construction: 4 checkouts measured per skill tool call (TASK-13896 plan F3). Production constructs the service per request or call at: Skills/context_integration.py get_skills_context_text_async (:59) and add_skill_tool_to_tools_list_async (:242), which /chat/completions calls on the event loop for every chat request with skills (chat.py:3634, :5011), the hottest path; the sync helpers at context_integration.py :43 and :210; api/v1/endpoints/skills.py:136; Chat/command_router.py:688 and :759; MCP skills_module.py:559. (handle_skill_tool_call builds it off the loop since TASK-13896.) Remember per handle and backend target that the table is ready, as _bootstrapped_backend_targets does for the schema; that removes the statements from every site at once. Lock behaviour of the repeated IF NOT EXISTS DDL under concurrent writes is not measured." \
  --ac "A second SkillsService for the same handle runs no table statement (test counts backend.execute calls)" \
  --ac "The first construction still creates the table on SQLite and PostgreSQL"
$B task create "Voice commands: the action's own ChaCha work checks out on the event loop" \
  --id TASK-<N4> -l voice,postgres,performance \
  -d "Found by TASK-13896's review (M4). TASK-13896 moved the voice-assistant WebSocket's own ChaCha calls (session start, the router's command load, session save, analytics and external-connection lookup) off the event loop, and its AC 4 was narrowed to those. The action a command runs still does its own ChaCha work wherever its module does it: a SYNC workflow's notes step (Workflows/adapters/knowledge/crud.py run_notes_adapter) calls NotesInteropService synchronously on the loop with a per-call handle it closes, so it checks out on the event-loop thread; MCP modules use per-call handles closed on the same thread, not measured on or off the loop. Under a full content pool such a checkout stalls the whole server for up to pool_timeout. plan_fu2_workflows.md (TASK-13897) keeps the notes adapter on the loop. Measure each on PostgreSQL first, then move what checks out on the loop into asyncio.to_thread with its own operation." \
  --ac "A voice command whose SYNC workflow runs a notes step takes no ChaCha checkout on the event-loop thread (PostgreSQL test), or the measurement shows it does not" \
  --ac "The MCP modules a voice command can call are measured, and any that check out on the loop are moved or listed"
$B task create "Tests: seven PostgreSQL ChaCha tests fail on dev and are tracked nowhere" \
  --id TASK-<N5> -l chacha,postgres,tests \
  -d "Found by TASK-13896/TASK-13899's review sweep, and re-run on origin/dev e06eeec68c with the same result (12 failed, 200 passed, 1 skipped across their 4 files; 5 of the 12 are TASK-13901's). Untracked: tldw_Server_API/tests/Character_Chat_NEW/unit/test_postgres_backends.py x4 (StubDB has no _verify_dictionary_schema_postgres) and tldw_Server_API/tests/DB_Management/test_character_behavior_snapshot_migration.py x3 ('CharactersRAGDB' object has no attribute '_owner_user_id'). Each sweep that touches ChaCha on PostgreSQL has to re-baseline them." \
  --ac "The 7 tests pass on PostgreSQL, or each is fixed or removed with its reason"
```
Expected: five `TASK-<Nk> - … (To Do)` lines.

Notes, AC4's rewording, ACs, DoD, final summary and status for TASK-13896. Fill the bracketed results from Step 2:
```bash
$B task edit TASK-13896 -s Done \
  --append-notes "Plan: Docs/superpowers/plans/2026-10-22-chacha-off-loop-and-transaction-guard.md. PR: <PR URL>. Branch fix/chacha-off-loop-and-transaction-guard." \
  --append-notes "Persona (AC1, AC2): all 13 on-loop calls on persona_scope_db (the 10 in async handlers and the 3 in the analytics helpers) run as await asyncio.to_thread(chacha_operation(independent=True)(fn), ...); the three helpers are async so their bookkeeping (persona_live_summary_sessions_seen) stays on the loop; their 9 call sites are awaited. The AST guards require that form everywhere in persona.py (off-loop count 8 -> 21) and an await on every helper call (mutation: a dropped await is reported as persona.py:10655). RED on origin/dev e06eeec68c: 4 failed, 3 passing pins: 'assert 8 == 21'; the 13 sites listed, first persona.py:9420; the shield guard ([], 0) == ([], 4); the PostgreSQL test (2-connection pool held 1.5 s, real persona_stream, voice_config) gave checkouts [(True, 1.5), (True, 0.0) x4] on the loop with a 1.53 s server-loop gap. GREEN: 7 passed; the same test gave [(False, 1.5), (False, 0.0) x4] and gaps of 0.021-0.030 s (3 runs)." \
  --append-notes "Persona writes (review B2): moving the writes to awaited threads let a cancel skip them. test_persona_ws.py::test_persona_audio_chunk_records_live_voice_commit_telemetry failed 3 of 3 on the first version (the client closes on VOICE_TURN_COMMITTED, which was emitted before the commit's analytics writes). Now the commit's and the manual-mode notice's writes run before their notice (6 of 6 passed); the 4 writes (summary upsert, event record, both preference saves) run under run_quiescent_operation(asyncio.to_thread(...)) (Notes_Graph/semantic_publication.py), so a queued or running write finishes before the cancel propagates; and the session finalizes its summaries after its turn tasks are drained. A new shield guard names any unshielded write (RED on the first version: 4 sites); a PostgreSQL pin cancels a queued preferences write on a one-worker executor (shielded: saved; plain to_thread: lost). Reads lost to a cancel are harmless. Loop sweep (32 files) under load up to 11: 328 + 317 passed." \
  --append-notes "Skill (AC3): handle_skill_tool_call builds SkillsService through asyncio.to_thread inside the lookup's independent operation. RED: (True, True, 4) == (True, True, 0), the 4 table statements of _ensure_skill_registry_table on the loop. GREEN: 0; with test_skill_mcp_integration.py 15 passed. The per-construction DDL itself is TASK-<N3>." \
  --append-notes "Voice (AC4): session start saves and loads in one independent operation on a worker thread, applies the registry cache on the loop, and degrades the session on any failure (except Exception, same sanitized warning). _process_text_command no longer wraps the command in an operation; the router runs its three ChaCha call groups (command load + new-session save; analytics + session save; external-connection lookup) through asyncio.to_thread(chacha_operation()(fn), ...), so it joins an HTTP request's operation (/voice/command) and owns each group in the WebSocket. VoiceCommandRegistry.refresh_user_commands is split into load_user_commands (database) and apply_user_commands (cache, on the loop); its HTTP callers are unchanged. Sessions handed to workers are model_copy(deep=True) snapshots. RED on e06eeec68c: 4 failed: session start 1 checkout on the loop; full pool ('PoolTimeout', [], 1.017 s loop gap: the socket died); command ([['MainThread']], 2, [], 0): checkout held through the action, 2 on the loop, analytics row lost; external lookup 1 on the loop. GREEN: 17 passed with test_ws_integration.py. PR B's test_a_text_command_returns_its_checkout was replaced: its stand-in router used the handle directly on the loop, the shape this task removes." \
  --append-notes "AC4 reworded (premise narrower than its words): 'command handling' read literally includes the action's own module. A SYNC workflow's notes step runs NotesInteropService on the loop with a per-call handle it closes (TASK-13897's area; its plan keeps the notes adapter on the loop), and MCP modules use per-call handles closed on the same thread. The description's fix shape covers the WebSocket and the router's call groups, which is what the new AC 4 states. The residual, the action modules' own on-loop checkouts, is TASK-<N4>." \
  --append-notes "Found while planning, filed: TASK-<N1> (voice commands are unusable on PostgreSQL: INTEGER deleted/enabled against the boolean rewrite; on dev it also killed every PostgreSQL voice session start, which this task's degrade now catches), TASK-<N2> (a failed statement outside a managed transaction poisons the operation's connection; it lost the command's analytics row on dev), TASK-<N3> (SkillsService DDL per construction, incl. the /chat/completions path), TASK-<N4> (a voice command's action modules check out on the loop), TASK-<N5> (seven untracked PostgreSQL test failures on dev)." \
  --append-notes "Sweep: the 263-file union of PostgreSQL ChaCha tests, persona/voice/skill callers and the semantic projector's tests (RAG/test_restricted_postgres_media_retrieval.py excluded: TASK-13900 hangs it on dev too), 6 shards, RUN_JOBS=1: [result] (dev baseline for any failure: [result]). ruff: no new findings (persona.py I001+SIM114, voice_assistant.py I001, router.py F401+F821x5 as on dev). Bandit: nothing in the product files; tests B101 LOW, voice test B105 LOW x3 ('valid-token'). Triage OK. Docs: ADR-080 Cost bullet and the ChaChaNotes guide; Docs/Published refreshed; tests/Docs 213 passed." \
  --remove-ac 4 \
  --ac "The voice-assistant WebSocket's session start and the voice router's ChaCha call groups (command load, session save, analytics, external-connection lookup) make no ChaCha checkout on the event-loop thread, and a command holds no checkout through its LLM call, MCP action or SYNC workflow" \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Persona, skill and voice-assistant ChaCha calls no longer check out on the event-loop thread, so a full content pool makes them wait on a worker thread instead of stalling the whole server for pool_timeout. Persona: 13 calls through asyncio.to_thread with their own independent operations, pinned by AST guards and a PostgreSQL test (1.53 s loop stall -> 0.03 s); its writes run before the notices a client may close on, and finish once queued even if the session's task is cancelled. Skill tool call: SkillsService built off the loop (4 on-loop checkouts -> 0). Voice: session start off the loop and degrading on a full pool instead of killing the socket; each command's three ChaCha call groups off the loop in their own operations, so no checkout is held through the LLM call, MCP action or workflow, and the analytics row is no longer lost after a failed read. Follow-ups: TASK-<N1> … TASK-<N5>."
```

Then for TASK-13899:
```bash
$B task edit TASK-13899 -s Done \
  --append-notes "Plan: Docs/superpowers/plans/2026-10-22-chacha-off-loop-and-transaction-guard.md. PR: <PR URL>." \
  --append-notes "Chose fail loudly, not serialize. ConnectionState gains tx_owner, the (thread ident, asyncio task) key use() already uses; BackendManagedTransaction claims it before reading tx_depth (enter_transaction), releases it in __exit__ and on a failed __enter__ (leave_transaction); the same key nests and joins as before; another key raises ConcurrentChaChaTransactionError (a RuntimeError, logged at ERROR). Legacy thread-local and SQLite paths unchanged. Why not a lock: a lock would deadlock a transaction that waits on another thread's, block the event loop, and, per thread, miss two tasks on one thread (mutation: a thread-only key loses writes in the two-task case exactly as dev). Raise versus log-only was the owner's Decision 1: raise (AC1 as written; refusal also covers overlaps that happened to commit on dev, so a missed caller becomes a 500 or a failed job)." \
  --append-notes "Overlap audit: the first scans (1228 sync transaction blocks in app/ on e06eeec68c, none awaits or hands work to a thread inside; the 2 that hold a ChaCha transaction across a yield have no awaiting caller) missed fan-outs. Widened (plan Design choice 1): 486 ChaCha methods that open a managed transaction, 135 fan-out sites (gather, create_task, TaskGroup, as_completed, wait, start_soon, run_in_executor, submit, run_coroutine_threadsafe, Thread, copy_context) in the 268 app files that name ChaCha, matched by name within 3 hops of their module; 17 flagged, each read. One overlaps: SemanticProjector.validate_conversion gathered two to_thread(list_visible_vector_ids), each opening db.transaction(), inside the request's operation, so POST /notes/{id}/links with semantic_conversion would answer 500 (the review measured 40 of 40 refused). It now awaits the two lookups in turn. The audit is name-based within one module and skips modules with no ChaCha marker; the reviews spot-read the gather/wait sites, a gap scan of direct-transaction fan-outs (13 sites), and the MCP and Workflows fan-outs, and found no second overlap. Cross-module chains are not exhaustively covered." \
  --append-notes "RED on origin/dev e06eeec68c: tests/DB_Management/test_chacha_postgres_concurrent_transactions.py 2 failed: ([], ['r1']) == (['ConcurrentChaChaTransactionError'], ['r1', 'r3']) for two threads and for two tasks on the loop thread (r2 and r3 rolled back with the operation). GREEN: 2 passed; with test_chacha_operation_scope.py 21 passed. Semantic conversion (tests/Notes_Graph/integration/test_semantic_conversion_postgres_operation.py): dev (None, 2), both lookups at once; guard without the fix ('ConcurrentChaChaTransactionError', 2) 3 of 3; with the fix (None, 1), 38 passed with the existing semantic projector and manual-conversion tests. AC2: the 13 operation-focused PostgreSQL files (operation scope, HTTP operation lifecycle, chat retry, enrichment, study pack, per-call handles, legacy checkouts, close_all, locking reads, notes-graph, character_chat, Jobs workers incl. the Chat_Macros concurrent-branch run) 162 passed with RUN_JOBS=1, 163 passed with the semantic test; the 263-file sweep (RAG/test_restricted_postgres_media_retrieval.py excluded: TASK-13900 hangs it on dev too) [result]." \
  --append-notes "Docs: ADR-080's 'Concurrent transactions' Consequences bullet now states the guard and its cost (Decision unchanged); ChaChaNotes_DB.md names the error; Docs/Published refreshed. Bandit: nothing in operation_scope.py, ChaChaNotes_DB.py or semantic_projector.py; the two tests have B101 LOW x1 each." \
  --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Inside one chacha_operation, a db.transaction() that enters while another thread's or task's transaction is open on the same ConnectionState now raises ConcurrentChaChaTransactionError instead of joining it; nesting on one thread and task joins as before. Before, the overlap left tx_depth wrong and every later write in the operation was silently rolled back (measured: r2 and r3 lost for both a two-thread and a two-task overlap). Raising rather than serializing avoids deadlocks and event-loop stalls, and covers two tasks on one thread; it also refuses overlaps that happened to work, so the one current caller, semantic conversion's two lookups, now runs them in turn. ADR-080's rule is now enforced."
```
Expected: each task prints with status `Done`, 4 and 2 ACs checked, and all 6 DoD items checked.

If the owner chose Decision 1's (B), log only:
- reword TASK-13899's AC1 first, with `--remove-ac 1 --ac "<the owner's wording>"` before `--check-ac`;
- replace the "Chose fail loudly" note with what was built.

The plan does not script (B).

Release the lock only after the task files are pushed and visible in the PR (Step 6): `rmdir $S/id_lock`.

- [ ] **Step 5: The plan copy**

```bash
sed -e 's/[[:space:]]*$//' "$S/plan_fu2_chacha_runtime.md" > Docs/superpowers/plans/2026-10-22-chacha-off-loop-and-transaction-guard.md
```
`Docs/superpowers` is not mirrored.

- [ ] **Step 6: Commit, review, push, open the PR**

```bash
git add -A backlog/tasks
git add Docs/superpowers/plans/2026-10-22-chacha-off-loop-and-transaction-guard.md
git commit -m "$(cat <<'EOF'
docs(backlog): close TASK-13896 and TASK-13899; file the follow-ups; add the plan

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
git diff --check origin/dev...HEAD && echo "no trailing whitespace"
```
Run `/superpowers:requesting-code-review` in a clean subagent with the branch diff and this plan, and address its findings before pushing. Then:
```bash
git push -u origin fix/chacha-off-loop-and-transaction-guard
gh pr create --base dev --title "fix: ChaCha calls off the event loop; an overlapping transaction in one operation raises (TASK-13896, TASK-13899)" --body "$(cat <<'EOF'
## Summary
- Persona live, the skill tool call and the voice-assistant WebSocket no longer take ChaCha checkouts on the event-loop thread. When other work holds the whole content pool, they wait on a worker thread instead of stalling the server for `pool_timeout`.
  - Measured on a held 2-connection pool, persona: a 1.53 s loop stall became 0.03 s.
  - Voice on a full pool: session start used to stall the loop 1.02 s and kill the socket with `PoolTimeout`; now the session degrades.
- A voice command no longer holds a checkout through its LLM call, MCP action or SYNC workflow. The router owns its three ChaCha call groups instead of the whole command.
- Persona writes now run before the notices a client may close on, and once queued they finish even if the session's task is cancelled. They use `run_quiescent_operation`.
- Inside one `chacha_operation`, a `db.transaction()` that overlaps another thread's or task's now raises `ConcurrentChaChaTransactionError`. Before, it joined, and every later write in the operation was silently rolled back (measured for two threads and for two tasks).
  - The refusal also covers overlaps that happened to work on dev; that was the owner's Decision 1.
  - The one current caller that overlapped, semantic conversion's two lookups (`POST /notes/{id}/links` with `semantic_conversion`, which would have answered 500), now runs them in turn.
  - A name-based audit of the 135 fan-outs in the 268 app files that name ChaCha, plus the reviews' spot reads, found no other. Cross-module chains are not exhaustively covered.
- ADR-080's Consequences are updated as status facts; the decision is unchanged.
- Follow-ups filed:
  - TASK-<N1> (voice commands are unusable on PostgreSQL);
  - TASK-<N2> (a failed statement poisons the operation's connection);
  - TASK-<N3> (SkillsService DDL per construction);
  - TASK-<N4> (a voice command's action modules check out on the loop);
  - TASK-<N5> (seven untracked PostgreSQL test failures on dev).

## Change summary
<!-- The owner writes this section (ADR-004). -->

## Test plan
- [ ] TASK-13899 PostgreSQL test: dev 2 failed (`([], ['r1'])`), branch 2 passed; mutation with a thread-only key fails the two-task case
- [ ] semantic conversion inside an operation: dev `(None, 2)`; guard without the fix `ConcurrentChaChaTransactionError`; branch passes (38 with the semantic tests)
- [ ] operation-scope and HTTP lifecycle files plus the semantic test: 163 passed (RUN_JOBS=1)
- [ ] persona: dev 4 failed (guard 8 != 21; 13 on-loop sites; 4 unshielded writes; 1.53 s loop stall), branch 7 passed; a dropped `await` fails the guard
- [ ] `test_persona_ws.py` telemetry test (broken by the first version): 6 of 6 passed
- [ ] skill: dev 4 on-loop checkouts, branch 0
- [ ] voice: dev 4 failed, branch 4 passed (17 with test_ws_integration.py)
- [ ] 263-file union sweep (TASK-13900's RAG file excluded): no failures beyond dev's baseline
- [ ] ruff no new findings; Bandit nothing in product files; triage OK; tests/Docs 213 passed

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Before merge:
- the owner's `## Change summary` (ADR-004). It is not covered by a recorded waiver unless the owner says so; ask.
- `gh variable get MERGE_QUEUE`, and follow CLAUDE.md's "Merging into dev".
- If the workflows PR (`plan_fu2_workflows.md`) merged first:
  - rebase only when this PR is next to merge;
  - keep both ADR-080 bullets at lines 51-52, and their `Docs/Published` copies;
  - re-run `refresh_docs_published.sh` and `tests/Docs`, then Task 1 Step 4.
- Confirm `MERGED` before deleting the branch.

---

## Follow-ups (filed at execution, Task 5 Step 4)

| Id | Title | Why |
|---|---|---|
| TASK-<N1> | Voice commands are unusable on PostgreSQL: INTEGER deleted/enabled against the boolean rewrite | Measured: `UndefinedFunction` from the four read and delete helpers, and `DatatypeMismatch` from `upsert_voice_command`. Create, list, get, count and delete all fail. On dev it also killed every PostgreSQL voice session start; with this PR the session degrades. |
| TASK-<N2> | ChaCha on PostgreSQL: a failed statement outside a managed transaction leaves the operation's connection aborted | Measured: on dev a voice command's failed refresh made its analytics insert fail too (`events == 0`). The same shape applies to any HTTP request. |
| TASK-<N3> | SkillsService re-runs its registry DDL at every construction | 4 statements and checkouts per construction (F3), at 8 production sites, including the `/chat/completions` path on the loop (`context_integration.py:59`, `:242`). |
| TASK-<N4> | Voice commands: the action's own ChaCha work checks out on the event loop | The AC4 residual (Design choice 6, the review's M4). A SYNC workflow's notes step is on the loop; MCP modules are unmeasured. |
| TASK-<N5> | Tests: seven PostgreSQL ChaCha tests fail on dev and are tracked nowhere | `test_postgres_backends.py` ×4 and `test_character_behavior_snapshot_migration.py` ×3. Every ChaCha PostgreSQL sweep re-baselines them (Task 5 Step 2). |

Not filed:
- **Persona: a `PoolTimeout` in a persona call ends the persona session.** That is dev's behaviour, consistent with HTTP's 500. This plan removes the loop stall; the outcome is a product choice nobody has asked for.
- **`Buddy/turns.py`'s detached tasks** (from the r2 re-review; read, not measured).
  - `_accept`, `_heartbeat` and `_drain` are started with `create_task` inside a `/turns` HTTP request. They inherit the request's operation and outlive it.
  - On PostgreSQL their `repository.db.transaction()` calls would then meet a closed operation: `ClosedChaChaOperationError`.
  - That is ADR-080's detached-task class (TASK-13897's area), not this guard. The module has no ChaCha marker, so the audit skipped it.
  - Measure it before filing it alongside TASK-13897.
- **`execute_query(read_only=True)`'s unclaimed read scope** (Review Focus 4). The one cross-thread case found, the clone job's cancellation read, takes turns with its clone thread except after a 5 s timeout. It is not measured; file it if the clone's timeout path is ever seen to overlap.

## Decisions for the owner

1. **TASK-13899: what an overlapping transaction inside one operation does.**
   - **Recommended, and what this plan implements: (A) raise `ConcurrentChaChaTransactionError`.**
     - It is AC1 as written ("either serialize or raise; neither loses a write").
     - It stops the silent loss: a crossing overlap leaves every later write in the operation uncommitted.
       - TASK-13861 measured this as 7 of 50 Chat_Macros replays. That was a configuration that never shipped: the run wrapped in one operation.
       - It is not a current production loss. The shipped Chat_Macros run gives each call its own operation.
     - **Cost:** a caller the audit missed becomes a 500 (HTTP) or a failed, possibly retried, job. That includes overlaps that happened to commit everything on dev:
       - nested in time;
       - a hand-off to a thread inside a transaction;
       - `asyncio.run` inside a transaction.

       The review's probe A showed each. Semantic conversion was such a caller, and Task 1 Step 6 fixes it.
     - No other caller was found, but the search was not exhaustive:
       - the widened audit is name-based within one module (Design choice 1);
       - the reviews spot-read the gather/wait sites, the direct-transaction fan-outs, and the MCP and Workflows fan-outs.
   - **(B) Log only** (an ERROR line, then join as today).
     - **Cost:** AC1 must be reworded through backlog-py, because it no longer prevents the loss, and the crossing interleaving's silent loss stays.
     - Task 1's test expectations, the ADR bullet and the TASK-13899 summary change to match, and Step 6's semantic fix is still worth keeping.
     - **A staged variant: log only now, raise later.** Ship (B) with the ERROR log, watch for it, then switch to (A) in a follow-up once the log is quiet.
       - It turns callers the audit missed into log lines instead of failures meanwhile.
       - It costs the same AC rewording, a follow-up task, and silent loss on crossing overlaps until the switch.
   - **Why it is the owner's call.** It trades a loud failure on paths that may have worked against silent data loss on paths that did not, for callers nobody has found yet.

The owner's `## Change summary` (ADR-004) is an input, not a decision. AC4's narrower wording follows the task's own fix shape (Design choice 6), and its residual is N4.
