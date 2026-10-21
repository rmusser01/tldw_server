---
id: TASK-13861
title: Legacy thread-local PostgreSQL ChaCha checkouts leak (thread exit, long-lived
  threads, close_all_connections before reset)
status: In Progress
labels:
- chat
- postgres
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-13841 (measured on origin/dev 7c94a2efb1). (1) Thread exit: outside a chacha_operation, CharactersRAGDB._connection_state() falls back to threading.local and _get_connection_for_state checks out a pooled PostgreSQL connection that belongs to that thread (ChaChaNotes_DB.py:8337-8343, :8652-8660). psycopg's Connection.__del__ returns early for pooled connections, so when the thread exits the checkout is never returned and its pool slot is lost (measured: holders 'AnyIO worker thread: DEAD', pool_available 0). AnyIO worker threads exit after MAX_IDLE_TIME = 10 s idle, so this leaks even in a long-running server. (2) Long-lived threads: a default executor thread or the event-loop thread outside an operation holds its legacy checkout until the handle closes, which can exhaust the pool without any thread exiting. (3) close_all_connections before reset: on psycopg_pool, CharactersRAGDB.close_all_connections() returns the checkout through putconn, which queues an asynchronous ReturnConnection, and then closes the pool at once; MaintenanceTask.run discards the task, so the connection is neither reset nor closed. In TASK-13841's test the discarded connection stayed referenced with an open transaction and blocked pg_restricted_backend's DROP OWNED BY until pytest-timeout (review: 19 of 52 runs). Product code calls close_all_connections() per call, for example the MCP modules (flashcards_module.py x13, notes_module.py x10). (4) Callers outside the middleware: ChaChaOperationMiddleware only wraps scope type 'http'. The WebSocket route /chat/stream (audio_streaming.py:1947) persists chats through _shim_get_chacha_db_for_user_id (:2415) with no chacha_operation; the persona.py and voice_assistant.py WebSocket routes come next. The only product callers of chacha_operation( are ChaCha_Notes_DB_Deps.py (3), conversation_enrichment.py (2), the repair scope in ChaChaNotes_DB.py (1) and study_pack_jobs_worker.py (1). None of these paths was exercised on PostgreSQL. (5) Test apps: 29 test files build a bare FastAPI() over ChaCha with Postgres params and no middleware (git grep -l 'app = FastAPI()' -- tldw_Server_API/tests | xargs grep -l 'CharactersRAGDB\|get_chacha_db_for_user' | xargs grep -l postgres | xargs grep -L ChaChaOperationMiddleware); each leaks one checkout per request and passes only because the default pool has 10 connections. Optional: pg_restricted_backend's teardown (12 test files) waits 300 s on a leaked idle-in-transaction session and then os._exits the session; SET lock_timeout before DROP OWNED BY, plus a report of the role's open sessions, would make it fail fast.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 /chat/stream persistence is exercised on PostgreSQL first, and every production caller that reaches the legacy thread-local path on PostgreSQL is listed, with evidence
- [ ] #2 No PostgreSQL checkout outlives its thread or pins a pool slot outside an operation: such callers run inside chacha_operation or return their checkout
- [x] #3 After CharactersRAGDB.close_all_connections() on psycopg_pool, no checkout it returned is left open without its reset
- [ ] #4 Test apps over PostgreSQL ChaCha install ChaChaOperationMiddleware, or are shown not to leak
- [x] #5 CharactersRAGDB.close_all_connections() never closes the shared PostgreSQL content pool
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR A (fix/chacha-pg-legacy-checkouts). close_all_connections() on PostgreSQL: (1) it closed the shared content pool. Every handle built without a backend uses the one cached content backend, so one MCP tool call (flashcards.decks.list through FlashcardsModule.execute_tool) left every later ChaCha access failing with 'Connection pool is closed' until restart. Callers: MCP flashcards x13, notes x10, quizzes x14, characters x2, slides x2, skills x1, persona_visuals x1, chats x2, rpg x1; Persona/memory_integration x3; persona.py /stream x1; Notes_Library x2; close_all_chacha_db_instances. Fix: on the shared backend only this thread's checkout is returned; content-backend shutdown owns the pool (ADR-020). (2) On a caller's pool the checkout was returned through putconn and the pool closed at once, so psycopg_pool discarded the queued reset and left the connection open INTRANS. Fix: close this thread's checkout (discard_connection) before closing the pool.
PR A legacy-checkout guard: a thread-exit fix plus a GC-timed backstop. A legacy thread-local checkout was lost when its thread exited (plain thread, AnyIO worker, asyncio default executor). A guard stored next to the checkout queues it on a queue.SimpleQueue when the thread-local slot is dropped, and the next ChaCha checkout or close_connection() in the process returns it; a caller already blocked in getconn does not benefit. Thread exit is deterministic (CPython clears a dead thread's locals at once; the tests use no gc.collect()). A dropped handle is cyclic garbage, so its guard runs only when the cyclic GC does: in a server-sized heap that can take arbitrarily long (review probe 4). It is a backstop; PR C makes the known per-call sites own their checkouts. The finalizer never calls the pool (a collection can run on a thread holding psycopg_pool's task-queue lock); every explicit return disarms it first.
PR A locking reads: get_workspace(for_update=True) and get_persona_profile(for_update=True) read the legacy thread-local tx_depth, so inside an operation (every production HTTP request) they raised 'locking reads require a transaction connection' even inside db.transaction(). On PostgreSQL every non-Character workspace chat creation, POST /chats/workspace-startup and persona admission returned 400, as did chat creation in a workspace under a chat quota. Found when PR D's middleware went into test_conversation_quota_count_backends.py's app. Fix: read self._connection_state().tx_depth.
PR A RED/GREEN on 3c4d734981: test_chacha_postgres_close_all_connections.py dev 2 failed, fix 2 passed. test_chacha_postgres_legacy_checkouts.py dev 5 failed 1 passed, fix 6 passed (v1's lock-taking guard fails the finalizer test). test_chacha_postgres_locking_reads_in_operation.py dev 2 failed, fix 2 passed. The three new files plus test_chacha_connection_typing.py on unchanged dev: 9 failed, 5 passed. Operation-scope pins with the guard: 109 passed. Sweep of the 108 files that call close_all_connections over PostgreSQL ChaCha, 4 shards: 1224 passed 1 skipped; 762 passed; 832 passed 5 skipped; 652 passed 4 skipped; no FAILED, ERROR or timeout (dev baseline: not needed, nothing failed). Bandit: B101 only in the new tests; none in the product files. The ADR for the whole series (ADR-NNN, numbered in PR C) records the close_all_connections rule.
PR A review fix (#3293, round 1): the guard's finalizer now also closes the dead thread's connection (PQfinish; psycopg 3.3.6 Connection.close() takes no lock and makes no pool call without close_returns), so a session the thread left idle in a transaction ends with the thread instead of at the next ChaCha call, and the queued putconn of the closed connection still recovers the pool slot (psycopg_pool discards it and adds a replacement). The guard records os.getpid() when armed and does nothing in a forked child, and a finalizer that fails is swallowed. Test: test_a_dead_threads_session_does_not_stay_idle_in_a_transaction (pg_stat_activity, with and without the caller holding a checkout through close_all_connections).
Known bound, not covered: inside a chacha_operation, close_all_connections() on a caller's own pool returns the operation's checkout through putconn and closes the pool at once, so that checkout can be left open without its reset; dev does the same, and only tests reach it, because production handles use the shared content backend, which returns early.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
