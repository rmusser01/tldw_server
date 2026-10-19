---
id: TASK-13861
title: Legacy thread-local PostgreSQL ChaCha checkouts leak (thread exit, long-lived
  threads, close_all_connections before reset)
status: To Do
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
- [ ] #3 After CharactersRAGDB.close_all_connections() on psycopg_pool, no checkout it returned is left open without its reset
- [ ] #4 Test apps over PostgreSQL ChaCha install ChaChaOperationMiddleware, or are shown not to leak
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
