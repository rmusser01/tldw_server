---
id: TASK-13841
title: Postgres chat import returns 500 when a second owner reuses chat ids
status: Done
labels:
- chat
- postgres
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tests/Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat[postgres] fails on dev 7b5ba7aa6d with `assert 500 == 201` at line 798; the sqlite variant passes. The 500 comes from POST /api/v1/chats/import, whose trace records error.type psycopg_pool.PoolTimeout. The test only runs when a Postgres is reachable, so CI without Postgres does not see it. Found while sweeping the TASK-13560 change (PR #3247); it fails on unchanged dev too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause of the pool timeout is fixed, not hidden by a longer timeout
- [x] #2 On PostgreSQL a second owner reusing the first owner's chat or message ids gets 409 (shared key space), imports with ids of its own get 201, and the first owner's chat and messages are untouched
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Cause: the test app in _client was a bare FastAPI() without the ChaChaOperationMiddleware that main.py:2796 installs. Without an operation, CharactersRAGDB falls back to a threading.local checkout. TestClient runs each request on a new portal whose AnyIO worker threads exit when the request ends, so each request leaked one PostgreSQL checkout. pg_restricted_backend sets pool_size=5 (max_size; max_overflow is ignored), so the 6th checkout waited the 30 s pool_timeout and raised PoolTimeout, giving a 500 at line 798. A probe of get_connection/return_connection showed 5 checkouts, 0 returns, pool_available 0, and holders 'AnyIO worker thread: DEAD' x3, portal x1, MainThread x1. Fix: _client installs ChaChaOperationMiddleware, so each request's checkout is returned when the response ends (probe: peak 3 outstanding). Pool size and timeout are unchanged. Product code is unchanged; HTTP in production already has the middleware.
Teardown hang uncovered by the fix: with the 500 gone, the body's own main-thread reads (lines 780-781, 793-794, 804-806) run outside any operation and leave both handles' legacy checkouts INTRANS. The owners teardown called first.close_all_connections(), which returns the checkout through putconn (queuing psycopg_pool's asynchronous ReturnConnection) and then closes the shared pool; MaintenanceTask.run discards a task on a closed pool, so the connection was neither rolled back nor closed. When it stayed referenced, its session stayed idle in transaction with a lock on message_images, and pg_restricted_backend's DROP OWNED BY (tests/_plugins/postgres.py:409) waited until pytest-timeout os._exited the session after 300 s. Fix: the owners teardown rolls back both handles' connections while the pool is open, inside try/finally around close_all_connections().
RED (dev e1f7770953, re-checked on 9ddc9d83f4): target 1 failed (postgres), 1 passed (sqlite); assert 500 == 201 at :798, error.type psycopg_pool.PoolTimeout (56 s, load 58). Whole file 1 failed, 270 passed (1732 s, load 60 falling to 19). Middleware alone: the body passes (2 passed) but teardown hangs at postgres.py:409 DROP OWNED BY; deterministic with a scratch plugin that starts queued returns 0.5 s late and keeps them referenced (1 discarded INTRANS return, 3 IDLE); without it, two loops of 10 side by side: 6/20 hung at postgres.py:409 at load 23-71 (6 passed; the other 8 hit the 60 s limit in fixture setup, schema migration or DROP DATABASE as load rose to 65-71, not in this teardown) (review: 19/52). GREEN with the teardown fix: the scratch plugin gives 2 passed (5 discarded returns, all IDLE); two loops side by side 20/20 passed, 0 hangs, at load 41-64 (review: 46/46); whole file 271 passed (3699 s at load 33-106, alongside the image file run). Ruff clean. Bandit (uvx bandit) on the file: B101 x290 and B105 x2, identical to dev, so no new findings.
AC #1 amended (Review M4): its old text said a second owner reusing the first owner's ids gets 201 on Postgres, which is false by design. PostgreSQL ids share one key space, so reused chat or message ids get 409 (lines 782-791); the 201 is the second owner's import with ids of its own (line 798), and the first owner's chat and message stay untouched (lines 804-808). Removed with --remove-ac 1 and re-added with that wording; it is now AC #2, and the old AC #2 is now #1.
ADR: not needed; a test-only change that creates no durable policy (Docs/ADR/README.md). Governing: ADR-005 (Bandit on the touched paths), ADR-059 (backlog-py), ADR-004 (the owner's Change summary on the AI-authored PR).
Follow-up: TASK-13859, for legacy thread-local checkouts (thread exit, long-lived threads), close_all_connections() closing a psycopg pool before its queued reset (MCP modules call it per call), the WebSocket /chat/stream persistence path, and the 30 other test apps without the middleware.
The follow-up first filed as TASK-13859 is TASK-13861: open PR #3271 took 13859 first.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The PoolTimeout was a per-request checkout leak in the test app, not in the import endpoint. The test built its app without ChaChaOperationMiddleware, so each request took a thread-local PostgreSQL checkout on an AnyIO worker thread that exited without returning it. The 5-connection restricted pool ran out at the 6th checkout. The test app now installs the same middleware as main.py. That let the body reach its own main-thread reads, whose open transaction could outlive the pool when teardown closed it, hanging DROP OWNED BY; the owners fixture now rolls those back while the pool is open. The target test passes without hangs and the whole file is 271 passed (3699 s at load 33-106, alongside the image file run) (dev: 1 failed, 270 passed). No timeout or pool size changed. AC #1 was amended to the 409/201 contract the test checks. Docs: not relevant (test-only). Known follow-up: TASK-13859.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
