# ADR-080: Every non-HTTP entry point owns its PostgreSQL ChaCha checkouts

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

The thread-local checkout outside an operation is a fallback. When its thread exits, that thread closes it and queues its return, lock-free. When its handle is collected on another thread (only when the cyclic GC runs), the return is queued but waits until the owning thread has exited: no thread closes or returns a checkout that a live thread owns. The next ChaCha checkout or `close_connection()` in the process makes the queued returns. `CharactersRAGDB.close_all_connections()` never closes the shared content pool, and leaves other live threads' checkouts to those threads.

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
| Close a dropped checkout from whichever thread drops it | A close from another thread while the owning thread is mid-statement is a use-after-free inside libpq; a probe crashed the process (SIGSEGV) in 3 of 4 runs. |

## Consequences

- **New entry points.** A new non-HTTP entry point that touches ChaCha (a worker, a WebSocket, a background loop, or a service that opens a handle per call) does two things:
  - it wraps its unit of work in `chacha_operation(independent=True)` or decorates it with `@owns_chacha_checkouts`. If it fans out ChaCha transactions, it gives each call its own operation instead;
  - it adds a PostgreSQL test that checks no checkout is left.
- **Threads.** `run_in_executor` does not carry the operation into the worker thread. Use `asyncio.to_thread`, or open the operation inside the callable. `@chacha_operation(...)` decorates plain functions only; async code uses `with` or `@owns_chacha_checkouts`.
- **Detached tasks.** A detached task that outlives the step or job that started it must open its own operation; otherwise it gets `ClosedChaChaOperationError`. The Workflows engine's detached run tasks (`engine.continue_run`, the human-step timeout and the orphan reaper) inherit their caller's operation today. So a wrapped owner they can reach uses `independent=True`, as the chatbooks and character_chat adapters do. The notes adapter does not yet, and TASK-13897 tracks the root cause: starting those tasks outside the caller's operation.
- **Cost.** Each per-call operation costs about 5 extra database round trips, plus a reset in the pool's worker. One that runs on the event-loop thread checks out synchronously there, so when other work holds the whole pool it can stall the loop for up to `pool_timeout`. Calls in async code therefore go through `asyncio.to_thread`, as the persona and voice-assistant WebSockets and the skill tool call do since TASK-13896.
- **Checkouts held through provider calls.** A job owner wrapped in one operation keeps its checkout, idle in a transaction, from its first ChaCha use until it returns, through every provider call it makes (for example research outputs, persona visual generation, writing annotation review, Data Tables). Dev held as long or longer: until job end, or for the life of the executor thread. A new owner with long provider calls should prefer an operation per ChaCha call, as the Chat_Macros run does, or returning its checkout after each step outside an operation, as the notes-graph job does.
- **Close-then-reuse is not caught everywhere.** `current_operation_is_open()` is also True on SQLite, where the operation does not own the handle's thread-local connection, so a cleanup that skips its close inside an operation leaves that SQLite connection to its thread. And `VNAssetPacksRepository.run_worker_replay_operation` (`VNAssetPacks_DB.py:2697-2719`) closes and then reuses its handle inside the VN job's operation. That is safe only while the VN repository rejects PostgreSQL handles (`_require_sqlite_chacha_db`); whoever adds PostgreSQL support there must fix it first.
- **The fallback.** Checkouts from code nobody has wrapped still come back on thread exit, at the next ChaCha call. A dropped handle's come back only through the cyclic GC, and not before their threads exit; for the event-loop thread or an executor thread, that is the life of the process.
- **Concurrent transactions are refused.** Since TASK-13899, a `db.transaction()` that enters while another thread or task holds the operation's transaction raises `ConcurrentChaChaTransactionError` instead of joining it, so a fan-out inside one operation fails loudly instead of losing writes. Nesting on the same thread and task still joins. The refusal also covers overlaps that happened to commit (one nested in time, or a hand-off to a thread inside a transaction), so a caller that overlaps gets an error, not usually-correct writes; semantic conversion's two lookups were the one such caller, and now run in turn. It raises rather than waits: a lock would deadlock a transaction that waits on another thread's, would block the event loop while it waited, and, being per thread, would not see two tasks on the loop.
- **An explicit shared backend.** A handle that is given the shared backend explicitly (`DB_Manager.create_chacha_database(backend=…)`, no callers today) does not know the backend is shared, so `close_all_connections()` on it would still close the shared pool.
