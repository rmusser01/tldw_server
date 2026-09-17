# UAT181 / TASK13260.118 — architecture reassessment after third native gate

## Conclusion

Stop adding SELECT flags. The durable boundary should be an explicitly owned **request or operation connection lease**, separate from the cached schema/database configuration. The present database object outlives requests and keeps a PostgreSQL connection in `threading.local`; its ownership cannot be recovered reliably from SQL text or PostgreSQL INTRANS state. The existing per-read opt-in correctly preserves caller work, but therefore cannot clear an inherited read transaction.

No production or test edits were made during this reassessment. The proposed design and causal test stages below require review before implementation.

## What the native evidence proves

All3667 files in `api61182-source.json` still match their recorded hashes. That receipt records committed3ce1b63d57, API61182 and no hot reload. The ordinary Notes/keywords/collections bootstrap returned200. At01:59:22.938759 UTC, PID37203 was idle in transaction with AccessShare locks on buddy_attachments, buddy_profiles, source_review_occurrences and source_review_plans; no Notes/keyword relation locks remained.

At02:01:11.394499 UTC, the next snapshot shows old idle transactions37202/37203 and initializer PID38214 waiting for AccessExclusive on source_review_occurrences. Native Notes/collections/keywords then returned500, with another Notes500 on retry. These are now observed500s, unlike the prior overlap that eventually returned200. The parent retained SIGTERM/replacement health and subsequent recovery separately.

The old API access log contains two actual successful `/api/v1/flashcards/source-review-plans/due` requests (lines1302 and1858 of backend-1789610298022.private.log). The corresponding endpoint is synchronous and calls two unscoped SELECTs on source_review_occurrences joined to source_review_plans. SourceReviewDuePanel enables this query when the Study panel is active; the query hook has a60-second refetch interval. Thus a real due-review read is a concrete possible transaction starter matching the retained relations, not a hypothesis based only on a browser tab title.

**Causal limit:** retained access logs do not map a request to PostgreSQL PID or identify which browser tab made it, nor do they retain the first statement of37202/37203. Buddy reads already use `read_only=True`; their retained locks are consistent with inherited transactions. That does not independently prove the exact first SQL statement or ordering for each PID. The monitor's `transactionReadOnlyVerified` describes its own metadata transaction, not the observed application's transactions. `causal-receipts.sanitized.json` preserves safe request/status, PID/lock and source-freeze evidence without queries, parameters or credentials.

## Why the flags cannot establish a lifecycle guarantee

1. `get_chacha_db_for_user` returns a cached CharactersRAGDB; it is not a yielding request resource. Non-request helpers also return the same cached object.
2. `_get_thread_connection` pins a pool checkout until explicit release. A sync endpoint leaves it in a worker thread; async handlers on the same event-loop thread can also share the thread-local object across request tasks.
3. `execute_query` and BackendConnectionWrapper use that connection as externally supplied. PostgreSQLBackend.execute intentionally does not settle an externally supplied connection.
4. The opt-in only creates a read scope when transaction status is IDLE and both tracked transaction depths are zero. Once any unflagged SELECT starts a transaction, later flags preserve it, just as they must preserve an implicit UPDATE or raw BEGIN from a legitimate caller.
5. An idle read transaction and caller-owned implicit work can both appear as INTRANS with zero wrapper depths. Changing defaults, checking SELECT text, or blanket rollback after a read/request on the shared object cannot distinguish their owners. SELECT functions, locking reads, CTE writes and RETURNING make SQL classification an additional independent risk.

The previous44-case test is valid for its direct handler chain. It is insufficient for the full runtime because the UI has independent background consumers and synchronous handlers reuse worker-thread connections. The next regression must traverse actual dependency/request lifetimes and overlapping owners, not only prepend another read to a direct-call test.

## Comparable designs already in this repository

| Existing pattern | Useful contract | Limit when adapting |
| --- | --- | --- |
| MediaDbFactory/MediaDbSession; `DB_Deps.get_media_db_for_user` and `managed_media_db_for_owner` | Cache factory resources; make a fresh scoped session; use finally for both HTTP and non-HTTP ownership. Runtime execution passes an explicit transaction connection when present, otherwise uses backend-owned execution. | Do not copy its constructor-per-request literally: ChaCha constructor runs substantial DDL. A thin proxy forwarding already-bound methods to the same shared thread-local object is also insufficient. |
| PostgreSQLBackend.execute/transaction | Ownership is explicit through supplied connection versus internally borrowed connection. Owned operations settle and return their checkout; external operations preserve their owner's work. | Reusing its SQL command classification as a global ChaCha default would change implicit-write and SELECT-function behavior. Reuse ownership and pool cleanup, not blanket execution semantics. |
| AuthNZ.DatabasePool.acquire/transaction and guarded sync pool lease | Acquisition/release is an explicit lexical lifetime. Cleanup survives errors; sync wrappers check owner identity before release. Statement autocommit is a separate explicit API. | Asyncpg behavior differs from psycopg. This is an ownership/lifetime example, not authorization to switch ChaCha to autocommit or transplant AuthNZ connections/fixtures. |

References: media_db/runtime/session.py:44,100; media_db/runtime/connection_lifecycle.py:94; media_db/runtime/execution_ops.py:116; API_Deps/DB_Deps.py:232,271; backends/postgresql_backend.py:905,997; AuthNZ/database.py:1183,1387,1444; AuthNZ/profile_user_sync_boundary.py:310,317.

## Options and recommendation

### A. Explicit request/operation lease — recommended ownership repair

Keep the cached initialized schema/backend configuration, but acquire an owned session for each HTTP request and a managed session for each background/job/WebSocket operation. Bind connection lookup **and transaction-depth state** to that owner. Reuse that session through its nested service/repository calls. At its end, return only its own checkout through the existing pool cleanup; do not commit merely because an HTTP response succeeded. Uncommitted work belongs to the operation owner, and all existing explicit commit/rollback behavior inside the operation remains intact.

An explicitly supplied external/caller session is borrowed, never auto-committed/rolled back/released by a nested scope. Implicit writes, raw BEGIN, locking reads and nested contexts continue using the same connection until that owner chooses commit/rollback. Caller work spanning service calls must have an explicit outer lifetime; a new HTTP request must not accidentally inherit it from a reused thread.

Implementation must address async dependency→sync worker execution and teardown explicitly. Merely adding `finally: db.close_connection()` to today's async dependency can close a different thread's connection or fail to close the worker's. ContextVar alone is also insufficient if a spawned task inherits a live mutable lease. Use an owner token and captured checkout registry/session, define child-operation inheritance, and test isolation. Existing repository stores retain references to their DB, so connection/transaction lookup must resolve the bound owner consistently; a superficial proxy will not do that.

Keep initialization/default-character maintenance under their own managed operation. New lease entry must not reconstruct CharactersRAGDB or rerun migrations per request. Non-request `get_chacha_db_for_user_id`/owner consumers need a managed counterpart and a call-site audit; leaving an unowned background route outside the contract would leave the guarantee incomplete. This is a bounded lifetime change in shared plumbing with actual-route tests, not another domain-SELECT inventory.

### B. Per-operation transient statements — not a safe drop-in

Media's execute-with-no-transaction pattern is appealing, but applying it to every ChaCha call would break current implicit multi-call writes and raw connection workflows unless every caller declared ownership first. It can be an internal consequence of optionA for explicitly owned standalone operations. It is not a lower-risk global replacement for the existing API.

### C. Verified already-current bootstrap without unconditional DDL — separate availability option

`_initialize_schema_postgres` locks the version row and runs a long list of `_ensure_*` methods even when already at target67. `_ensure_source_review_schema_postgres` recreates indexes/functions/triggers; RLS is also ensured unconditionally. This amplifies a read leak into cross-domain startup failure, including Notes requests whose own reads are already scoped.

A validated current-schema path could avoid writes when structural/security invariants already match, keeping migrations/repair under a separately owned transaction. But `current_version == target` alone is not sufficient: current initialization repairs storage, trigger/index and forced-RLS drift and some historical changes were ensured without a version bump. A shortcut must prove those invariants, preserve historical/fresh migrations, fail closed on incompatible state, and serialize any repair/recheck. It would reduce needless lock contention but would not fix idle transactions, stale snapshots or pool lifetime. Treat it as a distinct reviewed unit if pursued; do not weaken RLS/verification to make the replacement pass.

## Bounded causal test/design stages before implementation

1. **Reproduce actual lifetime leakage without modifying SQL:** official required-PG fixture, actual due-review route plus Buddy and complete Notes routes through real dependency wiring. Exercise empty and populated source plans. Keep the old application's connection alive while a second real constructor targets the same isolated DB with a bounded lock timeout. Record all public table locks and per-request/session owner IDs, never query parameters. Include async-handler and sync-worker execution, not just direct function calls. This determines which scoped boundary must own cleanup without attributing the native browser tab prematurely.
2. **Ownership prototype controls:** two requests on one worker, two async requests sharing an event loop, sync handoff and cancellation/failure; one owner's teardown cannot settle another owner's pending write. Preserve implicit UPDATE+read then explicit commit/rollback, raw BEGIN, existing outer/nested/backend transaction contexts, FOR UPDATE locks until owner exit, CTE writes/RETURNING and a side-effecting SELECT inside an explicit owner. Verify no automatic success commit, no forced release of borrowed sessions, and no SQL-classifier/default change. Cover a managed background operation and child task inheritance.
3. **Compatibility and real replacement:** official PostgreSQL plus SQLite data/rollback, existing quota/RLS/owner boundaries, source-review CRUD/sync/version tests and181 lifetime controls. Verify current-head reopen and genuine historical/fresh migration paths are unchanged, initializer not rerun per request, bounded pooled checkouts, and all old read-only request sessions settled before actual replacement. Keep caller-owned active transactions intentionally blocking until caller exit; that is a control, not a defect.
4. **Separate migration option only if selected:** prove a current complete schema reopens alongside a held legitimate read without acquiring unnecessary DDL locks; missing/wrong tables, indexes, triggers and forced-RLS policy must still be repaired safely or rejected. Preserve upgrade rollback/version bookkeeping. This cannot be tested with mocked schema initialization.

No candidate is implemented or declared passing here. The next approval should select a lifetime contract and test boundary, not a fourth collection of flags. Parent owns native recovery, integration and task tracking.
