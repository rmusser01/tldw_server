# PostgreSQL restart lock diagnosis — 2026-09-16

## Finding

The post-restart Buddy 500s came from a schema bootstrap waiting on a read transaction retained by the old API process. This is a connection/transaction lifecycle defect exposed by overlapping shutdown and restart. The captured blocker is unrelated to the Media/ChaCha sequence ownership deadlocks repaired under UAT147. Do not reopen that specific sequence defect based on this incident alone.

No source, runtime, browser, task or database-state changes were made during this investigation. Parent independently stopped the already-authorized old owned API; all database queries here were read-only.

## Direct evidence

- `ps` confirmed old PID68056 and new PID82576 alive. Old log records shutdown at local15:44:21.583, then `Waiting for connections to close` at15:44:21.684, with no application-shutdown completion.
- `lsof` showed old68056 retained approximately30 PostgreSQL connections and two established HTTP connections on18503, but no listening socket. New82576 owned the listener on18503. The HTTP connections alone do not prove they were SSE.
- At22:46:44.169857 UTC, `pg_stat_activity` and `pg_locks` identified new-era PG backend16272 (started22:44:32.655323) waiting for an AccessExclusiveLock while executing:

```sql
ALTER TABLE note_folders DROP CONSTRAINT IF EXISTS note_folders_path_key
```

- `pg_blocking_pids(16272)` was `[5829]`. Backend5829 was in the same preserved PGmulti database, started22:23:13.438655, and idle in a transaction opened22:24:36.498399. Its last query selected folders through `note_folders`, memberships, source memberships and sync suppressions. It held AccessShareLock on the exact relation OID721767 for which16272 requested AccessExclusiveLock.
- New app log records repeated ChaCha initialization failure for user2 after PostgreSQL query execution failure. The source bootstrap statement is `ChaChaNotes_DB.py:18109`.
- After parent force-stopped only old68056, the22:48:12.939938 UTC catalog check found5829 absent and no waiting locks in PGmulti. Former waiter16272 had progressed to idle-in-transaction with no blockers. The legacy constraint was absent at this after-recovery check; this does not independently prove its pre-stop presence/absence.
- Parent stop receipt is22:47:35.129 UTC. The first completed Buddy GET/attachment200 responses are22:47:35.290 UTC, 161ms later, completing previously waiting requests. By the22:50:29 snapshot, this log prefix contains70 earlier Buddy500 responses and144 subsequent Buddy200 responses; latest successes take33ms. Exact first/last results are in `uat147-restart-recovery-receipt-20260916.json`.

The pre-stop Docker network translation prevented direct client-port equality between OS `lsof` and PG backend5829. Its disappearance and bootstrap recovery immediately after the parent's targeted old-process stop provide the causal control, together with its pre-restart creation time and preserved-profile database identity.

## Bounded code trace

1. `CharactersRAGDB.get_note_folders_for_note` at34529–34554 runs the exact captured SELECT via `execute_query` with its default `commit=False`.
2. `CharactersRAGDB.execute_query` at8160 obtains the thread-pinned connection and runs its cursor. It only commits when requested; ordinary standalone reads have no cleanup boundary.
3. `BackendCursorWrapper.execute` at500–509 passes that pinned connection as the backend's `connection` argument.
4. `PostgreSQLBackend.execute` treats a supplied connection as external. Its implicit SELECT rollback at approximately1051–1062 is guarded by `managed_depth == 0 and not external_conn`, so it intentionally cannot release this caller-owned read transaction.
5. Notes `_run_db_call` at1984 simply uses `asyncio.to_thread`; the cached `get_chacha_db_for_user` dependency at805–844 returns the database instance and does not close the worker-thread transaction when that call returns.
6. Old graceful shutdown waited for its HTTP connections before completing resource cleanup. Its pending read lock therefore survived while the replacement process attempted bootstrap DDL.

New-process catalog evidence already shows other idle-in-transaction sessions after recovery, so stopping the old API clears this instance but does not establish a durable transaction lifecycle repair. A repair must preserve intentional explicit/nested write transactions; blanket rollback of every supplied connection would violate that contract.

## Next action and limits

- Parent has already performed the targeted old-process stop and the blocked bootstrap recovered. Keep new82576 unchanged while native checks resume.
- Track this as a distinct retained-read-transaction/restart-DDL issue after checking duplicates; the sequence repairs remain a separate verified contract.
- The browser's loading Flashcards editor is not independently explained by the database lock once no request is recorded. No claim is made that its pending UI action submitted, retried or completed. Parent owns browser evidence and actual retry.
- No need to terminate arbitrary PostgreSQL backends or restart the cluster. No such actions were performed.
- A first catalog serialization attempt failed because PostgreSQL `char` was returned as bytes; the successful retry explicitly casts `contype` to text. This was a reporting-only failure, with no database write.

## Private receipts

- `/private/tmp/uat147-restart-lock-diagnosis-20260916.json` — before-stop activity and locks, SQL string literals redacted.
- `/private/tmp/uat147-restart-relation-diagnosis-20260916.json` — after-stop relation/activity state.
- `/private/tmp/uat147-restart-recovery-receipt-20260916.json` — bounded HTTP result metadata and exact current-log prefix hash.
- Parent stop receipt: `.tmp/uat151-137-native-20260916/old-api-force-stopped.json`.

Logs remain private. Credentials were read only for an authenticated metadata connection and were never displayed.
