# Workspace Persona Strict Startup: Offline Cutover

**Tracking:** TASK-13245.8, parent TASK-13245, issue #2950.
**Status:** Release-candidate instructions, not deployment certification.
**Scope:** Stage 2C backend startup only. Stage 2D, frontend adoption and broader parity are not delivered by this procedure.

## Release Gates

Before cutover, require reviewed exact-head tests, current hosted checks, a requester-owned Change summary, and an up-to-date normal merge into `dev`. Check the final migration registry and ADR links after integration; planning numbers are not reservations. Receipts use SQLite 74/PostgreSQL 78 after Companion's 73/77, with the durable contract in ADR057.

Required capabilities in every running binary:

- Permanent owner/key receipts, forced owner-only PostgreSQL RLS and lifetime capacity checks.
- Atomic committed chat/receipt acceptance and bounded strict-only transport.
- Transaction-local invalidation in local and Sync identity/scope writers, hard-delete tombstones, restore/cascade admission and privacy erasure.
- Current Persona admission before ordinary chat side effects; explicit rejection of unsupported session generation.

Keeping the strict endpoint disabled does not make an older writer safe against migrated data.

## Drain And Back Up

1. Block new API/Sync/import requests at the deployment boundary and drain active requests and jobs. Stop every API worker, Sync worker, background task and administrative/import process that can open or mutate ChaCha storage. Include local scripts and cached handles, not only the HTTP listener.
2. Confirm no old process can reconnect or restart automatically. This is an offline upgrade, not a rolling migration. Schema checks on a newly opened handle do not fence an already-open driver.
3. Inventory the actual configured SQLite user databases or shared PostgreSQL content database and their trusted owner configuration. Do not infer authority from a device/writer `client_id`, migrate an unreviewed path, or use the test database URL against production.
4. Take a consistent database backup using the existing tooling and test restoration into a disposable target. For SQLite use `DB_Backups.create_backup` with the allowed configured paths; do not copy only the main file while WAL writers are live. For PostgreSQL the existing configured helper makes a custom-format logical dump with matching `pg_dump`/`pg_restore` clients, not a physical cluster backup or PITR archive:

```bash
source .venv/bin/activate
python Helper_Scripts/pg_backup_restore.py backup --label content
```

The helper reads the configured content backend. Verify that configuration and backup destination before running it. Its restore command replaces database objects by default; run it only against the intended drained/disposable target, never an active database.

## Migrate And Validate

1. Install the reviewed receipt-aware release without restarting writers. First rehearse on the restored copy with the same configuration and trusted owners.
2. Open each inventoried database using normal `CharactersRAGDB` initialization under that release. Registered migrations are the only schema installation path; do not apply ad hoc SQL or rewind `db_schema_version`. PostgreSQL initialization must use the normal shared-content backend and tenant context.
3. Check the final registry version, receipt table/indexes, ENABLE/FORCE RLS and owner-only USING/WITH CHECK policies. Confirm existing chats, identity/provenance and native history remain intact. Test tenant access using a non-superuser/non-bypass role, not only the migration administrator.
4. Reopen storage with the same compatible binary. In a disposable Workspace, verify fresh startup is 201, a matching retry is 200 with `Idempotency-Replayed: true`, and both return the same conversation. Change metadata and the saved default; replay must still use the original accepted identity and current metadata. Test revocation, hard-delete 410, conflicting key 409 and exhausted lifetime capacity without new rows.
5. Run the process-boundary, migration, lifecycle and HTTP acceptance suites against disposable databases. Use the repository's official fixtures and require live PostgreSQL; do not build another fixture cluster or waive a failure:

```bash
source .venv/bin/activate
TLDW_TEST_POSTGRES_REQUIRED=1 TLDW_TEST_NO_DOCKER=1 python -m pytest -q \
  tldw_Server_API/tests/DB_Management/test_workspace_chat_startup_concurrency.py \
  tldw_Server_API/tests/DB_Management/test_workspace_chat_startup_migration.py \
  tldw_Server_API/tests/DB_Management/test_workspace_chat_startup_lifecycle.py \
  tldw_Server_API/tests/DB_Management/test_workspace_chat_startup_repair.py \
  tldw_Server_API/tests/Workspaces/test_workspace_chat_startup_transport.py \
  tldw_Server_API/tests/Workspaces/test_workspace_chat_startup_api.py
```

Fixture tests are release evidence, not a substitute for validating the production inventory and drained writer state.

## Restart And Observe

Restart only compatible binaries after all databases pass validation. Re-enable jobs and request admission after confirming there are no old workers or cached handles. Observe bounded startup errors and chat/receipt counts without logging raw keys, bodies, profile data or content. No greeting, provider request or Workspace Sync publication belongs to this startup transaction.

`WORKSPACE_CHAT_STARTUP_RECEIPT_LIMIT_PER_USER` defaults to `10000`. Set a positive integer consistently on all workers. Every accepted key consumes lifetime capacity, including invalidated receipts, deleted targets and deleted Workspaces. Raise capacity deliberately and monitor storage; do not lower it to enforce eviction. Matching replay remains possible at capacity when current access/admission permits it.

## Rollback And Retention

- Before any strict acceptance, a fully drained restore of the pre-cutover backup can revert the entire data/release pair. Verify no acknowledged strict key is lost before choosing that boundary.
- After strict acceptance, rollback means a compatible receipt-aware binary that preserves the table, RLS, writer hooks and admission semantics. An old binary against migrated data is unsupported, even if it can read the schema. Do not remove receipts or lower the schema version to make it start.
- Backups and disaster recovery must preserve acknowledged receipts together with their conversations and tombstones. Restoring an older snapshot can lose later accepted keys and allow duplicate creation; recover the corresponding durable WAL/PITR history or keep the deployment unavailable until that data-loss boundary is explicitly resolved. A content export/Chatbook is not a receipt backup.
- Receipts contain hashes, references and timestamps, not request/response/profile snapshots. They remain private data: restrict and protect backups and follow the existing account-erasure policy. They are excluded from ordinary export/import/clone/Sync payloads.
- Never expire, recycle or manually delete keys to reclaim budget. Hard-deleted conversations null their references; Workspace deletion and binding invalidation retain lifetime tombstones. A repeated key is not permission to resurrect or rebind a chat.

## Evidence Limits

The cached-writer rehearsal deliberately uses an already-open unhooked driver: an identity change is detected on replay, but change-away-and-back can escape permanent invalidation. It demonstrates why the writer drain is mandatory; it does not certify old binaries or mixed writers. The SQLite physical-backup rehearsal checks the existing backup/restore path, retained keys and capacity. PostgreSQL logical-dump rehearsal, migration/RLS/process evidence, and any operator physical-backup/PITR exercise are separate evidence. Do not label a source review, thread-only barrier or restored modern database as process/crash/historical-upgrade certification.
