# Buddy v1 backend qualification

Revision under qualification: `1fc19c7c8384f38eca2becdf53fb8bbcd205be7a`, with the separately reviewed local model-catalog fix still uncommitted in the shared worktree.

## Targeted command and result

Run from `/private/tmp/tldw-server-buddy-persona-ux`:

```sh
CHAT_FORCE_MOCK=1 .venv/bin/python -m pytest \
  tldw_Server_API/tests/Persona/test_independent_buddies.py \
  tldw_Server_API/tests/Persona/test_buddy_turn_ledger.py \
  tldw_Server_API/tests/Persona/test_buddy_turns.py \
  -q --no-header --tb=short
```

Result: **52 collected; 51 passed, 1 skipped, 3 warnings in 113.78 seconds**.

- `test_independent_buddies.py`: 26 collected, 25 passed and 1 PostgreSQL case skipped.
- `test_buddy_turn_ledger.py`: 9 passed.
- `test_buddy_turns.py`: 17 passed.

Full captured output: `/private/tmp/buddy-v1-backend-qualification-tests.log`.

The skip was resolved with only the named case under explicit no-Docker mode:

```sh
TLDW_TEST_NO_DOCKER=1 .venv/bin/python -m pytest \
  tldw_Server_API/tests/Persona/test_independent_buddies.py::test_postgres_v65_upgrade_installs_buddy_storage_and_forced_tenant_policies \
  -q --no-header --tb=short -rs
```

Result: **1 skipped in 0.77 seconds** because `Postgres not reachable; skipping Postgres-backed tests`. Log: `/private/tmp/buddy-v1-backend-qualification-postgres-skip.log`. This run provides no fresh PostgreSQL migration evidence.

## Upgrade evidence classification

The SQLite upgrade test is a real predecessor-schema migration, not only a symbol/unit assertion:

1. It creates an on-disk SQLite database while `CharactersRAGDB._CURRENT_SCHEMA_VERSION` is pinned to 65.
2. It inserts a pre-v66 conversation titled `Keep this` and closes the database.
3. It reopens with the current schema head and verifies schema version 66, all six independent Buddy/turn tables, and the preserved conversation.

The adjacent rollback test also creates a v65 database, injects invalid v66 Buddy DDL, and verifies the migration transaction leaves version 65 with no partial `buddy_*` tables.

This is faithful source-bound v65→v66 upgrade/rollback evidence, but its legacy-data assertion covers one conversation row. It does not load a production database dump or validate a wider legacy conversation graph. `test_schema_head_v66_has_one_sqlite_and_postgres_step` is unit-only migration registration coverage. The PostgreSQL case is designed as a real v65→v66 upgrade plus RLS/CRUD/activity test, but it skipped in this qualification.

The remaining independent-Buddy tests exercise an actual SQLite repository and FastAPI routes with same-database user isolation, immutable copied art after source Persona deletion, stale profile/attachment versions, deleted/foreign targets, bounded workspace projection, and exact result acknowledgement.

The turn suites use the real authenticated Chat ASGI admission and persistence boundary with a controlled mock provider. They cover exact conversation persistence, accepted FIFO continuation after detach/transport cancellation, concurrent conversation queues, workspace/identity rechecks, idempotency, Stop/late-publication fencing, process-owner expiry/reconciliation, principal-scoped status, and pre-dispatch failures. They are not real external-provider evidence.

## Live navigation receipts supplied by the qualification

- `/private/tmp/buddy-v1-qualification-6_vfinnl/navigation-turns.json`: one turn, `status=completed`, nonempty `result_message_id`, `error_code=null`.
- `/private/tmp/buddy-v1-qualification-6_vfinnl/navigation-attachment.json`: attachment version 3, conversation scope, authorized target title present, `unavailable_reason=null`.
- The completed turn's `conversation_id` exactly matches the attachment's `scope_id`.

Together with the reported live Watchlists→Research Workspace navigation and closed-modal completion, these receipts show the accepted result remained bound to the selected conversation across navigation. The files alone do not prove packaged/native application behavior or a real external model provider.

No production or test files were changed for this backend qualification. The shared worktree's existing model-catalog source/test changes and root-owned qualification artifacts were left untouched.
