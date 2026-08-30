# TASK-13134 Task 2 Report

## Scope

Connected canonical Note and Sync mutations to the semantic persistence ledger.
The mutation path remains transaction-local and does not call Jobs, providers,
embeddings, or vector storage.

## Implementation

- Added no-op-when-unconfigured semantic lifecycle boundaries to
  `NoteSemanticStore`. They resolve only the supplied authoritative dataset and
  active generation, use the current Note transaction, and retain opaque
  generation identity in cleanup work after hard deletion.
- Added canonical fingerprinted dirty/tombstone hooks to `NoteStore` for
  content creation, content edits, restore, soft delete, hard delete, and Sync
  mutations. Relationship-only updates and idempotent writes do not advance a
  semantic revision.
- Threaded `SyncEnvelope.dataset_id` through the Notes materializer. Callers
  without dataset authority remain backward-compatible no-ops; no arbitrary
  dataset is selected.
- Added lifecycle and Sync coverage for generation coalescing, cleanup work,
  rollback, disabled/unconfigured datasets, restore with unchanged content,
  and a fresh import-style dataset with no server-local semantic state.

## TDD Evidence

The required RED command initially failed because Note mutations did not
publish semantic state. A later focused RED test exposed that an unchanged
content Sync restore did not advance its generation; the implementation now
treats restoration from a tombstone as semantic.

The required focused and ordinary regression command passed after the minimal
implementation:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/integration/test_semantic_note_lifecycle.py tldw_Server_API/tests/Sync/test_sync_v2_notes_semantic_lifecycle.py tldw_Server_API/tests/Notes_Graph/integration/test_graph_lifecycle_queries.py -q
```

Result: `13 passed`.

## Verification

- `python -m ruff check` on all changed production and focused test files:
  passed.
- `python -m bandit -r` on changed production files: `0` results.
- `git diff --check`: passed.
- Existing PostgreSQL semantic migration contract:
  `python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_semantic_migration_postgres.py -q`:
  `11 passed`.

## PostgreSQL Scope

No new live PostgreSQL Note/Sync lifecycle test was added. The existing live
fixture covers the semantic schema, owner/dataset RLS, and transaction-backed
store compatibility; the new lifecycle hooks use its backend-neutral connection
and transaction APIs. A dedicated PostgreSQL lifecycle suite would require
broader Sync/Notes fixture plumbing than this task's focused change.

## Backlog

`TASK-13134` already existed and was in progress. No task notes were added, so
no Backlog update was needed.
