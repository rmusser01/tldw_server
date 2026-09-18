# TASK13260.210 World Book lifecycle report

## Outcome

The supported World Book lifecycle now uses the existing portable `db.transaction()` boundary at each exercised local operation: entry create/read/update/delete, book soft/permanent delete, and character detach. Manual commits were removed from these nested-safe paths. Book create/update and character attach already used that boundary and remain unchanged.

The repair preserves cache invalidation, optimistic-version conflicts, invalid-reference behavior, attachment upsert/idempotency, and caller-owned transaction rollback. It does not alter database wrappers, SQL guards, schemas, timestamp policy, routes, permissions, configuration, or global state.

`restore_world_book` is not a supported service operation. Qualification therefore covers the existing soft-delete visibility contract and the supported `hard_delete=True` permanent-delete option; it does not add restore behavior.

## Causal evidence

The first official real-backend lifecycle test failed with four failures. The PostgreSQL trace showed `add_entry` attempting to enter `BackendConnectionWrapper` as a context manager. SQLite showed the same method committing an entry inside a caller transaction, leaving the entry after the caller raised for rollback. These are retained as `red-add-entry.log` in `.tmp/uat-repairs-231-246/worldbook-lifecycle210/`.

Independent operation cases then confirmed the same boundary defect for entry update/delete, character detach, and book delete before the local transaction replacement. A harness-only assertion correction changed the public entry key from `entry_id` to `id`; it did not change product code or causal findings.

## Verification

| Check | Result |
| --- | --- |
| Real lifecycle, SQLite + official PostgreSQL | 12 passed, 4 warnings, 0 skips |
| Combined lifecycle plus prior timestamp/init/character-read contracts | 57 passed, 5 warnings, 0 skips |
| Permission and negative controls | 10 passed, 3 warnings |
| Adjacent World Book unit controls | 10 passed, 2 skipped; not used for acceptance |
| Compile | passed |
| Bandit | 0 production findings; 34 low test-only B101 assertions |

The final combined check used the project official fixture runner with isolated label `worldbook210-final-20260917190855`. The command receipt SHA-256 is `d7cc1b4ff9d3607a2d45015df357451353afd0fb9d28ce05c4c3a5868496d150`; the private runner log was inspected only for the aggregate result and has SHA-256 `baea0c7e0aa4fb2f0c61a54cf7807ff90b8a1d2ce5f9afa618215ae55e0e27f7`.

Ruff reports `I001`, `SIM118`, and `SIM118` on this scope. A read-only scan of baseline `0d7f2a23c9` produces the same three diagnostics, so no new Ruff finding is attributed to this change.

## Pending attribution

The legacy mock suite command below currently gives 29 passed and 10 failed. The failures are not used as acceptance evidence and have not been changed. Root is running a read-only baseline source-selection comparison before they are classified as pre-existing or as a separate task.

```sh
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Character_Chat/test_world_book_manager_legacy.py -q --tb=short
```

## Frozen source hashes before independent review

```text
c6ae32d056d79c4fb5a529fae5e99d749ca855c0a975fc4d485fe4d72abca186  tldw_Server_API/app/core/Character_Chat/world_book_manager.py
a8ce91e3223370abfc68b5468d93c26bf8a1830e546b650062ff3bfd06687ce7  tldw_Server_API/tests/DB_Management/test_world_book_lifecycle_backends.py
```

No commit has been made. Independent review is required before parent-controlled commit.
