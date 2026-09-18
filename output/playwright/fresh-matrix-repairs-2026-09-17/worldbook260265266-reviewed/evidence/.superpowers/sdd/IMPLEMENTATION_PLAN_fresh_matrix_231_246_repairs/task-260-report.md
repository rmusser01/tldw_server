# UAT260 / TASK13260.202 World Book timestamp report

## Outcome

World Book API readbacks now emit explicit instants without altering global timestamp parsing, database session setup, pools, schemas, permissions, cache ownership, or route transactions.

- SQLite-generated naive `created_at` and `last_modified` values are marked UTC only at the World Book response boundary.
- PostgreSQL reads project those native `timestamp` columns with the current session policy: `AT TIME ZONE current_setting('TimeZone')`. PostgreSQL therefore applies its own non-UTC and DST rules before the response model serializes an explicit offset.
- Get, list, and character-attached reads use that shared projection; create and update routes use those reads for their response models.
- The frontend formatter is unchanged. Focused tests prove explicit `Z`, zero, negative, and fractional-hour offsets preserve their instant and render consistently across representative browser timezones.

## Stable-policy limitation

A naive PostgreSQL field contains no historical writer offset. The code can unambiguously apply the known current/writer session policy, including DST, but cannot reconstruct an instant if historical sessions used a different or unknown timezone. That needs a separately approved migration or provenance policy; this change does not guess or blanket-append `Z`.

## Separately diagnosed transaction repairs

The timestamp endpoint exposed two existing PostgreSQL-only transaction-boundary faults. Root created and authorized separate records before each product edit.

- **UAT265 / TASK13260.207:** `update_world_book` used `get_connection()` as a context manager. PostgreSQL's `BackendConnectionWrapper` does not implement that protocol. Replacing it with the existing portable `db.transaction()` preserves the optimistic-version check and caller rollback semantics.
- **UAT266 / TASK13260.208:** `attach_to_character` had the same wrapper misuse. It now uses `db.transaction()` and checks both existing foreign records within the transaction before the existing backend-specific idempotent upsert. This retains false validation results for missing references and rolls back a caller-owned attachment.

The final test coverage exercises create/read/update timestamp readbacks, optimistic conflict, update rollback, attachment idempotency, attachment validation, attachment rollback, endpoint permissions, and SQLite/PostgreSQL controls. An inventory found raw context-manager usages in other World Book operations, but none were executed or failed in this work; they remain unchanged.

## Test evidence

Evidence directory: `.tmp/uat-repairs-231-246/worldbook260/`.

| Check | Receipt | Result |
| --- | --- | --- |
| Intended SQLite red before implementation | `sqlite-red.log` | exit 1: 1 failed, 4 deselected |
| Focused final SQLite backend controls | `sqlite-final.log` | exit 0: 3 passed, 6 deselected |
| Mandatory official PostgreSQL fixture run | `postgres-final.log` | exit 0: 45 passed, 5 warnings; no skips |
| World Book permission controls | `permissions.log` | exit 0: 9 passed, 5 warnings |
| Focused frontend formatter controls | `frontend.log` | exit 0: 1 file, 26 passed |
| Python compile and diff whitespace check | `compile-final.log` | exit 0 |
| TypeScript diagnostic comparison | `tsc-comparison.json` | 90 baseline, 90 current, 0 added, 0 removed |

The mandatory PostgreSQL invocation was:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=worldbook260-final \
  node .tmp/fresh-uat-recovery-20260916/run-pg-tests-fixture-database.mjs \
  tldw_Server_API/tests/DB_Management/test_world_book_timestamp_responses_backends.py \
  tldw_Server_API/tests/DB_Management/test_world_book_initialization_backends.py \
  tldw_Server_API/tests/DB_Management/test_character_world_book_reads_backends.py \
  -q --tb=short
```

The runner sets its required PostgreSQL fixture environment; it was used with an isolated `worldbook260-final` fixture database label, not a substitute database setup.

## Static evidence

- `bandit-final.json`: zero findings in `world_book_manager.py`. The 20 findings are test-only `B101` assertions in the new pytest module; no production security finding was introduced.
- `ruff-final.log` / `ruff-baseline.log`: Ruff exits 1 for the identical pre-existing source findings: one `I001` import-order item and two `SIM118` items. The new test has no Ruff finding.
- `compile-final.log`: `python -m compileall -q` over the changed Python source and test exits 0.

## Reassessment record

1. The causal SQLite test failed as expected because response timestamps were naive.
2. A direct FastAPI function test supplied the framework `Query` default for `expected_version`; the test was corrected to pass `None` explicitly, with no product change.
3. The official PostgreSQL run exposed the separate update context-manager fault. UAT265 was recorded and authorized; the targeted transaction repair then passed.
4. The next PostgreSQL run exposed the separate attachment context-manager fault. UAT266 was recorded and authorized; the targeted transaction and validation repair then passed.

Each fault was re-scoped and tracked before its edit. No remaining unexecuted raw context-manager site was broadened into this change.

## Frozen source hashes

```text
1ffb3839115e34daa162abae2f9b6648741597814744798d435c70960010c870  tldw_Server_API/app/core/Character_Chat/world_book_manager.py
928dbf57c452f6ea020a09272c26359e4677450fc2d64ac0b7b0e3f16bd078cf  tldw_Server_API/tests/DB_Management/test_world_book_timestamp_responses_backends.py
3f051d0e8f05490434f51e184a5c9bfc13bc853da4c6e45ce8dcdd71de64c1ca  apps/packages/ui/src/components/Option/WorldBooks/__tests__/worldBookListUtils.test.ts
```
