# UAT247 / TASK13260.189 — author review packet

## Result and source boundary

The frozen helper advances Media serial sequences without rewinding allocated IDs hidden by RLS, abandoned by rollback, or reserved by caching. **117 tests passed; zero skips** (13.72 seconds, four existing warnings). The unchanged original worker/persistence suite separately passed **48 tests, zero skips** (9.64 seconds). Independent review and native acceptance remain pending. This packet owns one production helper and three test files; worker/persistence and their original regression were not changed.

`owned-manifest.json` SHA256: `ea50ce2285a2d4870b9b95bc422983ee265df96654f205bf148a7efc0e7ec254`.
Production helper SHA256: `20a6d438ee5d4406e6e0e008efcda0347d7007742a93b444a19f8daf9ef05d0b`.

## Repair and transaction contract

The UAT147 Media-owned serial allowlist is unchanged, independently checked by AST and actual schema inventory. The catalog is traversed in deterministic schema/name order. For each allowed serial, a quoted `ALTER SEQUENCE ... OWNED BY` reasserts its existing catalog-proven dependency, obtaining a lock that blocks ordinary concurrent `nextval`/`setval`. Under that lock the helper compares `last_value`/`is_called` with visible MAX and advances only when needed. Empty and already-ahead sequences are left alone; a large explicit-ID gap needs constant work.

Each temporary lock is confined to `conn.transaction(force_rollback=True)`. Only that savepoint is rolled back: caller rows and existing transaction ownership remain untouched. PostgreSQL `setval` intentionally survives the rollback. This releases sequence locks before subsequent schema DDL, avoiding the observed cross-module bootstrap deadlock. Actual helper tests prove another connection allocates while the caller remains open, caller rows still roll back, and a real query failure propagates while leaving the caller able to inspect and roll back its own work.

The maintenance caller must own the schema/sequence, as required by this DDL operation. Tests use a disposable NOSUPERUSER/NOBYPASSRLS/NOINHERIT schema owner with FORCE RLS and application `is_admin=0`, within the official fixture. No native role was elevated. This is not a general data-only grantee API and does not promise concurrent arbitrary explicit-ID import safety beyond the existing import contract.

## Causal evidence and retained reassessment

| Receipt label | Outcome / interpretation |
|---|---|
| `uat247-sequence-causal-red` | Original helper: six failures/four controls. Hidden tenant IDs, called/uncalled high water, rolled-back allocations, cache reservation and real concurrent allocation expose rewind. |
| `uat247-sequence-red` | Earlier characterization also retains unsupported no-option ALTER and its cleanup error; no production implementation used that statement. |
| `uat247-sequence-lock-proof` | Actual supported OWNED BY preserves sequence parameters/state/dependencies and demonstrably blocks allocation. |
| `uat247-sequence-final` / `uat247-sequence-deadlock-proof` | First production candidate: 113 pass/one failure; actual SQLSTATE40P01 in unchanged UAT147 concurrent bootstrap. `candidate_outer_lock.py` is retained. |
| `uat247-adjacent-baseline` | Original helper passes that same adjacent test, confirming introduced lock-lifetime regression. |
| `uat247-lock-lifetime-red` | Production-helper causal regression fails because allocation cannot finish before outer caller closes. |
| `uat247-savepoint-proof` | Real rollback-only savepoint preserves forward sequence advance and caller rollback while releasing allocation lock. |
| `uat247-final-verified` | Current second revision: **117 passed, zero skipped**. |

Receipts and exact command arrays are copied under `receipts/`; baseline modules, first RED tests, direct driver observer, and non-mutating baseline replay are retained separately. After the first introduced deadlock, work paused for explicit design reassessment before this second production revision.

## Tests and compatibility changes

The new 14-case actual-PG suite covers sequential cross-owner IDs; called/uncalled and pristine states; unrelated sequences; behind/equal-uncalled explicit IDs; allocations from rolled-back transactions; a 9-billion-ID gap with caller rollback; cache high water; unchanged parameters/dependencies; concurrent ordinary allocation; narrow lock lifetime; and real-error propagation.

The existing every-owned-serial test now starts sequences behind imported IDs rather than requiring an unsafe rewind from 9001. Its complete-schema allowlist assertion and foreign/blocked-table controls remain. Two fake-backend schema unit controls now return sequence state and model the rollback-only context; the invalid-MAX case expects no reset. Original failing stale expectations are retained. The unchanged ChaCha/Media ownership and concurrent initialization controls, plus existing migration synchronization test, run in the final117.

## Verification

- Ruff: zero findings across four owned files.
- Bandit: production zero findings/parse errors; tests zero findings/parse errors with only B101 excluded.
- Python compile: four files pass. Added patch lines have no trailing whitespace.
- New test file passes Ruff formatting. Each of the three existing files already fails whole-file formatting on its exact baseline and current bytes; no broad formatting was performed.
- AST outside `sync_postgres_sequences` is identical, including the UAT147 allowlist. Quoted identifiers remain sourced from the filtered PostgreSQL catalog; values use bound parameters.

Independent replay (official running isolated fixture, no Docker start or native database mutation):

```sh
source .venv/bin/activate &&
TLDW_UAT_EVIDENCE_LABEL=uat247-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs \
  tldw_Server_API/tests/DB_Management/test_media_postgres_sequence_monotonicity.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_sequence_ownership.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_ownership.py \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py::test_media_postgres_sequence_sync \
  -q --tb=short
```

Retry replayed its separate original worker48 against this frozen helper: **48 passed, zero skips**, with its three source/test hashes unchanged; receipt `uat238-233-with247-green.redacted.log`. No native browser/provider/service/held-profile actions were performed; actual queued two-owner upload acceptance remains a separate gate.
