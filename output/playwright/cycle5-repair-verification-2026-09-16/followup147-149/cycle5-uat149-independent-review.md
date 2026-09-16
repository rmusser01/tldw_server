# UAT149 / TASK-13260.87 — independent review

**Disposition: clear within the two-file Collections bootstrap scope.** No actionable findings.

Reviewed frozen production `tldw_Server_API/app/core/DB_Management/Collections_DB.py` and new `tldw_Server_API/tests/Collections/test_collections_postgres_bootstrap.py`, plus the existing helper, PostgreSQL backend inspection, official fixture lifecycle, and adjacent SQLite/PG controls. Both SHA-256 hashes match the 2026-09-16T16:37:26.311Z freeze; audit: `/private/tmp/cycle5-uat149-independent-hashes.json`.

## Findings and coverage

- Six column inventories now use existing `_table_columns`. PostgreSQL obtains actual `public` schema columns through backend `get_table_info`; fresh declared columns no longer enter inappropriate duplicate ALTER backfills.
- SQLite still calls the same `_sqlite_columns` helper. Removing its unused pre-create content inventory leaves the effective post-create inventory in place. No helper behavior or public error sanitization changes.
- Backfill remains conditional on actual missing fields: the new real-PG test physically drops output-template metadata, invokes schema setup, then verifies restoration.
- Explicit repeated schema setup retains the exact saved template ID/body; it does not rely only on the bootstrap memo. Catalog inspection failure is injected at the backend inspection boundary and propagates as DatabaseError.
- Existing real-PG integration preserves artifact/content round trips, idempotency behavior, and legacy notification-column upgrade. Two existing SQLite schema controls pass unchanged.

## Fresh independent verification

From repository root, using existing local official fixture credentials through the supplied wrapper (not printed):

```sh
node /private/tmp/cycle5-postgres-fixture-run.mjs \
  tldw_Server_API/tests/Collections/test_collections_postgres_bootstrap.py \
  tldw_Server_API/tests/Collections/test_collections_schema_bootstrap.py \
  tldw_Server_API/tests/Collections/test_collections_postgres_integration.py \
  -q -rs --show-capture=no
```

**7 passed, 0 skipped, 8 warnings, exit 0**, 14.96 seconds. Five tests used the official PostgreSQL fixtures; two used SQLite. Redacted output: `/private/tmp/cycle5-uat149-independent-tests.redacted.log`, produced with `/private/tmp/cycle5-postgres-report.mjs`. Existing old pytest temporary-directory cleanup warnings followed the successful summary.

Author static summary inspected: production Bandit 0 findings/errors; production Ruff nine baseline signatures unchanged; new test Ruff 0. These static results were inspected, not independently rerun. Author valid three-failure RED is retained separately in `/private/tmp/cycle5-postgres-149-red-confirmed.redacted.log`.

## Limits

No repository source/test/task edits, commits, held native-profile/database access, or runtime restarts. Tests create and remove their official isolated PostgreSQL databases. No new concurrency or schema migration framework was reviewed. The live app's reading-digest worker startup consequence remains root-owned native acceptance; this result establishes the bounded Collections bootstrap correction and its covered backfills, not a full-server startup pass.
