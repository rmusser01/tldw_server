# UAT177 / TASK-13260.114 — Study PostgreSQL diagnosis

## Proven causes

1. `get_flashcard_analytics_summary` uses `substr(fr.reviewed_at, 1, 10)` for its UTC streak query. PostgreSQL stores this column as a timestamp; the actual backend logs `UndefinedFunction`. This fails even for an empty review table. SQLite passes.
2. The method's SELECTs run on the persistent external thread connection without owning a read transaction. After the query error, that connection remains failed. In a controlled same-connection sequence, otherwise valid `list_decks`, selected-deck `list_flashcards`, and completed `list_flashcard_review_sessions` fail with `InFailedSqlTransaction`. The review-session maintenance transaction rolls back on its failure; the next assistant DB read then succeeds. An explicit isolated-test rollback restores all tested reads. This is a proven cascade mechanism, not proof of the exact worker-thread assignment in the native concurrent request burst.
3. Separate actual HTTP probes on fresh isolated databases reveal response-schema defects: assistant history rejects PostgreSQL `datetime` values in thread `created_at` and `last_modified`; populated completed review history rejects session `started_at`, `last_activity_at`, and `completed_at`. SQLite returns 200 for both. These are distinct from the original analytics query and must be explicitly scoped by the parent before repair. The native original assistant failure is not proven to have reached response validation; a poisoned connection could have failed first.

## Native corroboration (sanitized)

The retained native backend log has first PostgreSQL query failure at 2026-09-16T23:19:15.685Z; analytics summary HTTP500 at .700Z, decks500 at .707Z, completed review-sessions500 at .709Z. The review-session trace enters `abandon_stale_flashcard_review_sessions` and its update execution. Later at23:19:55.336Z analytics500 recurs, followed by filtered cards500 at .362Z and assistant500 at .368Z. Driver exceptions are redacted in the native log, so attribution relies on the separate actual PostgreSQL reproductions rather than invented native SQL details.

## Minimal proposed UAT177 repair

- In the analytics method alone, choose `(fr.reviewed_at AT TIME ZONE 'UTC')::date` on PostgreSQL; retain existing SQLite `substr`. The existing `str(row['review_day'])` produces the expected YYYY-MM-DD key. Explicit UTC avoids inheriting the database session timezone.
- Mark only the three known pure analytics SELECT calls `read_only=True`, using the existing UAT174 helper. It owns a transaction only when the connection is idle and no caller scope exists; it leaves existing caller writes/transactions to their owner. No translator, global rollback, datetime stringification, or SQL query rewriting.
- Tests: actual HTTP empty and populated analytics; selected-deck/global/workspace visibility; UTC midnight crossings under a non-UTC PG session; owned read cleanup and forced query failure cleanup; explicit and implicit caller commit/rollback preservation. Official PostgreSQL fixtures required, SQLite controls.
- Keep separate timestamp-contract defects explicit pending parent scope decision. Existing UAT167 schema before-validator pattern is the likely minimal fix if authorized, without modifying DB timestamp types.

## Evidence

- `test_study_probe.py`: real SQLite/PostgreSQL operations and same-connection cascade.
- `.tmp/fresh-uat-recovery-20260916/uat177-isolated-probe.redacted.log`: **2 expected PostgreSQL failures /12 passes /0 skips**; UndefinedFunction and InFailedSqlTransaction captured via existing backend log extra fields.
- `test_response_probe.py`: actual FastAPI router, real databases, dependency-owned synthetic fixture only.
- `.tmp/fresh-uat-recovery-20260916/uat177-response-probe-valid.redacted.log`: **2 PostgreSQL response-contract failures /2 SQLite passes /0 skips**.
- Earlier `uat177-response-probe.redacted.log` is a private probe import-collection error, not a product regression receipt; corrected by locally defining the same official-fixture-backed setup.

No browser, live request, service, model, production source, tracked tests, runtime database, or tracker mutation performed for this diagnosis. All database probes used official temporary fixture databases on the owned cluster.
