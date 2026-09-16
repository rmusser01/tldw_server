# UAT132 / TASK13260.73 — ready for independent review

SQLite recognized FTS failure now replaces MATCH at the same condition/parameter position with bound literal title/content LIKE filtering. Escaped percent/underscore/backslash remain literal. All existing visibility/field/filter predicates stay present. Count-error and results-only-error paths share the replacement; results-only fallback recomputes the count rather than retaining a different FTS total. Valid FTS and PostgreSQL paths are unchanged.

## Verification
- Actual SQLite repository regressions: RED **8 failed / 5 passed**, then **13 passed**. Expanded **15 cases** add missing-table and team/org/date controls.
- Final command: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_media_search_fts_fallback.py tldw_Server_API/tests/DB_Management/test_media_db_entrypoint_ops.py tldw_Server_API/tests/DB_Management/test_media_postgres_support.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_search_request_model.py -q` — **52 passed / 1 skipped**. PostgreSQL-dependent skipped node: `test_media_postgres_support.py::test_fresh_postgres_schema_enforces_transcript_run_history_uniqueness`. Its exact dynamic fixture reason was not emitted without `-rs`. User requires this gap resolved; root is preparing the official fixture, and no PostgreSQL acceptance is claimed or waived.
- Controls: own/foreign/unmatched marker, title/content/author, valid quoted query and NOT operator, empty query, pagination/count, literal punctuation, IDs/media-type/keywords/trash/deleted/team/org/date.
- `python -m ruff check` production + new test: **0 findings**, production baseline0.
- `python -m bandit` changed production: **0 findings**.
- Existing deprecation and pytest cleanup warnings remain. The result-only FTS failure is injected at the actual repository SQL execution boundary, then real SQLite executes fallback queries.

Logs/manifests share this prefix: `-red.log`, `-green.log`, `-final-green.log`, `-ruff-before.json`, `-ruff-after.json`, `-bandit.json`, `-manifest.json`. No runtime/browser/inference/frontend/staging/commit. Independent review and native acceptance remain pending.

## Required PostgreSQL follow-up
Root restored the official PostgreSQL18.6 fixture. Bounded backend32 and AuthNZ2 checks passed with REQUIRED=1/NO_DOCKER=1 and zero skips. The two formerly skipped nodes now execute. The bootstrap test assertion defect was corrected separately under TASK13260.75.1; see /private/tmp/cycle5-postgres-136-report.md for exact commands, scope and redacted evidence. This resolves the recorded fixture-execution gap for these controls, not native full-workflow acceptance. Product code for this unit remains unchanged.
