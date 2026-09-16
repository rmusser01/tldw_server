# UAT147 completion — reciprocal Media sequence ownership

Task TASK-13260.88. Original ChaCha tuple correction remains valid and committed41fcd5e4b3. This bounded continuation changes only `media_db/schema/postgres_sequence_maintenance.py` and adds `tests/DB_Management/test_media_postgres_sequence_ownership.py`.

## New native cycle and cause

After UAT150's repeated native API reads passed, concurrent normal startup exposed another ownership cycle. Sanitized actual PostgreSQL statements:

- Media process7891: `SELECT COALESCE(MAX("id"),0) AS max_id FROM "decks"`; waits for AccessShareLock held by ChaCha7897.
- ChaCha7897: `ALTER TABLE IF EXISTS chacha_keywords ENABLE ROW LEVEL SECURITY`; waits for AccessExclusiveLock held by Media7891.

Media's sequence helper discovers every public-schema serial sequence, then reads and resets all of them, including foreign ChaCha tables. Its earlier read of `chacha_keywords` holds a conflicting lock while it later waits for `decks`. This is reciprocal cross-component sequence ownership under the same incomplete UAT147 acceptance, not a failure of the original `keywords`→`chacha_keywords` correction.

Private official fixture proof also found actual Media reopen resets foreign decks nextID702→1. No held native databases or processes were changed.

## Minimal correction and canonical list

Added an immutable positive allowlist of39 table/column pairs and skip foreign catalog rows **before any MAX/setval**. Catalog discovery, legitimate Media synchronization, SQL escaping, policy SQL, RLS, backend and transaction behavior stay unchanged. No global lock, retry, SQL parser or negative ChaCha exclusion list.

Canonical sources:

- `media_db/media_database_impl.py`: `_TABLES_SQL_V1` (Media/Keywords/junctions/transcripts/chunks/documentversions/sync/templates), `_CLAIMS_TABLE_SQL`, `_MEDIA_FILES_TABLE_SQL`, `_TTS_HISTORY_TABLE_SQL`, `_DATA_TABLES_SQL`, email DDL.
- `schema/features/core_media.py`: base PG conversion/bootstrap.
- `schema/postgres_claims_collection_structures.py`: output templates, highlights, collection tags/content items and claims extensions.
- `schema/postgres_data_table_structures.py`, `schema/email_schema_structures.py`: corresponding extension tables.

The existing `_POSTGRES_REQUIRED_TABLES` registry has14 coretables and omits many actual owned serial pairs; it cannot safely serve as the complete allowlist. Fresh real Media-only schema discovery confirms39 current pairs. A behavioral inventory regression independently discovers every serial pair before adding foreign tables, forces each sequence out of sync, then checks maintenance returns the proper next maximum. A future owned pair omitted from the production allowlist will fail this test.

## RED/GREEN

Permanent RED: **2failed,1passed,0skips**,10.46s; `/private/tmp/cycle5-postgres-147-media-red.redacted.log`.

- Foreign ChaCha decks, chacha_keywords and arbitrary serialtable all incorrectly reset702→1.
- Actual Media constructor blocks under real ACCESS EXCLUSIVE foreign-table locks (5second bounded worker timeout; locks released before worker join).
- All-current-owned Media sequence maintenance remains the positive control; explicit keywordID701 advances next702.

Final required PostgreSQL18.6 GREEN: **137passed,0skips**,5files,16.50s,exit0. Includes new3 cases, original147 concurrent/sequence3 cases, schema bootstrap, Media PG support and current-policy validation.

```sh
node /private/tmp/cycle5-postgres-fixture-run.mjs tldw_Server_API/tests/DB_Management/test_media_postgres_sequence_ownership.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_ownership.py tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py tldw_Server_API/tests/DB_Management/test_media_postgres_support.py tldw_Server_API/tests/DB_Management/test_media_postgres_runtime_validation.py -q -rs > /private/tmp/cycle5-postgres-147-media-green.log 2>&1
node /private/tmp/cycle5-postgres-report.mjs /private/tmp/cycle5-postgres-147-media-green.log --save
```

Focused final regression rerun: **3passed,0skips**,4.67s,exit0; `cycle5-postgres-147-media-final-focused.redacted.log`. Four warnings in each run. Only one blank-line removal for Ruff occurred after behavioral suites; no semantic change.

Projectvenv Ruff production baseline0 and owned final0; production Bandit baseline0/final0. JSON artifacts `cycle5-uat147-media-{ruff-baseline,ruff-final,bandit-baseline,bandit-final}.json`.

## Limits / acceptance

These real-PG tests prove Media skips foreign sequences/table locks and maintains its canonical serial pairs; original147 concurrent constructor regression also passes. They do not claim every possible bootstrap race is solved. Independent review and repeated normal single/multi startup plus authenticated MCP/Chat/Notes acceptance remain parent-owned and pending. UAT150 source and held r3 profiles remain unchanged.
