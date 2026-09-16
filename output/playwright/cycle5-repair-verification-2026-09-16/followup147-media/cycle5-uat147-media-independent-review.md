# UAT147 reciprocal Media sequence ownership — independent review

## Verdict

**No actionable findings.** The frozen two-file correction is approved for integration. Native startup acceptance remains parent-owned; this review does not claim every possible shared-schema initialization race is resolved.

## Scope and ownership audit

Reviewed against HEAD `62456c09700d4597b936abbf14b4c3c921f9bcb9`:

- `tldw_Server_API/app/core/DB_Management/media_db/schema/postgres_sequence_maintenance.py`
- `tldw_Server_API/tests/DB_Management/test_media_postgres_sequence_ownership.py`

Both SHA-256 values match /private/tmp/cycle5-uat147-media-code-freeze.json (2026-09-16T17:00:51.493Z). Independent hash audit: /private/tmp/cycle5-uat147-media-independent-hashes.json.

The immutable positive allowlist contains **39 exact table/column pairs**. Independently cross-checked the canonical Media core, claims, TTS, media-file, data-table and email DDL in media_database_impl.py, the collection DDL in schema/features/core_media.py, and the corresponding explicit SERIAL/BIGSERIAL collection definitions in postgres_claims_collection_structures.py. The normalized canonical set and allowlist match with **zero missing and zero extra pairs**. Retained static inventory: /private/tmp/cycle5-uat147-media-independent-ddl.json. sync_log.change_id is retained as a legitimate Media pair; decks, chacha_keywords and arbitrary tables are excluded.

The membership check precedes any table MAX query or setval. PostgreSQL catalog discovery, escaped identifiers, parameterized sequence updates, owned-sequence behavior and transaction handling are unchanged. Therefore foreign catalog entries no longer cause Media to read/reset their sequences or acquire data-table locks. This is a minimal ownership correction; it does not replace the still-needed ChaCha-side keywords→chacha_keywords fix.

## Regression quality

The new positive test discovers actual serial pairs from a fresh **Media-only** official PostgreSQL fixture instead of reading the production allowlist. It advances every discovered sequence to9001, computes the real table maxima, and verifies maintenance restores the correct next value for every pair. Explicit keyword ID701 additionally verifies nonempty-table behavior. An omitted new canonical serial pair would fail this test.

The foreign-value test constructs real ChaCha tables and an arbitrary serial table, advances each foreign sequence to701, then invokes the actual Media constructor. All three must retain next value702. The lock test holds real ACCESS EXCLUSIVE locks on decks, chacha_keywords and an arbitrary table while the actual Media constructor runs on another thread. Completion must occur before the five-second bound; exceptions propagate through the future. On failure, transaction exit releases the locks before executor shutdown joins the worker, preventing the harness from hanging while it holds those locks. No SQL result, lock, or database error is mocked.

Inspected pre-fix RED /private/tmp/cycle5-postgres-147-media-red.redacted.log: **2 failed / 1 passed / zero skips**. The old code resets all three foreign sequences to1 and blocks under the foreign locks; the owned-pair positive control already passes.

## Independent verification

Command, using host access to the existing official disposable fixture:

`node /private/tmp/cycle5-postgres-fixture-run.mjs tldw_Server_API/tests/DB_Management/test_media_postgres_sequence_ownership.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_ownership.py tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py tldw_Server_API/tests/DB_Management/test_media_postgres_support.py tldw_Server_API/tests/DB_Management/test_media_postgres_runtime_validation.py -q -rs --show-capture=no`

**137 passed across 5 files, zero failed, zero skipped**, exit0,15.88s. Includes the original ChaCha overlap regressions, Media bootstrap/support, and current-policy startup validation. PostgreSQL is mandatory, Docker autostart disabled; project venv is activated by the official helper. Safe log: /private/tmp/cycle5-postgres-147-media-independent.redacted.log.

Inspected scoped static artifacts: Ruff baseline0/final0 and Bandit baseline0/final0 findings/errors, at /private/tmp/cycle5-uat147-media-{ruff-baseline,ruff-final,bandit-baseline,bandit-final}.json. Independent scoped git diff --check is clean. Static analyzers were inspected, not rerun.

## Limits

The tests cover the current canonical public-schema serial-column contract and the demonstrated foreign-table blocking/reset boundary. They do not certify all migration schedules, all PostgreSQL schema/search-path configurations, or normal single/multi authenticated runtime flows. No source/task/shared-document edits, held native database changes, process/profile changes, browser activity, inference or commits were performed.
