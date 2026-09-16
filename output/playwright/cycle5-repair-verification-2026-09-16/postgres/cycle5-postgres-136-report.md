# UAT136 / TASK13260.75.1 — PostgreSQL verification correction

**Frozen, ready for independent review.** One test file changed; no production changes. Official fixture on root-owned PostgreSQL **18.6 (Debian 18.6-1.pgdg13+2)**, confirmed by actual SHOW server_version through the same fixture. No manual DB/container setup. REQUIRED=1 and NO_DOCKER=1 supplied by the private controller helper.

## Cause and correction
The backend intentionally replaces driver errors with `PostgreSQL query execution failed`; the existing bootstrap assertion expected raw unique/duplicate-key text. The actual driver reports SQLSTATE23505 and the exact intended indexes. This is a verification/test failure, not a native product failure or weakened uniqueness enforcement.

A test-local cursor observer retains only SQLSTATE and constraint name before rethrowing the identical driver exception through unchanged backend redaction. The test asserts:
- the exact sanitized public error;
- 23505/idx_transcripts_media_run_id, then23505/idx_transcripts_media_idempotency_key;
- after both rejected transactions, the three valid committed rows remain, including both nullable idempotency keys, and subsequent backend reads work.

## Verification
- Original required-PG backend: **31 passed / 1 failed**, zero skips (`cycle5-postgres-backend-bounded.redacted.log`).
- Safe actual-driver diagnosis: **1 passed**; version and two exact23505/index tuples (`cycle5-postgres-constraint-diagnosis-final.redacted.log`). Initial probe used psycopg2 pgcode and retained None for psycopg3; corrected to sqlstate-or-pgcode before the permanent change. No raw exception text is retained in structured diagnostics.
- Corrected bootstrap: **1 passed**, zero skips (`cycle5-postgres-136-green.redacted.log`).
- Final required-PG backend: **32 passed**,17 deliberately deselected by postgres filter, **zero skips** (`cycle5-postgres-backend-final.redacted.log`). Includes previously skipped fresh Media schema and strict Chat image snapshot, transaction/pool controls, and actual PostgreSQL FTS backend mapping.
- Required-PG AuthNZ: **2 passed, zero skips** (`cycle5-postgres-auth-bounded.redacted.log`), real JWT principal/state and Prompt Studio authenticated path using existing isolated AuthNZ fixtures.
- Ruff **19 baseline / 19 final, no added findings**; Bandit **0** with B101 excluded because these are pytest assertions. Production untouched. git diff --check passes.
- Five sanitized evidence files scanned: **0 fixture-password/DSN/JWT matches** (`-scan.json`). Raw logs remain private0600 and must not be retained as public evidence.

## Exact commands
```sh
node /private/tmp/cycle5-postgres-fixture-run.mjs tldw_Server_API/tests/Chat/integration/test_chat_image_recovery.py::test_postgres_strict_image_snapshot tldw_Server_API/tests/DB_Management/test_chacha_postgres_transactions.py tldw_Server_API/tests/DB_Management/test_media_postgres_support.py tldw_Server_API/tests/DB_Management/test_database_backends.py -k postgres -q -rs
node /private/tmp/cycle5-postgres-fixture-run.mjs tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_prompt_studio_invariants.py -q -rs
node /private/tmp/cycle5-postgres-fixture-run.mjs tldw_Server_API/tests/DB_Management/test_media_postgres_support.py::test_fresh_postgres_schema_enforces_transcript_run_history_uniqueness -q -rs
```
Redirect private outputs, then sanitize with `node /private/tmp/cycle5-postgres-report.mjs <log> --save` before reading summaries. The helper activates the project venv and passes private fixture settings.

## Limits and bookkeeping
These are bounded real PostgreSQL controls, not a native PostgreSQL full workflow certification. The auth tests do not replace the separate pending native Prompt-collections acceptance. UAT129/132/133 source hashes are unchanged. Independent review pending; no staging/commit/runtime/browser/inference actions.

Canonical task75.1 was created through delayed MCP. CLI fallback raced it and created duplicate75.2, which was corrected and archived through the official CLI before implementation. No code belonged to the duplicate.
