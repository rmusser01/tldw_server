# UAT174 / TASK-13260.111 — safe duplicate-deck classification

## Frozen result
Ready for independent review. Production is exactly12 added lines across3 files; one permanent test file adds11 cases. `owned-manifest.json`, `owned.patch`, and `review-snapshot/` capture the exact release. Baseline copies match commit `cc0f12a5d8dc21e370a5a2034c44244fa53d09af`. The private approved design is `DESIGN.md`; parent owns shared design/tracker/task integration and commits.

PostgreSQLBackend.execute now retains only whether a caught trusted psycopg error has SQLSTATE23505. It raises a payload-free `UniqueConstraintError(DatabaseError)` outside the catch, with the exact existing `PostgreSQL query execution failed` message. This marker has no custom attributes, driver cause/context, SQLSTATE field, diagnostics, constraint identities, values, or SQL. Other failures retain the ordinary DatabaseError path. Rollback, pool ownership and logging are unchanged.

ChaCha's existing `_is_unique_violation` recognizes the marker before its previous SQLite/text fallback. Its existing add_deck ConflictError path and existing HTTP409 mapping now work. No other ChaCha method, execute_many path, endpoint, UI, schema, or exception payload changed.

## RED → GREEN
- Before production edits, real official PostgreSQL/SQLite suite: **4 FAIL /7 PASS /4 warnings /0 skips**. Actual duplicate POST returned500, real backend uniqueness classification failed, and two uniqueness-with/without-rollback-failure unit cases failed. SQLite duplicate, actual CHECK-constraint500 with subsequent recovery, and unrelated/foreign-runtime privacy controls passed. See `uat174-red.redacted.log` and command receipt.
- Initial GREEN: **29 PASS /4 files /4 warnings /0 skips**.
- Final frozen test bytes: **29 PASS /4 files /4 warnings /0 skips**, `uat174-final.redacted.log`, exit0.

The new HTTP test mounts the real Flashcards router in a TestClient with only its database dependency replaced by the official temporary test DB. It performs actual successful/duplicate POSTs, checks409 and unchanged original description/count, and creates a later distinct deck successfully. It is HTTP serialization/database behavior coverage, not authentication or browser coverage. The independent SQLite control uses a real file DB. The non-unique PG control adds a CHECK in its isolated fixture DB and verifies500, rollback and a later successful POST.

The backend privacy test uses an actual PostgreSQL UNIQUE violation, confirms the original row survives, the connection remains usable, the generic error string is exact, both cause/context are None, and the marker has no attributes. Captured logs/error text contain no private value, query, or constraint identity. Driver-error controls also cover a failed rollback and reject a foreign RuntimeError merely advertising sqlstate23505. Existing redaction and transaction-manager suites run unchanged.

## Static verification
- Scoped Ruff baseline and final: **0 diagnostics**. New test formatter check: PASS.
- Bandit production baseline/final: **0 findings /0 errors**.
- Bandit final new tests: **0 findings /0 errors**, B101 excluded only for pytest assertions. Initial test scan mistook a privacy sentinel variable named `secret` for a hardcoded password; it was renamed `private_value`, and final29 tests plus checks reran. The original finding remains in `bandit-tests.json`; final result is `bandit-tests-final.json`.
- Owned diff whitespace checks: no diagnostics. Untracked test diff returns1 because the file is added. The log is empty.
- One private manifest attempt hit Node's default1MB git-show buffer for the large ChaCha file; the successful manifest uses10MB and verifies all baseline hashes. No source mutation occurred in that failed artifact operation.

## Reproduce
From repo root, using the existing authorized local PG test cluster (network escalation required in the agent sandbox):

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat174-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs -q tldw_Server_API/tests/DB_Management/test_postgres_unique_conflict.py tldw_Server_API/tests/DB_Management/unit/test_postgresql_error_redaction.py tldw_Server_API/tests/DB_Management/unit/test_postgres_transaction_manager.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_transactions.py --tb=short
python -m ruff check tldw_Server_API/app/core/DB_Management/backends/base.py tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/DB_Management/test_postgres_unique_conflict.py
python -m bandit tldw_Server_API/app/core/DB_Management/backends/base.py tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json
python -m bandit tldw_Server_API/tests/DB_Management/test_postgres_unique_conflict.py -s B101 -f json
```

No AuthNZ test_db_pool or custom database provisioning is used. The runner makes PostgreSQL mandatory, disables Docker autostart and redacts its connection credentials. Private raw logs/configuration are not copied into this packet.

## Remaining acceptance
Independent review and parent-owned native duplicate-name/actionable error plus retained-draft save acceptance remain pending. No browser, runtime, service restart, tracker/task, staging or commit changes were performed by this agent. No broader recovery of constraint-specific callers or execute_many classification is claimed.
