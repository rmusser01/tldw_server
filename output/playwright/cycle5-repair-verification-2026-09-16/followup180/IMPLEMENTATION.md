# UAT180 / TASK-13260.117 — PostgreSQL flashcard lifecycle

## Frozen scope

Production changes exactly two statements in `ChaChaNotes_DB.py`: delete reads `id`/`version`/`deleted` by column name; reset reads `id`/`version` by name. PostgreSQL returns mappings, so the old integer offsets raised KeyError before a mutation. SQLite supports named access too. SELECTs, update fields, version checks, idempotent deletion, transaction boundaries and endpoint error mapping are unchanged. The adjacent normal flashcard update already uses this pattern.

One new permanent file, `test_flashcard_lifecycle_backends.py`, tests actual SQLite and the official isolated PostgreSQL fixture. The real Flashcards router is mounted with only the DB dependency overridden. This tests HTTP serialization and mutation behavior, not authentication or a browser. Twenty cases cover successful/idempotent deletion, reset defaults/content preservation, required HTTP version versus optional DB-level version, missing/stale conflicts, and caller rollback after both mutations.

The baseline source copy exactly matches parent commit `f817f4aa968d9d5ec5c01c5b4db53c924867c4fe`, after UAT177 analytics was integrated. The source patch is only these two statements. `owned-manifest.json`, `owned.patch`, and `review-snapshot/` define the frozen release.

## RED and GREEN evidence

- First RED: **8 failed / 10 passed**. Six failures were the actual PostgreSQL bug; two exposed an incorrect test assumption that reset's optional DB argument also meant optional HTTP input. Existing schema requires the HTTP version. The test was corrected to exercise omitted version at the DB boundary and explicitly preserve HTTP422/no mutation. This is recorded in `uat180-red.redacted.log`.
- Corrected permanent RED, before production: **7 failed / 13 passed / 4 warnings / zero skips**, `uat180-final-red.redacted.log`. All seven failing PostgreSQL cases reach the two positional-row statements: actual HTTP success/conflict paths produce500, direct DB reset/rollback controls show KeyError0. All SQLite and missing-target/required-version controls pass.
- GREEN: **33 passed / 3 files / 4 warnings / zero skips**, `uat180-green.redacted.log`:20 new cases plus13 existing PostgreSQL transaction controls.
- Existing SQLite endpoint controls: **8 passed / 173 deselected / 4 warnings**, `existing-endpoint-controls.log`. These cover successful delete/reset, missing/version conflicts and deliberate DB error mappings.

No tests or implementation changed after GREEN. Required PostgreSQL execution uses the existing private runner and official per-test databases. Only redacted logs and nonsecret command receipts were copied; no private configuration or raw logs were copied.

## Verification

From repository root:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat180-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs -q tldw_Server_API/tests/DB_Management/test_flashcard_lifecycle_backends.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_transactions.py tldw_Server_API/tests/DB_Management/unit/test_postgres_transaction_manager.py --tb=short
python -m pytest -q tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k 'delete_flashcard_returns or reset_scheduling' --tb=short
python -m ruff check tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/DB_Management/test_flashcard_lifecycle_backends.py
python -m ruff format --check tldw_Server_API/tests/DB_Management/test_flashcard_lifecycle_backends.py
python -m py_compile tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/DB_Management/test_flashcard_lifecycle_backends.py
python -m bandit tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json
python -m bandit tldw_Server_API/tests/DB_Management/test_flashcard_lifecycle_backends.py -s B101 -f json
```

- Ruff baseline/final:0 diagnostics. New test format check and Python compilation pass.
- Bandit full production baseline/final:0 findings,0 errors. New test:0 findings,0 errors; only B101 excluded for ordinary pytest assertions.
- Owned whitespace checks: no diagnostics. The untracked new-file diff command returns1 because it is an added file, not because of whitespace errors.
- No AuthNZ test_db_pool, manual database provisioning or runtime operations.

## Context and remaining acceptance

UAT179 concurrently changes only review-session/study-thread/study-message timestamp validators in an unowned schema file. The Flashcard timestamp validator used by reset responses already exists in the baseline commit. An earlier coordination message incorrectly attributed it to179; inspection of the actual schema diff corrected that claim. UAT180 does not require a new179 validator for its reset response.

Parent owns independent review, native PostgreSQL Manage delete/reset acceptance, task/tracker/shared design and commits. No browser, runtime, task/tracker, shared-plan, staging or commit changes were made by this agent.
