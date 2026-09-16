# UAT147 — ChaCha PostgreSQL keyword sequence ownership

Task: TASK-13260.88. Production change: one tuple entry in `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:924`, `keywords` → `chacha_keywords`.

## Cause and scope

Shared content PostgreSQL is documented (`Docs/Deployment/Postgres_Migration_Guide.md:145–164`, `Docs/Code_Documentation/Database-Backends.md:12–15`). ChaCha resolves that backend and transforms its keyword table to `chacha_keywords`; its sequence maintenance still targeted Media's `keywords`. This both rewound Media's sequence and failed to advance ChaCha's sequence. With concurrent Media schema initialization, the extra Media keyword lock completes the native lock cycle against shared `sync_log` RLS setup.

A private isolated driver observer captured SQLSTATE40P01 at `ALTER TABLE "sync_log" ENABLE ROW LEVEL SECURITY`. Sequential Media→ChaCha→Media succeeded. The first diagnostic barrier was accidentally reused on subsequent sequence passes; its later timeout is a harness artifact, not product evidence. The permanent regression uses one-shot barriers and catches actual database errors at both synchronized boundaries, including errors swallowed by a constructor.

No backend, Media, schema, policy, credential or RLS changes. No held native databases or runtime processes touched.

## Permanent RED and GREEN

New file: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_ownership.py`.

- Own explicit keyword ID501 must advance ChaCha's next ID to502.
- Existing Media keyword sequence at701 must remain next702 after ChaCha initialization.
- Actual concurrent ChaCha initialization and Media reopen, synchronized only at real SQL boundaries, must complete with no boundary DB errors. No responses/errors are mocked.

RED: 3 failed,0 skipped; values1≠502,1≠702 and actual initialization database error. `/private/tmp/cycle5-postgres-147-final-red.redacted.log`.

Required PostgreSQL18.6 GREEN: 19 passed,0 skipped,16.55s, exit0. `/private/tmp/cycle5-postgres-147-green.redacted.log`. Command:

```sh
node /private/tmp/cycle5-postgres-fixture-run.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_ownership.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_transactions.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_session_scope.py tldw_Server_API/tests/DB_Management/test_chacha_shared_sqlite_cleanup.py tldw_Server_API/tests/Services/test_startup_chacha_warmup.py tldw_Server_API/tests/Characters/test_chacha_postgres_sync_log_entity_column.py -q -rs > /private/tmp/cycle5-postgres-147-green.log 2>&1
node /private/tmp/cycle5-postgres-report.mjs /private/tmp/cycle5-postgres-147-green.log --save
```

Runner supplies official disposable fixtures, REQUIRED=1/NO_DOCKER=1; credentials stay private. Broader run had4 warnings and unrelated pytest garbage-directory cleanup warnings, exit0. Final test-only import ordering changed after the broader run; same3 regression tests rerun separately in `cycle5-postgres-147-final-focused.redacted.log`.

## Static checks and limits

Project venv Ruff: production baseline0/final0; final production+newtest0. Bandit production baseline0/final0. JSON logs: `cycle5-uat147-{ruff-baseline,ruff-final,ruff-owned-final,bandit-baseline,bandit-final}.json`.

The concurrent regression proves this reproduced lock cycle is removed; it does not claim all possible shared-schema bootstrap races are solved. Native normal API warm-up and independent review are still parent-owned acceptance. No native startup or frontend pass is claimed by this report.

Final formatted regression run: **3 passed,0 skipped,6.87s,exit0**. Production/test freeze:2026-09-16T16:38:42.564Z.
