# UAT146 / TASK13260.85 — current media PostgreSQL policy validation

## Result

**56 passed / 0 failed / 0 skipped**, exit0; official PostgreSQL18.6 fixtures,5.14s. Production/test freeze `/private/tmp/cycle5-uat146-code-freeze.json`. Independent review and parent-owned actual API restart remain pending; no native workflow signoff claimed.

## Diagnosis and minimal repair

Actual startup delegates through `startup_content_backend_validation` → `DB_Manager.validate_postgres_content_backend` → runtime factory. Factory construction performs the real canonical MediaDatabase schema bootstrap. Both fresh and existing-schema bootstrap call `ensure_postgres_post_core_structures` → `ensure_postgres_policies` → `schema/features/postgres_rls.py`.

That canonical helper deliberately removes the four legacy `media_scope_*` policies and creates `media_visibility_access`. It also creates `owned_clone_pending_keyword_access`; the four `sync_scope_*` policies remain. The validator required the removed media policies, aborting successful schema bootstrap.

Only production change: `media_db/runtime/factory.py` now requires the current media policy, all four existing sync policies, and the owned-clone pending-keyword policy. No policy SQL/predicates, schema, RLS configuration, backend selection, auth, or permissions changed. Failure still raises the existing actionable RuntimeError and closes the validator.

## Tests and evidence

New `tests/DB_Management/test_media_postgres_runtime_validation.py`:

- A normal Python subprocess with curated private config/cwd, explicit PostgreSQL content DSN from a new official `pg_temp_db`, and no pytest/test-mode signals invokes the actual startup service and DB_Manager wiring. It verifies current policy names, absence of legacy media policies, and enabled **and forced** RLS on media, sync_log, and operationownedclonekeywords.
- Six missing-policy cases bootstrap the actual schema, remove one policy in the disposable fixture immediately before validation, and require rejection of that exact policy. This boundary placement prevents schema repair from masking the failure.
- Six unreadable-policy cases use the real PostgreSQL schema and helper but inject a bounded BackendDatabaseError at the selected catalog-read boundary. These prove fail-closed handling; they do not claim a native database-permission-denial experiment.

Valid RED:13 failures in `/private/tmp/cycle5-postgres-146-red-confirmed.redacted.log`; normal startup reports `obsolete_media_policy=True`, and every negative case incorrectly encounters legacy media_scope_admin first. Two preliminary runs had test cleanup API errors (backend.close / backend.close_all); these were corrected to the actual backend.get_pool().close_all contract before the valid RED. They are not product proof.

Final GREEN: `/private/tmp/cycle5-postgres-146-green.redacted.log`,56passed/0skipped. Four test/dependency warnings plus pytest's existing old temporary-directory cleanup warning are retained honestly.

```sh
node /private/tmp/cycle5-postgres-fixture-run.mjs \
  tldw_Server_API/tests/DB_Management/test_media_postgres_runtime_validation.py \
  tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py \
  tldw_Server_API/tests/DB_Management/test_media_db_postgres_rls_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_support.py \
  tldw_Server_API/tests/Services/test_startup_content_backend_validation.py \
  -q -rs --show-capture=no > /private/tmp/cycle5-postgres-146-green.log 2>&1
node /private/tmp/cycle5-postgres-report.mjs /private/tmp/cycle5-postgres-146-green.log --save
```

Helper activates project `.venv`, supplies private official fixture settings, sets required PostgreSQL and no-Docker-autostart. No credentials are printed. The old `test_postgres_returning_and_workflows.py` startup smoke uses a global backend-env skip and does not create an isolated fixture; it was inspected, not used as a claimed pass. The new always-executed explicit-fixture startup control closes that gap.

## Static and scope

Project-venv Ruff: production2 unchanged baseline findings (I001,TRY203); new test0;0 added. Production Bandit0 (baseline0). Scoped diff-check clean. Structured outputs `/private/tmp/cycle5-uat146-ruff-{initial,final}.json`, `/private/tmp/cycle5-uat146-bandit-{baseline,final}.json`.

Owned files: production factory, new focused test, official task85. Original held PostgreSQL databases/native profiles and processes were not touched; test policy removal affected only disposable official fixtures. No staging or commits. Final owned manifest and scanned evidence manifest accompany this report.
