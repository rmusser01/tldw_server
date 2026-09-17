# Independent review — TASK13260.125 restricted PostgreSQL role harness

## Verdict

**Clear for the bounded harness qualification.** Independent checks: **36 Node guards passed**, **26 Python guards passed**, and **2 real disposable PostgreSQL cold controls passed / 0 skipped / 10.19s**. No new causal product/harness finding or failing test run occurred in this review; one later metadata-scan attempt is qualified below. All14 frozen source/snapshot/copied-file identities match before and after execution.

Author source manifest: `.tmp/uat-next-matrix-20260916/role125/source-manifest.json`, SHA256 `76b49014e0aee3eea3cc6b3c406e7099e4e02a626956384205454be8dffb450a`. This review uses exact copied bytes at the same directory depth in `.tmp/uat125-role-independent-20260917`, preserving the harness's repository-root calculation. The separate copy prevents its fixed cold-result/log paths from overwriting the author's evidence. `source-before.json` and `source-after.json` bind all paths and hashes. No author harness or product source was changed.

## Authority and privilege boundaries

- The adapter creates one generated direct LOGIN role; its SQL identifier and password are safely composed with psycopg. Initial flags explicitly include NOSUPERUSER, NOBYPASSRLS, NOINHERIT, NOCREATEROLE and NOREPLICATION. Temporary CREATEDB is used only so the **official** `pg_temp_db` / `pg_temp_db_session` fixtures can provision their databases. The adapter implements no CREATE/DROP DATABASE path.
- Qualification checks both fixture connection identities and official temporary database names, revokes CREATEDB, then uses fresh direct connections to verify current_user/session_user, database ownership, login, zero memberships, all forbidden privilege flags false and row_security=on. A held receipt and runtime credential file are written only after this qualification. Administrator credentials remain confined to the provisioning adapter; runtime config/receipt use the generated role. The receipt's provisioning fields contain identity metadata rather than the administrator password.
- Each initialize/backend/frontend launch first validates the private receipt/source/run/cell identity and then invokes the live role verifier with the explicitly selected copied interpreter. Failure stops before the child process or port check. Frontend receives its separate frontend environment; its required SQL qualification runs as a preflight. There is no SET ROLE fallback or automatic privilege widening.
- Official fixture dependency teardown removes the fixture databases before the adapter drops its exact generated role. The adapter uses neither DROP OWNED nor a broad role sweep. Cleanup errors remain visible. The actual cold tests additionally query the catalogs after teardown and assert that the exact generated role and both owned databases are absent. These assertions passed independently.

The official fixture implementation remains the sole owner of database creation/deletion and its existing connection termination behavior. The generated runtime owns its disposable databases to permit normal schema DDL. This is intentionally not a separate read-only runtime/migration-role deployment design, nor proof of tenant separation on every application resource.

## Real cold proof and limits

The isolated subprocess calls the normal AuthNZ initializer with test mode false and no PYTEST_/TLDW_TEST_ flags. It mounts the real auth/users routers with no dependency overrides:

| Mode | Independent authenticated result | Content proof |
| --- | --- | --- |
| single_user | API-key profile GET200 | Actual PostgreSQL database/role identity matches the fixture DSN; note persist/read succeeds |
| multi_user | Real login POST200 and bearer profile GET200 | Same real PostgreSQL identity and note round-trip assertions |

The multi-user fixture account is created through actual UsersDB/password/RBAC APIs. Content access deliberately calls the real accessor for owner1; it is not a test of mapping the logged-in fixture account to a tenant or of cross-account RLS isolation. The control performs existing ChaCha/Audit/AuthNZ shutdown before official fixture teardown.

The proof runs current checkout application source with existing dependencies and three explicit Python roots. It does not run the final archived source/dependency cell, main.py, a listening API/frontend, a browser or a provider. No final matrix prepare/initialize/backend/frontend CLI action was executed. The existing matrix protocol and final native gates remain unchanged.

## Privacy and static verification

Eight generated role/environment/log records across both disposable modes were verified mode0600. The packet's redacted logs were checked in memory against18 generated values from the private environment and role records; **zero matches**. Counts and file identities, never raw values, are retained in `secret-scan.json`. The private inputs themselves were not copied into this report packet. A later attempt to expand the scan to initialized env-file values and URL encodings could not complete because the pytest temporary directory was already gone; `secret-scan-followup-unavailable.json` preserves that limitation. The earlier successful18-value scan and eight mode checks remain retained. Source review also confirms the control redacts initialized env-file values before publishing its logs. Known-value scanning is a concrete check, not a claim that arbitrary secrets can never be logged.

Independent Ruff: **0 findings**. Bandit across the seven Python harness/control/test paths: **0 findings / 0 errors**, excluding B101 assertions only and preserving the inspected narrow existing annotations for synthetic credentials and fixed no-shell subprocesses. All copied Python files parse/compile; all three JavaScript files pass Node syntax checks. JavaScript behavior is covered by the fake-boundary guards; Bandit does not analyze it.

The author retained the original prefix error, omitted-shutdown timeout, target-qualification RED and synthetic receipt corrections. Those remain harness history, not application defects. The cold test and guards passed; the separate post-run metadata-read failure is recorded above and is not an application or role-cleanup failure.

## Reproduction and retained results

```sh
node --experimental-vm-modules --test .tmp/uat125-role-independent-20260917/matrix-launcher.test.mjs
source .venv/bin/activate
python -m pytest .tmp/uat125-role-independent-20260917/test_pg_role_adapter.py .tmp/uat125-role-independent-20260917/test_initialize_cell.py -q --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TLDW_UAT_EVIDENCE_LABEL=uat125-role-independent-cold node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs -p tldw_Server_API.tests._plugins.postgres --confcutdir .tmp/uat125-role-independent-20260917 -c .tmp/uat125-role-independent-20260917/pytest-holder.ini .tmp/uat125-role-independent-20260917/test_pg_role_cold_runtime.py -q --tb=short
```

The last command needs the approved local fixture access and was run with its output retained privately. Exact command metadata, Node/Python logs, redacted PostgreSQL receipt, safe cold results, credential modes, source hashes and static reports are in this packet. The source copy is a review artifact; it is not a final matrix profile.

Only private review artifacts and disposable official fixture state were created. No product/task/tracker/git/browser/final-profile/runtime configuration was changed. Final full-matrix acceptance remains the parent's responsibility.
