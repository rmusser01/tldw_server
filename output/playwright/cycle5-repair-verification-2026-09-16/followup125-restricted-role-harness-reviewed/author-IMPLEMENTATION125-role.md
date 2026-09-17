# TASK13260.125 — restricted matrix PostgreSQL runtime

## Scope and behavior

Private harness changes only; no product/task/tracker/git/browser/native-profile changes. The full matrix remains gated. The official `pg_temp_db` and `pg_temp_db_session` fixtures still create/drop every database. A holder-local session fixture adapts official `pg_server`: the provisioning administrator creates one random direct LOGIN role with NOSUPERUSER, NOBYPASSRLS, NOINHERIT, NOCREATEROLE, NOREPLICATION and temporary CREATEDB. After both fixture DBs exist, revoke CREATEDB before writing any held receipt. Both direct connections must identify that role, own their exact fixture databases, have no memberships, and report row_security=on.

The holder emits separate mode0600 `runtime.pg-config.private.json`. Only pg-holder takes administrator config; prepare/app launch takes the emitted runtime config. Receipt checks bind runtime/provisioning identities to the existing source/run/cell/venv ownership. Every initialize/backend/frontend action rechecks live flags with the explicit copied interpreter before spawning or opening a port. No SET ROLE fallback or privilege widening exists.

Official fixtures clean up their owned databases before the adapter drops only its exact generated role. No CREATE/DROP DATABASE implementation, DROP OWNED, broad role deletion, admin membership or automatic failure cleanup was added. Failures preserve private creation records and surface cleanup errors.

## Files

Changed existing private files: `matrix-launcher.mjs`, `pg-holder.mjs`, `test_hold_official_pg.py`, `matrix-launcher.test.mjs`, `LAUNCHER.md`, `ISOLATION.md`, plus manifests/validation records.

New private files: `pg_role_adapter.py`, `test_pg_role_adapter.py`, `test_pg_role_cold_runtime.py`, `role125/cold_runtime_control.py`, design/report/receipts. `initialize-cell.py` and its existing five guards are unchanged. `PROTOCOL.md` is unchanged: four serial fresh-state cells, existing named journeys, no invented A/B/C mapping. The copied-venv, three Python roots, three cell-local Bun trees, exact editable-file handling and archive/source binding contracts remain unchanged.

## Evidence

- `node-red.log`: original launcher plus new role controls **10 failed / 24 passed**. Existing23 passed; the old launcher accepted privileged/legacy receipts and omitted the live-role gate.
- `python-new-surface-red.log`: new adapter tests were authored before its module existed. This is a missing-surface result, not the causal behavioral RED.
- `python-target-red.log`: **5 failed / 16 passed**, proving qualification could touch a role before rejecting foreign fixture targets. Final adapter validates all four connection identity fields and fixture DB name first.
- Final Node guards: **36 passed / 0 skipped**. Existing23 assertions preserved; the existing positive PG fixture data was updated to the new identity contract. New coverage includes all privilege flags, legacy/admin receipt rejection, exact credentials/ownership, and blocked initialization/backend/frontend before child/port actions.
- Final synthetic Python guards: **26 passed** = existing5 initializer cases +21 role catalog/lifecycle/target controls. No real process/network in these guards.
- Actual required-PG controls: **2 passed / 0 skipped** on fresh official databases, single_user and multi_user. The existing normal AuthNZ initializer returns without test mode. Single API-key authenticated profile GET200; multi actual login POST200 and bearer profile GET200. The real content accessor initializes its schema, asserts actual PostgreSQL current_database/current_user match the content receipt, and persists/reads a note. After standard shutdown, official fixture teardown removes both DBs and the generated role; post-teardown catalog assertions verify absence.

Cold control limits: a fresh subprocess uses current checkout source and existing dependencies with three explicit Python roots; it is not the final archived/copied dependency cell. It mounts real auth/users routers without dependency overrides and calls the real content accessor; it does not start main.py, a listening server, browser or provider. Multi-user fixture account is created through the real UsersDB/password/RBAC APIs before actual login. No RLS tenant-separation acceptance is claimed.

## Harness corrections retained

`required-pg-first.redacted.log` records two failures from duplicated private-router prefixes after both normal initializers succeeded; no application repair was needed. `first-cold-*.redacted.log` preserves those child logs. Missing MCP secrets were generated only in those disposable env files; retained text now redacts those values. Subsequent controls supply all secure keys explicitly.

`required-pg-teardown-harness-failure.redacted.log` records one pass/one timeout: both authenticated paths and content roundtrips had succeeded, but the ASGI-only control omitted standard application resource shutdown. The control now invokes existing ChaCha and Audit shutdown functions, then closes AuthNZ; no product lifecycle code changed. `required-pg-final-green.redacted.log` retains the resulting2PASS. The final exact-source repeat with explicit content SQL identity assertions passed2/0skip in9.74s (`required-pg-identity-green.redacted.log`). First backend/frontend fake receipt probes omitted the existing exit0 field; correcting that fixture reached the intended role guard. No failure receipt is presented as a product defect.

## Commands and static checks

From repository root:

```sh
node --experimental-vm-modules --test .tmp/uat-next-matrix-20260916/matrix-launcher.test.mjs
source .venv/bin/activate
python -m pytest .tmp/uat-next-matrix-20260916/test_pg_role_adapter.py .tmp/uat-next-matrix-20260916/test_initialize_cell.py -q --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TLDW_UAT_EVIDENCE_LABEL=uat125-restricted-cold-identity node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs -p tldw_Server_API.tests._plugins.postgres --confcutdir .tmp/uat-next-matrix-20260916 -c .tmp/uat-next-matrix-20260916/pytest-holder.ini .tmp/uat-next-matrix-20260916/test_pg_role_cold_runtime.py -q --tb=short
```

The actual command needs approved local fixture network access. Runner output remains private until redacted against generated credentials. No Docker start/restart. Node syntax, Python AST/format/Ruff, and Bandit receipts are in this packet. Bandit excludes B101 test assertions; five narrow documented nosec annotations cover synthetic credentials and fixed no-shell subprocess use, after inspection of the original findings. Final results have zero findings/errors. Bandit does not validate JavaScript semantics.

## Remaining gate

Ready for independent review on the frozen manifest. No final prepare/initialize/backend/frontend CLI action has executed. Database ownership deliberately permits normal schema DDL; FORCE RLS and genuine two-account acceptance still require final native verification. No clean-machine installation claim: dependencies, Python, browser and model assets remain reused as disclosed. Any later migration failure must be diagnosed; never silently elevate the runtime role.
