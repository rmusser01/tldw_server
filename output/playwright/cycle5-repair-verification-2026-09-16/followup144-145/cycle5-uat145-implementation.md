# UAT145 / TASK13260.84 — canonical PostgreSQL bootstrap fixtures

## Repair

Exactly three existing fixtures in `tests/AuthNZ/integration/test_single_user_bootstrap_postgres.py` used direct users INSERTs rejected by `ProfileUserWriteRejected`, before their bootstrap assertions. Replaced them with the existing `tests/AuthNZ_SQLite/_user_fixtures.create_authnz_test_user`, which supports both backends through `VersionedUserWriteGateway`.

Fixed IDs, names, emails, empty password hash, active/verified states, roles, conflict-ignore semantics, primary-key preseeding, and all original assertions remain. No product code, helper, permission, or guard changed in this unit. The tests continue to check preseeded-key reuse/upgrading, extra-active-user rejection, multiple-primary-key rejection, and ordinary idempotent bootstrap.

## Evidence

- Initial current-code failure: `/private/tmp/cycle5-postgres-144-green.redacted.log`,39pass/3fail/0skip across42controls.
- Independent baseline-source comparison: `/private/tmp/cycle5-postgres-144-baseline-existing-valid.redacted.log`,3fail/1pass/0skip; same three guard failures.
- Private baseline loader `/private/tmp/cycle5_uat144_baseline_loader.py` substitutes only saved HEAD database source `/private/tmp/cycle5_uat144_baseline_database.py`, preserving original `__file__` for schema resources. No repository rollback. The first preliminary loader did not preserve `__file__` and failed schema-resource lookup; it is not counted as baseline proof.
- Baseline command: `PYTHONPATH=/private/tmp:$PWD node /private/tmp/cycle5-postgres-fixture-run.mjs -p cycle5_uat144_baseline_loader tldw_Server_API/tests/AuthNZ/integration/test_single_user_bootstrap_postgres.py -q -rs --show-capture=no`, redirected to the baseline-valid log above, then processed with `/private/tmp/cycle5-postgres-report.mjs <log> --save`.
- Final combined verification: **42 passed / 0 failed / 0 skipped**,82 warnings,125.78s; `/private/tmp/cycle5-postgres-144-145-final-green.redacted.log`.
- Exact five-suite final command is retained in `/private/tmp/cycle5-uat144-implementation.md`. Official disposable PostgreSQL18.6 fixtures only; no required-PG skips or held runtime database mutations.

Ruff on the touched existing test:5 findings (4 existing I001,1 existing F401), versus6 baseline (5 I001,1 F401);0 added. Raw structured evidence `/private/tmp/cycle5-uat145-ruff-{baseline,final}.json`. `git diff --check` clean. This unit changes tests only; production security validation is the separate UAT144 Bandit0, no new production security scope.

Freeze `/private/tmp/cycle5-uat145-code-freeze.json` contains the one test path; final owned manifest includes task84. No staging/commit. Independent review is clear:4/4 required-PG controls and all31 original assertion lines retained byte-identically/in order; `/private/tmp/cycle5-uat145-independent-review.md`. No native product-pass claim is attached to this fixture correction.
