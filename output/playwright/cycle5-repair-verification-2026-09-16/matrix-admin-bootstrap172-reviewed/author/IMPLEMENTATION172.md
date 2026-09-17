# TASK13260.172 — explicit normal admin bootstrap

## Outcome and scope

Added the private harness `bootstrap-admin` action and `bootstrap-admin-cell.py`. Normal preparation still writes planned credentials only, and `initialize` still calls the existing noninteractive AuthNZ initializer. The new explicit action invokes the existing supported `AuthNZ.create_admin.create_admin_user_non_interactive` function for multi-user cells after completed initialization. No product source, AuthNZ implementation, raw database writes, actual profile/account or native runtime was changed.

The five owned files are the launcher, its existing fake-process test, new helper and its new fake-import test, and `LAUNCHER.md`. Role125 runtime/holder/initializer files and browser166 files were not edited. The existing Node fixture now delegates its mocked inspection/environment callbacks dynamically so the added source-drift and exact-environment controls actually change the injected boundary; no production boundary is replaced differently.

## Guard contract

- Uses unchanged completed-preparation/initialization and source fingerprint guards; PostgreSQL still requires a fresh direct-role check. Uses the same exact runtime environment, frozen private interpreter and cell working directory.
- Refuses both single-user cells and already completed admin bootstrap. Credentials are loaded from the profile's mode0600 planned record; missing/invalid planned admin fields are rejected.
- Checks the actual imported module file resolves inside the frozen source root. Only the supported function's boolean `True` normal return permits an exclusive mode0600 attempt proof.
- Parent requires exit0, no signal/forwarded interruption, and matching action/token/preparation proof before writing `admin-bootstrapped.private.json`. Missing, stale, mismatched or incomplete proof cannot certify success. Failed attempts remain inspectable/retryable and get a new token.
- No planned credential values are placed in child argv or process/completion receipts. Existing runtime environment secrets and child logs remain private under the existing launch boundary; nothing reads real credentials during these tests.
- The completion receipt certifies the supported bootstrap call, not login or password reset. Existing matching admins keep their password. Alice/Bob are still created through `/admin/server` → Create user with User role, followed by normal login.

## RED and GREEN

The final new launcher controls replayed against unchanged baseline source produced **13 failures / 41 controls / 0 skips**: the old launcher has no bootstrap branch/proof handling (the five inherited preflight negatives already passed, as did the original36 controls). `node-red.log` retains exact failures. New Python helper tests first failed16 because the helper did not yet exist; this is missing-feature evidence, not a runtime product finding.

Final:

- **54 Node checks passed / 0 skipped** (36 existing +18 added), 162.62ms.
- **42 Python checks passed / 0 skipped**, 0.13s: new16 helper controls +existing5 initializer +existing21 synthetic PostgreSQL-role guards.
- Ruff0 on both owned Python files. Python compile2, Node syntax2 pass.
- Bandit helper0 findings/errors; test0 findings/errors with B101 only excluded. Initial test-only B105 reports on synthetic constant credentials/tokens were retained and removed by generating disposable random test values; no security suppression was added.
- Bandit does not analyze the JavaScript launcher; its scope is explicitly Python. JavaScript syntax and actual fake-I/O control flow were checked with Node.

All tests use disposable temporary files, fake subprocess/network boundaries, and fake supported-function imports. No real initializer/bootstrap module, database connection, fixture holder, profile preparation, archive/dependency copy, server, browser, or inference was executed. Full-matrix gate remains closed.

## Independent commands

From repository root:

```sh
node --experimental-vm-modules --test .tmp/uat-next-matrix-20260916/matrix-launcher.test.mjs
source .venv/bin/activate
python -m pytest .tmp/uat-next-matrix-20260916/test_initialize_cell.py .tmp/uat-next-matrix-20260916/test_bootstrap_admin_cell.py .tmp/uat-next-matrix-20260916/test_pg_role_adapter.py -q --tb=short
python -m ruff check .tmp/uat-next-matrix-20260916/bootstrap-admin-cell.py .tmp/uat-next-matrix-20260916/test_bootstrap_admin_cell.py
python -m bandit -q .tmp/uat-next-matrix-20260916/bootstrap-admin-cell.py
python -m bandit -q -s B101 .tmp/uat-next-matrix-20260916/test_bootstrap_admin_cell.py
node --check .tmp/uat-next-matrix-20260916/matrix-launcher.mjs
node --check .tmp/uat-next-matrix-20260916/matrix-launcher.test.mjs
```

Do not execute the launcher CLI or `test_hold_official_pg.py` for this review. A nonmutating Node RED replay is available using `MATRIX_TEST_SOURCE=.tmp/uat-next-matrix-20260916/admin172/baseline/matrix-launcher.mjs` with the same Node command.

## Freeze

`owned-manifest.json` binds the five final files and exact snapshots; `owned.patch` contains only the new action/helper/tests/docs. `verification-manifest.json` binds the report and retained receipts. Design stages1/2 and author verification are complete; independent review is the remaining gate. Parent owns task/tracker, integration and any eventual real execution.
