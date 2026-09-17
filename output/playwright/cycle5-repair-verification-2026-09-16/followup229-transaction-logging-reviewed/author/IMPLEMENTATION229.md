# UAT229 / TASK-13260.170 — transaction failure logging

## Result and frozen scope

A brace-bearing exception could cause Loguru formatting to throw before SQLite rollback, replace the original failure, and leave the pending write and transaction active. Four diagnostic calls in `TransactionContextManager.__exit__` now pass exception values as formatting arguments. Logger levels and extra fields, rollback/commit decisions, original exception propagation and commit-error causes are unchanged.

Only two files are owned: `ChaChaNotes_DB.py` and new `tests/DB_Management/test_transaction_error_logging_backends.py`. Baseline is the exact post168 source committed at `447edebb7b53d7489c2fbac5903db473c1f556d5`; the local keyword survivor implementation remains intact. `ast-scope.json` verifies that replacing the four logging argument lists makes the complete before/after module AST identical. No generic logging or transaction framework changes.

## Causal evidence

The initial actual SQLite probe is retained in `.tmp/uat228-repair-20260917/test_transaction_format_probe.py` and `uat228-transaction-format-probe.redacted.log`: one failure / one control. A plain `ValueError` propagated and rolled back; a brace-bearing `ValueError` became `KeyError`, with the actual transaction and pending row still present. The probe explicitly rolled back after observing the failure. This is the production Loguru formatting path, not pytest sink behavior.

Permanent first RED: `uat229-transaction-red.redacted.log`, **5 failed / 6 passed / 0 skipped, 11.43s**. Expanded final RED: `uat229-full-causal-red.redacted.log`, **6 failed / 9 passed / 0 skipped, 14.92s**. All failures reach the intended logging branches; actual PostgreSQL controls pass on the baseline. Logs and exact command JSON files are in `.tmp/fresh-uat-recovery-20260916/`.

## Verification

- Focused final suite: **15 passed / 0 skipped, 13.56s** (`uat229-transaction-green.redacted.log`).
- Focused plus adjacent transaction/operation controls: **47 passed / 0 skipped, 27.66s**, 20 warnings (`uat229-adjacent-green.redacted.log`).
- Ruff: **0 baseline / 0 current / 0 new-test findings** using the actual logical production filename for both baseline and current.
- Bandit: **0 production baseline/current findings or errors**; **0 test findings or errors**, with test assertions excluded through `B101` only.
- Both owned Python files compile; exact four-call scope verified independently of text formatting.

The 15 permanent cases include actual SQLite/PostgreSQL plain/brace and nested rollback, original exception identity, cleared transaction and absent pending writes, and successful commit. Real SQLite writes plus a connection proxy inject commit/rollback driver failures to test all four diagnostics, original commit-error cause, attempted rollback, and the intentionally active transaction after a driver rollback failure. The fixture explicitly cleans these intentionally faulted transactions. No driver failure injection is claimed as a PostgreSQL failure simulation.

### Exact independent command

Run from repository root:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat229-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/DB_Management/test_transaction_error_logging_backends.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_transactions.py tldw_Server_API/tests/DB_Management/test_chacha_operation_scope.py tldw_Server_API/tests/DB_Management/unit/test_postgres_transaction_manager.py -q --tb=short
```

The runner requires PostgreSQL, uses the official DB_Management fixtures on the existing owned cluster, and does not inject a Jobs database URL into legacy SQLite tests. No database was manually provisioned. The normal adjacent controls cover PostgreSQL commit/rollback and pool returns, operation ownership/finalization and the existing PostgreSQL transaction manager.

## Frozen identities and limits

`owned-manifest.json` binds current files and snapshots, their baseline and causal/green receipts. Production SHA256: `33f987c9e4c6acc3b1502c9f6e10dc66c6e17929f2258c4d67fa40e87b7fe06b`; test SHA256: `cd72155a039815bde0bb76be92c6c2a6f9ddb50ab8029af48aeea02a96af03f1`.

No native runtime, browser, config, Backlog or git action was taken. The unrelated UAT228 historical fixture correction is separately frozen/reviewed. This repair does not claim whole-application logger safety or alter non-SQLite transaction semantics. Independent review remains the integration gate.
