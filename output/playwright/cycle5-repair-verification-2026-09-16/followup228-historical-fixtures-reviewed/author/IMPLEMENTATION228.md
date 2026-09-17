# UAT228 — TASK13260.169 — historical keyword fixture correction

## Result and scope
Only the two existing tests/ChaChaNotesDB files changed: test_notes_organization_migration_v55.py and test_notes_organization_migration_v57.py. Production migrations and their guards are untouched. Final complete affected suites42PASS/0skip3.17s through the official required-PG harness; actual cases in these historical suites are SQLite or backend-adapter unit controls, so this is not a new real-PG migration claim. Task168 separately has actual PostgreSQL/SQLite migration coverage.

Both fixtures previously built the latest schema, dropped a few objects and relabeled its version. That left future storage incompatible with the real v59 guards. Exact pre168 source reproduces3FAIL/39PASS5.40s: v55 attachment-registry collision and two v57 task-catalog-drift failures. Source168's new CHECK merely caused the fake-v54 fixture to fail earlier when dropping sync_id; it did not create the underlying invalid historical fixture.

## Correction
Seed-only initializers execute the real V4 creation and registered steps through54 or56, under ordinary scoped pytest monkeypatch restoration. The normal initializer is restored before the upgrade. Historical preconditions assert the exact version and absent future attachment/identity metadata.

The maintained V4 template already contains two later V55 portable-ID declarations and unique indexes. For the v54 seed only, each exact declaration/removal count is asserted (2 columns,1 index for each of2 tables); those additions are removed before constructing the historical database. No latest-schema alteration or version relabel remains. Pre-V55 folders were an unversioned optional runtime backfill; the seed builds the historical folder and membership shapes required by the preserved parent/deleted-child tree. The registered54→55 migration handles that preexisting tree and adds its portable identity.

All original seeded active/deleted keyword and collection rows, versions/device labels, folder parent relation, collection link and note-folder membership remain. Every original assertion is present unchanged by AST comparison; additional preconditions verify historical state. V57 additionally executes the exact56→57 step and compares the full seeded row before the separate normal current-head reopen. No registry/collision guard or production migration changed.

## Receipts and attempts
All receipts are in `.tmp/fresh-uat-recovery-20260916`, each paired with command JSON:
- `uat168-organization-history-baseline`: exact pre168 modules,3FAIL39PASS0skip5.40s.
- `uat168-organization-history-first`: current source3FAIL39PASS.
- `uat228-historical-green`: first candidate41PASS1FAIL; precondition discovered retrofitted IDs in nominal V4 template.
- `uat228-final-green`: second candidate41PASS1FAIL; the registered historical path has no optional note_folders, which the original seed requires.
- `uat228-historical-final-green`: corrected full42PASS0skip3.17s.

The two fixture failures caused a bounded reassessment against the actual V4 template, registered migration and unversioned folder builder before the third candidate. No production retries or native mutations occurred.

## Secondary production finding, separate task
The first failed precondition's dict-containing assertion text triggered a second error at TransactionContextManager.__exit__ → installed Loguru message.format. This is a production formatting path, not a pytest sink artifact. A separate actual SQLite plain/brace probe proved1FAIL1PASS: brace-bearing ValueError becomes KeyError, leaves the transaction active and the pending write visible because logging occurs before rollback. The probe cleans up after observation. Parent associated UAT229/TASK13260.170 before repair. Its production fix is excluded from228.

Retained evidence: `test_transaction_format_probe.py`, `uat228-transaction-format-probe.redacted.log` and its command; original secondary traceback remains in `uat228-historical-green.redacted.log`. No claim that the secondary issue is harmless/harness-only.

## Static and review
Ruff0; Bandit0 findings/errors (B101 test assertions excluded), compile2PASS. assertion-parity.json proves zero removed original assertions. Exact two-file patch, baseline and snapshots are retained. No test count was dropped, no skip introduced. Independent review is pending; parent owns integration/tasks/native gates.

Independent command:
```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat228-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v57.py -q --tb=short
```
