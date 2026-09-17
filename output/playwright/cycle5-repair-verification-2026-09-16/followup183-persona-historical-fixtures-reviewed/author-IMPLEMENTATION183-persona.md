# UAT183 / TASK13260.120 — v25/v36 historical persona test fixtures

## Result and scope

The complete six-suite persona/memory/WorldBook regression command now reports **105 passed, 0 skipped, 0 deselected, 62.09 seconds**. This includes both previously failing migration cases and the separately frozen UAT200/203 controls. Exactly one test file changes for this UAT183 unit: `tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py`. No production migration, schema guard, database helper or other test function changed.

The old fixtures created the current complete schema, assigned an older version, and dropped selected tables. Newer Notes tables remained, so the intentional v59 registry collision guard correctly rejected the inconsistent database. The failures were present with exact pre-UAT200 PersonaStateStore bytes; baseline replay and current failing receipts were retained before fixture edits. Parent reopened the existing UAT183 task before this correction.

## Honest historical construction

A small seed-only helper follows the already reviewed v21/v39 test pattern: apply the actual V4 base and every registered SQLite migration through the requested historical version, asserting each version increment. A temporary initializer/version cap is restored before any ordinary current-head initialization. It does not fake a historical version, suppress a migration or weaken a guard.

- **v25:** assert the persona tables and later Notes registry are absent. Seed a historical character. Invoke the exact real25→26 migration and assert persona tables now exist/version26. Seed a persona and memory using only historical columns. Reopen with the ordinary current initializer and verify every original character/persona/memory column value survives. All original current-head, table, column and index assertions remain.
- **v36:** assert version36, absence of the later Notes registry, and absence of the later voice persona/connection columns. Seed real historical persona and voice rows. Invoke the real36→37 step, assert version37 and unchanged voice data. Reopen normally to current head, retain all original voice column/index/head assertions, compare every historical persona/voice value, and verify the newly added linkage columns default to null. The voice column repair is the existing current-initializer compatibility routine, not falsely attributed to36→37.

`assertion-comparison.json` proves all13 original v25 assertions and all4 original v36 assertions remain. All other functions are AST-identical, and the file remainder beginning at the first ordinary CRUD test is byte-identical. Only the now-unused sqlite3 import and excess import spacing are additionally removed.

## RED → GREEN and checks

- Original current broad run: 87 passed, these2 failed at the registry guard. Retained in the neighboring UAT200 packet.
- Exact pre-UAT200 loader replay: these same2 failed; that run also intentionally contained2 new PostgreSQL memory-control failures and2 SQLite passes. `baseline-red.redacted.log` is explicitly that combined receipt, not a claim of4 migration failures. Loader and original store bytes remain in the UAT200 packet.
- Corrected focused cases: 2 passed /0 skipped,2.14s. Only later formatting changed before the final combined run.
- Final unchanged six-suite command, with no exclusions: 105 passed /0 skipped /0 deselected,62.09s. `full-adjacent-green-command.json` records the exact command, official required-PG helper and test paths. The migration cases themselves are real SQLite cases; PostgreSQL is exercised by the adjacent backend cases.
- Ruff: baseline1 finding, current0. Bandit:0 findings /0 parse errors, with B101 excluded only for test assertions. Python AST parsing and scoped diff whitespace check pass. Receipts are in `static-summary.json` and underlying files.

Independent focused rerun:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat183-persona-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py::test_migration_v25_to_latest_creates_persona_tables tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py::test_migration_v36_to_latest_adds_voice_command_persona_columns -q --tb=short
```

Frozen source SHA256: `4e0b8f236bdfefc34608410d8af4ccc04eecd2f1692349324de701ff3191ce73`.
Frozen owned-manifest SHA256: `c31492013419b4631c9f346c67fa1a0b645083ed684f364bb2a3b62bd8484194`.

## Limits

These are synthetic historical schemas made using repository migration code, not recovered user databases. This test repair does not certify every historical upgrade or native UI. No browser, runtime restart, native database, provider, task/tracker, git or commit action. UAT200 and203 production edits remain separately attributed in their own packets. Independent review and integration are parent-owned.
