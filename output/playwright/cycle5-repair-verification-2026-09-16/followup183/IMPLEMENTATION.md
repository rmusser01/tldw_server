# UAT183 / TASK13260.120 — historical migration test fixtures

## Change

Only four existing migration tests in two files change. Production migrations, collision guards, SQL, runtime, and configuration are unchanged.

- Build the SQLite v39 seed from the real V4 base and each registered migration through38→39. The seed-only initializer verifies each version increment, actual version39, and absence of persona_buddies/note_attachments. It inserts a real conversation using historical columns. The local patch restores the normal initializer before reopening at current head; existing columns/index/FK assertions remain, and title/owner survive.
- The SQLite and PostgreSQL v65 tests still verify exact65→66 completion and all existing Buddy storage assertions. The real PostgreSQL test still checks forced row-level security and tenant policies for every Buddy table at66. Each then reopens with the normal current head and verifies completion plus retained conversation; existing PostgreSQL API/asset/attachment/activity/version-conflict controls remain.
- Rename the schema-head66 test to describe the permanent v66 migration step. Keep the exact SQLite65 registry mapping and PostgreSQL65→66 method checks; allow newer current heads.
- Remove sqlite3 from persona_buddy_db imports because the corrected fixture no longer uses the direct downgrade/drop-table workaround. No other test bodies change.

## Evidence

| Check | Result |
| --- | --- |
| Retained pre-repair four failures |4 failed,0 skipped (baseline-four-red.log) |
| First capped39 fixture attempt |3 passed,1 setup failure,0 skipped |
| Corrected historical construction and all four cases |4 passed,0 skipped,3.94s |
| Same adjacent suite |162 passed,0 skipped,139.59s |

The cap-only attempt failed before testing upgrade: current initialization unconditionally runs recent Persona compatibility repair, which replays39→40 even with a39 target. The retained receipt is uat183-four-green.redacted.log; its label records the intended gate, not a passing result. No production workaround was added. Seed construction now invokes real historical migrations directly; only the subsequent ordinary constructor is the acceptance boundary.

## Validation

- Scoped Ruff:3 baseline /3 current diagnostics,0 new. Existing I001 and duplicate/unused json issues remain outside the four repaired tests. Removing an unused import shifts an existing diagnostic's line reference only.
- Full Bandit:185 B101 findings (pytest assertions),0 parsing errors, no other finding type. A second security scan excluding B101 records0 findings/0 errors. Assertions remain enabled; no production security checks are excluded.
- Both Python files parse; git diff --check passes.
- owned-manifest.json records frozen hashes; review-snapshot contains exact copies; owned.patch contains only this test repair.

## Reproduce

Activate the existing virtual environment, then run the official fixture helper with the exact four node IDs in uat183-historical-four-green-command.json, or the full adjacent command recorded with its receipt. The helper requires PostgreSQL, forbids unavailable skips, and uses official per-test databases on the owned55475 cluster. It does not use the native content database or AuthNZ test_db_pool.

Independent reviewer sidebar155_review passed all four corrected nodes (4 passed,0 skips,4.07s) and confirmed preservation of the substantive assertions. Review receipt: .tmp/uat183-review-20260917/independent-review.md. No additional edits requested. Parent owns tracker, commit and native work. This test-only correction makes no new native acceptance claim.


### Focused independent command

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat183-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py::test_migration_v39_to_latest_creates_persona_buddies_table tldw_Server_API/tests/Persona/test_independent_buddies.py::test_v66_migration_remains_registered_for_sqlite_and_postgres tldw_Server_API/tests/Persona/test_independent_buddies.py::test_postgres_v65_upgrade_installs_buddy_storage_and_forced_tenant_policies tldw_Server_API/tests/Persona/test_independent_buddies.py::test_v65_database_upgrade_preserves_conversation_and_adds_independent_storage -q --tb=short
```

The full adjacent command is retained verbatim in uat183-adjacent-green-command.json and repeats the same nine suites previously yielding158 passes/four failures. No deselection or test exclusion was added.
