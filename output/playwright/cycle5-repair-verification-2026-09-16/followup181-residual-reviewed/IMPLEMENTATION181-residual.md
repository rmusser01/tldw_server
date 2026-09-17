# UAT181 residual read-transaction repair: frozen handoff

Existing TASK13260.118. [Design](DESIGN181-residual.md), [permanent RED](RED181-residual.md), [owned patch](owned.patch), [source/test hashes and snapshots](owned-manifest.json). Production/test files are frozen for independent review. Parent owns native acceptance, task/tracker updates, staging and commits.

## Result and minimal scope

Exactly six pure SELECT sites now opt into the existing `read_only=True` ownership helper:

| File | Sites |
| --- | --- |
| ChaChaNotes_DB.py | ensure_character_tables_ready normal and recovery verification reads |
| chacha/character_store.py | get_character_card_by_name normal and recovery reads |
| chacha/persona_state_store.py | list_persona_profiles and list_persona_buddies |

These reads previously started an implicit PostgreSQL transaction on the cached connection. Later Buddy/Notes reads correctly preserved that pre-existing transaction, retaining locks that blocked replacement initialization. The actual persona list includes both the profile list and the buddy projection SELECT, so fixing only the first would leave the chain open.

The existing helper owns only an idle connection outside explicit transaction depth. It preserves caller writes, nested scopes, backend transactions and locking/generic operations. No commit/rollback, dependency scheduling, endpoint, RLS, SQL selection/filtering or backend helper change was added. UAT187 schema changes are a separate reviewed unit.

## Causal evidence and controls

Final RED: **9 failed,22 passed,0 skipped**,45.70s. Four isolated starters retained their own transactions. Actual existing-default maintenance and the actual persona list endpoint followed by Buddy or Notes reads each blocked a real second CharactersRAGDB constructor. The actual async maintenance executor alone retained character_cards and blocked RLS initialization. No-predecessor Buddy/Notes replacement controls passed. All12 pending caller-write outcomes,6 explicit first-read scope controls and2 SQLite read/rollback controls passed.

After the six flags, the same31 permanent cases pass. This includes actual replacement constructors against the same disposable PostgreSQL database while the first connection remains alive. Only the production timeout budget is shortened in the fixture; the initialization SQL is real. Existing read-lifecycle, character/persona store, persona Buddy and dependency-error controls also pass.

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-residual-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_shell_read_lifecycle.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_study_read_lifecycle.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_note_read_lifecycle.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py tldw_Server_API/tests/Chat/test_chacha_db_deps_error_mapping.py -q --tb=short
```

Result: **184 passed,0 skipped,4 warnings**,222.47s. [Redacted GREEN log](../fresh-uat-recovery-20260916/uat181-residual-green.redacted.log). Official function-scoped PostgreSQL fixtures and temporary SQLite only; helper network escalation was limited to the existing test cluster. No native database, browser, service or inference action was performed by this agent. Endpoint calls use real fixture DB/user injection and an enabled feature flag; they do not certify auth or browser transport.

## Static verification

[Exact static commands/exit codes](static-command-results.json), [Ruff baseline comparison](ruff-comparison.json), [production Bandit](bandit-production.json), [test Bandit](bandit-test.json).

- Full touched Ruff exits1 for5 existing diagnostics; exact baseline replay with original file paths has the same5, zero added/removed. New regression Ruff exits0.
- Regression formatting, compilation of all4 owned files and owned diff whitespace checks exit0. Existing production formatting is preserved.
- Production Bandit reports zero findings and zero errors. Its existing nosec-comment warnings do not prevent parsing; no production exclusion was added. Test Bandit reports zero findings/errors with B101 excluded for pytest assertions.
- Diff/snapshot validation confirms exactly2 changed SELECT lines in each of3 production files. No other production content changed within this unit.

## Native boundary and remaining acceptance

Parent independently observed the unchanged-source overlapping replacement failure: old API32260 remained alive as38457 reached health200, real Notes returned500, and old session7234 held persona_profiles while replacement12217 waited for its AccessExclusiveLock. These receipts are in `.tmp/uat181-native-20260917/bounded-replacement-notes-38457.json` and matching wire/UI files. Earlier old-exited-before-new-started metadata was not an overlap failure and remains labeled accordingly.

The older Buddy-only session84146 still lacks a directly observed first statement; do not claim these fixture paths retrospectively identify that exact session. This bounded change repairs the proven maintenance/persona composition, not every SELECT in the repository. Parent must perform final native replacement reacceptance on the reviewed bytes. Independent review remains required before integration.
