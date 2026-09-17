# UAT181 / TASK13260.118 independent final review

## Verdict

Clear for the bounded source/test repair. No remaining correctness defect found in the 39 opted-in read callsites. Independent required-PostgreSQL run: **61 passed, 0 skipped, 4 existing warnings, 90.07 seconds**. Native acceptance remains a separate parent-owned gate.

The integrated file also contains the separately reviewed UAT182 named-column asset getter repair. UAT181 does not receive credit for that repair.

## Exact reviewed source

The reviewer independently parsed baseline and frozen source, removed only the additional `read_only=True` keywords, and compared the remaining ASTs. Exactly 39 additions, no other AST changes: ChaChaNotes_DB 31; Buddy_DB 5; persona_state_store 1; conversation_store 2. All frozen manifest hashes match. Current files were checked again after the independent test run.

| Path under tldw_Server_API | SHA256 |
| --- | --- |
| app/core/DB_Management/ChaChaNotes_DB.py, pre182 frozen181 | `0e71405bd53b8ea9b7fed933fa9647315721f41bc352614b89eb26e6899526dd` |
| app/core/DB_Management/ChaChaNotes_DB.py, tested181+182 | `17f1a2db3214b6488fd9cb1e6c137dcdfb08594faca869b71115b9afa799a84d` |
| app/core/DB_Management/Buddy_DB.py | `ffd4c2b1dfd7c10e7beed095d354b33b186f26b0515af2f6d3b70980521dffa3` |
| app/core/DB_Management/chacha/persona_state_store.py | `23f9d6959ac21052310d3051bb40daa7d41b4247456c34de7ff578fd87476abe` |
| app/core/DB_Management/chacha/conversation_store.py | `d089e822c3b32de000d66eb85c5f9f541530bcedd07a5c9796ee47531328166c` |
| tests/DB_Management/test_chacha_postgres_study_read_lifecycle.py | `c7efdc1af2520b040dfad006b5d6a83a297c98adcafba263eb4355a433f03e71` |

The current ChaCha bytes equal the frozen181 file with exactly `blob = row[0]` replaced by `blob = row["image_data"]`. Other source/test files are byte-identical to the freeze. See `final-source-check.json` and `final-review-evidence-manifest.json` for machine-readable attribution.

## Correctness and test review

- All 39 SQL sites are pure reads. No locking SELECT, side-effecting function, DML CTE, SQL/default/schema/write change, or blanket classification was introduced.
- The existing helper owns only a read that starts on an IDLE PostgreSQL connection with both explicit transaction depths zero. It preserves an implicit pending transaction, explicit/nested ChaCha scope, and an explicit backend scope. Buffered query results remain readable after an owned read commits.
- The added populated Buddy tests resolve real conversation/workspace attachments and stored assistant activity before Flashcards reads. Both workspace branches and both conversation methods are now included. This closes the earlier review finding in the real call chain.
- The suite observes PostgreSQL state and locks, exercises existing-column `ALTER TABLE` using a separate connection, and runs an actual replacement CharactersRAGDB constructor. Seed cleanup occurs before the tested read; tests do not close the tested connection to manufacture success.
- Pending UPDATE RETURNING data remains invisible to an independent observer until caller commit. Caller rollback remains effective. Locking SELECT, transaction-local set_config, and a DML CTE remain caller-owned. A failed standalone SELECT returns the connection to IDLE and permits a subsequent real chain.
- The SQLite inventory and rollback control pass. Empty and populated asset reads are included; populated bytes require the separate UAT182 fix.
- Existing mutation-containing wrappers are appropriately excluded from the wholly pure caller-inventory assertions: stale-session maintenance, session completion, and new assistant-thread creation retain their pre-existing write behavior. Only their flagged SELECT boundaries are changed. The earlier unchanged UAT171 helper suite supplies the raw BEGIN control; UAT181 adds no new transaction helper.

## Verification receipts

Reviewer command, using the existing official isolated fixture runner after activating the project virtual environment:

```sh
TLDW_UAT_EVIDENCE_LABEL=uat181-independent-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_study_read_lifecycle.py -q --tb=short
```

`independent-green-command.json` records required PostgreSQL, Docker autostart disabled, and reuse of the owned cluster. `independent-green.redacted.log` records exit 0 and 61 passes with no skips. These are fresh reviewer results.

Reviewed author evidence: original RED 37 failed/12 passed; populated-chain RED 8 failed/3 passed/49 deselected; final integrated181+182 73 passed/0 skipped. Production Bandit JSON has 0 findings and 0 errors. Author scoped Ruff retained the same 8 baseline production findings and no new test findings.

The adjacent suite is **not fully green**: 158 passed/4 failed/0 skipped. The reviewer inspected the nonmutating baseline replay and its receipt: it restores all 36 changed method bodies from the retained baseline into the disposable pytest process, then reproduces the same four failures. Three assert schema head66 against current67; the fourth builds current schema, relabels it39, and then encounters the existing note_attachments migration collision guard. These failures were neither edited nor skipped and are not caused by UAT181. They do not establish whether an authentic historical39 migration passes.

## Limits

This review validates the frozen repair and isolated database tests. It performs no browser, runtime, production-database, task, tracker, or git mutation and does not certify repaired native restart/recovery. Other unmodified callers can still open transactions; preserving those caller-owned transactions is deliberate. This is not a claim that the whole application has no idle transactions or that all adjacent tests pass.
