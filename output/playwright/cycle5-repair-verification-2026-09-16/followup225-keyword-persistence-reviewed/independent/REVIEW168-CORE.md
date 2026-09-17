# TASK13260.168 — independent local keyword survivor persistence review

## Verdict and scope

**CLEAR for the frozen five-file persistence core.** No correction to its production or tests is requested. This is the schema/persistence portion of UAT225, not acceptance of the complete inactive-Sync lifecycle. Adjacent test maintenance and Stage B consumer integration remain separately reviewable.

Frozen author manifest: `.tmp/uat225-keyword-merge-20260917/owned-core-manifest.json`, SHA256 `830b744fb6fbdd0f44574f69f611b402d3e352822e8976a1357c4f57b7bd0103`. All five current files and snapshots matched before and after verification.

| File | SHA256 |
| --- | --- |
| `ChaChaNotes_DB.py` | `1e20679f6f7576a02b4e4aed5b12d8bd8eef60d647d7b2da6fc98947f2ae8172` |
| `chacha/keyword_store.py` | `a7b8bba6e299695d122773617937a1e9c9d591ff1707aeef70e887be952cdba4` |
| `chacha/organization_sync_store.py` | `3742eb2acc0a0eca57950b23f2b9b27433b7a9a2f06a0dae0888e0d4ec9d5b24` |
| `test_local_keyword_merge_survivor.py` | `f3afb480c7887e3e4e5ca8e2cfb693086c1ccf4c32dd7e54f665417ea0cd82fd` |
| `test_local_keyword_survivor_migration.py` | `33a7c92dc80c8ede2a5024f8e70fc2e1fd88406771239359393264a7ec0b7ee2` |

## Source review

- SQLite 67→68 and PostgreSQL 69→70 add one nullable field, an active-row/self-reference constraint, and the exact version advance inside the existing initialization transaction. Historical fixtures construct capped prior schemas and assert both the old version and absent column before migration. Existing seeded rows, IDs, labels, versions, timestamps, indexes, triggers, and PostgreSQL policy/security metadata are compared after repeated reopen. Injected failure after the migration verifies column/data/version rollback before a successful retry. Historical tombstones receive NULL; there is no invented backfill.
- A local merge writes the immediate target's validated portable UUID in the same source CAS update and transaction as all four membership moves. Parent owner checks and source/optional-target version checks remain. Parent locks now use sorted IDs, matching the resolver's lock order.
- The resolver validates each identity and bounds traversal at 100 rows. PostgreSQL filters the selected owner at every hop; SQLite preserves its per-file/device-label behavior. It does not guess from labels. Missing, plain-deleted, malformed, foreign, cyclic, and over-limit chains fail closed. Live rows take precedence over old tombstone history.
- A locked resolution requires a caller connection, discovers a bounded chain, acquires sorted parent locks, and compares complete row snapshots. A concurrent merge, restore, delete, or rename changes that snapshot and raises a retryable conflict. It does not commit or roll back the caller's surrounding work. This review does not claim arbitrary untrusted connection objects are validated as active transactions.
- Ordinary add/undelete, canonical organization keyword upsert, and PostgreSQL flashcard tag restoration clear the redirect in the same activation update. Re-merging a restored identity records a new redirect; idempotent deletion of a merged tombstone preserves its existing result.
- The canonical organization resource projection and legacy keyword sync-trigger payloads remain explicit and unchanged. No canonical head, response schema, provider, Jobs, authority, or suggestion-decision implementation is part of this core.
- An independent AST comparison confirms that only ten approved methods, two schema-version constants, and the two keyword-validation imports differ from the retained baseline; the remainder of the three production modules is equal.

## Fresh independent verification

Official isolated PostgreSQL/SQLite command, after activating the project venv:

```sh
TLDW_UAT_EVIDENCE_LABEL=uat168-sidebar-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/DB_Management/test_local_keyword_merge_survivor.py tldw_Server_API/tests/DB_Management/test_local_keyword_survivor_migration.py -q --tb=short
```

**62 passed, 0 skipped, 4 existing warnings, 70.58 seconds.** No manual database setup or native runtime action was performed. Real PostgreSQL controls include opposing merges, four pre-lock chain mutations, and a NOSUPERUSER/NOBYPASSRLS selected-owner operation. SQLite caller rollback, device labels, all restore writers, four-family atomicity, fresh/reopen/migration failure, and invalid chains are covered. Restricted-role service controls do not establish arbitrary raw-SQL isolation.

Ruff: **0 findings** across all five files. Bandit: **0 findings / 0 parse errors** on production; **0 / 0** on the two tests with only B101 excluded. In-memory compilation of all five files passed. Exact commands, logs, hashes, AST output, and static JSON are retained in this directory.

## Causal and adjacent evidence

The original actual merge/reopen receipt shows **2 failures / 4 controls passed**, with no persisted survivor on either backend. The opposing-merges RED records one successful merge and one database/deadlock error; the frozen control now requires the serialized conflict result. The migration RED and later test-fixture corrections are retained by the author; their invalid fixture/exception expectations are not counted as additional product defects.

The author's broader adjacent run is **309 passed / 4 failed / 0 skipped**. The four failures are literal current-head expectations: three deck tests expect PostgreSQL 69 after unrestricted initialization now reaches 70, and one character test expects the global SQLite constant to remain 67. Narrow test maintenance is in progress, outside this frozen five-file verdict; exact capped historical transitions must remain asserted.

The separate historical suite is **39 passed / 3 failed**, and the same three nodes fail on the author's retained pre-repair modules. Two relabelled-v56 fixtures hit existing task-catalog drift. The relabelled-v54 fixture reaches an existing attachment-registry collision on baseline; with the new constraint it fails earlier when attempting to drop `sync_id`. These are baseline fixture defects assigned separately to UAT228/TASK13260.169, not proof that production migration guards should be relaxed. This reviewer inspected the paired receipts; did not rerun those baseline replays.

`causal-and-adjacent-inputs.json` binds the exact six reviewed logs. A local evidence-copy filename typo was corrected without rerunning tests or changing any source.

## Remaining gates

Stage B must still prove actual pending/late-published suggestion acceptance, immutable replay, authority retirement, and acceptance/enrollment concurrency using this resolver. Core resolver races alone do not establish those consumer invariants. No browser, native original-scenario acceptance, full matrix, or clean-whole-repository claim is made. Parent controls integration and runtime acceptance. No production/test/task/tracker/git/runtime/browser state was changed by this review.
