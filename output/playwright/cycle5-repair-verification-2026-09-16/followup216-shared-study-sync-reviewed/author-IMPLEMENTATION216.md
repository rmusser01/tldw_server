# UAT216 — StudyPack creation in the shared PostgreSQL schema

TASK13260.155. **Author source/test frozen for independent review: 55 tests pass, zero skips.** Native completion remains pending; original Alice job 2 was only inspected read-only and remains quarantined.

## Proven cause and repair

Normal Media-first initialization owns a shared `sync_log` whose identifier is `entity_uuid`. Standalone ChaCha initializes its own shape with `entity_id`. StudyPack's PostgreSQL trigger bodies always used `entity_id`, so the real shared schema rejected the first pack INSERT even though standalone worker tests passed.

The safe owned-profile inspection in `../uat216-diagnosis-20260917/owned-schema-job-function-shape.json` confirms: job2 has one note source, no workspace, and deck_mode=new; runtime sync_log has required entity_uuid and no entity_id; all three installed pack/membership/citation functions have three entity_id INSERTs each and zero entity_uuid INSERTs. Reads used DatabaseBackendFactory with verified transaction_read_only=on, existing profile/holder/target boundary checks, short timeouts and no writes. Credentials, source text, title and session data were never printed. Native driver detail remains redacted; no raw-log copy was made.

The official-fixture actual worker → generation service → pack persistence test reproduces the exact native `Failed to create study pack: PostgreSQL query execution failed` only with real Media initialized first. Standalone PostgreSQL and SQLite controls pass. A separate private official-fixture contrast captures only the payload-free `UndefinedColumn` exception category and shows that replacing these three trigger column references makes real pack, membership, citation and sync rows succeed. No database results/errors were mocked.

The production delta is confined to `_ensure_study_pack_schema_postgres`:

- Introspect the existing sync table once using the initializer's existing connection/transaction.
- Select only the fixed names entity_id or entity_uuid, following existing schema-index, generic-link and character-history compatibility patterns. Unsupported shapes fail instead of guessing another identifier.
- Use that choice in exactly nine INSERT column lists in `study_packs_sync_log_fn`, `study_pack_cards_sync_log_fn`, and `flashcard_citations_sync_log_fn`.
- Preserve all values, JSON fields, owner/version semantics, trigger timing, transactions, table schemas and schema versions.

The existing initializer runs on every current-head open and replaces trigger functions. The repair therefore updates old installed bodies on a normal reopen, without a new migration, manual native schema patch or global SQL rewrite. No worker or generation-service production code changed. The three SQL interpolation annotations document that the identifier comes only from a closed set; no user input is interpolated.

## Owned files and shared-source attribution

1. `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`: only the above method. Frozen method SHA256 `ed682318fa4cf77681446231e3fecdd470c108ab1f46a8fd6e7bc301d2173e81`.
2. `tldw_Server_API/tests/DB_Management/test_study_pack_shared_sync_schema.py`: nine new permanent tests. SHA256 `65cb83597c9cfa63a36b631c82b8f42bd359250f5a6acd43c7a3c6899b031504`.

The whole ChaCha snapshot also contains Source209's independently owned Notes changes. That combined file SHA is `245ee8718d6d7ba118b88e02efb75f473abcc74f28abe2b9ee27034cfd34b025`; it stayed stable through the final run. Source209 confirmed method nonoverlap. `owned.patch` is attributed only to216 (method delta +new test); `owned-method.patch` and baseline/current method snapshots support exact review. Whole-file snapshots are retained transparently, not claimed as all authored by216.

Owned manifest SHA256: `7d47fbb3913444cfe4b74a02ea8caca248a4fcac239f7e4da704b88ed1af1505`.

## RED/GREEN evidence and controls

- Initial worker RED: **1 failure /3 controls /0 skips**, 6.47s (`initial-red/worker-red.log`). Exact initial test bytes retained.
- Payload-free cause/candidate contrast: **2 controls pass**, 3.68s (`../uat216-diagnosis-20260917/column-causal-control.log`); only disposable fixture function definitions differ.
- Both-order expansion: **2 failures /4 passes**. The extra failure occurs in Media initialization before any StudyPack call and is separately tracked UAT218/TASK13260.156.
- Initial expanded/following-green reopen test incorrectly closed the shared backend pool before constructing the new instance. This fixture mistake is retained in `expanded-red.log` and `first-green-fixture-failure.log`; it is not a product failure. The test now releases the checkout, retaining the shared backend for the actual reopen.
- Final original-method replay uses an in-memory, hash-checked plugin with the corrected final tests: **4 expected failures /5 passing controls /0 skips**, 16.34s (`final-baseline-red.log`). The four failures are actual worker persistence, all-trigger payload lifecycle, caller rollback, and old-body replacement. No source file is swapped and no fake SQL response is supplied.
- Final current-source combined suite: **55 passed /0 skipped**, 68.20s (`final-green.log`). This includes all nine new cases plus existing canonical-worker-owner, operation lifecycle, membership-count/rollback and StudyPack storage tests.

New controls exercise actual worker/service/source-thread behavior with only the remote model seam and database accessor supply controlled. They verify persisted owner2 across pack/deck/card/membership/citation rows, exact source identity and citation text, and sync identifier/owner/version/payload linkage. Create/update/delete branches of all three triggers emit the expected versions1/2/3 and preserve their real payloads. An outer caller rollback removes the entire newly created graph and corresponding sync rows while keeping previously committed source data. The current-head reopen test installs the exact historical bad column references in disposable functions, then relies solely on normal reopening to replace them without changing schema version.

These owner controls preserve canonical output identity; they are not a claim of general StudyPack tenant isolation or native model quality. Existing restricted-role source controls run in the adjacent owner suite. No original card was rated or changed.

## Static checks

- Ruff: **0 findings** across the combined production file and new216 test.
- Bandit production: **0 findings /0 parse errors** (`bandit-production.json`). The initial three closed-set interpolation alerts and their correction are retained in `bandit-production-first.json`; annotations are confined to the three expressions with fixed identifiers.
- Bandit test: **0 findings /0 parse errors**, excluding expected assertion-only B101 (`bandit-test.json`).
- Python compilation: both owned paths compile; no Python bytecode/runtime service launch required.
- `git diff --check` on the two owned paths: pass. No git mutation performed.

## Exact final commands

From repository root, using the official disposable fixtures and required-PostgreSQL runner:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat216-final-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_study_pack_shared_sync_schema.py tldw_Server_API/tests/DB_Management/test_study_pack_worker_owner_contract.py tldw_Server_API/tests/DB_Management/test_study_pack_worker_operation_lifecycle.py tldw_Server_API/tests/DB_Management/test_study_pack_membership_count_backends.py tldw_Server_API/tests/StudyPacks/test_study_pack_storage.py -q --tb=short

PYTHONPATH="$PWD/.tmp/uat216-repair-20260917${PYTHONPATH:+:$PYTHONPATH}" TLDW_UAT_EVIDENCE_LABEL=uat216-original-method-final-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs -p uat216_baseline tldw_Server_API/tests/DB_Management/test_study_pack_shared_sync_schema.py -q --tb=short

python -m ruff check tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/DB_Management/test_study_pack_shared_sync_schema.py --output-format json
python -m bandit tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json
python -m bandit tldw_Server_API/tests/DB_Management/test_study_pack_shared_sync_schema.py -s B101 -f json
```

Reviewers should use a new evidence label to preserve these receipts. `final-green-command.json` and `final-baseline-red-command.json` retain actual runner invocation metadata without connection credentials.

## Explicit separate findings / acceptance limits

- **UAT218/TASK13260.156:** reverse ChaCha-first →Media initialization fails at Media's `idx_sync_log_entity_uuid` index against the other column shape. Exact failure is retained in `both-orders-red.log`; permanent `test_media_after_chacha_initialization_postgres.py` isolates that same actual initializer failure with guaranteed cleanup; original worker-case bytes/receipt remain retained. It is separate, unskipped, and remains failing pending its own design/repair. The216 green command deliberately lists only216 and adjacent accepted suites; it is not a claim that reverse initialization passes.
- Adjacent suggestion snapshot/generation-link triggers have the same literal by source inspection. They were disclosed for separate task association; no actual native failure or blanket repair is claimed.
- Full native StudyPack completion and original-scenario acceptance still belong to the parent after review/restart. Existing job2 was not retried or modified by this agent. No browser/session credential/runtime/provider/task/tracker/git mutation occurred.

Design and causal inspection are retained in `../uat216-diagnosis-20260917/DESIGN216.md`. Author implementation/verification is complete; independent review and native acceptance are pending.

## Final test-only cleanup

At parent review, the reverse-order case was fully decoupled from216's test helper:218 now has its own real initializer regression/cleanup, and the216 test lost only the obsolete `after` branch. Production is unchanged. The final nine216 cases pass with zero skips in14.67s (`final-cleanup-green.log`); fresh Ruff0 and Bandit test0/0 confirm the new test bytes. The preceding55-case combined receipt applies to the same production and the pre-cleanup test retained as `pre-cleanup-test.py`; no assertions were removed from the selected216 cases. The independent reviewer should rerun the listed55 command on the refreshed manifest. The standalone218 case still fails1/0 at the exact index boundary and is not skipped.
