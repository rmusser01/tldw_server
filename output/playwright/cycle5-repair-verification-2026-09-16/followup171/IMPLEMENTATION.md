# UAT171 / TASK-13260.108 — standalone Notes read ownership

## Frozen result

Ready for independent review; task remains In Progress pending parent-owned native acceptance. The repair changes two production files and adds one permanent test file plus the approved shared design section. Exact paths, source snapshots and hashes are in `owned-manifest.json` and `review-snapshot/`.

Known pure Notes reads can now opt into finishing the PostgreSQL transaction they start. The call owns that transaction only when the raw connection starts IDLE and both ChaCha and backend managed depths are zero. It reuses the existing backend transaction context, including failure rollback. Pre-existing implicit writes, raw BEGIN, explicit/nested scopes and generic public query calls retain caller ownership; SQLite follows its existing path.

Nine pure read call sites opt in: note by ID, list/count Notes, singular/bulk note keywords, and four folder lookups. The actual keyword implementations are delegated through `note_store.py`. No backend interface, QueryResult, schema, connection pool, HTTP API, migration, or SQL classifier was changed. `read_only` is an internal caller declaration for these known pure SELECTs, not an SQL authorization or read-only enforcement mechanism.

## Why this boundary

Native evidence showed a retained Notes folder SELECT blocking the exact note_folders bootstrap ALTER after replacement startup. A simple `with self.transaction()` around folder reads would commit a pre-existing caller write when ChaCha's depth was zero. Restricting ownership to an idle connection avoids that. Earlier Notes and keyword reads also need the opt-in, otherwise their existing implicit transaction would prevent the final folder read from owning its work. A blanket SELECT cleanup would change locking reads and function side effects, so generic calls remain unchanged.

Diagnosis is retained in `/private/tmp/uat147-restart-diagnosis-20260916.md`; it concerns transaction lifecycle, distinct from the earlier UAT147 sequence ownership repair. The approved design section is in `Docs/Design/2026-09-16-uat-cycle-5-repairs.md`.

## Meaningful RED and controls

Official DB_Management `pg_database_config` provisions a temporary database per test. No AuthNZ `test_db_pool` or hand-created database is used. The private runner sets PostgreSQL required, disables Docker autostart, reuses the existing owned cluster and redacts credentials. PostgreSQL runs use authorized network escalation.

- Baseline production: **7 FAIL /14 PASS /0 skips**. Six failures are real lock timeouts when an independent connection executes `ALTER TABLE note_folders DROP CONSTRAINT IF EXISTS note_folders_path_key` after the four folder reads or either actual Notes get/list handler. The seventh proves a failed read leaves its transaction INERROR. Evidence: `uat171-red.redacted.log`; baseline sources and the original test bytes are retained.
- Initial implementation: **19 PASS /2 FAIL**. Actual endpoint tests caught opt-ins applied to similarly named inactive keyword-store methods. Those edits were fully reverted and the actual NoteStore delegation was corrected. This is recorded in `uat171-green.redacted.log`; `keyword_store.py` is byte-identical to its captured baseline and is excluded from final owned paths.
- Corrected original suite: **21 PASS /0 skips** in `uat171-green-final.redacted.log`.
- Final expanded controls: **46 PASS across4 files /4 warnings /0 skips**, including the new24-case suite, existing PostgreSQL transaction/session controls and the SQLite note-folder suite. Evidence: `uat171-final-controls.redacted.log`.

The new controls verify exact committed data and folder/keyword results, the raw idle state and released relation locks, caller visibility before commit, chosen commit/rollback outcomes, implicit UPDATE and data-modifying CTE writes, raw BEGIN, nested ChaCha transactions, backend-managed scopes entered before any SQL, generic SELECT FOR UPDATE, a transaction-local `set_config` SELECT effect, error rollback/recovery, and actual SQLite get/list handler results with outer rollback.

The real Notes handlers execute their full database read chains with injected test identity and a permissive fake rate limiter. This is not HTTP authentication/serialization coverage or a native browser test. Queries, rows, lock acquisition, errors and database responses are real.

## Verification command

From the repo root after `source .venv/bin/activate`:

```sh
TLDW_UAT_EVIDENCE_LABEL=uat171-final-controls node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs -q tldw_Server_API/tests/DB_Management/test_chacha_postgres_note_read_lifecycle.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_transactions.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_session_scope.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py --tb=short
```

Use a distinct evidence label for independent review to preserve author logs. Command receipts are alongside each retained log.

## Static checks

- Ruff owned Python scope: **0 diagnostics**, baseline also0. New test formatter check passes.
- Bandit production scope: **0 findings,0 errors**, baseline also0. Existing nosec-comment parser warnings are retained in its private log.
- Bandit new tests: **0 findings,0 errors**, with B101 excluded only for pytest behavioral assertions.
- Owned `git diff --check`: PASS.

## Limits and handoff

- Native Notes read plus owned API restart acceptance remains pending with parent. No browser, runtime process, database-session termination, service restart, staging or commit was performed here.
- Other ChaCha domains and unopted query calls retain their existing lifecycle. This bounded repair does not claim global elimination of idle transactions.
- An already active caller transaction is intentionally retained, even if its previous operation was another unopted read. Closing it would violate caller ownership; acceptance must exercise the repaired fresh Notes chains.
- Test warnings and the intermediate unsuccessful implementation are retained explicitly. All final owned production bytes are frozen; no further author edits are planned before review.
