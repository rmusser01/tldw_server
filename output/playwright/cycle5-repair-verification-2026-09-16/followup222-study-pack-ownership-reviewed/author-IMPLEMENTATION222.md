# UAT222 / TASK13260.160 — StudyPack and provenance ownership

## Result and scope

The selected PostgreSQL owner now controls StudyPack metadata, membership and citation reads and their directly related writes. The existing detail and regenerate routes consequently hide a foreign pack through the normal missing-record response. An owned card cannot expose a foreign pack or citation via malformed historical child rows. SQLite continues treating client IDs as sync-device labels.

Only two repository paths are owned:

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`: twelve existing StudyPack/provenance methods and the closed `_require_selected_owner_row` helper extension.
- `tldw_Server_API/tests/StudyPacks/test_study_pack_owner_contract.py`:63 actual storage/route controls.

`ast-attribution.json` proves every other production AST node is unchanged from baseline SHA0bbf4442fa4f41d53ff61c3261b480387bcbdc866e1f15785244fbdb6065513c. No schema, endpoint, cache, source resolver, worker or transaction-manager changes. Parent-owned UAT223 note-link changes are present in the combined test source and are hashed separately.

## Implementation

Four reads bind the selected owner in PostgreSQL. Child reads additionally check their pack/card parents. Membership order and citation ordinal order remain unchanged; ownership filtering precedes the first-pack LIMIT so a later valid membership stays eligible. Existing deleted-row semantics and read-only flags are preserved.

Eight create/mutation methods validate parents inside their existing transaction. Destination deck validation reuses the existing locked owner check. The existing parent helper accepts only two additional fixed tables and selects the fixed UUID column only for flashcards. Membership append locks its pack, then all unique referenced cards in sorted order before any insert; insertion still uses the original caller order. Citation replacement updates only the selected owner's child rows. Delete preserves owned stale-version/idempotency behavior; supersession keeps deterministic pack-lock order. No method commits or rolls back caller work independently.

## Causal evidence

All receipts are retained in this packet; raw/private logs are excluded.

- Initial read candidate:11 failures/6 passes, including3 test-fixture errors (SQLite duplicate deck names and unavailable optional Notes schema in a standalone fixture). Preserved, not counted as11 product failures.
- Corrected read RED:8 PostgreSQL failures/9 controls passed/0 skips. Actual detail/regenerate and owned-card assistant routes are included.
- Initial write candidate:17 failures/3 passes, including7 missing test-exception-import errors. Preserved separately.
- Corrected write RED:10 PostgreSQL failures/10 SQLite passes/0 skips,17 previous cases deselected.
- Expanded RED:27 failures/28 passes/0 skips.26 PostgreSQL ownership/parent failures; one SQLite test incorrectly expected a caught nested executemany failure to independently roll back its partial insert. The unchanged SQLite manager delegates rollback to the outer caller. The resulting assertion with a dictionary also exposed an unrelated existing logger-format error; no transaction/logger production change was made.
- Corrected transaction/order RED:2 PostgreSQL failures/4 controls passed/0 skips,53 deselected. PostgreSQL denial is tested before any insert; propagated write failure and caller rollback are tested on both backends.
- Later historical-child control replay: original13 methods loaded only in the isolated pytest process,2 PostgreSQL failures/3 SQLite controls passed/0 skips,58 deselected. Confirms both replacement methods previously rewrote foreign citation rows. `uat222_baseline_replay.py` never writes production files.
- First GREEN:59 passed/0 skipped,80.68s.
- Final GREEN:63 passed/0 skipped,86.96s;4 existing dependency/deprecation warnings. These include true restricted-role PostgreSQL, same-file SQLite device mutation, foreign/missing/deleted references, versions, empty/duplicate inputs, original membership order, later valid historical membership, and sync-row restoration after rollback.

## Exact final command

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat222-final-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/StudyPacks/test_study_pack_owner_contract.py -q --tb=short
```

The official required-PG runner provisions disposable fixture databases. Explicit Jobs mode removes the global Jobs URL so existing SQLite JobManager fixtures remain SQLite. Regenerate tests enqueue fixture jobs only; they do not run a worker/model. Route dependencies supply fixture principals/databases; this is actual router/storage testing, not native login acceptance.

## Adjacent verification

Final adjacent run:204 passed/0 skipped,201.79s;9 existing warnings. Exact11-file command is retained in `uat222-adjacent-green-command.json`. Coverage includes existing storage/generation/provenance, completed-job/admin-owner route resolution, response serializers, actual worker ownership/lifecycle, membership counts and flashcard resource-owner controls.

## Static verification

- Scoped Ruff:0 findings.
- Production Bandit:0 findings/0 parse errors; test Bandit:0 findings/0 errors with B101 excluded for test assertions. SQL identifiers remain closed internal values and data remains parameter-bound.
- Compile:both owned Python files pass.
- Scoped `git diff --check`:pass.
- Owned source/test hashes and exact snapshots are in `owned-manifest.json`; all other AST nodes compare equal.

## Limits and remaining acceptance

Native original job5, pack2 and deck10 were not altered or read by this agent. Parent must repeat Alice-owned/Bob-foreign readback on the reviewed runtime. The prior native role is privileged/BYPASSRLS; this unit separately proves application predicates under an actual NOSUPERUSER/NOBYPASSRLS fixture role. It does not introduce table RLS or claim raw-SQL tenant protection.

Workspace/source-bundle contracts and legacy SQLite semantics remain as before. No broader source-reference authorization framework was introduced. Parent-owned UAT223 fixes Notes route generation separately. Source/test bytes are frozen for independent review; native acceptance remains parent-owned and pending.
