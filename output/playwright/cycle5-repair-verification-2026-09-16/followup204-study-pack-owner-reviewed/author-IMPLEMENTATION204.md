# UAT204 / TASK13260.142 — canonical StudyPack owner

## Result

**Author validation:30 passed,0 skipped,4 warnings,37.67s.** Source and tests are frozen for independent review. Production changes only the StudyPack caller: omit its optional custom `client_id` and use the existing accessor's canonical numeric default. The181 independent operation, acquisition-through-finally cleanup, service, transaction, cancellation and error behavior remain unchanged. No cache/helper/schema/SQL or other-worker changes.

Frozen paths/hashes:

- `app/services/study_pack_jobs_worker.py`: `8723ee86e578b05c45bb9a57ce2fd04feaebd6744dc5fc006e144ea0c0cdc78c`.
- New `tests/DB_Management/test_study_pack_worker_owner_contract.py`: `44a0ab3af2cf69ba642a47b70cce76413ddb726f8552ceda03e7563f62ea3ee8`.
- Existing `tests/StudyPacks/test_study_pack_jobs_worker.py`: `cd72f389e355c821ad8744b9b05f3f406a8e7c06bfb56c06c1c324191a5eb864`.

All paths above are under `tldw_Server_API/`. Owned manifest SHA256: `88ac4220398d781dab0bb89cf4390b81697442dc1de15f619bc9206c6a7d0520`. Snapshots and exact patch are included. AST comparison proves the sole production change is deletion of the custom call keyword. The existing job test only changes its media-lookup-failure stub to accept the default accessor signature, removing the obsolete prefix assertion while retaining user/error/cleanup assertions. The181 lifetime test is unchanged (SHA `554d98227c8884a8ba326f957a021c60df93e196d2c24736bc16d513df1c0ba4`).

## Causal proof and RED

The separate diagnosis packet `../uat-study-pack-owner-diagnosis-20260917/` retains the actual-factory privileged PostgreSQL cold/warm proof:1 cold failure/3 controls,0skip. Cold worker construction stores deck/card owner `study-pack-worker-2`, an independent canonical2 reader cannot get the deck, and the later canonical owner loader reuses the same wrongly labeled cached object. Warm owner-first uses2 and succeeds. The first private attempt accidentally closed the shared fixture pool; that harness failure remains clearly separate.

The permanent final test before production change reports **2 expected cold failures/2 warm passes,0skip,7.56s**. The PostgreSQL cold failure specifically reaches the actual source resolver under a verified NOSUPERUSER/NOBYPASSRLS role and cannot see canonical owner2's note because the connection scope is the worker prefix. Actual restricted-role deck/card inserts also store the prefix. The SQLite cold failure is the new intentional canonical-label contract, not a claim that the old per-file data was inaccessible. Before that assertion it proves historical prefixed rows remain readable.

## Test boundaries

Four backend/cache combinations use the real empty or real owner-warmed cache, factory/default bootstrap, worker operation, source `to_thread`, service validation, deck/card persistence, independent numeric owner reader and subsequent same-cache canonical owner loader. Model output and provider selection are controlled; no model inference. PostgreSQL comes only from official disposable fixtures; SQLite/visual assets use isolated temporary paths. The fake Media handle is unused by these note-source jobs.

Restricted-role controls use actual existing Notes RLS and source resolution on the acquired worker DB, followed by actual deck/card writes and reads. The disposable role is verified non-superuser/non-bypass and explicitly dropped with its grants. This demonstrates the relevant application read/write boundaries, not a complete direct restricted-login cold schema/bootstrap/model job. The full worker generation path uses the official privileged fixture.

SQLite preserves physical per-user-file semantics. A historical deck with `study-pack-worker-2` remains readable and its metadata unchanged. New cold worker records now carry2, matching the prior warm path. No row migration, sync-log rewrite or deletion occurs.

## GREEN and static checks

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat204-owner-contract-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_study_pack_worker_owner_contract.py tldw_Server_API/tests/DB_Management/test_study_pack_worker_operation_lifecycle.py tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py -q --tb=short
```

30PASS/0skip includes unchanged181 actual source/model failure, success/repeated cached jobs, in-flight cancellation/deferred return and unrelated outer pending-write commit/rollback ownership controls. No deselections.

Ruff:1 baseline/1 current finding,0 added; only the existing job test's untouched import block I001. Production and new test have0. New test formatter passes. Bandit production0 findings/0 errors; tests0/0 excluding only B101. Python AST parsing and scoped diff whitespace check pass. Raw per-check results, logs and command receipts are retained.

## Limits / handoff

Independent review and root integration remain pending. No native acceptance claim. Existing incorrectly labeled cached objects or historical PostgreSQL rows are not migrated by this caller fix; fresh canonical construction is the covered path. Other callers that pass custom labels were not repaired or certified.198 Flashcards predicates are retained unchanged; this repair supplies the correct upstream owner. The fixture run used shared ChaCha candidate6089c5e0… but owns none of its bytes. No runtime/browser/config/provider/task/tracker/git actions.
