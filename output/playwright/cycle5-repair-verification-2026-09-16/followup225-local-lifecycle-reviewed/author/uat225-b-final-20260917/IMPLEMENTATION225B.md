# UAT225 Stage B — local suggestion lifecycle and enrollment retirement

Task: **TASK13260.164**. Source/test freeze: `owned-manifest.json`, SHA256 `916e17df1a977d7531265736b3b031ab6f21abb3d67e6ae9affa2f0c93af2e7b`. Patch baseline HEAD `42a65b4dd14620b5653d01e25a1ff138361b65f8`. Nine production files and seven tests are the complete owned source scope. The patch includes the new untracked adapter/tests explicitly. No API/schema, keyword persistence, canonical organization store, runtime, browser, task/tracker or git changes are owned here.

**Status:** implementation frozen; final author173 focused +200 adjacent PASS, zero skips. Independent review underway. Native acceptance remains parent-owned and pending. This report does not close UAT225 or the fresh matrix gate.

## Problem and final behavior

Fresh inactive-Sync Notes had no local decision coordinator. Its exact `legacy:<owner>` review scope could not complete the documented admission/generation/acceptance lifecycle. A later canonical enrollment also needed to stop local work before taking Notes snapshots while retaining immutable Job/operation receipts and bounded cancellation obligations.

The existing local scope is now usable only while that owner has no canonical authority row. No fake legacy authority, namespace, table, payload rekey or new schema is introduced. The real factory supplies a guarded local mutation adapter. Existing Jobs admission, content-free payloads, worker retrieval/preparation/strict provider-result validation, publication, reads, accept/reject/reset, cancellation and reconciliation continue through their existing boundaries.

Product mutations and acceptance finalization share the existing transaction. Note and target changes invalidate local suggestions transactionally. Keyword creation alone never means acceptance. Deterministic UUID4 keyword identity and relationship identity preserve replay. Committed168 supplies durable merge-chain resolution; publication and the final membership guard re-resolve the portable identity, including merge followed by restore.

## Exact source scope

| File | Change |
|---|---|
| `DB_Management/chacha/note_graph_suggestion_store.py` | Exact local absence authority; canonical reservation and closed retired cleanup; local note invalidation/discovery; survivor-aware publication; fair retained late-enqueue discovery; two exact-local SQLite tag membership checks. |
| `Notes_Graph/suggestion_local_mutations.py` | New three-product adapter using existing Notes stores and guard/finalizer callbacks; local SQLite keyword lookup/membership preserves per-file device labels. |
| `Notes_Graph/suggestion_decisions.py` | Optional local adapter, merge resolution at acceptance and final product guard, local tag application, lazy product-db selection preserving link-only collaborators. Canonical route remains covered. |
| `Notes_Graph/suggestion_service.py` | Select the real local coordinator only for exact owner/local absence scope; existing canonical coordinator selection retained. |
| `Notes_Graph/suggestion_maintenance.py` | Closed retired-scope cancellation/staling/cleanup; late-enqueued Job discovery; narrow enrollment-race handling. No retired acceptance/publication. |
| `Sync/v2/service.py` | Re-read and validate the actual default personal chatbook dataset, validate the selected Notes DB owner, reserve that real target. |
| `Sync/v2/profile.py` | Invoke the common pre-snapshot fence in ordinary bootstrap and Personal Context new/supplied/resumed binding. |
| `Sync/v2/notes_link_bootstrap.py`, `notes_organization_bootstrap.py` | Same idempotent owner-validated precondition for direct bootstrap entrypoints. |

## Authority and transaction contract

- Local PostgreSQL operations acquire the existing authority table SHARE lock before product/run locks and recheck authority absence. Canonical reservation acquires the conflicting lock, preserves equal existing flags, rejects a different target and commits the actual all-false row before snapshots. Real PG barrier tests observe the pending relation lock in both orderings; SQLite transaction controls cover the equivalent serialized outcomes.
- The atomic operation is the **authority fence**. Local publication and mutation become inaccessible immediately. Old row/Job cleanup is bounded and follows afterward; there is no claim of a cross-database atomic migration.
- Generic enrollment still rejects first Notes default/org/link creation. Both actual profile default-creation paths and direct link/org bootstrap paths are covered. A later real task binder reuses the all-false same-target reservation.
- Closed maintenance may inspect only the exact retired owner-local scope. It cancels matching Jobs and stales pending/accepting work; it never reconciles acceptance or activates publication there. Foreign scopes and payload mismatches cannot be cancelled or mutated. Already accepted products and terminal receipt envelopes remain unchanged.
- Run/job UUIDs, idempotency payloads and terminal envelopes are never rewritten. Old rejected rows are not migrated into canonical suppression. Old scope is obsolete after authority change; cross-dataset replay/continuity is not promised.

## Late enqueue and retention

Admission commits before enqueue and bind. A producer may enqueue after enrollment and after the original ten-minute missing-job grace. Retirement therefore continues exact run-id idempotency discovery of failed unbound admissions through their **original** run expiry. Existing paired maintenance token/expiry fields schedule a five-minute retry; no new schema/cursor is added. Ordering by prior lookup expiry/creation lets untouched later records progress with budget one. A successful cancellation holds the lease through the original expiry. Discovery does not rewrite terminal receipts or extend the existing 30-day run and 90-day terminal receipt windows. Tests include a real late Job after the grace, bounded fairness, temporary Jobs outage, wrong payload, original-expiry cleanup and already terminal receipts.

## Review correction: SQLite keyword device labels

The initial candidate incorrectly reused canonical organization membership filtering for local SQLite keywords. Core168 and ordinary SQLite CRUD intentionally treat these as per-file identities, even when an earlier device wrote the keyword. Three actual consumer tests failed for existing, merged and same-label tags; a fourth showed duplicate publication of an already-linked prior-device tag.

The correction uses existing KeywordStore list/count/link and survivor APIs only in the local adapter; it reuses the canonical NFC/casefold normalizer. The decision's final guard and finalizer remain in the product transaction. Two suggestion-store membership reads omit keyword device filtering **only for exact local SQLite scope**. PostgreSQL selected-owner predicates and canonical Sync behavior remain unchanged. Tests preserve the old keyword's full row/version/device label, exercise Unicode label reuse, and force a failure after actual finalization to prove membership/decision rollback together. Broader historical prior-device Notes behavior is not changed.

Original provisional manifest/patch/snapshots are under `provisional-before-sqlite-review/`. The exact reviewer-caused RED and first candidate failure remain retained.

## Causal and intermediate evidence

All labels below refer to `.tmp/fresh-uat-recovery-20260916/<label>-command.json` plus `<label>.redacted.log`. The evidence manifest is an explicit safe allowlist; raw `.private.log`, configs and credentials are excluded. `verification-history.json` records every retained author result, including failed candidates.

| Boundary | RED / correction evidence |
|---|---|
| Disjoint adapter / fresh factory | `uat225-local-adapter-red`:10 FAIL; raw-link timestamp first candidate4 FAIL/4 PASS; existing timestamp normalizer then18 PASS with the two unwired factory controls explicitly deselected only at that early checkpoint. Final suite includes them. |
| Fresh admission / enrollment | `uat225-local-admission-red`:2 FAIL/2 controls; `uat225-enrollment-entrypoints-red`:6 FAIL; `uat225-b2-reservation-red`:14 FAIL. |
| Shared local factory/products | `uat225-b1-shared-red`:4 FAIL/20 controls; factory-products4 FAIL/6 controls. First deterministic UUID candidate2 FAIL/32 PASS because UUIDv5 violates existing UUID4 contract; corrected to deterministic UUID4 bytes. |
| Invalidation and local discovery | `uat225-b1-invalidation-discovery-red`:6 FAIL; subsequent24 PASS. |
| Merge consumer / product guard | `uat225-merge-consumer-red`:4 FAIL before168 consumer support; restore-under-guard2 FAIL; final6 PASS. |
| Retirement / late enqueue | `uat225-b2-retirement-red`:16 FAIL; >ten-minute late enqueue2 FAIL; maintenance enrollment race2 FAIL; corrected controls retained. |
| Late discovery first candidate | `uat225-b2-retention-fairness-green`:6 FAIL/44 PASS: four actual paired-lease constraint failures corrected by preserving token/expiry together; two fixture assertions expected an internal identity in a public response and were corrected to inspect persisted acceptance. |
| Adjacent compatibility | First final adjacent1 FAIL/199 PASS showed eager access to an organization db in a link-only collaborator; lazy selection fixed it. Final pre-review adjacent200 PASS. |
| SQLite device review | consumer3 FAIL; dedup-expanded4 FAIL; first correction28 PASS/2 assertion failures (empty tuple versus list, preserved missing/deleted error wording); final expanded32 PASS. |

Other retained harness corrections are explicit: one command named a nonexistent maintenance test file and collected no tests; actual route fixture initially used response `run_id` instead of existing `id` (2 FAIL/6 PASS); concurrent fixture referenced a nonexistent task-store property (2 FAIL/24 PASS). None is counted as a product defect or silently discarded.

### Stage A fixture evolution and teardown diagnosis

The real local factory intentionally changes old missing-coordinator expectations. The initial compatibility run retained20 FAIL/40 PASS and two teardown errors; an intermediate alignment retained2 FAIL/58 PASS and the same two errors. `NoJobs.__getattr__` called `pytest.fail` (a BaseException) when newly valid admission reached Jobs, terminating the TestClient portal; shutdown then raised `This portal is not running`. This was a fixture sentinel/portal failure, not observed PostgreSQL checkout leakage.

The missing-decision fixture now explicitly removes that collaborator. It permits only the existing read-only idempotency lookup that precedes validation, forbids enqueue/product mutation, and asserts the real failed-admission receipt. GET read-only guards remain unchanged. Actual local/canonical positive controls and retired/foreign negative controls stay enabled.

## Final verification commands and limits

Activate `.venv`; use the official fixture runner, which requires PostgreSQL and supplies credentials privately:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=<unique-label> node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs <file arguments below> -q --tb=short
```

Exact eight-file focused args: `uat225-b-reviewed-focused-command.json`. Exact nine-file adjacent args: `uat225-b-reviewed-adjacent-command.json`. These run real official PostgreSQL and SQLite persistence, actual local routes/Jobs/worker/factory and restricted-role controls. Provider replies are synthetic, as required for deterministic tests; there is no inference or native browser claim.

Final outcomes are recorded in `FINAL-VERIFICATION.json`: focused **173 PASS /0 skip /193.45s**, adjacent **200 PASS /0 skip /99.89s**, targeted32 PASS/0 skip; Ruff0; Bandit production0 findings/0 errors and tests0 findings/0 errors (only test assertion B101 excluded); compile16; owned diff whitespace check0. Earlier complete candidate runs165 then167 focused and200 adjacent are retained as intermediate evidence, not substituted for final bytes.

Dependencies: Stage A225/226 commit043b8cd1dd; separately reviewed keyword survivor168 commit447edebb7b53d7489c2fbac5903db473c1f556d5; unrelated228/229 already present in baseline42a65b4d. No keyword/schema/organization source change is included here. Parent must complete independent review and targeted native acceptance before UAT225 closure or the full fresh matrix.
