# PR 3016 VN Durability Review Implementation Plan

## Current Review Wave: Task 23

Full Qodo review5325946065 completed12:45:31Z on88aefac2ba95c9f743d3e67b34fd55858479315a,
acknowledgment5846379125. Request5846349629 fulfilled; no review pending.
One new finding4111405382/PRRT_kwDOL1aGf86mQtxJ;40threads1unresolved.
Tasks1-22 remain locally complete/frozen/independently approved. AC5 reopened.

### Task 23: Offload Complete Integrity Reconciliation

**Base:** 88aefac2ba95c9f743d3e67b34fd55858479315a.
**Tracking:** TASK-13369. **Spec:** Docs/Design/2026-09-25-vn-pr-3016-review.md.
**Scope:** VNAssetPacks_DB.py, VN_Assets/worker.py and focused existing VN tests.
No shared fixtures, global configuration, Jobs authority, storage or UI edits.

- [x] Add an awaitable repository boundary for the complete integrity write and
  per-slot reconciliation. Both async missing-recipe paths await it; synchronous
  parent fanout retains its synchronous operation. Never transfer an active
  connection, cursor or caller transaction between threads.
- [x] Preserve cancellation precedence, exactly-once terminal counters,
  reservation release, rollback, approved bytes/outcomes and sibling activity.
  Retain Jobs legacy-activity callback semantics and inline instance state.
  Preserve private-memory/active caller-transaction identity with documented
  owner-thread fallback, as approved for async observations.
- [x] Prove RED/GREEN responsiveness while reconciliation blocks, both worker
  call sites, owned resource closure/error propagation and compatibility modes.
  Run only affected covering modules once; exact-tier/doc/type metadata for all
  added tests/helpers. Use project venv and Bandit; qualify baseline warnings.
- [ ] Freeze delta/report/evidence; independent spec/quality and scoped changed-
  contract review. Controller normal commit after review; no hook bypass.
- [ ] Push, individual tested reply/resolution and one full exact-new-head Qodo
  review. All seven CI/current strict dev/human summary gates precede merge.

**Ruling:** The read-only Task22 helper cannot own this write transaction.
Choose a cohesive thread-owned write boundary retaining established repository
semantics, not fragmented SQL offloads or a new queue/pool authority. This is
a correction to approved durability, not a new feature. Cost if wrong: bounded
ownership/reconciliation rework with regression evidence.

Task23 frozen four-file implementation,62source/evidence hashes verified by
implementer/Main/independent reviewer. Two actual REDassertionfailures ->14GREEN;
293coveringpassed0skips5summarywarnings200.01s. Main fresh14passed135deselected
0skips4summarywarnings11.07s; productionBandit0findings0errors. Ruff1exactbase
BLE001/testBandit18exactbaseB106, no additions; warnings retained/qualified.
Huygens and Lovelace CLOSED; SPEC/QUALITY/changed-contract PASS/noactionable
findings. Dedicated single-operation thread retains repo/callback/context/inline
semantics; owns and closes only its handle, drains cancellation through cleanup.
Memory/active-caller fallback remains synchronous, no universalasync claim.
Currentdev freshlya2826f unchanged; normal seven-file hooks/commit/push next.
AC5open/AC6pending; no reply/resolution/newreview/mergeattempt yet.
## Historical Review Wave: Task 22

Full Qodo review5325783656 completed11:35:55Z on exact head
a3f62da0a29eac3743cd184341d97206e028e238, acknowledgment5845933607.
Six new findings;39 threads/six unresolved, prior33 preserved. AC5 reopened.
No review pending. Current dev a2826f103f remains the base; no new rebase.

### Task 22: Resolve Six Fresh Exact-Head Findings

**Base:** a3f62da0a29eac3743cd184341d97206e028e238.
**Tracking:** TASK-13369. One coordinated fix wave, then independent task
spec/quality and scoped cross-contract review. Tasks1-21 stay frozen.
**Files:** VN_Assets/worker.py, DB_Management/VNAssetPacks_DB.py,
DB_Management/jobs_failed_requeue.py, core/exceptions.py, affected VN/Jobs
regression tests, VNAssetsWorkbench.tsx and its existing tests. No shared
fixture, global pytest configuration, Jobs authority or unrelated changes.

- [x] Verify finding4111251955: synchronous outcome query inside the async
  worker blocks the event loop. Use the smallest cohesive off-thread read
  boundary preserving connection ownership and transaction scope. Do not
  transfer cursors/transactions across threads or introduce a global executor.
  Cover loop responsiveness with a controlled blocked query and real existing
  SQLite repository replay/fencing behavior; inspect in-memory/thread-local
  semantics and close newly owned resources appropriately.
  Preserve private-memory and already-active caller-transaction modes using a
  narrow documented owner-thread fallback when offloading would change their
  connection/transaction identity. Verify normal file-backed worker wiring has
  no outer transaction; do not claim universal nonblocking database access.
- [x] Verify finding4111251964: recipe-count or missing-recipe integrity failure
  terminalizes only the batch. Add an atomic DB_Management operation that
  terminalizes surviving unfinished recipes, releases their reservation
  capacity, adjusts failed counters once and reconciles all affected slots.
  Preserve completed/approved bytes and outcomes, historical completed/failed/
  cancelled counters, cancellation and sibling active work. Never invent
  outcomes for missing rows or silently replace approved assets. Keep missing
  ledger-row ambiguity explicit; no unsafe deletion of orphan published bytes.
  Both worker integrity-failure paths must use this boundary. Cover partial
  fanout with reserved items, repeated admission, rollback, completed approval,
  cancellation and a sibling active batch. Existing execution/publication
  fences must reject late workers after terminalization.
- [x] Verify finding4111251958: use a centrally defined Jobs-specific exception
  for retry-index lock timeout, definition collision and verification failure.
  Preserve RuntimeError compatibility, existing safe messages and native DB
  error propagation; avoid broad catch/reclassification. Cover each named
  failure and the real existing owned PostgreSQL ensure tests through official
  isolated_test_environment. Do not copy/alter fixture lifecycle.
- [x] Verify finding4111251961: every newly added executable test in the two
  cited VN modules must have exactly one accepted tier, including inherited
  markers. Real SQLite/database concurrency/schema tests use integration;
  database-free tests use unit. Preserve asyncio/parametrize metadata. Prove
  actual public collection RED/GREEN without a new general policy engine.
- [x] Verify finding4111251962: add immediate nonempty concise docstrings to
  new concurrency/replay test doubles and their methods lacking them, including
  EmptyGeneratedFiles and BlockingFirstImageAdapter. Scope to PR-added code,
  not an unrelated rewrite; do not change executable behavior for docs alone.
- [x] Verify finding4111251966: after successful cancel clear the matching
  owner/pack pending receipt and matching in-memory operation key. Snapshot the
  pending key before await; conditional cleanup must not erase a newer receipt
  or unrelated pack/owner state. Failed cancellation retains its key. Cover
  ambiguous start + failed reconciliation + successful cancellation + next
  start with fresh key, reload, failed cancel and selected-pack/key races.
- [x] Run focused RED/GREEN, then affected VN/Jobs/frontend scope once, scoped
  Ruff/Bandit/TypeScript/ESLint as applicable. Preserve logs/XML and truthful
  counts/skips/warnings; prior broad matrices remain historical. Use project
  venv and existing official PG fixture; unavailable required PG is a failure,
  not a successful skip. No Python3.14/whole-repo-green claim.
- [x] Freeze report/diff/evidence; independent spec/quality and changed-contract
  review. Fix verified blocking review feedback through original implementer
  and scoped re-review (max3 failed attempts before reassessment). Controller
  commits normally after review with associated tracking/design/plan records;
  no hook bypass, no unrelated dirty files staged.
- [ ] Push normal reviewed change; reply individually with tested evidence and
  resolve only verified findings. Request one full review on the new head,
  then require all seven CI contexts/current strict dev/human summary gates
  before normal authorized match-head merge. No admin bypass or skipped passes.

Task22 frozen implementation10files/93source-evidence manifest entries verified.
Covering305backendpassed0skips37warnings251.52s (17officialPG included),
65frontendpassed0skips8.97s; public135collected101addedcasesexactlyonetier.
ProductionBandit0; scopedRuffonebaselineBLE001/testBandit18baselineB106 no
new findings. TypeScript/scopedESLint/compile/diff passed; earlier failures and
warnings qualified in report. Erdos and Gibbs closed; independent SPEC/QUALITY/
changed-contract PASS, no actionable findings. Main bounded integration16passed
0skips plus fresh TypeScript/ESLint/Bandit passed, Ruff samebaselineBLE001.
Dev a2826f unchanged. Normal scoped commit next; AC5 open until tested evidence
replies and AC6/external gates pending. No merge attempted.

Task22 locally complete: normal13-file commit/FFpush/GitHubverified head
88aefac2ba95c9f743d3e67b34fd55858479315a; all93hashes match aftercommit.
Applicable precommitcheckspassed; normalcommit no hookoutput, no stageclaim.
Individual replies4111392924/2972/3012/3039/3073/3262 at12:39:54-12:40:04Z,
six verified threads resolved. Paginated39threads0unresolved/no remainingpages,
all24conversationcomments inspected. ONEfullrequest5846349629 at12:41:02Z
on88aefac2 PENDING, busy5846351222 at12:41:18Z; no duplicate. Pushsummary
5836873877 at12:39:25Z0bugs0rules24historicalomitted NOTfullreviewcompletion.
Verification-onlybodyupdate humanparagraph/allothersections/Cubic preserved.
Actual54checks33queued21done/sevenrequiredabsent/noactionablefailure; skipped/
cancellednotpasses. AC5checked/AC6pending, OPEN/BLOCKED/no mergeattempt.
Alltaskneededagents/tests/shellsessionsclosed. Onlylocalintegrationnotesdirty;
no trackingonlypush duringpendingreview. Preserveworktree/evidence/main.

**Ruling:** These are corrections to approved durability contracts, not a new
feature. Keep the worker/read and integrity paths in one coordinated wave to
avoid overlapping ownership. Preserve RuntimeError catch compatibility through
subclassing. Review the frozen working delta before controller normal commit
so integration records cannot be staged mid-edit. Cost if wrong: a bounded
follow-up fix/review, not a new queue authority or data migration.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all verified PR 3016 generation durability findings without weakening the existing API or Jobs contract.

**Architecture:** Jobs owns leases; V1 recipe rows own execution fences and outcome state. Storage registration converges by owner and source reference, while submission recovery and cancellation remain transactional in the VN database.

**Tech Stack:** FastAPI, SQLite ChaChaNotes, AuthNZ SQLite/PostgreSQL, Jobs WorkerSDK, pytest, Next.js/Vitest/Playwright.

**Spec:** `Docs/Design/2026-09-25-vn-pr-3016-review.md`

## Global Constraints

- Preserve V0 batch behavior and existing public VN error codes.
- Keep per-user VN metadata in ChaChaNotes and file records in AuthNZ.
- Use Jobs as the only queue and lease authority.
- Do not promise exactly-once external model execution after an expired lease.
- Complete `TASK-13369` and PR #3016 review/CI gates before merge.

---

## Stage 1: Recover submission and cancellation
**Goal**: Keep failed parent enqueue recoverable and clear cancelled reservation capacity.
**Success Criteria**: Same-key retries use the original batch; cancellation preserves completed/failed outcomes and excludes remaining reservations from capacity.
**Tests**: Add failing service/API and repository tests for enqueue failure and cancellation after reservation; run focused tests red, implement queued-with-error recovery and transactional V1 cancellation, then run green.
**Status**: Complete

### Task 1: Parent enqueue recovery

**Files:** `tldw_Server_API/app/core/VN_Assets/service.py`, `tldw_Server_API/tests/VN_Assets/test_generation_jobs.py`.

**Interface:** `recover_generation_receipt(record, pack_id, jobs_manager)` reuses `create_enqueue_batch_job` with the batch's deterministic key. It clears `enqueue_error` only after receiving a parent Job ID.

- [x] Add a failing test that makes `create_job` fail once, then retries the same receipt and asserts one batch and one parent Job.
- [x] Run the focused pytest node and confirm the existing failed-batch behavior causes failure.
- [x] Keep `status='queued'` with `enqueue_error` after parent enqueue failure; do not reopen a batch with variant outcomes.
- [x] Run the focused test green and broader receipt recovery tests.

### Task 2: Transactional cancellation

**Files:** `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`, `tldw_Server_API/app/core/VN_Assets/service.py`, `tldw_Server_API/tests/VN_Assets/test_vn_asset_packs_db.py`.

**Interface:** `cancel_batch(batch_id)` atomically sets the V1 batch and outstanding recipe outcomes to `cancelled`; `count_items_for_generation(pack_id)` excludes cancelled reservations.

- [x] Add a failing repository test for a reserved hidden item and repeated cancellation; mixed outcomes remain.
- [x] Run focused pytest nodes red.
- [x] Add the transactional repository operation and route service cancellation through it; block failed/completed transitions from overwriting cancelled outcomes.
- [x] Run focused repository tests green and broader generation API cancellation tests.

## Stage 2: Fence and reconcile variant execution
**Goal**: Prevent concurrent/stale workers from publishing duplicate assets and recover post-registration handoff errors.
**Success Criteria**: A V1 claim is reserved before the adapter call; only its owner can save/publish; newer Jobs leases can recover; one source reference has one live registered file.
**Tests**: Concurrent delivery with barriers, lost-lease takeover, post-registration DB failure, and storage registration collision; run red/green per boundary.
**Status**: Complete

### Task 3: Recipe execution fence

**Files:** `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`, `tldw_Server_API/app/core/VN_Assets/worker.py`, `tldw_Server_API/tests/VN_Assets/test_generation_jobs.py`.

**Interface:** Recipe claim stores an attempt token and Jobs `lease_id`; `claim_variant` reserves an item and conditionally claims a planned or replaced attempt; `complete_variant` requires the current attempt token.

- [x] Add failing concurrent and stale-lease tests using two worker attempts and a controllable adapter barrier.
- [x] Run the focused pytest nodes red.
- [x] Add migration columns and compare-and-swap repository claim/publication methods; validate the current Jobs lease before claiming, saving, and completing.
- [x] Run the focused tests and existing replay suite green.

### Task 4: Idempotent storage handoff

**Files:** `tldw_Server_API/app/core/Storage/generated_file_helpers.py`, `tldw_Server_API/app/core/AuthNZ/repos/generated_files_repo.py`, `tldw_Server_API/app/core/VN_Assets/worker.py`, relevant AuthNZ migrations and storage tests.

**Interface:** VN generated-file registration converges on one live owner/source-reference record; losing file writes are removed. A post-reservation persistence exception remains retryable for source-reference replay.

- [x] Add failing tests for a registered file followed by VN item update failure and for competing registration attempts.
- [x] Run those tests red.
- [x] Make storage registration idempotent and cleanup-safe; classify post-reservation errors as retryable without recording `failed`.
- [x] Run storage, replay, and migration tests green on supported backends.

## Stage 3: API compatibility and review quality
**Goal**: Address Qodo findings 5-8 without changing the public error format.
**Success Criteria**: Helpers have docstrings, the test fixture is typed, replay file stat is offloaded, and structured VN failures preserve existing error codes.
**Tests**: API error-code regression plus Ruff, mypy/TypeScript where scoped, and VN tests.
**Status**: Complete

### Task 5: Scoped quality fixes

**Files:** `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`, `tldw_Server_API/app/core/VN_Assets/worker.py`, `tldw_Server_API/app/core/exceptions.py`, `tldw_Server_API/tests/VN_Assets/test_generation_jobs.py`.

- [x] Add an API regression asserting the existing stable VN error code and a worker test for nonblocking file-stat behavior.
- [x] Run focused tests red where behavior changes.
- [x] Add concise helper docstrings, annotate `tmp_path: Path`, offload path stat with `asyncio.to_thread`, and introduce a typed VN error carrying identifiers while keeping public `detail` unchanged.
- [x] Run tests, scoped Ruff with no new findings, and Bandit with zero production findings.

## Stage 4: Final verification and merge
**Goal**: Close every review thread, pass required checks, and merge the PR.
**Success Criteria**: `TASK-13369` records verification; no actionable Qodo comment or failing required check remains; PR merges into `dev`.
**Tests**: VN backend suite, frontend VN suite, TypeScript, ESLint, Ruff, Bandit, Chromium smoke, GitHub checks.
**Status**: In Progress

Final local verification: 361 VN tests passed; 177 storage/AuthNZ tests passed;
all 15 required PostgreSQL cases passed with the official fixture; 44 frontend
tests, TypeScript, scoped ESLint, and four Chromium smoke tests passed. Production
Bandit found zero issues. Ruff has only two unchanged baseline BLE001 findings.
The initial full VN run's three obsolete fake-interface failures were corrected
without weakening production validation. Independent re-review is clean and
additionally passed 31 focused backend regressions.

The final review also identified and fixed terminal V1 cleanup on existing
NO ACTION schemas, invalid-byte replay (including already attached metadata),
and a browser pending receipt stuck after a definitive missing-slot 404. These
boundaries have RED/GREEN regression coverage. Original Qodo replies and rebase
onto `3f909e13` are complete; all eight original threads are resolved. Fresh
exact-head review added the eight findings tracked below. Checks and merge
remain pending.

- [x] Run the full scoped verification matrix and self-review the changed diff.
- [x] Reply in each original Qodo thread with the corresponding fix or technical reasoning.
- [x] Verify the requester-provided Change summary remains in the PR body.
- [ ] Confirm branch is rebased on current `origin/dev`, all required checks pass, then merge through GitHub.
- [ ] Record PR merge and final test evidence in `TASK-13369`.

## Fresh Qodo Review: September 26

Review 5324270842 on `87b80818` adds eight findings (5-12). Original findings
remain resolved. Merge is gated on this addendum, independent review, and CI.

### Task 7: Core receipt recovery ownership

- [x] Move receipt claim/recovery/completion decisions to the core service;
  keep HTTP response validation and error mapping in endpoints.
- [x] Preserve all existing idempotency scopes and JSON response compatibility;
  add core-level recovery tests and run generation API regressions.

### Task 8: AuthNZ boundaries and isolation

- [x] Move new VN item locking/lookup SQL behind a DB_Management abstraction,
  without changing transaction ownership or quota accounting.
- [x] Make PostgreSQL durability tests use `isolated_test_environment`,
  delegating lifecycle to the existing official fixture; rerun SQLite/PG cases.

### Task 9: Replay integrity and failure classification

- [x] Verify SHA-256 when a persisted checksum is available, off the event loop;
  test same-length corruption at storage and worker replay boundaries.
- [x] Definitive loss/corruption is nonretryable, while transient filesystem
  errors remain retryable. Fail an unfinished fenced recipe once, freeing its
  reservation; never reopen a completed outcome or demote an approved asset.
- [x] Test missing registered bytes, completed review preservation, sibling
  progress, explicit regeneration, and transient I/O behavior.
- [x] Bind non-sensitive cleanup identifiers and preserve the traceback without
  logging exception messages/locals that may contain secrets.
- [x] Add accepted tier markers and parameter/return annotations to new storage
  test doubles; retain existing test behavior.

### Task 10: Re-review and Integration

- [x] Independent review of the fresh delta, scoped backend/Storage tests,
  official PG cases, no new Ruff findings and zero production Bandit findings.
- [x] Commit/push, evidence-backed replies on all eight new comments, request
  fresh exact-head Qodo review.
- [ ] Confirm that review and required CI pass, then perform the authorized merge.

**Ruling:** Do not adopt Qodo's suggested automatic reset/regeneration of a
completed variant. The approved design makes completed outcomes and review
decisions immutable on redelivery. Silent byte replacement under an approval
would change what was approved. Definitive integrity failure instead stops Job
retries and allows explicit regeneration as a new draft through existing APIs.
Transient infrastructure errors still retry, and unfinished outcomes fail only
under their current fence. If this policy is wrong, recovery needs an explicitly
designed asset repair workflow rather than a hidden redelivery mutation.

Fresh local verification: full VN suite 395 passed; final Storage plus generation
worker scope 227 passed; shared-fixture SQLite/PostgreSQL durability cases 30
passed with zero skips; AuthNZ boundary/repository unit scope 28 passed. Final
independent review passed 67 narrow cases and found no outstanding actionable
finding. The Python 3.14 filesystem-error suppression issue was fixed using
explicit stat; its two obsolete off-loop test probes were updated without
weakening assertions, and the previously failing random seed passed all 227.
Production Bandit on all eight fresh source files has zero findings/errors;
compileall and diff checks pass. Ruff has only the two verified baseline BLE001
warnings. Earlier frontend verification remains applicable (no fresh UI edits).

A repository-wide attempt stopped at an unrelated existing MCP flashcard test
with KeyError 'rows'; that test and producer/exporter are identical to dev. The
producer returns 'No flashcards to export' before the fake exporter is called.
The isolated test reproduces the failure; no repository-wide green is claimed.
Dev advanced to 59bd584503 with unrelated MCP sanitizer changes; final rebase,
fresh exact-head external review, checks, and authorized merge remain pending.

Integration update: rebased cleanly onto dev 59bd584503 and pushed head
9e5fb2fdfc9e77fb51745d72aa36924160ccc597. Range-diff shows all six patches
identical; changed VN source blobs are unchanged. Post-rebase 58 focused cases
passed and commit-stage pre-commit checks passed. All eight new findings have
individual evidence replies; paginated GraphQL confirms all 16 threads resolved.
Requester Change summary remains verbatim. Fresh full Qodo review requested once
in comment 5842776017. PR is OPEN/BLOCKED with CI queued; review, checks and merge
remain pending, and skipped admission jobs are not verification passes.

## Full Qodo Review on 9e5fb2fd

Review 5324400373 completed at 03:33Z with seven new inline findings. Existing
16 resolved threads remain preserved. AC5 is reopened; merge remains gated.

### Task 11: Shared Slot State and Visibility Contract

**Files:** VNAssetPacks_DB.py, VN_Assets/worker.py, focused VN repository/worker tests.

- [x] Reproduce cancellation during generation, first completion/failure while
  siblings or another batch remain active, and stale worker slot mutation.
- [x] Derive affected shared slot state under the repository write transaction
  from all active work and published items, preserving established review-state
  precedence. Reconcile cancellation, completion and failure atomically.
- [x] Make V1 generating admission validate current Jobs authority and the
  current attempt token under the write lock before any visible mutation;
  retain V0 behavior, terminal outcomes, cancellation and approval preservation.
- [x] Document item_is_unpublished, its recipe predicate and propagating errors.
- [x] Run RED/GREEN and independent spec/quality review with no new Bandit findings.

Task 11 first independent review found a mixed-version compatibility gap:
V1 reconciliation ignores active V0 deliveries without recipe rows. Fix round
one must reproduce blocked legacy generation overlapping V1 cancellation and
preserve its active signal, including the reverse terminal transition, without
adding another lease authority or changing legacy generation/counter contracts.
Task 11 remains incomplete pending that fix and scoped re-review.

Fix round one reproduced four real blocked-adapter failures, not just a
metadata probe. Ruling A permits a bounded owner-scoped Jobs-read callback
for exact legacy display, with narrow service constructor/wiring ownership
handed from the now-frozen Task 12 scope. No receipt/admission code changes,
new persisted lease authority or ambiguous pending-as-generating fallback.
Execution-scoped inline display must clean up in finally and state its limits.
An append-only shared legacy Jobs-read helper in VN_Assets/jobs.py is permitted
to avoid duplicated readers or service/worker circular imports; Task 12 parent
recovery and admission logic remain frozen and outside this fix's ownership.

Task 11 fix round two addresses two confirmed handoff regressions. Exclude only
the exact finishing delivery and lease from its final display check, preserving
live siblings and replacement leases. A post-outcome display failure must not
replace a successful legacy generation return or its original error/cancellation
classification; log safe structured identifiers/stack information instead.
No additional model retry, job failure for a successful outcome, hidden queue
or persisted display authority. A transient display outage may require later
normal reconciliation; this does not authorize regeneration of committed assets.
The display exception must also be sanitized before the existing database
rollback logger observes it; test the full logging sink, not only the worker
diagnostic. Do not expand this fix into general ChaChaNotes logging changes.

Task 11 is locally complete. Archimedes approved both fix-round handoff findings,
clock delegation and the full-sink logging boundary. The final owned file passed
79 cases; Main's full integrated VN suite passed 509 cases with no skips.
Main's four-file production Bandit has zero results/errors; Ruff contains only
the two verified baseline BLE001 catches. External evidence replies remain pending.

### Task 12: Parent Job Health Recovery

**Files:** VN_Assets/service.py, VN_Assets/jobs.py, a narrowly scoped Jobs facade
method with a DB_Management query helper, and dedicated recovery tests.

- [x] Reproduce unfinished receipt recovery for a missing parent and exhausted
  parent after partial fanout using real Jobs semantics, not only create mocks.
- [x] Consult owner-scoped Jobs state. Restore incomplete active fanout through
  deterministic create or supported atomic retry operations; never reopen a
  terminal VN batch, revive deliberate admin cancellation, or requeue healthy
  completed full fanout. Keep failed recovery receipts unfinished.
- [x] Preserve completed receipt snapshot semantics, quotas, pause/drain controls,
  original recipes/outcomes, and persist the authoritative parent identity.
- [x] If existing Jobs primitives cannot requeue exhausted failures safely, add
  one explicit owner-scoped failed-job admission method preserving canonical
  identity, normal admission rules and counters; no general admin API change.
- [x] Count explicit retry-admission events in the existing per-minute creation
  quota through a DB helper, so interleaved creates/retries cannot bypass it;
  verify both backends, concurrent replay and rollback without double charge.
- [x] Verify concurrent recovery, wrong owner/type/queue and failure boundaries;
  run RED/GREEN plus independent review.

Task 12 is locally complete. Pascal approved lease health, first-completion
receipt CAS and failed PostgreSQL concurrent-index recovery. A genuine writer
timeout leaves an invalid index; ensure now verifies its canonical definition
and readiness, repairs only its own invalid index and rejects foreign collisions.
Snapshot-free advisory try-lock acquisition avoids the reproduced concurrent
partial-index deadlock and uses configured positive timeouts or a 30-second
fallback. Main's final follow-up passed 82 Jobs cases with required PostgreSQL
and zero skips; the broader admission/quota/migration matrix passed 215 cases
before the final index-only changes. Whole-delta review and external gates remain.

### Task 13: Embedded Runtime Test Contracts

**Files:** tests/AuthNZ/integration/test_vn_generated_file_idempotency.py and focused AST regression.

- [x] Add explicit types to every executable embedded-script helper, including
  variadic callbacks; retain runtime behavior and required shared PG isolation.
- [x] Parse the scripts to validate annotation coverage; compile and run all
  SQLite/required PostgreSQL cases without skips, plus scoped quality checks.

Task 13 is locally complete: two valid RED failures, four AST unit cases and
all 15 SQLite plus 15 required PostgreSQL cases passed with no skips. Averroes
independently approved spec/quality; annotations and docstrings alone changed
the scripts and fixture lifecycle is unchanged. Main independently ran all
four AST cases. Existing warning counts are not a warning-free claim; evidence
classification is recorded separately before integration.

### Task 14: Persisted Retry Identifier Validation

**Files:** frontend lib/vnAssetIdempotency.ts and its existing unit/workbench tests.

- [x] Reproduce zero/negative persisted retry slot IDs reaching reload recovery.
- [x] Reject non-positive or unsafe IDs, remove invalid stored state and verify
  no retry API call; retain positive IDs, owner scoping and valid key recovery.
- [x] Run focused frontend tests, TypeScript, scoped ESLint and independent review.

Task 14 is locally complete: six behavioral RED failures, then 76 VN frontend
tests passed with no skips; package and owned-file TypeScript and scoped ESLint
passed. Nash independently approved spec/quality with no actionable defect.
Baseline Node warning and unrelated whole-app lint warnings are qualified in
the report; zero-line Bandit is not TypeScript security assurance. Main also
independently passed all 76 VN frontend cases, TypeScript and scoped ESLint.
The external exact-head gates remain pending.

### Task 15: Whole Delta and External Integration

- [x] Independent task and whole-delta review, scoped backend/frontend matrix,
  production Bandit, no new Ruff findings and normal commit checks.
- [x] Commit/rebase/push, seven individual evidence replies and verified thread
  resolution. Request one new full exact-head Qodo review, then await CI.
- [ ] Merge only after review/required CI/human summary/current dev gates pass;
  finalize Backlog and pause the heartbeat on verified merge or closed PR.

Final local review is approved. Raman's sole final P2 (a published legacy
variant hiding a live replacement delivery) was reproduced with real Jobs and
blocked adapters, then fixed using an opaque exact-delivery fingerprint in
existing V0 item provenance. Unknown or mismatched historical provenance cannot
hide active work; raw lease tokens are not persisted and model inputs are
unchanged. Scoped re-review found no new actionable issue. Main's frozen full
VN suite passed 520 cases, zero skips, 10 existing warnings, in 307.06 seconds.
Final three-file Bandit has zero findings/errors; expanded manager scope has
one independently confirmed baseline B608, not a zero-findings claim. Ruff
has only two verified baseline BLE001 catches; compileall and diff checks pass.
Final normal commit-stage hooks passed on all 23 owned files; inapplicable
YAML/TOML/wizard hooks were skipped, not counted as passes. Push/replies remain
integration steps. Live dev
remains 59bd584503 and PR head remains 9e5fb2fd before this commit; no merge
readiness is claimed while exact-head external review and CI are pending.

Integration: normal fast-forward push verified GitHub head
4666d4994b13b32e5fda8f4642cef9ca60f1e48f. Fetched dev remains
59bd5845038342013a2d84d0130f6164f14b54fd and is an ancestor of this head;
no additional rebase was needed. Seven individual evidence replies are posted
and resolved; paginated GraphQL verifies 23 threads, zero unresolved, no
remaining thread/comment pages. The human summary remains verbatim. Full
Qodo review requested exactly once in comment 5843635886 at 05:46:59Z.
Edited summary 5836873877 reports zero bugs/rule violations after replies,
but is not yet a completed full review of this head. Actual check runs show
33 queued, no new actionable failure; required gate contexts are not yet
present. Skipped admission and cancelled audit runs are not passes. No merge
attempt; final integration records remain local pending merge finalization.

**Ruling:** Existing Jobs retry_now_jobs only accepts failures with retries left
and is not owner-scoped; deterministic create replays the same dead row. A
bounded explicit owner-scoped requeue admission method is required for exhausted
parent recovery, rather than raw SQL from VN or new random retry keys. It must
retain canonical identity, enforce admission/counters and never revive deliberate
cancelled/quarantined work. Cost if wrong: a small Jobs facade/query interface to
revise, not a second queue or hidden administrative bypass.

Task 12's bounded quota integration may touch the existing SQLite/PostgreSQL
admission quota helpers only to include the new explicit retry-admission events.
Their legacy admin retry API remains unchanged. This shared boundary requires
focused real-backend creation/retry/rate/rollback/counter coverage and independent
review; SQL for the new admission/count queries stays in DB_Management.

Task 12 fix round one adds a narrowly scoped partial retry-admission index on
both backends through established fresh/upgrade migrations. Independent review
found the new shared rate query otherwise scans all historical job events,
including when no retry events exist, inside admission locks. Index keys are
domain, owner_user_id and created_at, with the retry-admission event predicate.
Keep new SQL in DB_Management and test fresh/upgrade/idempotent migration,
SQLite indexed query plans and required PostgreSQL index/query support. No
performance-outage claim or broader admission refactor is warranted.

Consolidated Task 12 review also reproduced an expired final processing lease
accepted as healthy, and a delayed original completion overwriting a committed
recovery snapshot. Fix round one must use supported Jobs-authoritative lease
health (or fail closed pending normal Jobs reconciliation) and transactionally
conditional receipt completion. Permit a narrow read-only Jobs health method
if no supported primitive exists, using the Jobs clock rather than a separate
VN lease authority. DB completion method ownership is handed to Task 12;
Task 11's constructor/activity/slot helpers remain disjoint. Preserve payload
conflicts, owner/scope boundaries and completed response snapshots.

Effective dev rulesets require backend-required, security-required,
coverage-required, frontend-required, e2e-required, container-build-check and
frontend-license-policy/trusted/dev. Strict base integration applies, and only
the merge method is allowed. The legacy protection API returns 404 because
these controls are ruleset-based, not absent. Never use admin bypass.

## Full Qodo Review on 4666d4994b

Review 5324805354 completed at 05:50:11Z with four new findings. Prior 23
resolved threads remain preserved; AC5 is reopened. Do not redispatch earlier
completed tasks. One bounded implementer handles this fresh four-finding wave.

### Task 16: Legacy Review Precedence and Public Contracts

**Base:** 4666d4994b13b32e5fda8f4642cef9ca60f1e48f.
**Files:** VNAssetPacks_DB.py, VN_Assets/worker.py, core/exceptions.py,
DB_Management/jobs_failed_requeue.py, and narrowly scoped VN/Jobs tests.
Jobs/pg_migrations.py may change only if the actual migration defect reproduces.

- [x] Reproduce a failed legacy delivery after a completed V1 item on the same
  required slot is approved. Preserve approved/reviewing/skipped review
  precedence and readiness, current active/queued precedence, stronger derived
  failures and empty legacy terminal fallback; add actual worker mixed-version
  controls without changing outcomes, counters, approvals or model calls.
- [x] Centralize LegacyDisplayReconciliationError in core/exceptions.py and
  update DB/worker imports without altering safe rollback messages, retained
  internal type/traceback, SDK disposition or logging redaction. Regression
  coverage must use the centralized class and existing full-sink controls.
- [x] Verify the alleged PostgreSQL upgrade ordering on a real Jobs database
  with existing jobs but absent job_events, using the official Jobs fixture.
  Run the unmodified actual ensure_jobs_tables_pg entry point: its base DDL
  appears to create job_events first. If it passes, retain production behavior
  and add regression-backed rebuttal evidence; do not manufacture RED by
  replacing current DDL with an old script. If a real failure reproduces,
  minimally correct required ordering and fail-closed behavior with RED/GREEN.
- [x] Expand all three public retry-admission helper docstrings to describe
  parameters/shapes, supported backend/executor, transaction and connection
  ownership, return value, side effects and actual exceptions, including
  callback/policy/driver failure propagation. Keep runtime code unchanged.
- [x] Run bounded affected VN, centralized-exception and required real PG
  migration/index tests once, scoped Ruff/compileall/Bandit/diff checks and
  self-review. Report precise RED/GREEN, including any non-reproduced finding;
  no frontend, storage matrix, all-Jobs, whole-repo or native3.14 reruns.
- [x] Independent task spec/quality review with no actionable findings.

Task 16 is locally complete. Helmholtz independently approved spec compliance
and code quality with no actionable findings. The frozen affected matrix passed
309 cases with required official PostgreSQL and zero skips; Main's final VN and
central exception scope passed 554 cases, zero skips, 10 warnings, 327.30s.
Post-run source/test hashes match the freeze. Main production Bandit on four
files has zero findings/errors; Ruff has only the unchanged worker BLE001.
Compileall, diff checks and all applicable normal hooks on the eleven owned
files pass; inapplicable hooks and existing hook-stage warnings are qualified.
The PG allegation did not reproduce through actual current migration; its
production bytes remain unchanged. Final interaction review and external
integration gates remain pending, so this is not merge readiness.

**Rulings:** A fallback cannot override an existing published review state or
skip/stronger failure; actual legacy failure still supplies an empty terminal
outcome. Verify the helper's established failure-versus-cancellation precedence
with focused controls rather than rewriting the state machine. The PG claim
is provisional until a real pre-events installation exercises current DDL.
Changing proven-correct migration ordering solely to match a bot suggestion is
not required. Cost if wrong: a small predicate or migration correction, not
silent approval regression or unnecessary shared-schema behavior change.

### Task 17: Fresh Integration Gates

- [x] Independent final review of only this fresh wave and its interaction
  boundaries; no duplicate whole-branch rediscovery of completed work.
- [x] Main affected verification, normal hooks, safe/current dev integration,
  commit/push, four individual evidence replies and verified resolution.
- [ ] Request one full exact-head Qodo review, pass all required checks and
  human summary gate, authorized normal merge, truthful finalization/pause.

Harvey's final fresh-wave review approved local spec/quality with no actionable
findings or unresolved named risks. It independently inspected the unchanged
shared-work/readiness, rollback/SDK, PostgreSQL ensure/index and native helper
ownership boundaries and retained evidence, without duplicate suites. Excluded
whole-branch/other platform matrices, global infrastructure logging, exhaustive
historic PG versions and exactly-once model execution are unchanged qualified
limits, not claims added by this fix. External gates remain pending.

Integration: normal commit/push produced exact GitHub head
83c6a451cc8f0c430e2f058d60c15c612c03a436 on unchanged dev59bd584503.
All four new findings have individual tested fix/rebuttal replies and are
resolved; paginated GraphQL verifies 27 threads, zero unresolved and no
remaining review/thread/comment pages. Human summary remains verbatim and
only Verification was updated. One full exact-head Qodo request5843969815
posted at 06:38:50Z, pending. Edited summary06:36:16Z still showed one PG
allegation after three resolved findings; it is not a completed review of
this head. Actual55checkruns33queued22completed, no actionable failure;
all seven required contexts absent. Skipped/cancelled runs are not passes.
No merge attempt. Integration records stay local pending true finalization.

## Full Qodo Review on 83c6a451cc

Review5324956395 completed06:41:45Z. Prior findings cleared; one new fixture
lifecycle rule finding4110480430/PRRT_kwDOL1aGf86mObdo remains. AC5 reopened.

### Task 18: Shared PostgreSQL Isolation for Owned Jobs Tests

**Base:** 83c6a451cc8f0c430e2f058d60c15c612c03a436.
**Files:** tests/Jobs/conftest.py, test_job_retry_admission_index.py,
test_failed_job_requeue_admission.py, narrowly scoped fixture regression tests.
All production code and unrelated test modules are frozen.

- [x] Verify both explicit jobs_pg_dsn and autouse _pg_jobs_db_url routes; add
  failing isolation/identity guards that expose alternative allocation.
- [x] Add the smallest opt-in Jobs adapter for the PR-owned modules delegating
  database lifecycle to isolated_test_environment via the existing safe bridge.
  Preserve legacy Jobs routes outside this opt-in, SQLite no-PG allocation,
  actual migration and native transaction/error assertions. Do not add another
  database creator, raw DSN from an unrelated environment or manual cleanup.
- [x] Verify wrapper/connection database identity against the shared fixture,
  real required PG absent-events and native callback/driver controls, plus the
  two affected modules once, zero skips. Cover no alternate autouse allocation,
  connection cleanup and legacy path preservation without broad unrelated suites.
- [x] Run scoped Ruff/compileall/Bandit/diff checks and self-review. All new
  fixtures/helpers are typed and documented, accepted test tiers preserved.
- [x] Independent spec/quality and final scoped interaction review of fixture
  ordering/lifecycle/global selection, with no actionable findings. No duplicate
  review of the unchanged production branch or already completed VN matrix.
- [ ] Main integration verification, normal commit/push, individual tested
  reply/resolution, one full exact-head Qodo request, required CI/current-dev/
  human gates, authorized normal merge and truthful task finalization/pause.

**Ruling:** The existing Jobs fixture is per-test but does not satisfy the
required shared AuthNZ lifecycle. Correct only this PR-owned fixture chain,
including autouse routing, rather than migrating the entire Jobs suite.
Cost if wrong: a small fixture adapter correction, not production or global
test lifecycle changes. Prior verification remains real evidence for its old
fixture, not proof that the shared lifecycle was used.

Task 18 implementation is frozen. The substantive affected matrix passed 96
cases without skips, including 54 shared-fixture database create/drop pairs;
three historical-route controls retain the old lifecycle. A subsequent
module-local registration-only correction is proven unchanged by whole-module
AST comparison apart from pytest_plugins. Each owned module then passed its
standalone default-invocation PG control without an extra plugin flag; six
unchanged Logging isolation controls also passed. Main's final combined normal
invocation passed 15 cases, zero skips, 46 warnings, 35.09s. The earlier 96
matrix is substantive evidence, not a byte-identical final-file matrix.

All applicable normal seven-file hooks passed; inapplicable hooks were skipped.
The final 16-file manifest matches, with no production/shared-fixture/global
configuration change. Scoped Ruff retains exactly five verified baseline
Jobs-conftest diagnostics. Test-scope Bandit retains one verified baseline B608
and no errors or new findings; no zero-total or warning-free claim is made.
The one failed early probe's fixture-owned disposable database was cleaned
through the existing official helper and verified absent, with evidence retained.
Independent fixture spec/quality/final-interaction review and external gates
remain pending; no new push, thread resolution or merge is claimed yet.

Task 18 is locally approved after one scoped fix round. Independent review
found historical PG controls bypassed the Jobs-disabled collection gate; a
function-local jobs marker now preserves that gate without early shared
allocation. Actual selection RED/GREEN, deliberate default-gate skips before
fixture setup, and enabled native historical PG3 passed without skips cover
the change. Main reverified the disabled gate and applicable changed-file
hooks. Changed-guard Ruff/Bandit are clean; other test-file baselines remain
qualified. The prior matrix/default probes are pre-marker evidence, preserved
by whole-module AST comparison apart from the decorator. All 16 freeze hashes
match. Anscombe's scoped re-review approves spec, quality and final fixture
interaction with no new actionable finding. Production behavior and previous
branch reviews are unchanged; external integration gates remain pending.

Task 18 integration: normal commit/push bf319f48e4513b15958316c06dc96eb5d8e9ffe8
on unchanged dev59bd584503; all16 freeze hashes verified after commit. Individual
evidence reply4110596431 posted07:35:49Z and finding4110480430 resolved.
Paginated GraphQL28 threads0unresolved with no remaining pages. Edited Qodo
summary07:35:26Z says0bugs0rules after replies, with13 historical omissions;
this is not a completed full new-head review. One full /agentic_review request
5844316017 at07:37:42Z is pending. Human summary remains verbatim; only the
Verification section was updated. Exact-head55 checks33queued22completed,
no actionable failure; seven required contexts absent, skipped/cancelled not
passes. PR OPEN/BLOCKED, no merge attempted. AC5 checked; AC6/finalization pending.

## Full Qodo Review on bf319f48e4

Review5325085624 completed07:41:07Z with three new test-only findings;
prior28 threads remain resolved, AC5 reopened, no review request pending.

### Task 19: Narrow Shared Fixture Registration

**Base:** bf319f48e4513b15958316c06dc96eb5d8e9ffe8.
**Files:** the three owned Jobs modules, a narrow test-only fixture bridge,
and bounded registration/routing regression controls. Production, Jobs conftest,
AuthNZ conftest, existing full bridge and global config are frozen.

- [x] Reproduce full-plugin autouse leakage in actual fixture selection before
  the fix; verify shared fixture identity and its explicit dependencies.
- [x] Export only the original isolated_test_environment fixture through a
  narrow bridge and update all three module registrations. Do not duplicate
  lifecycle, register AuthNZ conftest or globally change existing plugins.
- [x] Convert the three historical routes to typed database-free unit controls
  exercising the existing fixture bodies with sentinel resolution. Prove no
  alternative/shared database is instantiated; preserve explicit, autouse and
  environment-override selection assertions. Remove redundant print diagnostics.
- [x] Verify standalone normal invocation, native shared-PG identities and
  cleanup, SQLite non-allocation, unrelated Jobs fixture selection and coexistence
  with normal AuthNZ collection. Run the affected fixture/Jobs scope once under
  required PG, plus bounded defaults; no repeated VN/storage/frontend matrices.
- [x] Scoped checks and independent spec/quality/final fixture interaction review.
- [ ] Normal integration, three individual evidence replies/resolutions and one full
  exact-head Qodo request; CI/current-dev/human gates and merge still required.

**Ruling:** The Task18 full bridge is discovery-safe but not selection-neutral.
Replace only its new registrations with the narrow original-fixture export.
Historical native controls were valid evidence, but the new route-selection
tests must no longer allocate the alternate lifecycle; unit sentinels suffice
for that boundary while owned SQL assertions remain real shared-PG tests.
Cost if wrong: a bounded fixture-export or unit-test correction, not production
changes or weakening of native migration/transaction coverage.

Task19 is locally approved: actual leakage RED1failed2passed; the one affected
required-PG run99passed1namespace-harness-failure remains explicitly not fully
green. Correcting that new assertion to inspect actual fixture markers passed
five covering cases in each import order, without repeating the99 passing
cases. Final standalone nativePG3, disabled units8 and coexistence7 passed;
54 matrix plus3 standalone fixture database names had matching drops and were
verified absent. Main final normal-default combined integration15passed,
zero skips,14warnings33.71s. The57-name catalog observation does not include
Main's additional combined-run names; normal fixture teardown applies there.

Ramanujan independently approved spec, quality and final scoped fixture
interactions. Original native bodies/shared fixtures/config remain unchanged;
all63 source/evidence hashes match after Main's verification. Applicable normal
eight-file hooks passed, scoped Ruff5files clean, Bandit retains only the
byte-identical B608 baseline and no errors/new findings. Warning noise and
broader collection/version permutations remain qualified, not pristine or
whole-suite claims. Both agents and all covering sessions are closed. Normal
commit/push and individual external evidence replies are next; merge gates pending.

Task19 integration: normal commit/push a003fe395eb3090c666e47a946e09f581c7bb7d5
on unchanged dev59bd584503, all63 freeze hashes match after commit. Three
individual evidence replies4110677065/4110677221/4110677407 posted08:14:22-31Z;
paginated GraphQL31threads0unresolved with no remaining pages. Summary08:13:44Z
says0bugs0rules16historicalomitted after push, not a completed full new-head
review. One full request5844537850 at08:16:10Z is pending. Human paragraph
remains verbatim; only Verification changed. Exact-head54checks33queued21done,
no actionablefailure, seven required contexts absent; skipped/cancelled not
passes. OPEN/BLOCKED, no merge attempted. AC5 checked; AC6/finalization pending.

## Full Qodo Review on a003fe395e

Review5325161826 completed08:18:18Z, ack5844550270. Prior31 threads remain
resolved; one new testability finding4110684831 is verified, AC5 reopened.
No review request is pending. Required CI still has33 queued checks and the
seven gate contexts absent; no merge attempted.

### Task 20: Observable Fixture Resolution Guards

**Base:** a003fe395eb3090c666e47a946e09f581c7bb7d5.
**Files:** test_retry_admission_fixture_registration.py and
test_shared_retry_admission_pg_fixtures.py only, plus a narrowly owned test
helper if demonstrably necessary. Production, fixture implementations,
original bridge, native Jobs modules and global configuration are frozen.

- [x] Replace private fixture-manager/FixtureDef inspection and direct
  fixture-wrapper calls with bounded probes through normal pytest resolution.
  Keep actual AuthNZ-reset/event-loop pollution negative controls, explicit
  shared lifecycle ownership and no unexpected PostgreSQL allocation.
- [x] Exercise historical explicit/autouse/environment routing using normal
  fixture resolution and sentinel database/I/O boundaries, not a second DB
  creator or copied fixture body. Cover shared env-bypass rejection before
  normal setup; preserve real native shared-PG identity and closed connections.
- [x] Prove negative controls fail for the actual forbidden behavior and pass
  after restoration, with any harness failures qualified. Run only changed
  guards, one representative owned native-PG case and bounded coexistence;
  do not repeat the previous completed broad matrices.
- [x] Scoped checks, source-boundary preservation, frozen evidence package and
  independent spec, quality and final fixture-interaction review.
- [ ] Normal commit/push, individual tested reply/resolution, one full exact-head
  review and required CI/current-dev/human gates before authorized normal merge.

**Ruling:** These guards protect observable registration, isolation and DSN
selection, not pytest's private representation of those behaviors. A bounded
normal-resolution sentinel harness preserves their sensitivity without calling
.__wrapped__, request._fixturemanager, FixtureDef._autouse or private pytest
fixture-marker APIs. Sentinel schema I/O remains a routing unit boundary;
existing actual migration/transaction tests stay native and unchanged. Cost
if wrong: a small guard correction, not a new lifecycle or production change.

Task20 implementation is frozen. Worker final scope16passed0skips12warnings
40.25s, Jobs-disabled units10passed3deselected; nested negative controls retain
their actual assertion/setup-failure sensitivity and are not added to outer
totals. Initial config-path, random loop-order and native seeder-order harness
failures are qualified, diagnosed and corrected. Main's final admission-first
normal required-PG integration16passed0skips12warnings42.46s, exit0. Applicable
five-file normal hooks passed; no-file hooks skipped, existing warnings retained.
Main Ruff/Bandit/compile/diff checks on two touched guards passed, Bandit zero
findings/errors. All268 frozen hashes match after Main's verification/hooks;
production/fixtures/native admission/index/globalconfig remain unchanged.

The failed seeder's disposable database tldw_test_44e59370 was verified owned
and inactive, cleaned only through the existing official helper, and freshly
verified absent with closed observer. No other DB or lifecycle was touched.
Independent spec/quality/final interaction review remains pending; no new push,
external evidence reply/resolution or merge is claimed yet.

Task20 Mencius independent spec/quality/final scoped interaction PASS;
no actionable findings. Audited all268 hashes, archive/actual-delta identity,
outer and nested outcomes, native and sentinel setup/teardown, Main's final
different-admission probe and exact-name cleanup without repeating suites.
Named limits and prior failed attempts remain qualified. Reviewer closed;
all task-needed agents/tests/shell sessions closed. Dev59bd584503 remains
unchanged ancestor. Normal commit/push and evidence reply next; external
exact-head review, CI/current-base/human gates and final merge still pending.

Task20 integration: normal five-file commit/push produced GitHub-verified
f8d4109600d52c02f5c0a9df93a1e50a7fc83726 on unchanged dev59bd584503.
All268 hashes match after commit. Individual evidence reply4110769088 at
08:56:24Z posted and thread resolved; paginated GraphQL32threads0unresolved,
no remaining review/thread/comment pages. Prior31 findings remain resolved.
All17 conversation comments inspected for new/edited state; edited summary
08:55:58Z says0bugs0rules17historicalomitted after push, not a completed full
new-head review. One full request5844787105 at08:57:21Z is pending, with busy
ack5844788046; do not duplicate. Only Verification was updated; human paragraph
verified verbatim afterward. Exact-head55checks33queued22done, no actionable
failure, seven required contexts absent; skipped/cancelled are not passes.
OPEN/BLOCKED, no merge attempted. AC5 checked; AC6/finalization pending.

External follow-up09:11Z: Qodo busy comment5844788046 was edited09:09:35Z
to an explicit service-side failure; request5844787105 did not complete a
full review. No new review/inline findings,32threads remain resolved, no
remaining pages. Following Qodo's manual-retry instruction, ONE retry request
5844895225 posted09:12:33Z on unchanged f8d4109600 is pending. This is the
first service-side retry, not a duplicate pending request. Required CI remains
55runs33queued22done/no actionablefailure and seven contexts absent. No source,
new head, merge attempt or completion claim. Do not duplicate the retry.

## Full Qodo Review on f8d4109600

Retry5844895225 completed review5325352669 at09:16:30Z, ack5844925279.
Prior32 threads resolved; new finding4110861531 verifies generated probes lack
accepted tier labels. AC5 reopened, no review request pending, required CI
still queued and no merge attempted.

### Task 21: Classify Generated Probe Tests

**Base:** f8d4109600d52c02f5c0a9df93a1e50a7fc83726.
**Files:** the same two fixture guards only. Runtime bodies, fixtures, plugins,
native SQL modules, production and global configuration are frozen.

- [x] Prove through actual public pytest collection that generated test items
  currently lack exactly one accepted tier; retain a failing classification
  check before the label-only fix, without database allocation.
- [x] Add unit classification to every generated database-free test; retain
  pg_jobs solely as an additional routing marker. Register unit in the local
  probe ini to avoid unknown-marker warnings. No new general policy engine.
- [x] Verify actual collected tiers and one bounded changed-unit run, retain
  negative-control sensitivity, and prove executable bodies/routing/native
  SQL remain unchanged apart from test metadata. No repeated PG/VN/Storage/UI.
- [x] Scoped checks, frozen report/evidence and independent spec/quality/final
  marker-interaction review; bounded Main integration and normal hooks.
- [ ] Normal commit/push, individual evidence reply/resolution, one full exact-
  head review and all external CI/current-dev/human gates before normal merge.

**Ruling:** Classification applies to subprocess-generated executable tests,
not just their outer wrappers. These probes perform no database I/O, so unit
is their accepted tier and pg_jobs is only routing metadata. Add labels and
local registration rather than refactoring the established fixture harness.
Cost if wrong: a few marker corrections, not a new lifecycle or behavior.

Task21 implementation frozen: actual public collection RED14missing tiers,
GREEN14items exactly unit with routing markers retained. Worker changed-unit
run10passed3deselected5warnings17.32s; nested9pass3expectedfail2expectederrors
are separate outcomes, not added to outer counts. Source preservation proves
only six decorators, ini registration and local literal wrapping changed;
runtime/generated bodies, fixtures and native SQL remain unchanged.

Main final covering generated-probe selection6passed5warnings17.86s, exit0;
applicable normal five-file hooks passed, no-file hooks skipped. Scoped Ruff,
compile/Bandit/diff checks passed, Bandit zero findings/errors. All203 hashes
match after Main verification/hooks; production/native/fixture/config diff
unchanged. No PG lifecycle was allocated or connected; existing import-time
temporary SQLite initialization and warning/formatter baselines are qualified,
not a literal zero-filesystem-side-effect claim. Independent Noether review
pending; no push/reply/resolution/merge yet.

Noether independent Task21 spec/quality/final marker-interaction PASS, no
actionable findings. Audited all203 hashes/archive/actual patch, actual
RED/GREEN collection records and exact old-versus-new imports/bodies. Inherited
warning/formatter/SQLite side effects remain qualified, no suppressed behavior.
Reviewer closed; all task-needed agents/tests/shell sessions closed. Fresh dev
59bd584503 unchanged ancestor. Normal five-file integration and individual
evidence reply next; exact-head Qodo/CI/current-base/human/merge gates pending.

Task21 integration: normal five-file commit and FF push produced GitHub-verified
2450adb17888b67b1c393581b01b011483e7cbec; dev59bd584503 remains unchanged.
All203 hashes matched after commit. Explicit applicable normal five-file
pre-commit checks passed; commit-stage hook execution is not claimed because
the normal commit produced no hook output. No bypass was used.
Individual tested reply4111014500 posted09:52:38Z. Fresh paginated inventory
33threads0unresolved, no remaining review/thread/comment pages; the latest
thread was already auto-resolved after push. All20 conversation comments
inspected, including edited summary09:46:09Z0bugs0rules18historicalomitted,
which is not a completed full new-head review.
Only Verification updated; human paragraph and all other body sections
preserved and freshly verified. ONE full request5845250951 at09:53:39Z on
2450adb is pending, with busy acknowledgment5845252770; do not duplicate.
Exact-head54checks33queued21completed/no actionablefailure, seven required
contexts absent. Skipped/cancelled are not passes. OPEN/BLOCKED, no merge
attempt. AC5 checked through official CLI after the read-only MCP stalled;
AC6/finalization pending. All task-needed agents/tests/shell sessions closed.
Only local integration records retained; no tracking-only push.

09:56 external check: full request5845250951 completed with Qodo exact-head
acknowledgment5845262987 at09:55:07Z. Busy5845252770 was removed (fresh404);
summary5836873877 updated09:55:04Z with exact2450adb footer, zero bugs/rules,
and visible historical findings resolved/dismissed. No new formal Qodo Review
object was emitted for this zero-finding run; the terminal request/ack sequence
establishes completion, not merely the edited push summary.
Fresh paginated inventory33threads0unresolved/no remaining pages/no new inline
feedback; all20 conversation comments inspected. No review pending; no repeat
request on unchanged head. Dev ref remains59bd584503 and human summary verbatim.
Actual exact-head55checks33queued22completed/no actionablefailure; all seven
required contexts absent. OPEN/BLOCKED/no merge attempt. AC6 remains pending.
No source, commit, push, rebase, agents or test reruns; retain local records.

## Current Dev Integration: a2826f103f

11:26 heartbeat found dev advanced through unrelated VZ startup-drill PR3017.
No overlap with owned source/tests, fixtures, global config or CI configuration.
Clean rebase12commits produced a3f62da0a29eac3743cd184341d97206e028e238 on
a2826f103f02a67f57adb40ed048dbfa2ecfc6e5; range-diff all12 patches identical,
owned source/test/config blobs unchanged and all203 frozen hashes match.
Only the two integration records were stashed and restored byte-for-byte
(diff SHA2569448fd08f511b078ed59c00ed634f6ea613a4da93740aa5fb874948e94154207).
Backup ref codex/vn3016-before-dev-a2826f-2450adb and scoped stash
018ffdd24a9acd92e07499297a084268c7bcd5dc are retained; do not reapply the stash.

Bounded post-rebase units10passed38deselected0fail/errors/skips5warnings17.85s.
The unit filter deselected35 VN integration cases and3 native-PG guards; the
two VN files were then run separately35passed0fail/errors/skips4warnings25.31s.
Logs/XML /tmp/vn3016-post-rebase-a3f62da and -vn retained. No PG/broad rerun,
no production edits, inherited cleanup/import-time SQLite warnings qualified.
Exact force-with-lease2450adb push succeeded, GitHub head/base verified.
Only PRVerification updated; human summary/all other body sections preserved.
Fresh paginated33threads0unresolved/no remaining pages; all22 conversation
comments checked. ONE full request5845913123 at11:32:34Z on a3f62da is pending,
busy5845914083 at11:32:44Z. Push summary11:31:20Z0bugs0rules is not completion;
prior2450adb full review does not satisfy the new-head gate. Do not duplicate.
Exact-head55checks33queued22completed/no actionablefailure/7required absent.
OPEN/BLOCKED/no merge attempted; AC5 checked/AC6pending. All task-needed
agents/tests/shell sessions closed; only local integration records retained.
