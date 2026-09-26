# PR 3016 VN Durability Review Implementation Plan

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
- [ ] Commit/rebase/push, seven individual evidence replies and verified thread
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
