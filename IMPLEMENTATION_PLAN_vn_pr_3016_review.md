# PR 3016 VN Durability Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve Qodo's eight VN generation findings without weakening the existing API or Jobs contract.

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
boundaries have RED/GREEN regression coverage. GitHub review replies, rebase,
fresh PR checks, and merge are still pending.

- [x] Run the full scoped verification matrix and self-review the changed diff.
- [ ] Reply in each Qodo thread with the corresponding fix or technical reasoning.
- [ ] Verify the requester-provided Change summary remains in the PR body.
- [ ] Confirm branch is rebased on current `origin/dev`, all required checks pass, then merge through GitHub.
- [ ] Record PR merge and final test evidence in `TASK-13369`.
