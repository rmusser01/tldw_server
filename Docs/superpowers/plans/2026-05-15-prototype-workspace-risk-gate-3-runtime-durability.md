# Prototype Workspace Risk Gate 3 Runtime Durability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for implementation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden prototype workspace runtime job orchestration and preview lifecycle behavior so retries, cancellation, validation failures, and preview replacement are deterministic and safe.

**Architecture:** Keep the existing `PrototypeWorkspaceJobs`, `jobs_worker`, `PrototypeWorkspaceService`, and `PrototypePreviewBroker` boundaries. Use the shared Jobs manager and `WorkerSDK` semantics instead of creating a prototype-specific queue. Treat full production hosting and frontend UI work as later gates; this slice defines backend state contracts and tests.

**Tech Stack:** FastAPI backend, AuthNZ prototype repository, shared Jobs `JobManager`/`WorkerSDK`, pytest, Bandit.

---

## Stage 1: Job Contract And Failure Semantics

**Goal:** Make prototype job payloads and worker failures explicit enough for retries, cancellation, and operator inspection.

**Success Criteria:** Each prototype job type has documented retryability, timeout/cancellation behavior, and stable result/error shape. Worker exceptions can be classified as retryable or terminal without relying on generic exception strings.

**Tests:** Extend `tldw_Server_API/tests/PrototypeWorkspaces/test_runtime_jobs.py` with failing tests for retryable preview bootstrap failures, terminal validation failures, cancellation-sensitive archived/revoked state, and stable worker result payloads.

**Status:** Complete

- [x] Add failing tests for a retryable worker exception carrying `retryable=True` metadata.
- [x] Add failing tests for terminal prototype domain errors carrying `retryable=False`.
- [x] Add a small typed exception/result helper in `tldw_Server_API/app/core/Prototype_Workspaces/jobs_worker.py` or `models.py`.
- [x] Document job-type retry/cancel/timeout semantics in `Docs/API-related/Prototype_Workspaces_API.md`.

## Stage 2: Idempotent Runtime State Transitions

**Goal:** Ensure branch bootstrap, preview boot, and snapshot-save jobs can be safely retried without duplicating active sessions, preview handles, or snapshots.

**Success Criteria:** Retried branch bootstrap reuses the same active session. Retried preview boot with the same idempotency key returns or preserves one active handle per scope. Retried snapshot save with the same request id does not produce duplicate snapshots or roll a session pointer backward.

**Tests:** Add service/job tests covering duplicate branch bootstrap, same preview boot request, same snapshot save request id, and retry after expired/revoked session state.

**Status:** Complete

- [x] Add a failing test for preview boot retry with identical payload preserving one active handle per scope.
- [x] Add a failing test for snapshot-save retry using explicit retry-safe snapshot identity.
- [x] Extend `PrototypeWorkspaceJobs` preview payload metadata so runtime profile version matches preview idempotency and reuse checks.
- [x] Update service/broker helpers only as needed to preserve monotonic session snapshot and preview-handle state.

## Stage 3: Preview Lifecycle Durability

**Goal:** Harden preview lookup, replacement, renewal, revocation, and cache refresh across process restarts and rollback paths.

**Success Criteria:** Preview renewal and validation recover from persistent records after memory cache loss. Active-handle replacement revokes old handles and restores previous handles only when the scope is still unclaimed. Revocation updates both persistent state and in-memory cache. Renewal of revoked/expired/inactive actor/session state returns a stable failure.

**Tests:** Extend `test_preview_broker.py` with persistent lookup, renewal after cache clear, replacement rollback, revoked actor/session renewal, and active-handle replacement assertions.

**Status:** Complete

- [x] Preserve existing tests for renewing after persistent lookup when cache is empty and revoked actor/session invalidation.
- [x] Preserve existing tests for preview replacement rollback not overwriting a concurrent active handle.
- [x] Ensure `PrototypePreviewBroker` reuses identical active handles and preserves persisted handle state on retry/replacement.
- [x] Document preview lifecycle/runtime result categories in the contract matrix.

## Stage 4: Publish Validation And Promotion Safety

**Goal:** Prove failed publish validation, stale candidates, and post-preview persistence failures never advance canonical or last-known-good pointers.

**Success Criteria:** Validation failures mark validation state without moving canonical pointers. Stale candidates do not promote. If preview grant succeeds but canonical pointer persistence fails, the new preview is revoked and workspace state is restored. Promotion request status reflects the terminal outcome.

**Tests:** Extend `test_promotion_service.py` with failed validator, stale baseline, promotion request state, preview rollback, and retry-safe promote idempotency cases.

**Status:** Complete

- [x] Preserve existing tests for failed publish validation preserving canonical and last-known-good pointers.
- [x] Preserve existing tests for preview-grant success followed by canonical update failure.
- [x] Confirm existing `PrototypeWorkspaceService.promote_candidate` compensation boundaries pass the focused promotion suite.
- [x] Ensure worker result payloads expose stable `failure_code`/`retryable` fields for frontend/operator surfaces.

## Stage 5: Verification And PR Closeout

**Goal:** Finish Risk Gate 3 with focused test evidence, security checks, and a PR linked to GitHub issue #1455.

**Success Criteria:** Focused prototype runtime tests pass, Bandit runs on touched backend files, TASK-389 is updated, and the PR body includes the human-owned change-summary reminder.

**Tests:** `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q`; focused runtime/preview/promotion tests added above; Bandit on touched backend paths; `git diff --check`.

**Status:** In Progress

- [x] Run focused red/green tests as each stage lands.
- [x] Run full `tldw_Server_API/tests/PrototypeWorkspaces -q`.
- [x] Run Bandit on touched backend paths.
- [ ] Update TASK-389 with verification/final summary.
- [ ] Open/update PR linked to GitHub issue #1455.
