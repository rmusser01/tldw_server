# VN Play Recovery Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class VN Play WebUI controls for checkpoint creation, checkpoint restore, retry-last-turn recovery, and branch/checkpoint inspection.

**Architecture:** Reuse the existing VN Play API client and workspace shell. Keep server-authoritative `scene_version` behavior: after checkpoint, restore, retry, stale-version, or in-progress recovery, refresh session, events, branches, and checkpoints from the backend instead of locally inventing state. Add focused React tests around the workspace because the backend endpoints and API client already exist.

**Tech Stack:** Next.js/React 18, Vitest, Testing Library, existing `@web/lib/api/vnPlay` client, existing VN Play FastAPI endpoints.

---

## Source Inputs

- Backlog: `TASK-151`
- GitHub issue: `https://github.com/rmusser01/tldw_server/issues/1401`
- Parent tracker: `https://github.com/rmusser01/tldw_server/issues/1391`
- API docs: `Docs/API-related/VN_PLAY_API.md`
- Runtime spec: `Docs/superpowers/specs/2026-05-01-vn-play-runtime-design.md`
- Existing implementation plan: `Docs/superpowers/plans/2026-05-01-vn-play-runtime-implementation.md`

## Current Code Shape

- `apps/tldw-frontend/lib/api/vnPlay.ts` already exposes:
  - `retryLastVNPlayTurn`
  - `createVNPlayCheckpoint`
  - `listVNPlayCheckpoints`
  - `restoreVNPlaySession`
  - `listVNPlayBranches`
- `apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx` currently loads sessions and events, handles turn responses, and reloads selected session/events after a turn.
- `apps/tldw-frontend/components/vn-play/SceneInspector.tsx` accepts `branches` and `checkpoints`, but `VNPlayWorkspace` does not yet load or pass them.
- Existing tests cover session creation and turn submission but not checkpoint/restore/retry/conflict recovery.

## Stage 1: Load Branch And Checkpoint Metadata

**Goal:** The selected session loads session details, events, branches, and checkpoints through one refresh path.

**Success Criteria:** Selecting or reloading a session updates runtime inspector counts from backend branch/checkpoint endpoints.

**Tests:** Add a failing `VNPlayWorkspace` test that mocks one branch and one checkpoint and expects the inspector to render `Branches 1` and `Checkpoints 1`.

**Status:** Complete

Steps:
- [x] Add `listVNPlayBranches` and `listVNPlayCheckpoints` to the workspace API mock.
- [x] Write failing workspace test for branch/checkpoint loading.
- [x] Extend `VNPlayWorkspace` state with `branches` and `checkpoints`.
- [x] Replace session/event refresh with `refreshSelectedSession(sessionId)` that loads session, events, branches, and checkpoints in parallel.
- [x] Pass branch/checkpoint arrays into `SceneInspector`.

## Stage 2: Checkpoint Create And Restore UX

**Goal:** Users can create and restore checkpoints without raw API calls.

**Success Criteria:** Runtime inspector exposes a compact checkpoint panel with create and restore controls, and restore refreshes session, events, branches, and checkpoints.

**Tests:** Add failing workspace tests for creating a checkpoint and restoring one.

**Status:** Complete

Steps:
- [x] Add mocked API calls for `createVNPlayCheckpoint` and `restoreVNPlaySession`.
- [x] Write failing test for creating a checkpoint from the current scene.
- [x] Write failing test for restoring a checkpoint and refreshing session/events/checkpoints/branches.
- [x] Add create-checkpoint label state and a restore handler in `VNPlayWorkspace`.
- [x] Extend `SceneInspector` with callback props for checkpoint create/restore.
- [x] Render checkpoint rows with restore buttons and stable accessible labels.

## Stage 3: Retry And Conflict Recovery UI

**Goal:** Recoverable failed/interrupted turns expose explicit retry/reload/poll actions.

**Success Criteria:** Stale scene version and in-progress turn conflicts render targeted guidance and actions; retry-last-turn uses a fresh idempotency key and refreshes state.

**Tests:** Add failing workspace tests for stale-version recovery, turn-in-progress recovery, and retry-last-turn.

**Status:** Complete

Steps:
- [x] Add mocked `retryLastVNPlayTurn` API call.
- [x] Write failing test for a `stale_scene_version` turn error showing a reload action.
- [x] Write failing test for a `turn_in_progress` turn error showing a poll action.
- [x] Write failing test for retry-last-turn calling `/retry-last-turn` with current `sceneVersion`.
- [x] Replace the generic `turnStatus` banner with structured recovery state.
- [x] Add reload/poll/retry handlers that call the unified refresh path.

## Stage 4: Smoke Coverage And Closeout

**Goal:** Preserve the existing VN Play smoke path and cover the new recovery controls where feasible.

**Success Criteria:** Focused Vitest coverage passes; smoke coverage is updated or blockers are documented; task and plan record verification.

**Tests:** Run focused VN Play tests and, if the local Playwright server can bind, run `e2e/smoke/vn-play.spec.ts`.

**Status:** Complete

Steps:
- [x] Update the VN Play smoke mock to respond to checkpoint and branch endpoints if the rendered page now calls them.
- [x] Run `bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayApi.test.ts __tests__/vn-play/SceneStage.test.tsx`.
- [x] Run `bunx playwright test e2e/smoke/vn-play.spec.ts --reporter=line` if the local environment permits it.
- [x] Run `git diff --check`.
- [x] Update `TASK-151` with verification notes and final summary.
