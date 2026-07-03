# Chat Workspace Source Rail Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete GitHub issue #2032 by making `/chat-workspace` source browsing, staging, unstaging, state rendering, and staged send behavior work against real workspace sources.

**Architecture:** Keep Chat Workspace as the active chat surface and route source-management actions to canonical Research Workspace/Media routes. The page owns browsed/staged source state, the rail exposes per-source lifecycle actions, and the chat panel continues to translate staged context into structured media ids plus fallback summary text.

**Tech Stack:** React, Zustand workspace store, React Testing Library/Vitest, Playwright smoke tests.

---

## Design Review Adjustments

- A source with `status: "ready"` but an invalid or missing `mediaId` remains stageable. It cannot be carried structurally, so existing fallback-summary behavior must cover it.
- Staging is disabled for processing, error, and unknown/unavailable source states, but unstaging remains enabled for any already-staged source.
- Browse should not surprise users by navigating away from Chat Workspace. It should mark the source as browsed and prime the workspace store focus target; explicit Add/Open links handle route changes.
- Source loading/error state should come from `useWorkspaceStore` (`sourcesLoading`, `sourcesError`) instead of invented local state.
- Workspace-switch guards must also protect any new individual unstage callback, matching the existing stale-clear protection.
- Duplicate source titles need disambiguated accessible action names for new buttons, matching the current browse/stage suffix behavior.

## Stage 1: Lifecycle Unit And Component Tests

**Goal:** Add failing coverage for source unstage, lifecycle states, empty states, and route actions.
**Success Criteria:** Focused Vitest tests fail for missing unstage/loading/error/link behavior before implementation.
**Tests:** `WorkspaceRail.test.tsx`, `ContextStagingCard.test.tsx`, `ChatWorkspacePage.test.tsx`, `staging.test.ts`.
**Status:** Complete

- [x] Add a failing `staging.test.ts` case for removing one staged source while preserving others.
- [x] Add failing `WorkspaceRail.test.tsx` cases for Add/Open route links, unstage action, ready-with-invalid-media stageability, processing/error disabled staging, loading state, source error state, no-source empty state, and filtered-empty state.
- [x] Add a failing `ContextStagingCard.test.tsx` case for removing one staged source by source id.
- [x] Add a failing `ChatWorkspacePage.test.tsx` case showing individual unstage is ignored for stale previous-workspace callbacks.
- [x] Run focused Vitest and confirm the new tests fail for the expected missing behavior.

## Stage 2: Source Rail And Staging Lifecycle Implementation

**Goal:** Implement the minimal UI and state plumbing needed to satisfy the failing component tests.
**Success Criteria:** Stage 1 tests pass without changing the existing successful send contracts.
**Tests:** Same focused Vitest files as Stage 1.
**Status:** Complete

- [x] Add `unstageWorkspaceSource` to `staging.ts`.
- [x] Extend `ContextStagingCard` with an optional `onRemoveSource` callback and per-source Remove buttons.
- [x] Extend `WorkspaceRail` props for `sourcesLoading`, `sourcesError`, `onUnstageSource`, and canonical source-management route URLs.
- [x] Render Add source and Open library as route links to `/research-workspace?tab=sources` and `/media`.
- [x] Render clear source lifecycle states and disable only non-actionable stage buttons.
- [x] Pass source loading/error state and unstage handlers from `ChatWorkspacePage` through `ChatWorkspaceConsole`.
- [x] Guard individual unstage against stale workspace state.
- [x] Run focused Vitest and confirm the Stage 1 tests pass.

## Stage 3: Browser Workflow Proof

**Goal:** Add browser coverage for stage, unstage, re-stage, and send with staged context.
**Success Criteria:** A deterministic Playwright smoke test proves unstage removes the source before send and re-stage sends ready media ids structurally.
**Tests:** `apps/tldw-frontend/e2e/smoke/chat-workspace-live-backend.spec.ts`.
**Status:** Complete

- [x] Extend the existing Chat Workspace live-backend fixture with a stage/unstage/re-stage workflow.
- [x] Assert unstage removes the source from the staged context and prevents accidental send with stale context.
- [x] Re-stage the ready source, send, and assert the RAG request includes the expected `include_media_ids`.
- [x] Keep the route-link proof local to Chat Workspace; do not depend on full Research Workspace source UI in this smoke test.
- [x] Run the focused Playwright spec.

## Stage 4: Verification And Backlog Finalization

**Goal:** Record verification evidence and complete the Backlog task once tests pass.
**Success Criteria:** Focused Vitest, focused Playwright, and whitespace checks pass; Bandit is documented as not applicable because no Python changed.
**Tests:** Focused Vitest, focused Playwright, `git diff --check`.
**Status:** Complete

- [x] Run focused Vitest for Chat Workspace component and staging tests.
- [x] Run focused Playwright Chat Workspace smoke spec.
- [x] Run `git diff --check`.
- [x] Mark Bandit not applicable for this TypeScript-only scope.
- [x] Update `TASK-12132` with touched files, verification results, completed acceptance criteria, and final summary.
