# Chat Overlay And Quick Ingest Execution Sequence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` for single-stream execution, or `superpowers:subagent-driven-development` only if the user explicitly asks to split the streams across agents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining chat character overlay follow-ups and Quick Ingest UX remediation slices as small, reviewable units without mixing unrelated UI surfaces in one PR.

**Architecture:** This plan does not replace the existing detailed implementation plans. It sequences the remaining Backlog tasks, defines review boundaries, and calls out the verification required before each slice is handed off.

**Tech Stack:** Next.js WebUI, React package UI, TypeScript, Vitest, Playwright, Backlog.md, GitHub PR workflow.

**Primary References:**
- `Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md`
- `Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md`
- `Docs/superpowers/specs/2026-05-16-quick-ingest-ux-remediation-stages-design.md`
- `Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md`

---

## Current Dev Reassessment

Reassessed against freshly fetched `origin/dev` at `61124888aa` on 2026-06-03 before implementation:

- `TASK-448` / `backlog/tasks/task-448 - Implement-overlay-send-path-and-snapshot-on-apply-behavior.md` is already `Done`.
- `TASK-449` / `backlog/tasks/task-449 - Add-desktop-character-control-rail.md` is already `Done`, and later `TASK-444` reconciles the standalone rail direction as superseded by the current cockpit runtime/context rails.
- The requested `TASK-457` overlay e2e title is not the current task file on clean `dev`; the same work exists as `TASK-487` / `backlog/tasks/task-487 - Add-WebUI-chat-end-to-end-verification-for-overlay-and-tracked-identity.md`, already `Done`.
- Quick Ingest `TASK-394.2` through `TASK-394.7` are already `Done` on clean `dev`.
- The clean `dev` Backlog has duplicate task ids for several numeric ids, so exact task file paths are safer than `task_id`-only MCP edits for this slice family.

Do not execute the implementation tasks below from a fresh `dev` branch unless a task is explicitly reopened or a new follow-up Backlog task is created. Treat the remaining task sections as historical sequencing guidance for review or regression triage.

---

## Scope Rules

- [ ] Keep chat overlay follow-ups and Quick Ingest remediation in separate branches and PRs unless the user explicitly asks to combine them.
- [ ] Preserve the in-stream order: `TASK-448` -> `TASK-449` -> `TASK-457` for chat overlay.
- [ ] Preserve the in-stream order: `TASK-394.2` -> `TASK-394.3` -> `TASK-394.4` -> `TASK-394.5` -> `TASK-394.6` -> `TASK-394.7` for Quick Ingest.
- [ ] Rebase each active branch onto latest `dev` before final verification and before addressing PR review.
- [ ] Move a slice's Backlog task to `In Progress` before code edits begin for that slice.
- [ ] Update each Backlog task with touched files, verification, blockers, and final summary before marking it Done.
- [ ] Treat frontend-only slices as Bandit not applicable; if backend Python is touched, run Bandit on the touched scope from the project virtual environment.

---

## Task 0: Preflight And Worktree Setup

**Goal:** Start from clean, isolated work areas so the existing dirty main worktree does not leak unrelated changes into these PRs.

**Backlog Tasks:** `TASK-448`, `TASK-449`, `TASK-457`, `TASK-394.2`, `TASK-394.3`, `TASK-394.4`, `TASK-394.5`, `TASK-394.6`, `TASK-394.7`

**Steps:**
- [ ] Inspect `git status --short` in the root and avoid staging unrelated dirty files.
- [ ] Fetch latest remote refs.
- [ ] Create or refresh a dedicated chat branch/worktree from latest `dev`, for example `codex/chat-overlay-followups`.
- [ ] Create or refresh a dedicated Quick Ingest branch/worktree from latest `dev`, for example `codex/quick-ingest-ux-remediation`.
- [ ] Confirm each Backlog task still matches the planned slice before editing code.
- [ ] Record this execution-sequence plan path in task notes if the task is updated during the slice.

**Success Criteria:**
- [ ] No unrelated root-worktree files are staged.
- [ ] Each stream has an isolated branch or worktree based on current `dev`.
- [ ] The first task in each stream has enough context to begin TDD.

---

## Task 1: Chat `TASK-448` - Overlay Send Path And Snapshot-On-Apply

**Goal:** Make overlay application and message send routing use one stable assistant snapshot so later rail and e2e work do not depend on implicit UI state ordering.

**Detailed Plan Source:** Chat implementation plan Task 3, "Implement Snapshot-On-Apply Overlay Resolution And Send Routing."

**Likely Files:**
- `apps/packages/ui/src/utils/assistant-overlay.ts`
- `apps/packages/ui/src/hooks/useMessage.tsx`
- `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts`
- `apps/packages/ui/src/components/Playground/AssistantSelect.tsx`
- Related `__tests__` files under `apps/packages/ui/src`

**Steps:**
- [ ] Write or update failing tests for snapshot-on-apply behavior and send-time override routing.
- [ ] Move overlay resolution behind a small helper that returns a typed applied snapshot.
- [ ] Route normal chat sends from the snapshot rather than recomputing from scattered UI fields.
- [ ] Keep the API surface narrow; avoid adding broad send options that callers must manually coordinate.
- [ ] Update `TASK-448` with verification and remaining risk.

**Verification:**
- [ ] Run targeted Vitest for the changed assistant overlay and chat send modules.
- [ ] Run `git diff --check` for touched files.
- [ ] Run Bandit only if backend Python files are touched.

**PR Boundary:**
- [ ] Open a PR containing only `TASK-448` unless the user asks to batch chat tasks.

---

## Task 2: Chat `TASK-449` - Desktop Character Control Rail

**Goal:** Add the desktop character control rail on top of the stable overlay snapshot contract from `TASK-448`.

**Detailed Plan Source:** Chat implementation plan Task 4, "Add The Desktop Character Control Rail."

**Likely Files:**
- `apps/packages/ui/src/components/Playground/CharacterControlRail.tsx`
- `apps/packages/ui/src/components/Playground/Playground.tsx`
- `apps/packages/ui/src/components/Playground/PlaygroundForm.tsx`
- `apps/packages/ui/src/utils/chat-surface-coordinator.ts`
- Related component and contract tests

**Steps:**
- [ ] Rebase onto the merged `TASK-448` branch or latest `dev` after `TASK-448` merges.
- [ ] Add failing tests for rail visibility, assistant selection, overlay apply/clear, and disabled states.
- [ ] Implement the rail as a presentation/control component that consumes the existing overlay snapshot APIs.
- [ ] Keep state ownership in the existing chat/playground coordinator rather than duplicating character identity state in the rail.
- [ ] Browser-check desktop layout for overlap, scroll behavior, and focus order.
- [ ] Update `TASK-449` with verification and remaining risk.

**Verification:**
- [ ] Run targeted Vitest for rail/coordinator behavior.
- [ ] Run a local browser check for the desktop `/chat` or playground surface.
- [ ] Run `git diff --check` for touched files.
- [ ] Run Bandit only if backend Python files are touched.

**PR Boundary:**
- [ ] Open a PR containing only `TASK-449` after `TASK-448` is merged or otherwise available on the base branch.

---

## Task 3: Chat `TASK-457` - `/chat` End-To-End Verification

**Goal:** Add WebUI `/chat` end-to-end coverage proving overlay and tracked identity behavior works through the actual user flow.

**Detailed Plan Source:** Chat implementation plan Task 5, limited here to the `TASK-457` e2e verification scope.

**Likely Files:**
- Existing `/chat` Playwright specs under `apps/tldw-frontend`
- Existing e2e helpers and fixtures
- Related UI test IDs or accessibility labels only where selectors are not stable

**Steps:**
- [ ] Rebase onto latest `dev` after `TASK-448` and `TASK-449` are merged.
- [ ] Locate the current `/chat` e2e spec and helper patterns before adding new selectors.
- [ ] Add coverage for selecting a character, applying an overlay, sending a message, and verifying tracked identity is preserved.
- [ ] Add coverage for clearing or changing the overlay if that behavior is user-facing in the merged UI.
- [ ] Coordinate with any still-open sidepanel/mobile parity task, but do not take over unrelated sidepanel implementation work in this slice.
- [ ] Update `TASK-457` with verification and remaining risk.

**Verification:**
- [ ] Run the targeted Playwright spec for `/chat` overlay/tracked identity.
- [ ] Run any targeted Vitest needed for selector or fixture helper changes.
- [ ] Run `git diff --check` for touched files.
- [ ] Run Bandit only if backend Python files are touched.

**PR Boundary:**
- [ ] Open a PR containing the e2e coverage and selector/test-helper support needed for `TASK-457`.

---

## Task 4: Quick Ingest `TASK-394.2` - Entry And Destination Copy

**Goal:** Clarify Quick Ingest entry labels, destination copy, and review text without changing ingestion behavior.

**Detailed Plan Source:** Quick Ingest remediation plan Task 2, "Clarify Entry Points, Destination Copy, And Review State."

**Likely Files:**
- Quick Ingest trigger/button component
- Quick Ingest add-content and review steps
- Related integration/component tests

**Steps:**
- [ ] Rebase the Quick Ingest branch/worktree onto latest `dev`.
- [ ] Add failing tests for updated entry labels and destination copy.
- [ ] Replace ambiguous copy with destination-specific wording that matches the current data flow.
- [ ] Keep behavior unchanged except for user-facing clarification.
- [ ] Update `TASK-394.2` with verification and remaining risk.

**Verification:**
- [ ] Run targeted Quick Ingest Vitest coverage.
- [ ] Browser-check the modal entry and review copy.
- [ ] Run `git diff --check` for touched files.
- [ ] Run Bandit only if backend Python files are touched.

**PR Boundary:**
- [ ] Open a PR containing only `TASK-394.2` unless the user explicitly asks to batch Quick Ingest copy-only changes.

---

## Task 5: Quick Ingest `TASK-394.3` - Result Handoff And Recovery Actions

**Goal:** Improve success/failure handoff after ingestion so users can recover or continue without relying on hidden state.

**Detailed Plan Source:** Quick Ingest remediation plan Task 3, "Improve Result Handoff And Recovery Actions."

**Likely Files:**
- Quick Ingest wizard modal
- Results step component
- Result action helpers
- Related component and e2e tests

**Steps:**
- [ ] Rebase onto latest `dev` after `TASK-394.2` merges.
- [ ] Add failing tests for successful result actions, failed result recovery, and close behavior.
- [ ] Make result actions explicit in the results step rather than requiring callers to infer next steps.
- [ ] Preserve existing ingestion API contracts unless the detailed task uncovers a hard blocker.
- [ ] Update `TASK-394.3` with verification and remaining risk.

**Verification:**
- [ ] Run targeted Quick Ingest result tests.
- [ ] Browser-check success and failure result states.
- [ ] Run `git diff --check` for touched files.
- [ ] Run Bandit only if backend Python files are touched.

**PR Boundary:**
- [ ] Open a PR containing only `TASK-394.3`.

---

## Task 6: Quick Ingest `TASK-394.4` - Offline, Cancel, And Progress States

**Goal:** Correct offline, cancel, and progress state transitions so the wizard does not depend on users or callers following fragile timing.

**Detailed Plan Source:** Quick Ingest remediation plan Task 4, "Correct Offline, Cancel, And Progress States."

**Likely Files:**
- Quick Ingest wizard modal
- Add-content and processing steps
- Floating progress widget
- Related state and interaction tests

**Steps:**
- [ ] Rebase onto latest `dev` after `TASK-394.3` merges.
- [ ] Add failing tests for offline entry, cancel during active work, progress text, and close/reopen state.
- [ ] Centralize state transitions enough that cancel/close/progress cannot drift across components.
- [ ] Keep any long-running ingestion cancellation semantics aligned with the current backend behavior; do not imply server-side cancellation if only UI cancellation exists.
- [ ] Update `TASK-394.4` with verification and remaining risk.

**Verification:**
- [ ] Run targeted Quick Ingest state tests.
- [ ] Browser-check offline, cancel, and progress flows.
- [ ] Run `git diff --check` for touched files.
- [ ] Run Bandit only if backend Python files are touched.

**PR Boundary:**
- [ ] Open a PR containing only `TASK-394.4`.

---

## Task 7: Quick Ingest `TASK-394.5` - URL And File Input Hardening

**Goal:** Harden URL and file validation at the wizard boundary so invalid inputs fail early with clear recovery paths.

**Detailed Plan Source:** Quick Ingest remediation plan Task 5, "Harden URL And File Input Validation."

**Likely Files:**
- Quick Ingest add-content step
- Queue validation helpers
- Ingest payload helpers
- Validation constants
- Related validation and interaction tests

**Steps:**
- [ ] Rebase onto latest `dev` after `TASK-394.4` merges.
- [ ] Add failing tests for malformed URLs, unsupported schemes, duplicate inputs, empty file lists, unsupported file types, and oversize files if current limits exist.
- [ ] Use structured validation helpers rather than scattering ad hoc validation across components.
- [ ] Be truthful about current client-buffered upload limits; do not introduce a transport redesign in this slice unless the task owner changes the scope.
- [ ] Update `TASK-394.5` with verification and remaining risk.

**Verification:**
- [ ] Run targeted Quick Ingest validation tests.
- [ ] Browser-check validation errors and recovery interactions.
- [ ] Run `git diff --check` for touched files.
- [ ] Run Bandit only if backend Python files are touched.

**PR Boundary:**
- [ ] Open a PR containing only `TASK-394.5`.

---

## Task 8: Quick Ingest `TASK-394.6` - Current-Flow Verification Coverage

**Goal:** Refresh stale selectors and add focused coverage for the active wizard states touched by the remediation slices.

**Detailed Plan Source:** Quick Ingest remediation plan Task 6, "Refresh current Quick Ingest wizard verification coverage."

**Likely Files:**
- Quick Ingest e2e specs
- Quick Ingest helper fixtures
- Component tests whose selectors no longer match the active UI

**Steps:**
- [ ] Rebase onto latest `dev` after `TASK-394.5` merges.
- [ ] Inventory stale Quick Ingest selectors and classify each as update, delete, or quarantine with documented reason.
- [ ] Add or update coverage for default, URL, text, file, validation, success/failure, close, and cancel paths where feasible.
- [ ] Prefer accessible selectors and stable test IDs over brittle visual structure selectors.
- [ ] Update `TASK-394.6` with verification and remaining risk.

**Verification:**
- [ ] Run targeted Quick Ingest e2e and component tests.
- [ ] Run `git diff --check` for touched files.
- [ ] Run Bandit only if backend Python files are touched.

**PR Boundary:**
- [ ] Open a PR containing only verification/test updates and selector support required for `TASK-394.6`.

---

## Task 9: Quick Ingest `TASK-394.7` - Final Review And Closeout

**Goal:** Finalize the Quick Ingest remediation stream with complete verification, Backlog evidence, and a PR-ready summary.

**Detailed Plan Source:** Quick Ingest remediation plan Task 7, "Close out Quick Ingest UX remediation verification."

**Steps:**
- [ ] Rebase onto latest `dev` after `TASK-394.6` merges.
- [ ] Run the final agreed verification set across the Quick Ingest touched scope.
- [ ] Update parent `TASK-394` and child tasks with completion evidence, known skips, residual risks, and PR links.
- [ ] Prepare a concise PR-ready summary listing what changed, why those choices were made, and how it was verified.
- [ ] Confirm no TODOs without issue numbers were added.
- [ ] Confirm no unrelated root-worktree changes are included.

**Verification:**
- [ ] Run final targeted Vitest/Playwright checks for Quick Ingest.
- [ ] Run `git diff --check` for touched files.
- [ ] Run Bandit if backend Python files were touched; otherwise document frontend-only skip.

**PR Boundary:**
- [ ] Open the closeout PR only after the implementation slices it verifies have merged or are included on the branch intentionally.

---

## Final Cross-Stream Checks

- [ ] Each PR has a focused scope and references the matching Backlog task.
- [ ] Each task has final verification recorded before being marked Done.
- [ ] The chat e2e PR lands after the chat behavior it verifies.
- [ ] The Quick Ingest verification/closeout PR lands after the remediation behavior it verifies.
- [ ] Any PR materially authored by AI waits for a human-written change summary before merge, per repo policy.
