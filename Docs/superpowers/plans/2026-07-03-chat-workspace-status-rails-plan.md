# Chat Workspace Status Rails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete GitHub issue #2033 by making `/chat-workspace` inspector and status rails reflect real runtime, degraded, hydration, and failed-send states.

**Architecture:** Keep the change inside the existing Chat Workspace component boundary. `WorkspaceChatPanel` remains the source of chat runtime state; `ChatWorkspacePage` lifts that state; `ChatWorkspaceConsole` passes it to `WorkspaceStatusStrip` and `InspectorRail`.

**Tech Stack:** React, TypeScript, Vitest, Playwright.

---

## Stage 1: Runtime Contract

**Goal:** Add failed-send state to the shared Chat Workspace runtime contract.
**Success Criteria:** Send failures can be lifted from the chat panel and rendered outside the composer.
**Tests:** `WorkspaceChatPanel.test.tsx` covers failed-send runtime reporting.
**Status:** Complete

- [x] Add `sendError` to `ChatWorkspaceRuntimeState`.
- [x] Report `sendError` from `WorkspaceChatPanel`.
- [x] Clear/report state through existing workspace and send transitions.

## Stage 2: Inspector Rail

**Goal:** Replace misleading placeholder inspector sections with real runtime and recovery state.
**Success Criteria:** Inspector does not show inactive approval/task placeholders, and it explains backend, hydration, send failure, model, and persona states.
**Tests:** `InspectorRail.test.tsx`.
**Status:** Complete

- [x] Add workspace-readiness and send-error props.
- [x] Add deterministic runtime precedence.
- [x] Add concise recovery copy.
- [x] Remove placeholder approval/task-progress panels.

## Stage 3: Status Strip

**Goal:** Make the footer status strip match the actual ability to send workspace chat.
**Success Criteria:** Hydration never renders as ready; failed sends and missing setup states surface as status pills.
**Tests:** `WorkspaceStatusStrip.test.tsx`.
**Status:** Complete

- [x] Add workspace-readiness, send-error, model, and persona props.
- [x] Add status precedence for server unavailable, workspace hydration, send failure, streaming, and ready.
- [x] Add recovery/status pills for reconnect, workspace identity, model selection, and optional persona state.

## Stage 4: Browser Evidence

**Goal:** Extend existing live-backend browser proof to assert visible rail/status runtime state.
**Success Criteria:** Playwright verifies streaming and send-failure status reaches the route UI.
**Tests:** `apps/tldw-frontend/e2e/smoke/chat-workspace-live-backend.spec.ts`.
**Status:** Complete

- [x] Assert status strip and inspector show streaming while stop generation is visible.
- [x] Assert failed send shows status-strip failure and inspector retry recovery copy.
- [x] Re-run Stage 5 Chat Workspace gate.

## Stage 5: Review Remediation

**Goal:** Address PR review comments after rebasing onto latest `dev`.
**Success Criteria:** Runtime rail contracts avoid string sentinels, workspace readiness is required by rail callers, and stop generation proves delayed stream output does not land after abort.
**Tests:** `InspectorRail.test.tsx`, `WorkspaceStatusStrip.test.tsx`, `WorkspaceChatPanel.test.tsx`, `apps/tldw-frontend/e2e/smoke/chat-workspace-live-backend.spec.ts`.
**Status:** Complete

- [x] Pass selected model details through `InspectorRail` and add an explicit `hasModelSelected` runtime flag.
- [x] Make `workspaceReady` required for `InspectorRail` and `WorkspaceStatusStrip`.
- [x] Assert aborted workspace streams do not append delayed streamed output or leave streaming rail state visible.

## Stage 6: Hydration and Offline Follow-up

**Goal:** Address the final PR review findings with one readiness source of truth and browser-level offline transition evidence.
**Success Criteria:** A persisted workspace ID cannot bypass store hydration, chat sending and both rails share the same readiness boolean, and active streaming rails switch coherently to server-unavailable state.
**Tests:** `ChatWorkspacePage.test.tsx` and `apps/tldw-frontend/e2e/smoke/chat-workspace-live-backend.spec.ts`.
**Status:** Complete

- [x] Add a failing page regression for a non-empty workspace ID while `storeHydrated` is false.
- [x] Derive readiness in `ChatWorkspacePage` from `storeHydrated` plus normalized workspace identity and pass it through `ChatWorkspaceConsole`.
- [x] Add a browser transition from active streaming to an unreachable connection and assert both rails suppress stale streaming state.
- [x] Re-run focused tests, TypeScript, lint/diff checks, and record verification in TASK-12135.

## Stage 7: Bounded Missing-Model Follow-up

**Goal:** Align otherwise-ready idle rails and typed/staged send admission with the existing selected-model value.
**Success Criteria:** No selected model shows `Select a model`, disables both send paths before preparation, and preserves draft/context. Actual selection restores readiness without sending. Existing higher-priority runtime states, explicit Auto routing and optional persona remain unchanged.
**Tests:** The three owning rail/panel suites plus all Chat Workspace regressions; real Chrome raw-CDP desktop/mobile checks against the live authenticated API and immutable production build.
**Status:** In Progress

- [x] Record the requester's bounded design approval in TASK-12135 before implementation.
- [x] Add failing model-readiness and typed/staged restoration regressions, then reuse the classifier and existing submission guard across four production files.
- [x] Verify 273 owning tests, production TypeScript, seven-file lint and production build/token checks with unchanged bundle budgets.
- [x] Verify seven native no-model desktop/mobile cases and a separate four-step selection/reload/cleanup continuation, with zero inference sends and screenshots inspected. Retain the first picker timeout as failed, not a successful full run.
- [x] Rebase onto PR3071's test-only correction and verify runtime source equivalence and protected data, tabs, draft, services and stashes.
- [x] Publish separate stacked PR #3159 against the parent branch and link it in TASK-12135.
- [ ] Address actual review/CI findings, retarget/rebase onto latest `dev` after the parent merge, and obtain this PR's own requester-owned `Change summary` before normal merge.

Evidence is recorded in TASK-12135 and `/private/tmp/chat-workspace-model-readiness-20261003-sVaPyQ`. The default-heap TypeScript attempt failed; the established 8 GB heap run passed. Unit test doubles are not UAT. No Python source changed, so Bandit is not applicable. The old-origin model-clear writeback hypothesis remains unproven and unfixed; the original Settings tab was never navigated or closed. Historical all-browser-target preservation remains failed, separately from the current protected-tab checks.
