# Main /chat Cockpit Single-PR Completion Plan

> **Scope:** Continue all remaining main `/chat` cockpit work inside draft PR #1582. Do not create a second PR, and do not touch the extension sidepanel/sidebar routes.

## Stage 1: Scope Reconciliation

**Goal:** Make the docs and tracking state match the user's single-PR constraint.

**Success Criteria:** Spec and Backlog state clearly say PR #1582 is the vehicle for the remaining main `/chat` cockpit work.

**Tests:** Documentation diff review.

**Status:** Complete

## Stage 2: Independent Cockpit Rail Visibility

**Goal:** Let users keep cockpit mode while hiding only the context rail, only the runtime rail, or both, without losing the focus preset.

**Success Criteria:** Context/runtime rail visibility is controlled by keyboard-accessible buttons, persisted separately from focus/cockpit mode, and covered by component tests.

**Tests:** `Playground.cockpit-shell.test.tsx`, `Playground.cockpit-a11y.test.tsx`, and real-server `/chat` smoke coverage.

**Status:** Complete

## Stage 3: Remaining Direct Rail Controls

**Goal:** Add only direct controls that map to existing shared `/chat` state and handlers.

**Success Criteria:** Context rail can clear active context types where existing setters support it; runtime rail exposes model/persona/runtime controls without duplicating state; status strip reflects operational state passed from the current chat surface.

**Tests:** Focused rail/status component tests plus `Playground.cockpit-controls.test.tsx`.

**Status:** Complete

## Stage 4: Real-Server Merge Verification

**Goal:** Prove the PR works against the running server without mocked API data.

**Success Criteria:** Playwright exercises `/chat` with the real backend, verifies cockpit/focus state survives a real chat attempt, verifies new rail controls, and confirms no route interception is used.

**Tests:** `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`.

**Status:** Complete

## Stage 5: PR Handoff

**Goal:** Leave PR #1582 draft but updated with implementation and verification evidence.

**Success Criteria:** Focused tests pass, real-server verification passes or records a real recoverable provider/server error, baseline blockers are separated, Backlog task is updated, and the branch is pushed.

**Tests:** Focused Vitest, real-server Playwright, `git diff --check`, frontend type check if practical, Bandit only if Python files are touched.

**Status:** Complete

**Verification Notes:** Focused cockpit Vitest coverage passed; the degraded readiness gate test passed; the real-server Playwright `/chat` cockpit spec passed against `http://127.0.0.1:8000` with no route interception; an additional one-off real-server screenshot capture returned a `200` chat completion and saved `/private/tmp/tldw-chat-cockpit-real-server-20260512.png`.
