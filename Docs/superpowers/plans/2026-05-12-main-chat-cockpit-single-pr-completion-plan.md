# Main /chat Cockpit Single-PR Completion Plan

> **Scope:** Continue all remaining main `/chat` cockpit work inside draft PR #1582. Do not create a second PR, and do not touch the extension sidepanel/sidebar routes.

> **Current merge bar:** Stages 1-5 document the earlier first-slice completion. The user has since raised the merge bar to a fully mature main `/chat` cockpit, so stages 6-10 are required before PR #1582 should be considered merge-ready.

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

## Stage 6: Mature Cockpit IA And Test Targets

**Goal:** Replace first-slice merge language with the clarified fully mature cockpit bar and lock the component/browser test expectations to that bar.

**Success Criteria:** Spec, plan, and Backlog task all distinguish first-slice history from the current fully mature cockpit merge requirement.

**Tests:** Documentation diff review and focused component test review.

**Status:** Complete

## Stage 7: Context Rail Maturity

**Goal:** Make the context rail a source-oriented work surface for the next turn, not just a summary and launcher.

**Success Criteria:** The rail shows an actionable source inventory for active web, file, knowledge, media, and research context; per-source remove/open actions use existing shared `/chat` state handlers; empty and inactive states clearly describe what will affect the next reply.

**Tests:** `Playground.cockpit-maturity.test.tsx`, `Playground.cockpit-controls.test.tsx`, and real-server `/chat` assertions for source inventory after a real web-search toggle.

**Status:** Complete

## Stage 8: Runtime Rail Maturity

**Goal:** Make the runtime rail an operational inspector for the current turn.

**Success Criteria:** The rail shows provider/model route, scoped provider:model settings summary, character/persona state, tool availability entry, and stop/regenerate recovery controls where existing handlers support them.

**Tests:** `Playground.cockpit-maturity.test.tsx`, existing runtime inspector tests, and real-server `/chat` assertions for route/settings/tools visibility.

**Status:** Complete

## Stage 9: Status Strip And Responsive Cockpit Maturity

**Goal:** Turn the status strip and mobile cockpit from passive summaries into deliberate diagnostic/action surfaces.

**Success Criteria:** The strip prioritizes streaming, degraded, error, no-model, unsaved, and context-active states with direct actions where applicable; mobile uses context/runtime tabs or an equivalent sheet pattern while preserving composer usability.

**Tests:** `Playground.cockpit-maturity.test.tsx`, `Playground.cockpit-shell.test.tsx`, `Playground.cockpit-a11y.test.tsx`, and real-server mobile `/chat` checks.

**Status:** Complete

## Stage 10: Visual QA And Merge Verification

**Goal:** Verify the mature cockpit against the real running server and record remaining baseline blockers separately from this PR's changed behavior.

**Success Criteria:** Focused Vitest, real-server Playwright, desktop/mobile screenshots, `git diff --check`, changed-scope type/design-system checks, and Backlog notes are complete. Bandit is recorded as not applicable if only frontend/docs files are touched.

**Tests:** Focused Vitest cockpit suite, `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`, screenshot capture, `git diff --check`, and changed-scope `tsc`/design-system-state filtering where repo-wide baselines still fail.

**Status:** Complete

**Verification Notes:** Focused mature cockpit Vitest coverage passed: `Playground.cockpit-maturity`, `Playground.cockpit-controls`, `Playground.cockpit-shell`, `Playground.cockpit-a11y`, and `playground-locale-mirror` (20 tests). Real-server Playwright passed against `http://127.0.0.1:8000` with no route interception (3 tests). The screenshot harness returned a `200` chat completion and captured desktop/mobile evidence at `/private/tmp/tldw-chat-cockpit-desktop-20260513.png` and `/private/tmp/tldw-chat-cockpit-mobile-20260513.png`; the captured cockpit status showed `2 messages` with two rendered chat articles. `git diff --check` passed. Filtered `tsc` output showed no changed Playground-file errors. `verify:design-system-state` still exits non-zero for existing Chatbooks/shared-product-state baseline findings; the only Playground output after this slice is the allowed existing `PlaygroundForm.tsx` baseline entry. Bandit was not applicable because this slice touched frontend/docs/e2e files only.
