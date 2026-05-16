# Chat Cockpit Mobile Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the main WebUI `/chat` mobile cockpit a deliberate context/runtime/focus flow that preserves the composer draft and keeps controls reachable.

**Architecture:** Extend the existing `PlaygroundCockpitShell` mobile tabs instead of introducing a new sheet or sidebar surface. Keep all mobile controls wired through the same left/right rail React nodes already used by desktop, and add proof around mobile state preservation rather than new state stores.

**Tech Stack:** React, TypeScript, Tailwind utility classes, Vitest with Testing Library, Playwright real-server workflow.

---

## Scope Guard

This task is only for the main WebUI `/chat` route:

- Route entry: `apps/tldw-frontend/pages/chat/index.tsx`
- Main UI: `apps/packages/ui/src/routes/option-chat.tsx`
- Chat cockpit: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Mobile shell: `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`

Do not edit browser-extension sidepanel/sidebar files, `/settings`, character-library pages, onboarding, backend APIs, MCP Hub pages, evaluation pages, media ingestion, or repo-wide architecture.

## File Map

- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`
  - Add mobile panel status metadata, larger tap targets, and a compact mobile help/status line that clarifies draft preservation without adding new product concepts.
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx`
  - Add focused mobile shell tests for draft preservation copy, active panel state, and touch target classes.
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`
  - Add accessible mobile tab/tabpanel relationship coverage if it does not fit the maturity test cleanly.
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
  - Expand the existing real-server mobile test to prove draft preservation across context/runtime/focus transitions and capture active-conversation screenshots.
- Update: `backlog/tasks/task-402 - Polish-main-chat-mobile-cockpit-experience.md`
  - Record touched files, verification, and completion state.

## Stage 1: Prove the Mobile Shell Gap

**Goal:** Capture the missing mobile affordance at the component level before implementation.

**Success Criteria:** The test fails because the mobile shell does not yet expose a mobile panel summary/status line and does not provide the expected mobile metadata/touch target treatment.

**Tests:** `bun run test:run ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx`

**Status:** Complete

- [x] Add a failing test that renders `PlaygroundCockpitShell` in cockpit mode with context/runtime rails and a child composer draft.
- [x] Assert the selected mobile panel has stable metadata such as `data-mobile-panel="context"`.
- [x] Assert a compact mobile summary states that the context panel is active and the composer draft remains available below.
- [x] Assert mobile tabs have a minimum touch target utility of at least `min-h-[44px]`.
- [x] Run the focused Vitest command and confirm the new test fails for the missing behavior.

## Stage 2: Implement Mobile Panel Clarity

**Goal:** Make the mobile cockpit communicate which rail is active and that switching panels does not discard the draft.

**Success Criteria:** Mobile context/runtime tabs expose active-panel metadata, a concise status line, and usable tap targets while desktop behavior remains unchanged.

**Tests:** `bun run test:run ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx`

**Status:** Complete

- [x] Add a localized mobile panel summary in `PlaygroundCockpitShell.tsx`.
- [x] Set `data-mobile-panel` on `playground-cockpit-mobile-rails`.
- [x] Increase mobile tab target classes to `min-h-[44px]` without changing desktop rail toggles.
- [x] Keep the rail content mounted only for the active mobile panel to preserve current rendering cost behavior.
- [x] Run the focused Vitest command and confirm the new test passes.

## Stage 3: Extend Real-Server Mobile Proof

**Goal:** Prove the mobile cockpit preserves real `/chat` draft and control reachability against the running backend.

**Success Criteria:** The mobile Playwright workflow enters text once, switches context/runtime/focus/cockpit, opens prompt/persona/model/MCP controls, and observes that the same draft remains in the composer.

**Tests:** `bun run e2e:pw -- e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line` with real `TLDW_E2E_SERVER_URL`, `NEXT_PUBLIC_API_URL`, and `.env` API key.

**Status:** Complete

- [x] Update the existing mobile real-server test, do not add mocked network routes.
- [x] Fill `chat-input` with a unique draft before opening mobile cockpit panels.
- [x] Verify the draft survives context tab, prompt selector open/close, runtime tab, character/persona selector open/close, MCP settings open/close, focus mode, and return to cockpit.
- [x] Capture an active-conversation or active-draft mobile screenshot in addition to the existing context/runtime/focus screenshots.
- [x] If the real health response is degraded, assert the warning copy allows chat instead of blocking it. If the server is healthy, record that no degraded state was observable in the real-server run.

## Stage 4: Verification and Closeout

**Goal:** Record evidence and leave the slice ready for the next cockpit maturity step.

**Success Criteria:** Focused unit tests, real-server mobile Playwright, diff hygiene, relevant lint, and backlog updates are recorded.

**Tests:**

- `bun run test:run ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx`
- Real-server Playwright command from Stage 3
- `git diff --check`
- Targeted ESLint for touched TS/TSX files
- Bandit skip documented because this slice touches frontend TypeScript and docs only

**Status:** Complete

- [x] Run all verification commands fresh before making completion claims.
- [x] Update `TASK-402` acceptance criteria, implementation notes, verification, and final summary.
- [ ] Commit the mobile cockpit polish slice.
