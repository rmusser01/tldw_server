---
id: TASK-485
title: Fix /chat rails regression coverage and sidepanel handoff target
status: In Progress
labels:
- webui
- chat
- sidepanel
- ux
priority: High
documentation:
- Docs/superpowers/specs/2026-05-31-chat-siderail-collapse-design.md
- Docs/superpowers/plans/2026-05-31-chat-siderail-edge-expand-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-05-31-chat-siderail-collapse-design.md
- Docs/superpowers/plans/2026-05-31-chat-siderail-edge-expand-implementation-plan.md
- apps/extension/tests/e2e/sidepanel-options-handoff.spec.ts
- apps/packages/ui/src/components/Layouts/Layout.tsx
- apps/packages/ui/src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts
- apps/packages/ui/src/components/Option/Playground/Playground.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/ArtifactsPanel.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/SidepanelHeaderSimple.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.webui-handoff.test.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/form.tsx
- apps/packages/ui/src/services/__tests__/sidepanel-chat-webui-handoff.test.ts
- apps/packages/ui/src/services/tldw/sidepanel-chat-webui-handoff.ts
- apps/tldw-frontend/components/layout/WebLayout.tsx
- apps/tldw-frontend/e2e/workflows/chat-rails-collapse.spec.ts
- apps/tldw-frontend/extension/routes/sidepanel-chat.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 1 from the fresh /chat UX re-evaluation: preserve restored /chat cockpit rails and fix the directly connected sidepanel chat handoff so visible sidepanel actions target WebUI /chat with draft/context handoff instead of /options.html#/chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 0/1: add failing regression coverage for /chat desktop rails and sidepanel handoff route; trace existing model selector/handoff controls; implement the minimal route/visible-action changes so sidepanel Continue in WebUI opens /chat?handoff=<id>; verify with focused tests and browser smoke.

Design addendum 2026-05-31: collapsed /chat siderails must disappear from layout and leave same-side edge-mounted expand buttons. Left chat rail collapse releases left width and shows a left-edge expand button. Right artifact rail collapse releases right width and shows a right-edge expand button when an artifact is available. Both collapsed states must keep chat/composer vertically anchored and visibly recoverable from both edges.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec committed and reviewed through the brainstorming spec-review loop. Initial review flagged ambiguous 768-1023px behavior and shared OptionLayout scope. Spec was updated to scope edge-mounted expand buttons to lg-and-wider /chat side-rail behavior, preserve md/tablet behavior, and require /chat/Playground-scoped layout changes. Re-review status: Approved.

Follow-up spec review clarified two planning details before implementation: the right-edge expand button is only visible when an active artifact exists, and browser verification must include layout measurements for chat width, chat-shell top stability, and composer bottom docking.

Implementation plan drafted at Docs/superpowers/plans/2026-05-31-chat-siderail-edge-expand-implementation-plan.md before runtime code changes.

Plan review loop completed. Initial review required the Playwright plan to reuse auth/setup seeding and provider endpoint stubs, plus verify right-edge absence below lg with an active artifact. Plan was updated and re-review status was Approved.
Implementation completed in commits cb2a400815, d6608e4b7e, and 89f7c6b72c. The shared options/extension shell and the Next WebUI shell now remove the desktop chat rail from layout when collapsed and expose the left-edge expand button. The artifact rail exposes a right-edge expand button only when an active artifact exists and the panel is collapsed. The new Playwright workflow covers desktop width release/top stability/composer docking plus medium and mobile absence of desktop edge controls.

Verification recorded: `bunx vitest run src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.jump-source.guard.test.ts` passed 4 files / 16 tests. `TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18081' TLDW_WEB_URL=http://127.0.0.1:18081 bunx playwright test e2e/workflows/chat-rails-collapse.spec.ts --project=chromium --reporter=line` passed 2 tests. Scoped whitespace checks for the touched WebLayout and E2E files passed. Full `git diff --check` is blocked by unrelated pre-existing trailing whitespace in `Docs/Design/Agents.md:155`. Bandit skipped because this slice touched frontend TypeScript/TSX and Playwright coverage only, with no Python runtime or tests changed.

Screenshot evidence captured after the fix: `/private/tmp/tldw-chat-left-rail-collapsed.png` and `/private/tmp/tldw-chat-right-rail-collapsed.png`. The stubbed provider state intentionally shows the existing no-provider warning while exercising layout behavior.
User rejected the closeout screenshots and validation because they did not show the required visual states: both rails expanded as a baseline, one rail expanded while the opposite same-side expand affordance is visible, and the composer did not match the current /chat surface. Reopened for live-state reproduction and correction.
2026-06-01 slice: implemented sidepanel chat WebUI handoff helper and wired the visible sidepanel full-screen/header action plus composer overflow Open full app action through `/chat?handoff=...` instead of the extension options hash. The payload preserves draft text, server chat id via existing `settingsServerChatId`, and chat context controls such as model, prompts, chat mode, web search, tool choice, temporary chat, OCR, and title. WebUI `/chat` now decodes the handoff, applies supported context state, seeds the composer with `tldw:set-composer-message` using `ifEmptyOnly`, and cleans the `handoff` query param. Updated extension e2e coverage to assert the sidepanel opens `http://127.0.0.1:8080/chat` with decoded draft/context rather than `options.html`. Verification: focused Vitest set passed 6 files / 19 tests; extension `bun run compile` passed; rail-collapse Playwright e2e passed 2 tests after elevated local-server permission; scoped `git diff --check` passed. Extension handoff Playwright e2e was not completed because its global setup repeatedly entered the existing WXT production build hang and had to be terminated; no test assertion ran. Bandit skipped because this slice touched frontend TS/TSX and Playwright only, no Python runtime files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented same-side desktop edge expand affordances for collapsed /chat rails across the shared layout, artifact panel, and Next WebUI shell. Added focused Playwright coverage for left and right rail collapse/expand behavior, chat width release, vertical stability, composer docking, and no desktop edge buttons below the lg breakpoint.
2026-06-01 sidepanel handoff slice: visible sidepanel chat actions now open real WebUI `/chat` with encoded handoff draft/context and WebUI `/chat` restores that handoff. Focused unit coverage and rail-collapse Playwright verification pass; extension handoff e2e is updated but remains blocked by the existing WXT production build hang before test execution.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
