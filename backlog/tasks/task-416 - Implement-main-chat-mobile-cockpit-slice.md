---
id: TASK-416
title: Implement main chat mobile cockpit slice
status: Done
labels:
- chat
- webui
- cockpit
- ux
- frontend
priority: medium
references:
- Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
- https://github.com/rmusser01/tldw_server/pull/1811
documentation:
- Docs/superpowers/plans/2026-05-17-chat-cockpit-mobile-cockpit-plan.md
modified_files:
- apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx
- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
- Docs/superpowers/plans/2026-05-17-chat-cockpit-mobile-cockpit-plan.md
- backlog/tasks/task-416 - Implement-main-chat-mobile-cockpit-slice.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR6 from the main WebUI /chat cockpit maturity roadmap. Scope is strictly the main /chat page: make mobile cockpit/focus behavior deliberate, usable, and visually verified without adding bottom controls or touching the browser-extension sidepanel/sidebar.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mobile /chat defaults to a chat-first focus experience while preserving explicit access to cockpit context/runtime rails.
- [x] #2 Mobile users can inspect and change prompt, persona/character, model, context, and tool state without losing an unsent draft.
- [x] #3 Mobile rail tabs/sheets preserve keyboard focus, accessible names, focus return, and non-overlapping composer behavior.
- [x] #4 Mobile status/degraded/error/session state remains visible without creating a bottom bar or composer-adjacent replacement summary.
- [x] #5 Focused Vitest coverage and real-server /chat Playwright screenshots cover mobile focus, context rail, runtime rail, active conversation, and key degraded/error states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-chat-cockpit-mobile-cockpit-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the implementation plan for the main /chat PR6 mobile cockpit slice. Baseline focused Vitest after installing apps workspace dependencies passed: Playground.cockpit-maturity, Playground.cockpit-a11y, and Playground.cockpit-shell, 26 tests. Impeccable PRODUCT/DESIGN context files are absent in this worktree; this slice follows the existing cockpit roadmap, WebUI patterns, and design-system conventions instead of inventing a new visual direction.
Implemented the mobile /chat cockpit slice strictly inside the main WebUI chat surface. PlaygroundCockpitShell now keeps mobile context/runtime tabpanels mounted with stable tab/panel ids, valid aria-controls/aria-labelledby relationships, hidden inactive panels, and a panel-local Return to focus chat control. No extension sidepanel/sidebar files were touched and no bottom bar, bottom summary, or composer-adjacent replacement UI was introduced. Real-server proof required escalation because sandboxed Playwright could not bind the Next.js dev server on 0.0.0.0:8080; the escalated run used the real running backend at http://127.0.0.1:8000 with no route mocks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the main /chat mobile cockpit slice. Mobile context/runtime panels now have stable accessible tab/tabpanel semantics, inactive panels remain mounted but hidden, and mobile users can return from the panel area to focus chat via a keyboard-reachable in-panel control. Expanded unit coverage for tab semantics, focus return, and keyboard activation; expanded real-server Playwright proof for mobile context/runtime/focus/active conversation states and explicit no-bottom-control assertions. Verification: focused Vitest 26/26 passed; adjacent cockpit Vitest 53/53 passed with existing mocked-client stderr; git diff --check passed; real-server Playwright mobile grep passed 2/2 against http://127.0.0.1:8000. Bandit not applicable because this slice touched TS/TSX/E2E/Markdown only. Screenshot evidence captured under apps/tldw-frontend/test-results for chat-cockpit-mobile-context.png, chat-cockpit-mobile-runtime.png, chat-cockpit-mobile-active-draft.png, chat-cockpit-mobile-focus.png, and chat-cockpit-mobile-conversation.png.
Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1811. The PR remains draft pending human-owned Change summary and review.
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
