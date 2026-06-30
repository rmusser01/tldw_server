---
id: TASK-518
title: Route sidepanel full-screen chat handoff to /chat
status: Done
labels:
- chat
- extension
- frontend
- ux
priority: high
documentation:
- Docs/superpowers/plans/2026-05-27-chat-rails-ux-rebaseline-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Sidepanel/Chat/SidepanelHeaderSimple.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx
- Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md
- backlog/tasks/task-518 - Route-sidepanel-full-screen-chat-handoff-to-chat.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Change the sidepanel chat full-screen action to open the rail-enabled /chat route and add focused coverage proving the extension options URL targets /chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The sidepanel chat full-screen action opens `/options.html#/chat`.
- [x] #2 Focused component coverage proves the Chrome extension URL contract targets `/chat`.
- [x] #3 Extension sidepanel screenshot evidence is captured or a blocking reason is documented.
- [x] #4 Verification results, skips, and any baseline failures are recorded in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added focused jsdom regression coverage in `SidepanelHeaderSimple.fullscreen-route.test.tsx` using `MemoryRouter`, hoisted `wxt/browser` mocks, and narrow mocks for i18n, Ant Design Tooltip, message/notification/capability hooks, and `TtsClipsDrawer`.
- Red verification before the production change: `bunx vitest run src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx` failed because `browser.runtime.getURL` received `/options.html#/` instead of `/options.html#/chat`.
- Updated `SidepanelHeaderSimple.tsx` so the sidepanel full-screen button calls `browser.runtime.getURL("/options.html#/chat")`. The existing dashboard button remains `/options.html#/flashcards`; dashboard route cleanup is explicitly outside Task 4 pending separate audit evidence.
- Green verification: `bunx vitest run src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.tts-clips-lazy-mount.test.ts` passed with 2 test files and 2 tests; final rerun after task updates also passed with 2 test files and 2 tests.
- `git diff --check` passed after code, test, audit, and task updates.
- Updated `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md` to classify C9 as still reproducing before Task 4 and fixed by Task 4 for the full-screen handoff.
- Extension sidepanel screenshot evidence is deferred to Task 5 because this Task 4 slice is a focused component contract fix and does not run the extension sidepanel browser capture.
- Bandit skipped: touched executable code is TypeScript/React plus Markdown/backlog documentation, with no Python files in scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 4 routes the extension sidepanel full-screen action to the rail-enabled `/chat` route and locks the extension URL contract with focused component coverage. The audit now records C9 as reproduced before the fix and fixed by Task 4 for the full-screen handoff, while leaving dashboard `/flashcards` cleanup out of scope until separately audited. Screenshot evidence is documented as deferred to Task 5.
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
