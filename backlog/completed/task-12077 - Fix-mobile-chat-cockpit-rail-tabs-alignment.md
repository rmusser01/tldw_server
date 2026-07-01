---
id: TASK-12077
title: Fix mobile chat cockpit rail tabs alignment
status: Done
labels:
- webui
- chat
- ux
priority: High
modified_files:
- apps/tldw-frontend/components/layout/WebLayout.tsx
- apps/tldw-frontend/__tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
- apps/tldw-frontend/e2e/workflows/chat-rails-collapse.spec.ts
- apps/packages/ui/src/components/Layouts/Layout.tsx
- apps/packages/ui/src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts
- apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the WebUI chat cockpit collapsed rail tabs shown in the supplied mobile screenshot: raise the collapsed Chats tab and keep the context rail restore tab attached to the side edge instead of floating over the composer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collapsed Chats rail tab appears near the upper half of the mobile/edge layout rather than down near the page middle/bottom.
- [x] #2 Collapsed context rail restore tab is anchored to the screen edge and does not float over or overlap the composer.
- [x] #3 Regression tests cover the rail positioning behavior where practical.
- [x] #4 Rendered WebUI validation confirms the corrected mobile layout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR: https://github.com/rmusser01/tldw_server/pull/2562
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the WebUI chat cockpit collapsed rail placement. The Chats rail handle now uses an upper viewport band instead of vertical centering, and the context rail restore handle is flush to the cockpit edge rather than offset into the content. Added/updated unit guards and the rail-collapse Playwright workflow setup to cover the corrected behavior. Verification: focused Vitest suite passed (14 tests); Playwright rail-collapse workflow passed (3 tests); WebUI lint scope passed; shared-package lint via frontend config exited cleanly with the expected Next pages-directory notice; Bandit attempted on touched TS/TSX scope produced no findings but TypeScript parse errors, with no Python files touched for this task.
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
