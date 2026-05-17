---
id: TASK-405
title: Implement chat sidebar tools-first expansion behavior
status: In Progress
labels:
- webui
- extension
- frontend
references:
- TASK-401
- TASK-404
documentation:
- Docs/superpowers/specs/2026-05-17-chat-sidebar-tools-first-expansion-design.md
- Docs/superpowers/plans/2026-05-17-chat-sidebar-tools-first-expansion-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Common/ChatSidebar.tsx
- apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx
- apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx
- apps/packages/ui/src/components/Common/__tests__/ChatSidebar.coordinator.test.tsx
- apps/packages/ui/src/components/Layouts/Layout.tsx
- apps/packages/ui/src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts
- apps/tldw-frontend/components/layout/WebLayout.tsx
- apps/tldw-frontend/__tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the shared ChatSidebar tools-first expansion behavior: shortcuts expanded and recent conversations collapsed on every open, with lazy history preserved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shared ChatSidebar resets to tools-first on direct mount, collapsed-to-expanded transition, and explicit open reset signal.
- [ ] #2 Recent conversations disclosure gates search/tabs/lists, server selection controls, coordinator visibility, and history overview fetching through a single recentHistoryVisible contract.
- [ ] #3 WebUI and shared layout shells pass an explicit reset signal for open/foreground events.
- [ ] #4 Focused Vitest coverage passes for tools-first behavior, lazy history, coordinator visibility, and layout reset signal.
- [ ] #5 Verification and any skipped browser/Bandit checks are recorded in Backlog.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation and verification are complete pending human visual approval of the rendered screenshot. Fresh verification passed: apps/packages/ui focused sidebar/layout Vitest suite (4 files, 14 tests), apps/tldw-frontend WebLayout Vitest suite (1 file, 4 tests), and Playwright browser check against the existing Next dev server at http://127.0.0.1:3000/chat. Browser check confirmed opening the collapsed sidebar yields Shortcuts aria-expanded=true, Recent conversations aria-expanded=false, no search while Recent is collapsed; manual Recent expansion shows search; collapse/reopen resets back to Shortcuts expanded and Recent collapsed. Screenshot artifact: output/playwright/chat-sidebar-tools-first-expanded.png. Bandit skipped because the touched implementation is frontend TS/TSX plus Backlog markdown. Root-level Vitest invocation was not used as evidence because it globbed unrelated .worktrees; package-scoped commands were used instead.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
