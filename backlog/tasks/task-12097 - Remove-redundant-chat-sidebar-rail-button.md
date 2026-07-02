---
id: TASK-12097
title: Remove redundant chat sidebar rail button
status: Done
assignee: []
created_date: '2026-07-02 06:27'
updated_date: '2026-07-02 06:37'
labels:
  - webui
  - chat
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the chat-specific collapsed rail edge button from the web layout because the app-wide sidebar popout button already handles this action.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The /chat layout no longer renders the chat-sidebar-edge-expand rail button
- [x] #2 The shared ChatSidebar can still receive openResetKey wiring for the app-wide popout
- [x] #3 Focused layout tests cover the removal
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation notes:
- Removed the dedicated chat-sidebar-edge-expand rail button from both the Next WebLayout and the shared extension Layout.
- Removed the special desktop chat collapse path so the normal shared ChatSidebar collapsed state remains the app-wide control.
- Removed the obsolete CHAT_RAIL_EDGE_TRIGGER_CLASS export and narrowed its positioning contract to the cockpit restore tab that still uses that helper file.
- Updated the chat rail Playwright workflow to assert the redundant edge button is absent while preserving cockpit restore and artifacts edge behavior.

Verification:
- Red: WebLayout chat-scroll contract failed because chat-sidebar-edge-expand still rendered.
- Red: shared Layout guard failed because useChatEdgeCollapse / chat-sidebar-edge-expand still existed.
- Green: bun run test __tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
- Green: bun run test src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts src/components/Layouts/__tests__/chat-rail-positioning-contract.guard.test.ts
- Green: npx playwright test e2e/workflows/chat-rails-collapse.spec.ts --project=chromium --reporter=line --workers=1
- Green: git diff --check
- Bandit: /tmp/bandit_task12097.json, zero findings; touched scope is TS/Playwright so 0 Python LOC.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the redundant chat-specific left-edge rail button and the special desktop collapse path from WebUI and extension layouts. The normal shared ChatSidebar control remains, and focused unit plus Playwright workflow coverage now assert the old edge button stays absent.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Acceptance criteria completed
- [x] #8 Focused tests recorded
- [x] #9 Bandit run or non-Python scope documented
<!-- DOD:END -->
