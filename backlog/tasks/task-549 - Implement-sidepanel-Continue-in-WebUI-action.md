---
id: TASK-549
title: Implement sidepanel Continue in WebUI action
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-29 06:52'
labels:
  - chat
  - extension
  - implementation
dependencies: []
references:
  - TASK-546
  - TASK-547
  - TASK-548
documentation:
  - Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
  - >-
    Docs/superpowers/plans/2026-05-29-sidepanel-chat-webui-handoff-implementation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from the sidepanel chat WebUI handoff plan: add a sidepanel ControlRow Continue in WebUI action that creates a handoff package, opens /chat with the handoff id, preserves the existing route-only full-app action, passes draft and visible page context from the sidepanel form, and covers the flow with focused regression tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ControlRow keeps the existing Open full app action route-only with no handoff creation or handoff query parameter.
- [x] #2 ControlRow provides a Continue in WebUI action that creates one handoff package, merges the handoff id into the current /chat route, preserves character route params, and avoids serializing draft/context content into the URL.
- [x] #3 Continue in WebUI handles empty context, storage failure, and rapid duplicate clicks without opening unintended tabs.
- [x] #4 Sidepanel form passes the current draft and visible page context callback based only on selected tab mentions and active tab title/URL, with no page-body capture.
- [x] #5 Focused sidepanel handoff and existing full-app route regressions pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented chat-continue-in-webui in ControlRow.tsx using the Task 1 handoff service and existing full-app route opener. Added draftMessage, hasVisiblePageContextForHandoff, and getVisiblePageContextForHandoff props to ControlRow. Added selected-document and active-tab title/URL context construction in form.tsx; no page body text is captured. Added ControlRow.chat-handoff.test.tsx covering route-only full-app behavior, handoff creation, character route merge, URL privacy, disabled empty state, stale context warning, storage failure, and duplicate-click prevention. Spec compliance review passed. Code-quality review requested an in-flight guard; fixed with a synchronous ref guard plus pending UI state and deferred-promise regression, then re-review passed. Verification: focused sidepanel suite passed with 14 tests; broader handoff suite passed with 24 tests after including service tests; UI typecheck passed after TASK-550 fixed service parser narrowing. Bandit is not applicable to TypeScript/TSX UI changes; worker attempt produced parser errors for TSX.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 2 is complete. The sidepanel now has a distinct Continue in WebUI quick action that stores the current draft/context in a handoff package and opens `/chat` with only the handoff id in the URL, while the existing Open full app action remains route-only. The sidepanel form supplies visible context from selected tabs and active tab metadata only, and the action now guards duplicate clicks, empty context, and storage failures.
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
