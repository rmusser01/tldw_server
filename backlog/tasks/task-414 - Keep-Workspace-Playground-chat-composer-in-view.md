---
id: TASK-414
title: Keep Workspace Playground chat composer in view
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 16:58'
labels:
  - webui
  - ux
  - workspace-playground
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the Workspace Playground chat composer remains inside the viewport-bounded workspace shell so users do not need to scroll the page to reach the input.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WebUI and extension Workspace Playground route wrappers constrain height and hide page-level overflow for the workspace shell.
- [x] #2 ChatPane keeps the composer as a non-scrolling flex footer under the transcript.
- [x] #3 Focused WorkspacePlayground desktop layout and ChatPane tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for route wrapper overflow bounds and the ChatPane composer footer contract.
2. Update the shared WebUI and extension route wrappers plus ChatPane footer classes.
3. Run focused desktop layout and ChatPane verification.
4. Record verification and known skips on the Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Bound the shared WebUI and extension Workspace Playground route wrappers with h-full and overflow-hidden, and made the ChatPane composer footer shrink-0 so the transcript owns internal scrolling while the composer stays in view. Added regressions for both the route shell contract and the composer footer contract. Verification: red test failed on missing wrapper h-full/overflow-hidden and missing composer shrink-0; focused desktop layout + ChatPane stage1 run then passed 2 files / 20 tests; broader focused WorkspacePlayground desktop + ChatPane suite passed 4 files / 47 tests; browser smoke at http://127.0.0.1:3000/workspace-playground confirmed the composer is visible without page scrolling. Console smoke still shows an existing AntD Modal destroyOnClose deprecation warning. git diff --check passed. Bandit skipped because touched code is frontend TypeScript/TSX only.
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
