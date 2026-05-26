---
id: TASK-413
title: Fix Workspace Playground collapsed sidebar restore affordances
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
Keep the Workspace Playground Sources and Studio restore controls visible, associated with their panels, and easy to re-open after users collapse the sidebars.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collapsed Sources and Studio restore controls remain visible as persistent desktop rails instead of disappearing into the workspace row.
- [x] #2 Restore rails expose accessible labels and panel associations so users can understand what each control reopens.
- [x] #3 Existing restore click behavior still expands the collapsed pane.
- [x] #4 Focused WorkspacePlayground desktop layout verification passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing desktop layout regression for persistent collapsed-pane restore rail affordances.
2. Update WorkspaceRestoreRailButton styling and accessibility attributes to keep the controls visible and associated with their panels.
3. Run focused WorkspacePlayground desktop layout tests and follow with a visual/browser check if the local app is available.
4. Record verification and any skipped gates on the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed Workspace Playground collapsed Sources/Studio controls from small inline buttons into persistent desktop restore rails with aria labels, panel associations, visible rail sizing, and existing click-to-restore behavior preserved. Added a desktop layout regression covering the persistent rail contract. Verification: red test failed on missing aria-controls/sticky rail contract; focused desktop layout suite then passed 9 tests; broader focused WorkspacePlayground desktop + ChatPane suite passed 4 files / 47 tests; browser smoke at http://127.0.0.1:3000/workspace-playground confirmed collapsed Sources and Studio show visible restore rails and reopen. git diff --check passed. Bandit skipped because touched code is frontend TypeScript/TSX only.
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
