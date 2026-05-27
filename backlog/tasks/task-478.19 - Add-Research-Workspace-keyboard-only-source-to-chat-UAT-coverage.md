---
id: TASK-478.19
title: Add Research Workspace keyboard-only source-to-chat UAT coverage
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 01:54'
labels:
  - research-workspace
  - uat
  - keyboard
  - e2e
dependencies: []
parent_task_id: TASK-478
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close RW-UAT-017 by adding a dedicated keyboard-only source-to-chat walkthrough/regression for Research Workspace. The coverage should prove users can move through the canonical /research-workspace surface without pointer interaction, reach source controls, preserve visible focus, reach the chat composer, and submit or stage a grounded-source question through keyboard interaction. Update the UAT matrix and Backlog notes with evidence while preserving unrelated MCP/ACP/Sandbox gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dedicated Research Workspace keyboard-only E2E/regression covers the source-to-chat path without pointer interaction.
- [x] #2 Test reaches a seeded source control by keyboard and verifies focus remains visible/logical.
- [x] #3 Test reaches the chat composer by keyboard, submits a selected-source question, and verifies the RAG request includes the selected media ID.
- [x] #4 UAT matrix RW-UAT-017 is updated based on coverage.
- [x] #5 Verification results are recorded; Bandit skipped for TS/TSX/Markdown-only scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Acceptance criteria:
- [x] Dedicated Research Workspace keyboard-only E2E/regression covers the source-to-chat path without pointer interaction.
- [x] Test reaches a seeded/visible source row or source control by keyboard and verifies focus remains visible/logical.
- [x] Test reaches the chat composer by keyboard, stages or submits a selected-source question, and verifies the intended keyboard interaction path.
- [x] UAT matrix RW-UAT-017 is updated honestly based on the resulting coverage.
- [x] Verification results are recorded; Bandit is skipped because this slice touched TS/TSX/Markdown only.

Plan:
1. Add a failing Playwright regression for the keyboard-only source-to-chat path. Complete.
2. Implement the smallest accessibility/focus-order changes needed for the test to pass. Complete.
3. Update the UAT matrix and Backlog evidence. Complete.
4. Run focused verification and commit. Verification complete; commit pending.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented TASK-478.19 keyboard-only Research Workspace source-to-chat coverage. Added a dedicated Playwright regression that reaches the Research Workspace skip link by Tab, focuses Sources with Enter, tabs to a named source checkbox, selects it with Space, tabs to the chat composer, submits with Enter, and verifies the RAG request includes only the selected media ID. Updated the page shell skip links to explicitly focus their target regions, made pane targets programmatically focusable, added source-specific accessible names/tab order to source checkboxes, and marked RW-UAT-017 Pass in the UAT matrix.

Verification: focused Playwright keyboard test passed on isolated port 18080; nearby Vitest suites passed 31/31; git diff --check passed. Full stubbed research-workspace.spec.ts currently has unrelated baseline failures in global search shortcut, older chat-completion expectation after RAG, and Studio compare button lookup, while the new keyboard-only test passed in that run. Bandit skipped: TS/TSX/Markdown-only scope.
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
