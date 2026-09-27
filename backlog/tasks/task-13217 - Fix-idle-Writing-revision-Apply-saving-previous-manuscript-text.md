---
id: TASK-13217
title: Fix idle Writing revision Apply saving previous manuscript text
status: To Do
created_date: 2026-09-07 23:08
references:
- TASK-13216
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed while verifying TASK-13216, and confirmed as an unchanged-path P2 by final reviewer. Idle Apply updates the editor and persists revision status as applied, but the saved session payload can contain the previous prompt. useWritingRevisions.ts calls applyEditorText(plan.nextText) then persists revisions; applySessionPayloadPatch resolves the callback's captured editorText in useWritingSessionManagement.ts. This is separate from continuation scope protection; do not change continuation behavior to fix it. Reproduce with real mutation-enabled component harness and inspect updateWritingSession payload after Apply/debounce.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Regression reproduces idle Apply persisting old prompt despite updated editor and applied revision state.
- [ ] #2 Applied manuscript text and revision status persist consistently without breaking rejected/pending proposals, rich editor handling, or continuation ownership guards.
- [ ] #3 Focused Writing revision/session tests and relevant quality checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
No fix attempted in the Predict/Fill Service Prompts branch. Review evidence: task-2-report final fix-wave observation and final scoped review at3f67f71fa4. Duplicate search via MCP did not return within bounded wait; filename search for revision persistence/save tasks found none.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
