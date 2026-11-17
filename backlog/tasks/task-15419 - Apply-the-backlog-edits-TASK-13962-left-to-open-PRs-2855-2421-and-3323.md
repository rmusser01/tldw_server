---
id: TASK-15419
title: 'Apply the backlog edits TASK-13962 left to open PRs #2855, #2421 and #3323'
status: To Do
labels:
- backlog
dependencies:
- TASK-13962
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13962 re-parented the subtasks 3a660d4f1d orphaned and pointed bare-id references at the renumbered copies they mean, but could not change three task files that open PRs change. Each PR should make its file's change; if a PR closes without it, apply it here with task edit. Docs/Development/Backlog_Task_Id_Renumbering.md ('Left to open PRs') lists the lines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 task-13145 (canonical admin webhook producers) depends on TASK-15132, not TASK-13111: by #2855, or with task edit --dep once #2855 closes without the change
- [ ] #2 task-2281 (LSP-backed MCP tools) depends on and references TASK-15254, not TASK-2387: by #2421, or with task edit --dep and --ref once #2421 closes without the change
- [x] #3 task-13421 (Jobs completion SQL ownership) references TASK-15149, not TASK-13215: by #3323, or with task edit --ref once #3323 closes without the change
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
AC 3 done on 2026-11-05 in the TASK-13952 PR: #3323 merged without the change.
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
