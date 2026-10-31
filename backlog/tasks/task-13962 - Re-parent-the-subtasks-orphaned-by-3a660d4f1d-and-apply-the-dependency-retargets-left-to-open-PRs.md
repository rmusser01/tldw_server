---
id: TASK-13962
title: Re-parent the subtasks orphaned by 3a660d4f1d and apply the dependency retargets
  left to open PRs
status: To Do
labels:
- backlog
dependencies:
- TASK-13960
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On 2026-07-05, 3a660d4f1d renumbered active parents without their subtasks, so 43 subtasks (407.1-.6, 410.1, the completed 418.14, 418.15-.16, 463.1-.5, 481.1-.19, 528.1-.9) still name a parent id that now belongs to an unrelated task. Docs/Development/Backlog_Task_Id_Renumbering.md lists them under Subtasks attributed by topic, with no copy as owner and the parent's new id. Re-parenting needs a backlog-py way to rewrite parent_task_id and move each id under the new parent. TASK-13960 also left two dependency retargets to the open PRs that change those files (#2855, #2421); the record lists them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every subtask the record lists under no copy names its real parent in its id and parent_task_id, changed through backlog-py
- [ ] #2 Every dependency retarget the record leaves to an open PR is applied: by that PR, or with task edit --dep if the PR closes unmerged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
