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
- [ ] #3 TASK-15183 carries its title, status and labels in its frontmatter (task edit --title/-s/-l), and its leftover body block is resolved through backlog-py
- [ ] #4 Each bare-id references: entry that names a renumbered id is checked, and those meaning the renumbered copy are changed to its new id through backlog-py
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From TASK-13960's Task 4 review (2026-10-31): (1) backlog/tasks/task-15183 (was task-13511, 'Remediate WebUI and extension performance review findings') was hand-written on dev without an opening '---', so backlog-py read it as having no frontmatter; the renumber put a new frontmatter block (id: TASK-15183) on top and left the old block (id: TASK-13511, title, status: Done, labels, a stray ---) as body text. Backlog tools see TASK-15183 with no title, status or labels, as they saw the original with none. (2) 155 bare-id references: entries in task files name a renumbered id and now resolve to the keeper, even where they meant the renumbered copy (e.g. task-12968.2 references TASK-12971 though its dependency was retargeted to TASK-15121; task-12301 references TASK-407, whose Workspace Playground copy is now TASK-15005). The plan left references to old ids as they are; each renumbered file's note names its old id.
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
