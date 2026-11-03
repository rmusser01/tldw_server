---
id: TASK-13962
title: Re-parent the subtasks orphaned by 3a660d4f1d and finish the renumbering's
  leftovers
status: Done
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
- [x] #1 Every subtask the record lists under no copy names its real parent in its id and parent_task_id, changed through backlog-py
- [x] #2 TASK-15183 carries its title, status and labels in its frontmatter (task edit --title/-s/-l), and its leftover body block is resolved through backlog-py
- [x] #3 Each bare-id references: entry that names a renumbered id is checked, and those meaning the renumbered copy are changed to its new id through backlog-py
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From TASK-13960's Task 4 review (2026-10-31): (1) backlog/tasks/task-15183 (was task-13511, 'Remediate WebUI and extension performance review findings') was hand-written on dev without an opening '---', so backlog-py read it as having no frontmatter; the renumber put a new frontmatter block (id: TASK-15183) on top and left the old block (id: TASK-13511, title, status: Done, labels, a stray ---) as body text. Backlog tools see TASK-15183 with no title, status or labels, as they saw the original with none. (2) 155 bare-id references: entries in task files name a renumbered id and now resolve to the keeper, even where they meant the renumbered copy (e.g. task-12968.2 references TASK-12971 though its dependency was retargeted to TASK-15121; task-12301 references TASK-407, whose Workspace Playground copy is now TASK-15005). The plan left references to old ids as they are; each renumbered file's note names its old id.
Plan: Docs/superpowers/plans/2026-11-03-reparent-orphaned-subtasks-13962.md. Follow-up for AC 2: TASK-15419. Owner answers (2026-11-03): Decision 1 (Change summary) waiver covers this PR; Decision 2 (AC 2's home) follow-up task; Decision 3 (TASK-13534) repair it.
AC 2 (the dependency retargets left to #2855 and #2421) moved to TASK-15419 on 2026-11-03, with the references entries left to #2421 and #3323: all three PRs were open at execution. The record's 'Left to open PRs' names the line each PR must change.
AC 1: 43 subtasks re-parented with task renumber (TASK-12295.1-.6, TASK-12303.1, TASK-12318.1-.3, TASK-12418.1-.5, TASK-12446.1-.19, TASK-12561.1-.9), in their old order; 418.17 (the old 418.14) first. backlog-py's renumber now sets parent_task_id for a subtask id. Each moved file and each new parent has a note. No dependency or references entry named an old subtask id. reparent_13962.py --verify: 43 of 43. parent_check: 12 before and after; those 12 are outside this task (TASK-12020.46-.52, 12116.1, 12178, 12179, 12570, 18.82.1).
AC 3: TASK-15183 and TASK-13534 (same missing opening ---) got title, status and labels with task edit, and the old block removed with the new task edit --remove-preamble.
AC 4: 157 bare-id references entries in 126 files checked (Docs/Development/Backlog_Task_Id_References.md): keeper 73, neither 32, renumbered copy 50, unclear 2. 45 changed with the new task edit --ref; 3 in backlog/completed left (task edit cannot reach them); 2 left to #2421 and #3323 (TASK-15419).
Verification: backlog-py 158 -> 5 failed, 158 passed (RED) -> 163 passed; CI suite 1622 passed, 5 skipped -> 1627 passed, 5 skipped; shard coverage new_uncovered=0; test-quality ratchet OK; check_backlog_task_ids exit 0; Bandit No issues identified (Medium 0, High 0).
Final review I1 (2026-11-03): task edit --remove-preamble alone on a task with no frontmatter (TASK-13534's old shape) deleted its only title and status. It now refuses unless the same call gives -t (guard in edit_task, so the one-call -t -s -l --remove-preamble repair still works); test_remove_preamble_needs_a_title_when_the_task_has_no_frontmatter RED -> GREEN; backlog-py 164 passed; CI suite 1628 passed, 5 skipped.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Re-parented the 43 subtasks 3a660d4f1d left under reused parent ids (now TASK-12295.x, 12303.1, 12318.1-.3, 12418.x, 12446.x, 12561.x) with task renumber, which now also sets parent_task_id. Gave TASK-15183 and TASK-13534 their frontmatter and removed the stray block with the new task edit --remove-preamble. Checked all 157 bare-id references entries that name a renumbered id and pointed the 45 that mean a renumbered copy at its new id with the new task edit --ref (table: Docs/Development/Backlog_Task_Id_References.md). The two dependency retargets and two reference retargets held by open PRs #2855, #2421 and #3323 moved to TASK-15419; the record names each line.
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
