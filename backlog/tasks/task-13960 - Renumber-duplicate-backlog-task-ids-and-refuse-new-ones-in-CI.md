---
id: TASK-13960
title: Renumber duplicate backlog task ids and refuse new ones in CI
status: In Progress
labels:
- backlog
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On 2026-10-30, 235 task ids on dev were held by more than one file (528 files; 293 to renumber), so task TASK_ID, dependencies and parent links are ambiguous. backlog-py cannot renumber, task create checks only backlog/tasks, and no CI check looks at ids.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 backlog-py task renumber PATH --id NEW_ID renumbers one task file in place, keeps its directory and file-name tail, appends a note, and refuses an id any task file holds
- [ ] #2 task create and task edit check ids and dependencies across backlog/tasks, backlog/completed and backlog/archive/tasks, and task create without --id allocates above all three
- [ ] #3 No two task files under backlog/tasks, backlog/completed and backlog/archive/tasks share an id on dev
- [ ] #4 A PR that adds a duplicate id fails backend-required, including a backlog-only PR
- [ ] #5 Docs/Development/Backlog_Task_Id_Renumbering.md records every keeper choice, dependency decision and rewritten or dead link, and each renumbered task's notes name its old id
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Dry run on origin/dev c251b6f978 (2026-10-31, open PRs fetched that day), exit 0: 235 duplicated ids, 293 files to renumber, 1 owner rows, 50 dependency entries (28 retargeted, 2 left to open PRs), 40 links in 39 files, 22 dead links and 33 dead self-links in task files, 68 duplicate files in open PRs (0 already on dev)
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
