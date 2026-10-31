---
id: TASK-13960
title: Renumber duplicate backlog task ids and refuse new ones in CI
status: Done
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
- [x] #1 backlog-py task renumber PATH --id NEW_ID renumbers one task file in place, keeps its directory and file-name tail, appends a note, and refuses an id any task file holds
- [x] #2 task create and task edit check ids and dependencies across backlog/tasks, backlog/completed and backlog/archive/tasks, and task create without --id allocates above all three
- [x] #3 No two task files under backlog/tasks, backlog/completed and backlog/archive/tasks share an id on dev
- [x] #4 A PR that adds a duplicate id fails backend-required, including a backlog-only PR
- [x] #5 Docs/Development/Backlog_Task_Id_Renumbering.md records every keeper choice, dependency decision and rewritten or dead link, and each renumbered task's notes name its old id
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Dry run on origin/dev c251b6f978 (2026-10-31, open PRs fetched that day), exit 0: 235 duplicated ids, 293 files to renumber, 1 owner rows, 50 dependency entries (28 retargeted, 2 left to open PRs), 40 links in 39 files, 22 dead links and 33 dead self-links in task files, 68 duplicate files in open PRs (0 already on dev)
Task 1: RED 10 failed, 32 passed (test_task_mutations.py); GREEN 42 passed; backlog-py suite 155 passed.
Task 2: RED ModuleNotFoundError; GREEN 4 passed. Task 3: RED ImportError; GREEN 6 passed. ruff clean; Bandit -ll no issues.
Task 4: generator line '235 duplicated ids, 293 files to renumber, 1 owner rows, 50 dependency entries (28 retargeted, 2 left to open PRs), 40 links in 39 files, 22 dead links and 33 dead self-links in task files, 68 duplicate files in open PRs (0 already on dev)'; check exit 0; normalize --check on 310 touched paths exit 0; wiring test RED 1 failed, 5 passed, then GREEN 235 passed; tests/CI + tests/Docs + backlog-py + generator test 1282 passed, 5 skipped; test-quality ratchet 1 passed; backend-required ratchet steps: tenant isolation 56 passed (exit 0), CI contracts 975 passed, 1 skipped plus WebSocket auth 190 passed (exit 0). Dead links: 22 found by the generator, plus 1 added by hand (23 in the record): TASK-15152's YAML-folded reference to the renumbered 13001 copy, now TASK-15271.
Task 5: ADR-084 (Accepted, as approved for v2's ADR-075 on 2026-10-18) and its Docs/Published mirror. Retargets left to open PRs: 2 (#2855: task-13145, TASK-13111 to TASK-15132; #2421: task-2281, TASK-2387 to TASK-15254).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
backlog-py gains task renumber, and create/edit check ids and dependencies across tasks, completed and archive; create without --id allocates above all three. Helper_Scripts/renumber_duplicate_task_ids.py renumbered 293 files (keeper rule a-d, rule d by arrival on dev, 1 owner decision, 18 topic groups), retargeted 28 dependency entries and left 2 to the open PRs that change those files, rewrote 40 links and left 22 dead links (plus 1 added by hand, 23 in the record) and 33 dead self-links in task files, all listed in Docs/Development/Backlog_Task_Id_Renumbering.md. backend-required's changes job now fails any PR with a duplicate id (ADR-084). Follow-ups: TASK-13961 (renumber's remote-id check once #3263 lands) and TASK-13962 (re-parent the orphaned subtasks; apply any left retarget whose PR closes unmerged).
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
