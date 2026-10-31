---
id: TASK-13961
title: 'Make task renumber refuse ids held on origin/dev or in open PRs once #3263
  lands'
status: To Do
labels:
- backlog
dependencies:
- TASK-13960
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13960 shipped backlog-py task renumber with a local id check only, so that the renumbering did not wait for #3263 (TASK-13854), which adds the origin/dev and open-PR check to task create. Once #3263 is on dev, renumber_task should run the same check (first plan review, M3). Helper_Scripts/renumber_duplicate_task_ids.py must then pass check_remote=False, because backlog-py's live check sees the renumbering PR's own block (second plan review, I1). The appendix of plan_dupids_v3.md holds the code and tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 task renumber refuses a top-level id that origin/dev or an open PR holds, with the same message as task create
- [ ] #2 renumber_task(..., check_remote=False) skips that check, and the generator's apply passes it
- [ ] #3 Tests cover the refusal, the skip, and apply taking the ids its own open PR shows
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
