---
id: TASK-13854
title: Allocate task ids across origin/dev and open PRs in backlog-py
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Root-cause fix for the concurrent-session task-id collisions (13403-13408 clobbered; 12905/12946 pre-existing; even cross-branch self-collision observed). _next_task_id now takes the max across local + origin/dev (ls-tree) + open-PR files (gh pr list); explicit --id colliding with a remote source is rejected with the source named; unreachable sources degrade to local-only with a stderr warning. 7 new tests; full backlog-py suite green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
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
