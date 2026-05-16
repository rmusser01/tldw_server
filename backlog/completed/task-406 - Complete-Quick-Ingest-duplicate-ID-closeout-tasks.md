---
id: TASK-406
title: Complete Quick Ingest duplicate-ID closeout tasks
status: Done
labels:
- quick-ingest
- backlog
- cleanup
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move completed Quick Ingest parent/planning/review Backlog records out of active tasks by exact file path because their TASK-392 through TASK-395 IDs collide with unrelated active records. Do not use ID-based completion for the duplicate records.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Completed Quick Ingest planning, parent, and PR-review records are moved out of active tasks by exact file path.
- [x] #2 Unrelated duplicate-ID active records remain untouched.
- [x] #3 Verification records that active TASK-392 through TASK-395 IDs now resolve to the unrelated active records only.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Moved the completed Quick Ingest records for TASK-392, TASK-393, TASK-394, and TASK-395 from `backlog/tasks` to `backlog/completed` with exact path-based `git mv` commands because Backlog ID-based completion was unsafe while duplicate active IDs existed. Left the unrelated active duplicate-ID records in place.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the remaining Quick Ingest Backlog closeout by moving the completed design, implementation-plan, parent implementation, and PR #1751 review records to completed storage. This resolves the active duplicate-ID ambiguity for TASK-392 through TASK-395 without renumbering or modifying unrelated active tasks. Verification was metadata-only: exact file moves, active/completed path checks, `git diff --check`, and status review.
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
