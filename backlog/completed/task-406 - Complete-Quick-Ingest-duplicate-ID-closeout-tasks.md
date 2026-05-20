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
Move completed Quick Ingest parent/planning/review Backlog records out of active tasks by exact file path because their task IDs collide with unrelated active records. Follow-up passes also move completed Quick Ingest, bulk-conference ingest, PR review, and related design-system task records for the same reason. Do not use ID-based completion for duplicate records.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Completed Quick Ingest planning, parent, and PR-review records are moved out of active tasks by exact file path.
- [x] #2 Unrelated duplicate-ID active records remain untouched.
- [x] #3 Verification records that active TASK-392 through TASK-395 IDs now resolve to the unrelated active records only.
- [x] #4 Completed Quick Ingest TASK-403 and TASK-407 records are moved out of active tasks by exact file path.
- [x] #5 Completed bulk-conference ingest and QuickIngest design-system records are moved out of active tasks by exact file path.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Moved the completed Quick Ingest records for TASK-392, TASK-393, TASK-394, and TASK-395 from `backlog/tasks` to `backlog/completed` with exact path-based `git mv` commands because Backlog ID-based completion was unsafe while duplicate active IDs existed. Follow-up on 2026-05-19 moved the already-completed Quick Ingest TASK-403 batch conference metadata record and TASK-407 extension playlist handoff record by exact path for the same reason. Follow-up on 2026-05-20 moved the remaining completed bulk-conference ingest records TASK-400, TASK-401, TASK-402, TASK-404, TASK-405, TASK-406, TASK-408, TASK-409, TASK-410, TASK-418, plus QuickIngest design-system records TASK-45.44.2.1 and TASK-45.44.2.3 by exact path. Left unrelated duplicate-ID records in place.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the remaining Quick Ingest Backlog closeout by moving the completed design, implementation-plan, parent implementation, PR #1751 review, batch metadata, extension playlist handoff, bulk-conference ingest, PR #1814 follow-up, and QuickIngest design-system records to completed storage. This resolves Quick Ingest and bulk-conference active-task ambiguity without renumbering or modifying unrelated duplicate-ID tasks. Verification was metadata-only: exact file moves, active/completed path checks, `git diff --check`, and status review.
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
