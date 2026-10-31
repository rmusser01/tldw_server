---
id: TASK-15195
title: Close ChaChaNotes SQLite/Postgres schema parity gap (v68 vs v72)
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recreation under fresh global max ID (original 13403 clobbered; first recreation 13639 collided with unmerged PR #3243 - allocated without checking open-PR branches). Batch 3 Stage 1-2 per Docs/Plans/2026-10-01-db-parity-structural-debt-implementation-plan.md: audit PG-only migrations v69-72 vs SQLite v68 (ChaChaNotes_DB.py:755-756), classify PORT/DESCOPE, execute with parity tests. No bulk refactor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered from TASK-13643 to TASK-15195 on 2026-10-31 (TASK-13960): TASK-13643 now names only "Move shared chat wire contracts into core ownership" (backlog/tasks/task-13643 - Move-shared-chat-wire-contracts-into-core-ownership.md). Before 2026-10-31 both tasks used TASK-13643; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
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
