---
id: TASK-12017
title: Implement Jobs backend parity refactor first slice
status: In Progress
created_date: 2026-06-24 21:44
labels:
- jobs
- implementation
- refactor
priority: medium
references:
- TASK-12015
- TASK-12016
- Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
- Docs/superpowers/plans/2026-06-24-jobs-backend-parity-refactor-implementation-plan.md
documentation:
- Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
- Docs/superpowers/plans/2026-06-24-jobs-backend-parity-refactor-implementation-plan.md
modified_files:
- Docs/Design/JOBS_BACKEND_PARITY_INVENTORY_2026_06_24.md
- backlog/tasks/task-12017 - Implement-Jobs-backend-parity-refactor-first-slice.md
updated_date: 2026-06-24 21:57
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the first safety-net PR from the Jobs backend parity implementation plan. Scope includes inventory, shared SQLite/Postgres parity scenarios, public admin and Chatbooks mapping contract tests, JobsSettings semantics, operation result contracts, and verification gates. Production SQL extraction is explicitly out of scope for this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 inventory created at Docs/Design/JOBS_BACKEND_PARITY_INVENTORY_2026_06_24.md. It classifies admin direct SQL, read-model SQL, service/worker operational SQL, and first-slice domain mapping coverage.
Follow-up inventory update added the public admin stale-processing read-model boundary and clarified the Prompt Studio status dashboard first-slice action.
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
