---
id: TASK-15150
title: Migrate acquired-job completion callers to UUID preconditions
status: To Do
created_date: 2026-09-07 20:05
dependencies:
- TASK-15149
labels:
- jobs
- hardening
- compatibility
priority: Medium
references:
- TASK-13215
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After TASK-13215 establishes the optional complete_job expected_uuid contract, inventory direct non-WorkerSDK callers that complete previously acquired jobs and migrate suitable paths to pass the acquired UUID. Preserve numeric-ID-only compatibility for intentional admin completion and update strict test doubles without adding a fallback that silently drops the UUID precondition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every production direct complete_job caller is classified as acquired-worker, intentional admin, or another explicitly documented category.
- [ ] #2 Each acquired-worker caller with a durable UUID passes expected_uuid and has stale-row rejection coverage.
- [ ] #3 Intentional admin callers retain documented numeric-ID compatibility unless their contract is explicitly changed.
- [ ] #4 Strict test doubles are updated to the new keyword contract; production code never retries completion after dropping expected_uuid on TypeError.
- [ ] #5 Focused caller tests, relevant Jobs regressions, and scoped Bandit pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered from TASK-13216 to TASK-15150 on 2026-10-31 (TASK-13960): TASK-13216 now names only "Expose Writing Predict and Fill prompts through Service Prompts" (backlog/tasks/task-13216 - Expose-Writing-Predict-and-Fill-prompts-through-Service-Prompts.md). Before 2026-10-31 both tasks used TASK-13216; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
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
