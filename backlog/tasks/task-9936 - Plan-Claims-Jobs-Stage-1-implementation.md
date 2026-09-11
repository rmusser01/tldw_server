---
id: TASK-9936
title: Plan Claims Jobs Stage 1 implementation
status: Done
created_date: 2026-06-25 02:48
labels:
- claims
- jobs
- refactor
- plan
priority: high
references:
- Docs/superpowers/specs/2026-06-24-claims-jobs-operational-control-plane-design.md
- tldw_Server_API/app/core/Claims_Extraction
- tldw_Server_API/app/core/Jobs
documentation:
- Docs/superpowers/plans/2026-06-25-claims-jobs-stage1-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-25-claims-jobs-stage1-implementation-plan.md
- backlog/tasks/task-9936 - Plan-Claims-Jobs-Stage-1-implementation.md
updated_date: 2026-06-25 03:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a concrete TDD implementation plan for Stage 1 of the Claims Jobs refactor: Claims job contracts, enqueue helpers, handlers, service routing, worker startup, tests, rollout verification, and keeping Jobs as the only queue/lifecycle owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan is saved under Docs/superpowers/plans with a unique Claims Jobs Stage 1 filename.
- [x] #2 Plan maps file responsibilities before task steps.
- [x] #3 Plan decomposes Stage 1 into TDD tasks with exact file paths, commands, and expected results.
- [x] #4 Plan explicitly preserves Jobs ownership of queue/lifecycle behavior and avoids Claims-side queue mechanics.
- [x] #5 Plan includes verification, Bandit, rollout, and commit guidance.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote the Stage 1 Claims Jobs implementation plan at Docs/superpowers/plans/2026-06-25-claims-jobs-stage1-implementation-plan.md. The plan maps file responsibilities, decomposes the work into TDD tasks, keeps Jobs as the only queue/lifecycle owner, and includes focused pytest, Jobs owner/idempotency, Bandit, and git diff verification commands. Self-review tightened the plan to remove vague fixture guidance, replace broad exception guidance with the existing Claims noncritical tuple, and document PostgreSQL-safe monitoring event ID return behavior. Verification: git diff --check passed; unfinished-marker scan over the plan found no matches for TODO/TBD/deferred implementation wording.
Reopened for pre-subagent plan review. Validated plan issues before execution: bulk review needs owner tracking for rebuild Jobs, job handlers should avoid initializing missing user DBs, alert Slack delivery must preserve existing ratio/details payload, and the alert delivery handler should avoid importing private claims_service helpers when a smaller extraction seam can be planned.
Pre-subagent plan review complete. Updated the plan so review notification delivery opens existing DBs with initialize=False, alert delivery is extracted into a reusable domain helper instead of imported from claims_service, alert handler tests verify Slack payload preservation and no DB initialization, and bulk review rebuild enqueueing carries owner_user_id per media id. Verification: git diff --check passed for plan/task files; stale helper/private-import/placeholder scan returned no matches.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the pre-subagent review and hardening pass for the Claims Jobs Stage 1 implementation plan. The plan now closes the validated handoff issues around owner propagation, existing-DB handling, alert payload parity, and alert delivery module boundaries, and is ready for subagent-driven execution.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Backlog task created and updated.
- [x] #2 Implementation plan written and self-reviewed.
- [x] #3 Plan contains no TODO/TBD placeholders.
- [x] #4 Docs-only verification passes.
- [x] #5 Plan is committed separately from implementation.
<!-- DOD:END -->
