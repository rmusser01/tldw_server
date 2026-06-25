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
updated_date: 2026-06-25 02:58
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Claims Jobs Stage 1 implementation plan. It is saved under Docs/superpowers/plans, linked to the hardened design spec, and covers contracts, enqueue helpers, rebuild/notification/alert handler seams, service routing, WorkerSDK lifecycle startup, dashboard read-only Jobs summaries, verification, Bandit, rollout notes, and commit boundaries.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Backlog task created and updated.
- [x] #2 Implementation plan written and self-reviewed.
- [x] #3 Plan contains no TODO/TBD placeholders.
- [x] #4 Docs-only verification passes.
- [x] #5 Plan is committed separately from implementation.
<!-- DOD:END -->
