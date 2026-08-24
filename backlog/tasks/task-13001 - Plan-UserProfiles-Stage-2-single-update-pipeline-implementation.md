---
id: TASK-13001
title: Plan UserProfiles Stage 2 single-update pipeline implementation
status: Done
created_date: 2026-07-26 03:50
dependencies:
- TASK-13000
labels:
- UserProfiles
- planning
- architecture
priority: High
references:
- TASK-13000
- Docs/superpowers/specs/2026-07-20-userprofiles-single-update-pipeline-stage2-design.md
- commit e54dfcb542
documentation:
- Docs/superpowers/specs/2026-07-20-userprofiles-single-update-pipeline-stage2-design.md
- Docs/superpowers/plans/2026-07-25-userprofiles-single-update-pipeline-stage2-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-07-25-userprofiles-single-update-pipeline-stage2-implementation-plan.md
- backlog/tasks/task-13001 - Plan-UserProfiles-Stage-2-single-update-pipeline-implementation.md
updated_date: 2026-07-26 04:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Translate the requester-approved UserProfiles Stage 2 design into an implementation-ready, test-driven plan reconciled with current origin/dev. This task creates planning and tracking artifacts only; it does not change runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan is reconciled with current origin/dev files, tests, migrations, and caller boundaries.
- [x] #2 Plan decomposes delivery into independently reviewable storage/transaction, membership, typed pipeline/effects, adapter migration, and removal/gate work packages.
- [x] #3 Every design requirement is mapped to an explicit TDD task, verification command, and commit checkpoint.
- [x] #4 Plan contains exact repository paths, concrete interfaces or code shapes, expected failing/passing outcomes, and no placeholders.
- [x] #5 Plan includes SQLite and PostgreSQL concurrency, migration, privacy, cache-generation, compatibility, structural-boundary, Bandit, and regression gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
The master plan defines five sequential, independently reviewed packages: (1) storage and transaction foundations, (2) deterministic membership writer protocol, (3) typed planner/executor/effects pipeline with evaluations fencing, (4) migration of all five adapters plus bulk anchor participation, and (5) transitional removal and release gates. Execution creates one TASK-13001 child per package and permits only one package in progress.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan reconciled against origin/dev 2e0d3f1a2cfcad9798008f5bd249d91bbac43f07. Self-review checked every approved design section for a corresponding task, removed interface ellipses/placeholders, added a cycle-free evaluations models module, centralized AuthNZ transaction policy, closed the transitional direct-writer gap, moved FastAPI-free domain taxonomy earlier, and made reservation/apply subprocess ordering explicit. Verification: 18 tasks across 5 work packages; 130 balanced Markdown fence lines; ASCII-only; placeholder scan clean after excluding valid tuple ellipsis; git diff --check clean. Bandit skipped because this task changes planning/tracking Markdown only. No blockers; implementation must start from a fresh then-current origin/dev worktree.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and committed the approved UserProfiles Stage 2 implementation plan at Docs/superpowers/plans/2026-07-25-userprofiles-single-update-pipeline-stage2-implementation-plan.md. The plan is reconciled with current origin/dev, contains 18 TDD tasks across five sequential review packages, requires one Backlog child per package, and includes SQLite/PostgreSQL concurrency, migrations, all-writer boundaries, evaluation generation fencing, all five adapters, privacy/OpenAPI/Bandit gates, and final transitional removal. Self-review corrected transaction-policy duplication, an evaluations import-cycle risk, the transitional writer gap, domain FastAPI coupling, and ambiguous subprocess ordering. Planning-only verification passed; Bandit was not applicable. Implementation handoff must begin in a fresh worktree from the then-current origin/dev.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Implementation plan is self-reviewed for spec coverage, placeholder-free steps, dependency ordering, and type consistency.
- [x] #8 Planning artifacts pass markdown/ASCII/diff checks and are committed on the isolated design branch.
- [x] #9 Backlog task records the final plan path, verification evidence, and implementation handoff.
<!-- DOD:END -->
