---
id: TASK-12803
title: Design Jobs backend parity refactor
status: Done
created_date: 2026-06-24 17:46
labels:
- jobs
- design
- refactor
priority: medium
modified_files:
- Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
- backlog/tasks/task-12015 - Design-Jobs-backend-parity-refactor.md
updated_date: 2026-06-24 19:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Capture the approved design for a long-term stability/pragmatism focused Jobs refactor: parity harness first, then incremental backend operation extraction behind the existing JobManager facade.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec documents goals, constraints, architecture, data flow, error handling, and testing strategy.
- [x] #2 Spec preserves REST/API response compatibility as the primary boundary.
- [x] #3 Spec includes reviewed improvements: real Postgres subset, typed outcomes, explicit side-effect cardinality, and no schema churn in the first slice unless required.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write the Jobs backend parity refactor design spec under Docs/superpowers/specs. 2. Self-review for placeholders, contradictions, scope creep, and ambiguous requirements. 3. Commit the spec and Backlog task together. 4. Ask the user to review before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote approved design spec at Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md. Self-review checked for placeholders, contradictions, scope creep, and ambiguous requirements; no spec placeholders or whitespace issues found. Documentation-only change; Bandit not applicable to this spec-only task.
Follow-up design review found two implementation-planning risks and amended the spec: existing Jobs admin endpoints still contain direct backend-specific SQL outside JobManager, and a JobsSettings extraction could accidentally freeze env-derived behavior currently changed per test/runtime path. Added endpoint-SQL inventory, existing paired-test consolidation, existing pg_jobs/jobs_pg_dsn fixture alignment, minimum first-PR scope, and explicit settings snapshot/refresh requirements.
Second follow-up review widened the direct-SQL boundary: Jobs table access exists outside jobs_admin.py in status, metrics, webhook, worker, and domain-service read/operational paths. Amended the spec to require classifying direct runtime SQL as state-changing, read-only/status, service/worker operational, or migration/bootstrap. Also recorded that Jobs-marked tests require RUN_JOBS=1 and that core parity tests should remain independent of FastAPI startup because the global lifecycle reset fixture imports the main app.
Third follow-up review found the spec still treated public API compatibility too generically. Existing domain adapters intentionally translate core Jobs status/id fields, with seed mappings documented in Docs/Product/Job_System_Unification_Mapping_Matrix.md. Amended the spec to require a domain mapping inventory and at least one non-identity domain adapter contract before production extraction begins.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Captured and amended the approved Jobs backend parity refactor design: parity harness first, field-level REST/domain compatibility, typed backend operation outcomes, facade-owned public compatibility, transactional durable outbox facts, narrow real-Postgres semantic coverage, direct runtime SQL inventory/classification, domain status/id mapping inventory, explicit JobsSettings snapshot/refresh semantics, existing Jobs test fixture alignment, and incremental admission/lifecycle extraction behind JobManager.
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
