---
id: TASK-12072
title: Design Scheduled Tasks Phase 4C Recurring Question execution
status: In Progress
labels:
- scheduled-tasks
- phase-4c
- design
- ux
- api-first
priority: high
references:
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md
- Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md
- Docs/ADR/003-jobs-vs-scheduler-default.md
- Docs/superpowers/specs/2026-06-30-scheduled-tasks-phase4c-recurring-question-execution-design.md
documentation:
- Docs/superpowers/specs/2026-06-30-scheduled-tasks-phase4c-recurring-question-execution-design.md
modified_files:
- Docs/superpowers/specs/2026-06-30-scheduled-tasks-phase4c-recurring-question-execution-design.md
- backlog/tasks/task-12072 - Design-Scheduled-Tasks-Phase-4C-Recurring-Question-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the API-first product/UX and backend dependency design for Scheduled Tasks Phase 4C: executable Recurring Question tasks. This phase should define how users/API clients create recurring unanswered-question monitors over newly ingested/searchable data, how runs execute, how results surface, and what backend work must exist before implementation. Keep GitHub/YouTube as examples only, preserve Watchlists as a separate persona/job, and do not include Agent Task execution beyond dependency boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec is grounded in the current Phase 4 contract, Phase 4B API foundation, Jobs vs Scheduler guidance, RAG/search capabilities, and results inbox/home surfacing model.
- [ ] #2 Spec defines Recurring Question task lifecycle, schedule configuration, supported scopes, prompt/question configuration, preview/safety checks, run/result model, visibility policy, retention/audit/RBAC expectations, and WebUI demo/main-client behavior.
- [ ] #3 Spec explicitly identifies backend/API dependencies without over-designing implementation details and separates Phase 4C from Watchlists and Phase 4D Agent Tasks.
- [ ] #4 Spec includes risks, open questions, proposed defaults, acceptance criteria, and staged implementation recommendations suitable for a follow-up implementation plan.
- [ ] #5 User reviews and approves the design before implementation planning starts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design-first task. Implementation plan will be written only after the Phase 4C design spec is reviewed and approved.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Drafted Phase 4C Recurring Question execution design. Incorporated user-approved decisions: capability-driven RAG scopes, Mark solved in first slice, findings_only default surfacing, Run now, evidence-only fallback when generation is unavailable, and durable run records for every execution. Spec review loop found and fixed capability status vocabulary drift and solved/reopen lifecycle ambiguity, then returned Approved with no blocking issues. Verification: placeholder scan found no TODO/TBD/FIXME markers; required Phase 4C terms and referenced files were present; Bandit skipped because this task touched documentation/backlog only and no executable code.
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
