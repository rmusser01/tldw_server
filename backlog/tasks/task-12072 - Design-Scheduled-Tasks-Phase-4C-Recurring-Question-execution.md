---
id: TASK-12072
title: Design Scheduled Tasks Phase 4C Recurring Question execution
status: Done
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
updated_date: 2026-08-24 06:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the API-first product/UX and backend dependency design for Scheduled Tasks Phase 4C: executable Recurring Question tasks. This phase should define how users/API clients create recurring unanswered-question monitors over newly ingested/searchable data, how runs execute, how results surface, and what backend work must exist before implementation. Keep GitHub/YouTube as examples only, preserve Watchlists as a separate persona/job, and do not include Agent Task execution beyond dependency boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec is grounded in the current Phase 4 contract, Phase 4B API foundation, Jobs vs Scheduler guidance, RAG/search capabilities, and results inbox/home surfacing model.
- [x] #2 Spec defines Recurring Question task lifecycle, schedule configuration, supported scopes, prompt/question configuration, preview/safety checks, run/result model, visibility policy, retention/audit/RBAC expectations, and WebUI demo/main-client behavior.
- [x] #3 Spec explicitly identifies backend/API dependencies without over-designing implementation details and separates Phase 4C from Watchlists and Phase 4D Agent Tasks.
- [x] #4 Spec includes risks, open questions, proposed defaults, acceptance criteria, and staged implementation recommendations suitable for a follow-up implementation plan.
- [x] #5 User reviews and approves the design before implementation planning starts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design-first task. Implementation plan will be written only after the Phase 4C design spec is reviewed and approved.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Drafted Phase 4C Recurring Question execution design. Incorporated user-approved decisions: capability-driven RAG scopes, Mark solved in first slice, findings_only default surfacing, Run now, evidence-only fallback when generation is unavailable, and durable run records for every execution. Spec review loop found and fixed capability status vocabulary drift and solved/reopen lifecycle ambiguity, then returned Approved with no blocking issues. Verification: placeholder scan found no open placeholder markers; required Phase 4C terms and referenced files were present; Bandit skipped because this task touched documentation/backlog only and no executable code.

Additional planning-risk review found and addressed four ambiguity risks before implementation planning: empty-scope dry runs are explicitly out of scope for 4C and must fail as `scope_empty`; `all_searchable_library` now resolves only to capability-reported searchable sources readable by the owner; scheduled/manual run creation now revalidates active owner context and current readable scope; `generation_mode` now uses existing RAG defaults/profiles or preview-validated safe overrides rather than adding a provider/model selection UX.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and approved the API-first Scheduled Tasks Phase 4C Recurring Question execution design. The spec defined capability-driven scopes, safe preview and owner revalidation, durable every-run history, findings-only Home surfacing, mark-solved/reopen behavior, Watchlists separation, backend dependencies, risks, and staged implementation guidance. The design was subsequently implemented and merged in PR #2566. Documentation-only verification and the documented spec review loop passed; Bandit was not applicable to the design slice.
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
