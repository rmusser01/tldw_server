---
id: TASK-2349
title: Design Scheduled Tasks Phase 4B backend API foundation
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-09 23:47
labels:
- scheduled-tasks
- api
- design
dependencies: []
references:
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md
documentation:
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md
priority: high
modified_files:
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md
- backlog/tasks/task-2349 - Design-Scheduled-Tasks-Phase-4B-backend-API-foundation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the Scheduled Tasks Phase 4B backend/API foundation design spec for API-owned Recurring Question and Agent Task definitions, durable previews, lifecycle management, audit, and WebUI reference-client behavior without execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Phase 4B design spec captures approved scope: Scheduled Tasks-owned persisted definitions, durable previews, lifecycle management, audit, and WebUI reference-client behavior without execution.
- [x] #2 Spec review loop findings are addressed or surfaced for user review.
- [x] #3 Spec document is committed on an isolated worktree branch.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Drafted the Phase 4B backend/API foundation spec from the approved brainstorming sections.
Created isolated worktree branch codex/scheduled-tasks-phase4b-api-foundation-spec from origin/dev to avoid the dirty main checkout.
Ran three spec-review subagent passes. Passes 1 and 2 found blocking/important issues and the spec was revised. Pass 3 found two remaining important issues; both were patched locally and recorded in the Spec Review section.
Spec status is Needs User Review because the review workflow capped at three subagent iterations and the final corrections need human review before implementation planning.
User-requested self-review found additional implementation-risk gaps: owner scoping in the data model/idempotency contract, health precedence, duplicate audit determinism, disabled-source duplicate guardrails, and normalized `automation_definition` status mapping for the existing WebUI list model. The spec was revised and the addendum was recorded in the Spec Review section.
Implementation-plan review added `disabled_lock_kind` and `disabled_reason` to make the disabled duplicate guardrail implementable.
Verification: git diff --check HEAD^ HEAD passed after the amended self-review commit; git status --short --branch showed the worktree clean and ahead of origin/dev by one commit.
Bandit: not run because this task touched only documentation and Backlog metadata, no Python/code paths.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and reviewed the Phase 4B backend/API foundation design spec. The spec is approved for implementation planning.
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
