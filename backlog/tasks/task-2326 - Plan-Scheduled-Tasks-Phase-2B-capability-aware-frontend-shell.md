---
id: TASK-2326
title: Plan Scheduled Tasks Phase 2B capability-aware frontend shell
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 01:55'
labels:
  - scheduled-tasks
  - webui
  - ux
  - phase-2b
  - implementation-plan
dependencies: []
references:
  - TASK-2324
  - TASK-2325
  - >-
    Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase2b-capability-aware-frontend-shell-implementation-plan.md
modified_files:
  - >-
    Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase2b-capability-aware-frontend-shell-implementation-plan.md
  - >-
    backlog/tasks/task-2326 -
    Plan-Scheduled-Tasks-Phase-2B-capability-aware-frontend-shell.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write an implementation plan for Scheduled Tasks Phase 2B.2 Capability-aware frontend shell. The plan should translate the hardened Watch/Ingest product contract into frontend-first work: runtime capability state modeling, Limited availability display, generated copy from metadata, source-intent capability handling, notification/result destination copy, redaction-safe preview messaging, and tests that prevent Watch/Ingest from becoming Available until all gates pass. Scope remains planning/docs only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan is saved under Docs/superpowers/plans with exact file paths, tasks, tests, commands, expected outcomes, and commit boundaries.
- [x] #2 Plan keeps Phase 2B.2 frontend-shell scoped and does not implement Watchlists backend contracts or promote Watch/Ingest to Available before all gates pass.
- [x] #3 Plan covers template capability states, Limited availability, source-intent capability metadata, generated result/notification copy, redaction rules, and extension-sized behavior.
- [x] #4 Plan includes focused tests for capability state gating, copy generation, unavailable/limited states, template filtering, route behavior, and regression protection against overpromising Home/search/RAG/notifications.
- [x] #5 Verification is recorded and Bandit is documented as not applicable for docs-only planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the Phase 2B.2 capability-aware frontend shell implementation plan. The plan keeps the slice frontend-only, adds a pure capability overlay model, Limited availability, metadata-generated result/notification copy, redaction helpers, and tests that prevent Watch/Ingest from being treated as Available before all gates pass. Plan-document-reviewer subagent was not spawned because the available subagent tool requires explicit user permission for delegation; performed a local self-review instead.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Phase 2B.2 capability-aware frontend shell implementation plan. It maps the hardened Watch/Ingest contract into TDD tasks for capability modeling, Limited availability, generated copy, Create panel integration, page wiring, and focused verification while explicitly excluding backend contracts and Watchlists creation adapters. Verification: git diff --check passed; unresolved planning marker scan passed. Bandit is not applicable because this is documentation/backlog-only planning work.
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
