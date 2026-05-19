---
id: TASK-431
title: Add pollable setup readiness provisioning
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-18 23:38'
labels:
  - implementation
  - setup
  - backend
  - api
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
  - >-
    Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the fourth backend slice from Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md: persisted setup readiness state and a pollable provision endpoint that applies confirmed preview config updates, queues installer work without blocking downloads, and exposes status polling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Setup readiness state persists selected profile, lanes, overlays, preview/provision metadata, and operation status via a JSON store.
- [x] #2 Preview returns a preview_id and persists only sanitized readiness state.
- [x] #3 Provision requires confirmed=true, applies previewed config updates, queues non-empty install plans without blocking downloads, and returns 202 with /api/v1/setup/readiness/status.
- [x] #4 Readiness status exposes persisted operation_id and operation_status for polling.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 4 implementation from Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md. Verification: pytest setup readiness slice passed with 17 tests; Bandit JSON at /tmp/bandit_first_time_readiness_provision.json has zero findings.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added SetupReadinessStore with atomic JSON writes and known lane/status/overlay validation. Preview persists sanitized state with preview_id. Provision applies confirmed config updates, queues installer work through BackgroundTasks, and merges persisted operation state into readiness status.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pollable setup readiness provisioning is implemented for the first-run backend path. Preview now returns a preview_id and persists sanitized readiness state, provision requires confirmed=true, applies previewed config updates, queues non-empty install plans through BackgroundTasks, and exposes operation status through /api/v1/setup/readiness/status. Targeted setup-readiness tests pass and Bandit reports zero findings for the touched backend files.
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
