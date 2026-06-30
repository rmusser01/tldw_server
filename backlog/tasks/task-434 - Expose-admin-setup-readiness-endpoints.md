---
id: TASK-434
title: Expose admin setup readiness endpoints
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-18 23:45'
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
Add admin-gated setup readiness profiles/status/preview/provision/verify endpoints so the same readiness screen can be used after first-run setup completes, matching the design decision that first-run local access is unauthenticated only while /setup is required and post-setup readiness is admin-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Admin setup readiness profiles/status/preview/provision/verify endpoints exist under /api/v1/setup/admin/readiness/*.
- [x] #2 Admin setup readiness routes require admin-equivalent setup access.
- [x] #3 Admin readiness routes work after first-run setup is completed or disabled.
- [x] #4 First-run readiness routes keep the local setup access guard.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Added Task 5.5 coverage and implementation for admin setup readiness route parity. Verification: pytest setup readiness slice passed with 26 tests; Bandit JSON at /tmp/bandit_first_time_readiness_admin.json has zero findings for setup.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shared the first-run readiness helpers with admin routes by adding allow_completed_when_disabled handling and routing admin endpoints through require_shared_audio_installer_access.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Admin setup readiness route parity is implemented. The same profiles/status/preview/provision/verify behavior is available under /api/v1/setup/admin/readiness/* with the shared admin setup dependency, while first-run routes remain local-only and unauthenticated only while setup is required.
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
