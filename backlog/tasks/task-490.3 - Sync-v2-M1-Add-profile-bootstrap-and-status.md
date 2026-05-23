---
id: TASK-490.3
title: 'Sync v2 M1: Add profile bootstrap and status'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- api
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add read-only profile status and explicit profile bootstrap for Chatbook server-connected modes, including default personal dataset creation, device/profile registration, per-domain status, protocol version, and honest server_trusted_v1 at-rest coverage reporting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 GET /api/v1/sync/profile is read-only and never creates durable sync state.
- [ ] #2 POST /api/v1/sync/profile/bootstrap idempotently registers the device/profile and creates or returns the default personal dataset.
- [ ] #3 Profile/status responses include protocol version, domains, cursors, encryption posture, device status, per-domain counts, conflicts, and apply health.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-3-add-profile-bootstrap-and-status
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
