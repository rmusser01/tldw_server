---
id: TASK-490.13.2
title: 'Sync v2 M3: Device lifecycle and acknowledgments'
status: To Do
labels:
- sync
- sync-v2
- m3
- devices
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Sync v2 devices user-manageable and add authorization plus per-device acknowledgment primitives required for background sync, retention, and safe revocation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can list, update, pause, authorize, and revoke registered devices with profile/domain status details.
- [ ] #2 Revoked devices fail closed across push, pull, restore, blob, conflict, repair, and key recovery APIs while historical envelopes remain auditable.
- [ ] #3 Per-device domain and blob acknowledgments are persisted idempotently and exposed for later retention/GC decisions.
<!-- AC:END -->

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
