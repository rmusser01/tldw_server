---
id: TASK-490.13.4
title: 'Sync v2 M3: Workspace dataset foundation'
status: To Do
labels:
- sync
- sync-v2
- m3
- workspace
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Introduce workspace-scoped Sync v2 datasets with permission and key-policy boundaries before enabling broad collaborative content sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dataset scope supports personal and workspace datasets with fail-closed workspace membership checks.
- [ ] #2 All dataset-scoped sync, blob, restore, conflict, repair, and key APIs re-check workspace permission for workspace datasets.
- [ ] #3 Initial workspace domains are limited to workspace metadata/source references until collaborative Notes/Chat semantics are separately designed.
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
