---
id: TASK-490.13
title: 'Sync v2 M3: Polished multi-device sync'
status: In Progress
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m3
- roadmap
- multi-device
priority: medium
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
modified_files:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
- backlog/tasks/task-490.13 - Sync-v2-M3-Polished-multi-device-sync.md
- backlog/tasks/task-490.13.1 - Sync-v2-M3-Refine-requirements-and-implementation-plan.md
- backlog/tasks/task-490.13.2 - Sync-v2-M3-Device-lifecycle-and-acknowledgments.md
- backlog/tasks/task-490.13.3 - Sync-v2-M3-Background-sync-policy-and-status.md
- backlog/tasks/task-490.13.4 - Sync-v2-M3-Workspace-dataset-foundation.md
- backlog/tasks/task-490.13.5 - Sync-v2-M3-Broader-domain-expansion.md
- backlog/tasks/task-490.13.6 - Sync-v2-M3-Stricter-encryption-and-key-rotation.md
- backlog/tasks/task-490.13.7 - Sync-v2-M3-Retention-GC-and-observability.md
- backlog/tasks/task-490.13.8 - Sync-v2-M3-End-to-end-verification-and-docs.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roadmap epic for Milestone 3 after M1/M2 mature: scheduled/background sync, workspace datasets, broader domain coverage, richer conflict UX, passphrase/device-key unlock, client-only encrypted datasets, device authorization/revocation, key rotation, retention/GC, and observability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 M3 requirements are refined after M1 and M2 outcomes are known.
- [ ] #2 Workspace dataset and permission/key model are designed before implementation.
- [ ] #3 Background sync, device lifecycle, retention, and observability have explicit product and operational success criteria.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TASK-490.13.1 completed the M3 planning gate with design, API draft, implementation plan, and child tasks.
- Implementation should proceed with TASK-490.13.2 first: device lifecycle and acknowledgments. That slice is a prerequisite for background sync leases/status and later retention/GC safety.
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
