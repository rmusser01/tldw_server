---
id: TASK-490.13
title: 'Sync v2 M3: Polished multi-device sync'
status: Done
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
- tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roadmap epic for Milestone 3 after M1/M2 mature: scheduled/background sync, workspace datasets, broader domain coverage, richer conflict UX, passphrase/device-key unlock, client-only encrypted datasets, device authorization/revocation, key rotation, retention/GC, and observability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 M3 requirements are refined after M1 and M2 outcomes are known.
- [x] #2 Workspace dataset and permission/key model are designed before implementation.
- [x] #3 Background sync, device lifecycle, retention, and observability have explicit product and operational success criteria.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TASK-490.13.1 completed the M3 planning gate with design, API draft, implementation plan, and child tasks.
- TASK-490.13.2 through TASK-490.13.8 completed the M3 foundation in planned order: device lifecycle/acknowledgments, background sync status, workspace datasets, broader metadata domains, stricter key policy/rotation, retention/diagnostics, and E2E closeout.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
M3 polished multi-device sync foundation completed across child tasks TASK-490.13.1 through TASK-490.13.8. Delivered planning docs, device lifecycle and acknowledgments, background policy/leases/status, workspace dataset foundation, source-cache/media/workspace metadata domains, stricter key metadata and key rotation, retention dry-run/guarded compaction, diagnostics, and end-to-end closeout coverage. Explicit deferrals remain documented: physical blob byte deletion, destructive envelope audit-log deletion, broad workspace Notes/Chat materialization, conflict summary/preview endpoints, passphrase/device-key unlock UX, and full client-only encrypted editing. Final Stage 8 verification: ruff check tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py passed; pytest tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py -q passed with 6 tests; pytest tldw_Server_API/tests/Sync -q passed with 412 tests; git diff --check passed; stale-doc phrase scan returned no matches. Bandit for Stage 8 skipped because only docs/tests/backlog were touched in the closeout slice; prior production-code M3 slices recorded their Bandit runs.
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
