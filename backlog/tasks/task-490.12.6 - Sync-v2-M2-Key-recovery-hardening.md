---
id: TASK-490.12.6
title: 'Sync v2 M2: Key recovery hardening'
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 18:34'
labels:
  - sync
  - sync-v2
  - m2
  - security
dependencies: []
documentation:
  - Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md
  - >-
    Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
parent_task_id: TASK-490.12
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden server-unlocked key recovery readiness and validation for M2 while leaving passphrase/device-key and client-only encryption modes for M3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recovery bundle validation checks dataset ownership, key purpose, wrapping metadata, device association, and revoked status.
- [x] #2 Restore manifest/preview report active recovery readiness and warn when a selected dataset has no active recovery bundle.
- [x] #3 Tests verify wrapped key material is not exposed in API errors or logs from touched key recovery paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Started from Stage 5 of the Sync v2 M2 restore completeness/blobs implementation plan.
- Added service-level key recovery bundle validation for dataset ownership, dataset_recovery purpose, registered device association, non-empty wrapping/KDF metadata, wrapped-key size, and active non-revoked rotation target references.
- Restore preview now emits sync_key_recovery_missing when a selected dataset has no active recovery bundle while manifest/key_status continue to report active readiness.
- API validation failures map to the generic sync_validation_failed response and tests capture Loguru warning output to verify wrapped key material and KDF secrets are not exposed.
- Updated M2 design docs and Stage 5 plan status to document the M2 server-unlocked recovery contract and M3 deferrals.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Sync v2 M2 key recovery hardening. Recovery bundle writes now fail closed for unsupported key purpose, missing wrapping metadata, unregistered device association, inaccessible datasets, oversized/empty wrapped material, and missing or revoked rotation targets. Restore preview now surfaces sync_key_recovery_missing for selected datasets without active recovery bundles while manifest readiness excludes revoked records. API validation errors remain generic and the new endpoint test verifies wrapped key material and KDF secrets do not appear in HTTP responses or captured warning logs. Verification: targeted red/green tests passed, the full Sync suite passed (313 passed, 6 warnings), restore e2e passed (4 passed), Ruff passed on touched files, Bandit passed with no findings at /tmp/bandit_sync_v2_m2_key_recovery.json, and git diff --check passed.
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
