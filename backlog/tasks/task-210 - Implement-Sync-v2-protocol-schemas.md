---
id: TASK-210
title: Implement Sync v2 protocol schemas
status: Done
assignee: []
created_date: '2026-05-10 02:28'
updated_date: '2026-05-10 02:43'
labels:
  - sync
  - api
  - schemas
dependencies:
  - TASK-209
documentation:
  - >-
    Docs/superpowers/plans/2026-05-10-chatbook-sync-engine-implementation-plan.md
  - Docs/superpowers/specs/2026-05-10-chatbook-sync-engine-prd-design.md
  - tldw_Server_API/app/api/v1/schemas/sync_server_models.py
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync v2 Pydantic models are added in tldw_Server_API/app/api/v1/schemas/sync_v2_models.py.
- [x] #2 Schema tests cover private-payload plaintext rejection, per-envelope push outcomes, attachment upload response models, and required adapter_version on envelopes.
- [x] #3 The models include capabilities, device registration, dataset enrollment, restore manifest, envelope, push, pull, attachment, conflict, and recovery-bundle request/response shapes.
- [x] #4 Focused pytest for tldw_Server_API/tests/Sync/test_sync_v2_models.py passes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Sync v2 schema models and focused schema tests in commits 132202f8b and aceed2993. Spec compliance review passed. Code-quality review initially requested validation hardening; follow-up commit replaced private payload clear-field denylist with an allowlist and rejects mismatched push/envelope dataset IDs. Final code-quality review approved.

Verification: python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py -v passed with 7 tests. git diff --check passed. Worker reported Bandit on touched schema/test files passed with 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 1 Sync v2 protocol schemas and tests. The models cover capabilities, devices, datasets, restore manifest, envelopes, push/pull, attachments, conflicts, and key recovery. Private client payload clear metadata now uses a conservative allowlist and push requests reject cross-dataset envelope mismatches.
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
