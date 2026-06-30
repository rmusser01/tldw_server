---
id: TASK-230
title: Add Sync v2 restore e2e coverage and docs
status: Done
assignee: []
created_date: '2026-05-10 15:29'
updated_date: '2026-05-10 15:34'
labels:
  - sync
  - docs
  - e2e
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-10-chatbook-sync-engine-implementation-plan.md
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 End-to-end restore test covers device registration dataset enrollment encrypted envelope push second-device restore manifest selected pull and duplicate idempotency
- [x] #2 Test verifies restore manifest and server-visible representations do not expose private plaintext
- [x] #3 Sync v2 docs describe endpoint overview protocol invariants encryption expectations legacy send/get policy restore flow conflict policy and known limits
- [x] #4 Focused Sync v2 tests plus e2e restore coverage pass
- [x] #5 Bandit on touched Sync v2 backend scope reports no new findings
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing server e2e test for the Chatbook Sync v2 restore scenario. 2. Fill only integration gaps uncovered by the e2e test. 3. Add Sync v2 backend docs covering protocol invariants restore and conflict behavior. 4. Run focused Sync v2 tests e2e coverage Bandit and diff checks. 5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a server e2e restore roundtrip covering device A registration dataset enrollment encrypted note chat workspace source-ref and source-cache envelope push duplicate idempotency device B restore manifest selected pull key recovery readiness and private plaintext exclusion. The red run failed because SyncRestoreManifestDataset dropped attachment_availability at the API schema boundary; implementation preserved that service-computed field in the response schema. Added Sync v2 API/design docs and expanded the core Sync README with protocol invariants restore conflict privacy and operational-limit details.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Sync v2 restore e2e coverage and documentation. The new e2e validates the Chatbook-style restore path across register enroll push restore-manifest pull and duplicate idempotency while asserting private plaintext and wrapped key material stay out of server-visible manifest/recovery metadata. Docs now cover Sync v2 endpoints protocol invariants encryption expectations legacy send/get policy restore flow conflict policy limits and architecture. Verification: red e2e failed on missing attachment_availability; after fix, tldw_Server_API/tests/Sync plus tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py passed 133/133; Bandit on touched Sync backend scope reported 0 findings; git diff --check clean.
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
