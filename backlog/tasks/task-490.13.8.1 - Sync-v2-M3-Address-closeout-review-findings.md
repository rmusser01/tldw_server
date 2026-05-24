---
id: TASK-490.13.8.1
title: 'Sync v2 M3: Address closeout review findings'
status: Done
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m3
- review-fix
priority: medium
parent_task_id: TASK-490.13.8
modified_files:
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
- backlog/tasks/task-490.13.8 - Sync-v2-M3-End-to-end-verification-and-docs.md
- backlog/tasks/task-490.13.8.1 - Sync-v2-M3-Address-closeout-review-findings.md
- tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
- tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
- tldw_Server_API/tests/Sync/test_sync_v2_models.py
- tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address code review findings from the Sync v2 M3 closeout slice: expose documented encryption_policies in the capabilities API response, correct Stage 8/Backlog wording around deferred conflict preview behavior, and strengthen E2E acknowledgment retention-safety assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /sync/capabilities response schema and endpoint tests include documented encryption_policies without leaking private material.
- [x] #2 Stage 8 plan and Backlog wording distinguish implemented conflict resolution/listing coverage from deferred conflict preview endpoints.
- [x] #3 M3 E2E asserts device acknowledgments are recorded at the expected sequence and clear retention unacknowledged-device blockers.
- [x] #4 Focused tests, full Sync suite, Ruff, Bandit for touched production scope, and diff checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Verified review findings before implementation: SyncV2Service returned `encryption_policies` but `SyncCapabilitiesResponse` dropped it; Stage 8/Backlog wording overstated conflict preview coverage; the E2E ACK assertions needed to target retention-safety behavior rather than stateless explicit-cursor background status.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed Sync v2 M3 closeout review findings. SyncCapabilitiesResponse now preserves the documented encryption_policies field from core capabilities and endpoint/register tests cover it. Stage 8 plan and TASK-490.13.8 now state conflict resolution coverage with documented preview endpoint deferral. The M3 E2E now proves retention candidates are blocked before device ACKs, ACK responses record the expected notes.note through_server_sequence, and retention candidates clear the unacknowledged-device blocker after ACKs. Verification: focused schema/endpoint capability tests passed with 3 tests; targeted M3 E2E passed; full restore E2E file passed with 6 tests; full Sync suite passed with 412 tests and 6 warnings; Ruff passed on touched Python files; Bandit on tldw_Server_API/app/api/v1/schemas/sync_v2_models.py reported 0 findings at /tmp/bandit_sync_v2_m3_closeout_review.json; git diff --check passed; stale wording scan returned no matches.
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
