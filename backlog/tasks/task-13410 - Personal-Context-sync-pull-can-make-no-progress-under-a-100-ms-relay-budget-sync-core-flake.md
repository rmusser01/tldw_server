---
id: TASK-13410
title: >-
  Personal Context sync pull can make no progress under a 100 ms relay budget
  (sync-core flake)
status: Done
assignee: []
created_date: '2026-10-01 17:53'
updated_date: '2026-10-02 18:33'
labels:
  - bug
  - sync
  - testing
  - flaky
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tests/Sync/test_sync_v2_endpoints.py::test_personal_context_endpoints_use_real_factory_bootstrap_and_complete_flow fails intermittently: after a successful push it polls /api/v1/sync/pull up to 10 times back-to-back and every response is envelopes=[], next_cursor='0', has_more=true. Seen in CI (sync-core, #3063 run 36693239728, 2026-09-30) and locally 3/4 and 2/4 failures, including on a frontend-only tree. PersonalContextRelay.relay_profile bounds each pull to row_budget=100 and wall_time_ms=100. If the fixed per-pull work exceeds 100 ms on a slow host, the cursor never advances, which is a possible liveness bug and not only a test issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root cause identified: whether a pull can exhaust its budget without advancing the relay cursor
- [x] #2 Either the relay guarantees forward progress per pull, or the test waits on relay state with a deadline instead of 10 tight polls; the test passes reliably
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (product liveness bug, not the pull budget). This test freezes the pull clock (_recovery_clock_ns = lambda: 0), so the pull's 100 ms budget never expires here. The stranded row comes from the relay inside prepare_personal_context_activation, which uses the real monotonic clock. _relay_owned checked the deadline after every step of a row, including right after stage_authority had committed the hidden home-authority envelope to Sync. On a slow host it abandoned the row there: the Sync envelope stayed apply_status=pending and record_staged_row never ran. Normally a later relay re-stages and finishes such a row, but the activation install then marks every batch up to its watermark covered_by_activation, so no relay ever revisits that batch. The pending envelope stays forever. Effects: (1) require_materialization_predecessors_applied rejects every later projection in the dataset (push returns apply_status=pending, sync_projection_predecessor_unresolved). (2) The PC pull scan stops at the unapplied ingress, so every pull returns envelopes=[], next_cursor='0', has_more=true, with relay state 'complete'. That is the CI symptom. Reproduced deterministically with a temporary 60 ms sleep before stage_authority (3/3 failures); instrumentation showed the manifest staged at cursor 2 and abandoned when the post-stage deadline check failed.

Fix (tldw_Server_API/app/core/Sync/v2/personal_context_relay.py): the deadline now stops a row only before its Sync write. Once the current attempt has staged a row, the later _renewed_current checks for that row run without the deadline (fence=None), so the row is recorded, acknowledged and finalized and at most one row overruns the budget. These still stop at the deadline as before: the next row, carried-over staged/acknowledged rows, batch completion, and the record-failure compensation path.

Tests: added test_relay_finishes_a_row_whose_staging_crossed_the_deadline in tests/Sync/test_sync_v2_personal_context_recovery_budget.py; it fails without the fix. Renamed test_relay_rechecks_deadline_after_each_successful_current_row_check to test_relay_current_row_deadline_stops_only_rows_it_has_not_staged; its three post-stage cases now expect the row to be finished, and the before-stage and acknowledged cases are unchanged. Results: before the fix the target test failed 3/13 plain runs (plus 4/8 instrumented). After the fix it passed 20/20 consecutive runs, 10 of them under heavy concurrent load, and 5/5 with the 60/120 ms slow-staging injection. Relay files (recovery_budget, relay, relay_recovery) 156 passed; Personalization/test_personal_context_activation.py 32 passed. Full tests/Sync (xdist): 2985 passed, 1 skipped, 30 failed. All 30 failures are PostgreSQL tests that timed out inside the pg_server fixture on 'docker rm -f tldw_postgres_test' (the host's shared Docker daemon is wedged; no local Postgres), before any product code runs. Their SQLite variants pass, and PostgreSQL relay coverage is left to CI. ruff clean. Bandit: uvx bandit -ll personal_context_relay.py reports no issues. PR #3078.

Known gap, not changed: failure paths (acknowledge_row/finalize_authority raising, or uncertain record after the deadline) can still leave a staged row hidden and unfinalized. A later relay retries it unless an activation covers the batch first; covered_by_activation batches have no orphan cleanup (only purge_terminal batches do). Docs: no documentation change needed (DoD #3 not applicable).

Landing: the relay fix itself reached dev through the release-0.1.46 sync PR #3088 (commit 3b5051d9fb), which reused #3078's repair after hitting the same handshake failure. #3078 then merged with only this task's close-out. Qodo's docstring findings on #3078 (the deadline_open helper and the regression test's stage/finalize callbacks) were addressed in the follow-up PR chore/followups-13410-13416. The remaining gap is tracked separately (staged rows orphaned when acknowledge_row/finalize_authority raise and an activation covers the batch).

Follow-up for the orphaned staged-row gap: TASK-13422.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the sync-core flake in test_personal_context_endpoints_use_real_factory_bootstrap_and_complete_flow. The cause was a real relay liveness bug, not pull timing. The activation's relay could abandon a row right after staging it into Sync when its 100 ms deadline expired, and the activation then covered that batch, so the hidden pending envelope blocked every later projection and pull scan in the dataset forever. The relay now finishes any row it has staged (at most one row past the deadline). Added a regression test, and the target test passed 20/20 runs. PR #3078.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
