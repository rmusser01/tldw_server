---
id: TASK-13431
title: 'Personal Context relay: a staged row is orphaned when acknowledge/finalize
  raises before an activation covers its batch'
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-13410 (#3078, landed via #3088). PersonalContextRelay._relay_owned now always finishes a row it has staged (deadline checks only stop rows before their Sync write). But if acknowledge_row or finalize_authority raises after stage_authority wrote the hidden home-authority envelope, the row is left staged with the Sync envelope at apply_status=pending. If a later activation install marks that batch covered_by_activation before a relay retry, no relay revisits it, and only purge_terminal batches have orphan cleanup. A pending ingress then blocks require_materialization_predecessors_applied for the dataset, the same symptom TASK-13410 fixed for the slow-host path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A relay or activation path finishes or compensates a staged-but-unfinished row even after its batch is covered by an activation, or activation refuses to cover a batch with a staged-unfinished row
- [x] #2 A regression test injects an acknowledge_row/finalize_authority failure after staging and shows later projections in the dataset still apply
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Known limits: the fix prevents new strands but does not repair a dataset that is already stranded. Ongoing-sync activation is still gated off before rollout (ongoing_sync_version defaults to 0), so none should exist. In the rare race path the prepared activation stays and fences the relay until it goes stale (60 s); the relay then expires it, finishes the row, and the next activation succeeds. PR #3143 (base dev).
Root cause (confirmed by a failing test before the fix). After stage_authority writes the hidden home-authority envelope, acknowledge_row or finalize_authority can raise. The journal row is then left staged or acknowledged, and the Sync envelope stays apply_status=pending. A later relay attempt normally finishes it. But a first activation marks every incomplete batch up to its watermark covered_by_activation; no relay revisits a covered batch, and only purge_terminal batches get orphan cleanup. The pending envelope then stays forever, and require_materialization_predecessors_applied rejects every later projection in the dataset. A probe confirmed this state for both injection points: batch covered_by_activation, row 0 staged/acknowledged at cursor 1, Sync cursor 1 (server-origin scope authority) pending, and the next push at cursor 2 coming back apply_status=pending.
Fix (activation refuses to cover). In tldw_Server_API/app/core/Sync/v2/personal_context_activation.py, prepare_activation refuses while Sync holds a pending home-authority row of the profile. It raises the existing PersonalContextActivationPendingError, which clients receive as the retryable personal_context_activation_required. It checks twice: (1) before activations.prepare, because a prepared activation fences the relay that would finish the row, so the client's retry converges immediately; (2) inside install(), under the profile lease, just before coverage commits, so a relay failure that races in after check 1 is still caught (no relay can stage while the lease is held). The read is the new Sync_DB.has_pending_personal_context_authority, with a SyncV2Store wrapper (server-origin, accepted, apply_status pending, role home_authority, same profile_id; portable SQL, JSON filtered in Python). The relay stays the only writer that finishes or compensates its rows; its lease and current-row guards and the TASK-13410 deadline semantics are unchanged. Rejected alternatives: (a) extending orphan cleanup to covered batches: discard_pending_personal_context_authority only removes a row that is still its object's head, so once a later batch moved the head the relay's cleanup loop (which runs before every batch) would fail on every attempt and wedge the profile; (b) finishing rows of covered batches: needs loosening row_is_current and the finalization guard (_source_claim_matches requires batch status relaying); (c) deleting the orphan in Sync during install: a crash before the canonical coverage commit would leave the journal pointing at a missing envelope. The design spec now records the new activation precondition.
Tests (tests/Sync/test_sync_v2_personal_context_activation.py, each on SQLite and Docker PostgreSQL via the new linked_activation_on_each_store fixture): test_activation_cannot_strand_a_staged_authority_row_left_unfinished[acknowledge|finalize] injects a failure after staging, activates the way a client would (restarting when told to wait), then pushes a record and asserts it applies. test_install_refuses_coverage_when_a_racing_relay_left_a_row_unfinished has a relay fail after check 1 and asserts the install-time check refuses, with no Sync install and no covered batch. All 6 cases fail on origin/dev code and pass with the fix; removing only the install() check makes the race test fail. Results: activation + relay + relay_recovery + recovery_budget + Personalization/test_personal_context_activation.py: 219 passed. TASK-13410 target test: 10/10 consecutive passes. Full tests/Sync (xdist, Docker PostgreSQL): first run 3047 passed, 1 skipped, 3 failed (authority_identity reuse drift[sync], bootstrap transport watermark, sqlite adapter-state concurrency). All 3 pass serially, and none of them reaches prepare_activation, the only caller of the new code. A baseline run with the PR's files set back to origin/dev had 1 load failure in the same authority_identity file. Second run on the branch: 3050 passed, 1 skipped, 0 failed. ruff clean. Bandit: uvx bandit -ll on personal_context_activation.py, store.py and Sync_DB.py reports no issues.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A relay failure after staging can no longer be stranded by activation coverage. Activation now refuses (retryable personal_context_activation_required) while Sync holds a pending home-authority row of the profile. It checks before preparing, so the retry converges immediately, and again under the profile lease just before coverage commits, so a racing relay failure is still caught. The relay keeps sole ownership of finishing its rows with its guards unchanged. New SQLite+PostgreSQL regression tests fail without the fix; the full Sync suite passed 3050/3050 on the second run. PR #3143.
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
