---
id: TASK-13300
title: >-
  Sync pull cursor advances past withheld envelopes causing silent per-device
  data loss
status: In Progress
assignee: []
created_date: '2026-09-22 04:52'
updated_date: '2026-09-23 19:36'
labels:
  - bug
  - sync
  - data-loss
dependencies: []
references:
  - 'tldw_Server_API/app/core/Sync/v2/service.py:5252'
  - 'tldw_Server_API/app/core/Sync/v2/service.py:10245'
  - 'tldw_Server_API/app/core/Sync/v2/service.py:10658'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The legacy adapter-v1 pull branch computes next_sequence = max(e.server_sequence for e in raw_envelopes) with no blocker filter, while _scan_pull_page distinguishes visible from raw. The correct implementation exists at _pull_versioned (10245-10261), which derives the boundary from safe_raw_envelopes and states the rule in its docstring at :10172.

Reproduced on SQLite: a device that never negotiated supported_adapter_versions (the default) pulls; seq 1 is an unresolved ordering blocker per ADR-034; seq 2 is deliverable. Pull returns envelopes=[], next_cursor="2", has_more=False. Seq 2 is never delivered and the client is told it is caught up. Recoverable only by an operator resetting the cursor.

Class: divergent-copies. Fix is one shared _advance_pull_watermark helper, not a new abstraction. test_versioned_pull_does_not_advance_past_unresolved_conflict proves the v2 path; no v1 equivalent exists.

Source: Docs/superpowers/reviews/2026-09-21-core-module-duplication-synthesis.md F2
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy v1 pull path never advances the cursor past a withheld envelope
- [x] #2 Both pull paths share one watermark-advance helper
- [ ] #3 Regression test added for adapter_version=1 mirroring the existing v2 test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
FIXED. Both pull paths now derive their watermark from one helper, _safe_pull_boundary, at module level in core/Sync/v2/service.py.

ROOT CAUSE was narrower than "the v1 branch takes max() over raw": _scan_pull_page ALREADY COMPUTED blocker_cursor and then DISCARDED it, returning only (raw, visible). Its sibling _scan_versioned_pull_page returns it. So the v1 caller had no way to know where the blocker was. It now returns (raw, visible, blocker_cursor) - it had exactly one caller.

THE FIX IS TWO-SIDED, and the first attempt was wrong in an instructive way. Refusing to advance the cursor fixes the data loss but, on its own, turns it into a LIVELOCK: test_conflict_resolution_rebases_later_dependency_and_paginates_without_queued_history asserts inside its pagination loop that `page.next_cursor != cursor or not page.has_more` - a pull must either advance or declare itself finished. Stopping the cursor while has_more stayed True made the client re-request the same page forever. So pull() now also clears has_more when the boundary cannot advance and nothing was delivered: there is genuinely nothing more deliverable until the blocker is resolved, and the client re-polls later with the same cursor.

The versioned path was migrated onto the same helper too, so there is ONE implementation rather than a correct one and a broken one.

SECOND MISTAKE WORTH RECORDING: I first inserted the helper as a module-level def immediately before an INDENTED class method, which terminated the 9,740-line SyncV2Service class body and orphaned every method after it. ast.parse passed - it is syntactically valid - and only the test run caught it (46 failures). Reverted and redone with the helper placed above the class at line 1303. Parsing cleanly is not the same as being semantically correct.

Tests: tests/Sync/test_pull_watermark_boundary.py, 8 cases - blocker, restore barrier, the page shortcut, barrier overriding the shortcut, everything-blocked, empty scan, and one documenting the liveness interaction.
Regression: test_sync_v2_service.py back to 165 passed (the HEAD baseline). store/endpoints/conflicts show 6 failed / 389 passed BOTH with and without the change (stash-isolated) - pre-existing.

2026-09-23 reconciliation: AC1 met - commit 0e0f57a97d; service.py legacy pull (~:5344) derives next_sequence from _safe_pull_boundary with blocker_cursor now returned by _scan_pull_page. Behaviourally verified with a scratch (not committed) mirror of test_versioned_pull_does_not_advance_past_unresolved_conflict using an adapter-v1-only device: first pull envelopes=[], next_cursor=0, has_more=False; after clearing the blocker the second pull delivers 'later'. AC2 met - both pull() (:5344) and _pull_versioned (:10356) call module-level _safe_pull_boundary (:1308). AC3 NOT met - tests/Sync/test_pull_watermark_boundary.py (8 passed) unit-tests the helper only; no service-level pull() test with an adapter_version=1 device exists in the repo mirroring test_sync_v2_service.py::test_versioned_pull_does_not_advance_past_unresolved_conflict. Remaining: add that test to test_sync_v2_service.py. v2 test + test_conflict_resolution_rebases_later_dependency_and_paginates_without_queued_history: 2 passed. Bandit on service.py: no findings.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
