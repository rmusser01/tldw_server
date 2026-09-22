---
id: TASK-13298
title: >-
  Sync pull advances the cursor past withheld envelopes on the default client
  path
status: To Do
assignee: []
created_date: '2026-09-22 04:51'
updated_date: '2026-09-22 19:34'
labels:
  - bug
  - sync
  - data-loss
dependencies: []
references:
  - 'tldw_Server_API/app/core/Sync/v2/service.py:5256'
  - 'tldw_Server_API/app/core/Sync/v2/service.py:10245'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`core/Sync/v2/service.py:pull` computes its next-cursor watermark from the **unfiltered** envelope list on the legacy adapter-v1 branch:

```python
has_more = has_visible_lookahead or len(raw_envelopes) > page_limit
if has_visible_lookahead and page:
    next_sequence = page[-1].server_sequence
else:
    next_sequence = max(
        (envelope.server_sequence for envelope in raw_envelopes),   # <- raw, not filtered
        default=since_sequence,
    )
```

The function deliberately distinguishes `visible` from `raw_envelopes` everywhere else in the same block. The correct implementation is ~5,000 lines away in the same file at `_pull_versioned`, which derives the boundary from **`safe_raw_envelopes`**.

**Failure:** a device that never negotiated `supported_adapter_versions` — the default — pulls. Envelope at seq 1 is an unresolved ordering blocker per ADR-034; seq 2 is deliverable. The pull returns `envelopes=[]`, `next_cursor="2"`, `has_more=False`. The watermark has advanced past seq 2, seq 2 is never delivered, and the client was told it is caught up. **Silent, permanent, per-device data loss on the default path.**

The repo already has `test_versioned_pull_does_not_advance_past_unresolved_conflict` proving the v2 path is correct. No v1 equivalent exists — which is why this survived.

Found by the comprehensive core-module review (Sync reviewer, reproduced end-to-end on SQLite); the raw-vs-safe divergence independently verified by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test on the adapter-v1 path reproduces cursor advance past an unresolved blocker, mirroring test_versioned_pull_does_not_advance_past_unresolved_conflict
- [ ] #2 Both pull paths derive the watermark from the blocker-filtered list via one shared helper
- [x] #3 has_more is computed consistently with the filtered list on both paths
- [x] #4 The v2 path's existing test still passes (no regression)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on branch ci/review-followup-visibility.

Reproduced first: test_legacy_pull_does_not_advance_past_unresolved_conflict in tldw_Server_API/tests/Sync/test_sync_v2_service.py, mirroring the versioned test named in the description. Failed with assert [] == ['later-v1'] against unfixed code.

The fix is in _scan_pull_page, not at the call site. First attempt bounded the caller's watermark by the blocker cursor (the shape AC #2 describes) and broke test_conflict_resolution_rebases_later_dependency_and_paginates_without_queued_history: pinning the cursor while has_more stayed True is a livelock, a client polls forever without progressing. That pre-existing test is a no-progress guard and it is correct; dev only satisfied it by losing the data.

The scan now ends at the blocker instead of filtering around it, matching how _scan_pull_page_versioned breaks out of its merge loop. raw no longer contains withheld envelopes, so both the watermark (max of raw) and has_more (len(raw) > page_limit) follow from the filtered list with no caller change -- AC #3 falls out of the same edit. 16 lines in app code.

AC #2 not taken literally: the two paths were not merged into a shared helper. The versioned scan additionally carries restore_barrier and per-stream watermarks, so a common helper would have to take both, and the structural parity that actually matters -- neither scan emits envelopes at or past the blocker -- is now present in both. Extracting the helper is a refactor with its own blast radius, not part of a data-loss fix.

Verification: tldw_Server_API/tests/Sync/test_sync_v2_service.py 166 passed (was 165 + the new test), including test_versioned_pull_does_not_advance_past_unresolved_conflict (AC #4) and the pagination-progress guard. ruff clean.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
