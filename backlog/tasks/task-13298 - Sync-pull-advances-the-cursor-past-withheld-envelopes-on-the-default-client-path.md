---
id: TASK-13298
title: >-
  Sync pull advances the cursor past withheld envelopes on the default client
  path
status: To Do
assignee: []
created_date: '2026-09-22 04:51'
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
- [ ] #1 A failing test on the adapter-v1 path reproduces cursor advance past an unresolved blocker, mirroring test_versioned_pull_does_not_advance_past_unresolved_conflict
- [ ] #2 Both pull paths derive the watermark from the blocker-filtered list via one shared helper
- [ ] #3 has_more is computed consistently with the filtered list on both paths
- [ ] #4 The v2 path's existing test still passes (no regression)
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
