---
id: TASK-13300
title: >-
  Sync pull cursor advances past withheld envelopes causing silent per-device
  data loss
status: To Do
assignee: []
created_date: '2026-09-22 04:52'
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
- [ ] #1 Legacy v1 pull path never advances the cursor past a withheld envelope
- [ ] #2 Both pull paths share one watermark-advance helper
- [ ] #3 Regression test added for adapter_version=1 mirroring the existing v2 test
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
