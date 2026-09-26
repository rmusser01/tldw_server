---
id: TASK-13358
title: Make VN asset variant publication replay-safe
status: Done
assignee: []
created_date: '2026-09-25 16:36'
updated_date: '2026-09-25 17:01'
labels:
  - vn-assets
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the server worker replay portion of issue #2021: duplicate delivery or worker restart must not produce duplicate visible items or inflate batch counters, including interruption around AuthNZ generated-file registration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A repeated completed variant returns its original item without a second image call or counter increment.
- [x] #2 An interrupted item/storage handoff can reconcile a registered file by owner and source reference.
- [x] #3 Batch counters are computed from committed variant outcomes and remain correct after retry.
- [x] #4 Focused fault-injection and owner-isolation tests cover the recovery boundary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replay ledger uses recipe rows; hidden reservations are excluded from public listings and failed reservations from item capacity. Parent fanout preserves child-updated status. Verification: 124 scoped backend tests passed plus owner-isolation test; Ruff E,F,I clean; Bandit 0 findings; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Versioned VN variants now publish exactly once, recover registered files by owner/source ref after interruption, and derive batch counters transactionally. Duplicate or late reports cannot regress completed outcomes.
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
