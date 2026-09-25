---
id: TASK-13360
title: Harden VN variant terminal and item visibility invariants
status: Done
assignee: []
created_date: '2026-09-25 17:24'
updated_date: '2026-09-25 17:41'
labels:
  - vn-assets
  - backend
  - review
dependencies:
  - TASK-13358
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address independent review findings in VN generation durability: retain sibling execution after one failure, prevent replay demoting reviewed items, keep cancelled batches from publishing reserved items, guard direct item APIs, and preserve queued recipe slots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One variant failure does not strand planned sibling outcomes.
- [x] #2 Completed-job redelivery preserves approved/preferred item state.
- [x] #3 Cancelled batches cannot publish a reserved variant on replay.
- [x] #4 Unpublished reserved items cannot be read or mutated through direct item APIs.
- [x] #5 A slot needed by an active batch cannot be deleted.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review fixes verified in the 301-test VN backend suite and 37-test frontend VN suite. Ruff, ESLint, TypeScript, Bandit (0 findings), and 4 Chromium smoke tests passed. Full repository suite not run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened replay and publication invariants: failed siblings continue, completed items keep review decisions, cancelled batches cannot publish reservations, direct item APIs hide unpublished reservations, and active recipe slots cannot be deleted.
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
