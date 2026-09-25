---
id: TASK-13359
title: Recover VN generation submissions after browser reload
status: Done
assignee: []
created_date: '2026-09-25 17:01'
updated_date: '2026-09-25 17:42'
labels:
  - vn-assets
  - backend
  - frontend
dependencies:
  - TASK-13358
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
  - 'https://github.com/rmusser01/tldw_server/pull/3016'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish issue #2021 generation submission recovery. Link an in-progress idempotency receipt to its batch transactionally, recover that batch and its deterministic parent Job on same-key retry, and preserve browser keys across reload until acknowledged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A same-key retry after a crash between batch creation and response completion returns the original batch without creating another batch.
- [x] #2 A batch whose parent Job was not recorded can be safely re-enqueued using the deterministic Jobs key.
- [x] #3 A browser reload after an ambiguous start or slot retry uses the original key and shows the authoritative batch state.
- [x] #4 Different owners and payloads cannot reuse another generation receipt.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
IMPLEMENTATION_PLAN_vn_generation_durability.md, Stage 4
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified 301 VN backend tests, 37 VN frontend tests, TypeScript, ESLint, Ruff, Bandit (0 findings), and 4 Chromium smoke tests. Full repository suite not run; targeted VN suite covers this scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Generation receipts now link to their immutable batch in one transaction; same-key retries recover that batch and its deterministic parent Job. The workbench preserves owner-scoped pending keys and selected pack across reload, replays once, and refreshes authoritative status.
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
