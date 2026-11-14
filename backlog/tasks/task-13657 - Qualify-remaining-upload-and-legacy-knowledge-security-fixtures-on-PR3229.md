---
id: TASK-13657
title: Qualify remaining upload and legacy knowledge security fixtures on PR3229
status: Done
labels:
- chat
- testing
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reproduce and correct stale test setup contracts without relaxing quota-off upload admission, native-ledger validation, or foreign-parent ownership rejection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Quota-off sparse upload reaches processing with distinct valid native usage and stale legacy user rows.
- [x] #2 Historical foreign parent is rejected before note or flashcard writes; current DB admission rejects the same malformed link.
- [x] #3 Affected suites, static security gates, and independent review pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR https://github.com/rmusser01/tldw_server/pull/3229. RED: 2 failed/22 passed across persistence cancellation/quota and chat knowledge suites. ADR assessment: no new ADR; fixture-only qualification preserves DB admission, native retained usage, and existing chat ownership boundaries.
Qualified full affected suites: 24 passed, 29 warnings, 0 skips, 72.86s. Ruff passes; scoped Bandit 0 findings/0 errors (B101 required test assertions excluded). Touched test functions Black-formatted with identical Python AST. Independent backend reviewer found no Critical/Important findings; bounded fixture review only. Original malformed parent refusal and API403/zero writes preserved; no production source change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Corrected two stale backend fixtures without production changes. Real quota-off sparse upload is admitted with a valid independent native ledger row. Live DB refuses foreign-parent admission; historical API read still returns403 with zero notes/cards. Full affected suites24/24, Ruff, Bandit and independent bounded review pass.
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
