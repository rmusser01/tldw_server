---
id: TASK-15691
title: Restore notifications pagination coverage for the kind filter
status: Done
labels:
- testing
- notifications
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The direct endpoint test fails because its notification fixture predates kind-filtered listing and counting. Cover canonical pagination with and without a kind filter in a test-only PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unfiltered and kind-filtered responses assert the returned page and matching total, continuation, and offset.
- [x] #2 Endpoint and database production code stay unchanged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reproduced TypeError for kind on dev. ADR assessment: no new ADR is needed for a regression test of the existing notification API contract.
Use an isolated notification store to exercise real listing and counting through the endpoint, then run the Collections and Notifications suites.
All four pagination cases passed against SQLite. The complete Collections and Notifications suites passed all 700 tests, including the PostgreSQL fixtures. Ruff format and lint, git diff --check, and Bandit passed (B101 excluded for test assertions). Independent Superpowers review found no outstanding issues.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced the obsolete endpoint test double with an isolated SQLite notification store. Verified ordered rows and canonical and legacy paging fields on unfiltered, filtered, terminal, and empty pages. This test-only change exercises the existing API contract.
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
