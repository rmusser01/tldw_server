---
id: TASK-15450
title: Make magic-link redemption a durable single-winner operation
status: Done
references:
- https://github.com/rmusser01/tldw_server/pull/3430
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consume a verified one-time sign-in token before account or session effects, using the existing blacklist unique token identifier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Concurrent authorized submissions with the same token yield at most one successful grant.
- [x] #2 A failed durable write or invalid redemption metadata cannot return credentials or mutate account state.
- [x] #3 Existing one-time, active-account and registration behavior remains covered.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Enforces the existing one-time sign-in contract using the existing blacklist table. Plan: reproduce duplicate and failed claims, implement a committed conditional insert before grants, verify the handler and both SQL dialects.
Review follow-up: accept only explicit insert outcomes. Unknown PostgreSQL command tags and SQLite row counts return 503. Two regressions returned 400 before this change. Final verification after review: 58 SQLite/adjacent tests and 7 PostgreSQL 18 integration tests passed. Ruff, compile, whitespace and changed-code Bandit checks passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The existing unique token identifier is claimed on the sign-in connection before any account or session changes. Confirmed duplicates return 400; uncertain or failed writes return 503. Concurrent requests and persistence failure reproduced incorrect success on the base. Final verification: 58 SQLite and adjacent AuthNZ tests, seven PostgreSQL 18 integration tests, Ruff, compile, whitespace, and no newly introduced runtime Bandit findings (base 11, branch 10).
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
