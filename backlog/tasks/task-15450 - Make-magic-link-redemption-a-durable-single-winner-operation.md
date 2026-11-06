---
id: TASK-15450
title: Make magic-link redemption a durable single-winner operation
status: Done
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A conditional blacklist insert on the existing autocommit sign-in connection makes redemption single-winner before account and session effects. Concurrent HTTP submissions and unavailable persistence both reproduced incorrect success on SQLite before the fix. The SQLite and adjacent AuthNZ suites passed 56 tests; all seven PostgreSQL 18 magic-link integration tests passed. Ruff, compile and whitespace checks passed. Bandit reported 10 runtime findings versus 11 on the base, with none introduced. The endpoint documentation describes 400, 503 and requesting a fresh link after a later failure.
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
