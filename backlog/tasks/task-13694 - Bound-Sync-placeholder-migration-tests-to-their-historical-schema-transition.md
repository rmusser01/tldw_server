---
id: TASK-13694
title: Bound Sync placeholder migration tests to their historical schema transition
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Historical SQLite75-to76/PostgreSQL79-to80 repair tests currently replay later native migrations when reopening or rewinding schema version. Pin their intended target, preserving whole-row equality, rollback, idempotency and cross-owner RLS checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Historical tests execute exactly the documented migration and preserve all existing equality and isolation assertions
- [x] #2 SQLite and real PostgreSQL cases pass with independent review
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no; test-only boundary correction changes no schema or durable rule. ADR-059 governs task tracking. PR https://github.com/rmusser01/tldw_server/pull/3229.
Independent B verified historical target pin; real production version floors are asserted before pinning to avoid tautological registration coverage. A and B clear final source; exact migration cases14passed so far within continuing full backend qualification. All original whole-row equality, idempotency, rollback and shared-owner RLS assertions remain unchanged. Production migrations and source incarnation guards untouched.
All14 historical migration cases passed in current actual SQLite/PostgreSQL qualification, including superuser/restricted-owner isolation; independent A and B clear. Ruff0; exact existing Black formatter edits unchanged; Bandit scan has no errors or new security issue (only deliberate pytest version-floor assertions).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pinned this historical test module to SQLite76/PostgreSQL80 after checking real version floors. All original exact-row, idempotency, rollback, registration and RLS checks remain;14 real-backend cases pass. No production schema or incarnation changes.
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
