---
id: TASK-13253
title: Add stable cursor pagination to email search
status: Done
assignee: []
created_date: '2026-09-13 19:03'
updated_date: '2026-09-13 19:30'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User authorized resolving email validation gaps. Implement FR-SEARCH-004 keyset pagination ordered by internal_date DESC, email_message_id DESC, preserving existing offset clients and tenant scope. Synthetic tests only, no Gmail/network/model calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cursor traversal has stable tie-break ordering without duplicate or skipped rows across inserts
- [x] #2 Validate malformed and incompatible cursors with clear errors while retaining offset compatibility
- [x] #3 Real SQLite and API tests cover null dates, filtering and tenant/deleted visibility
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bounded cursor design: cursor omitted preserves offset; cursor empty starts keyset mode; nonempty token binds resolved tenant, query, deleted visibility, frozen relative-time reference and date/ID position. Stable NULLS LAST ordering, limit+1 lookahead and strict invalid-token/offset checks. Dedicated real SQLite and HTTP tests; synthetic offline runner. Files: email.py, media_db/runtime/email_query_ops.py, new cursor helper and dedicated cursor tests.

Implementation and focused verification complete (awaiting parent integration/review; no staging/commit). TDD initial new suite: 15 failed, 2 passed with missing DB cursor/ignored API cursor; final scope: 56 passed, 4 existing warnings, 0 outbound attempts via /tmp/email_offline_pytest_13250.py. Tests: dedicated DB cursor + API cursor + existing DB query, endpoint and endpoint error mapping suites. Property coverage: 12 insertion-order/page-size examples. Ruff check and format check pass across 5 touched Python files. Bandit production and tests (B101 excluded only for tests) both have zero findings; JSON /tmp/bandit_email_cursor_13253.json and /tmp/bandit_email_cursor_tests_13253.json. Design/API contract: Docs/Design/email-search-cursor-pagination.md. Existing pytest cleanup warnings under unrelated Kokoro temp directories are environmental. Live PostgreSQL not run under offline constraint. Cursor is unsigned scoped position hint, not snapshot; SQL reapplies tenant/filter/deleted constraints; existing row date changes can move across cursor boundary. All repository edits confined to assigned cursor scope.

Final diff self-review removed unrelated Ruff formatter churn from existing query/detail helpers; Ruff lint still passes on all touched files. New helper/tests and endpoint remain formatted, while unchanged legacy query formatting is preserved deliberately. Follow-up date normalization issue reported to parent: existing graph persistence ignores email.internal_date and RFC-only parser returns NULL for ISO values; parent handles separate fix, cursor supports canonical persisted dates and NULL already.

Independent review date-overflow finding resolved with two red/green regressions; cursor and identity focused suite passed 46 tests with zero outbound attempts. Final integrated evidence in Docs/Operations/Email_Core_Validation_2026-09-13.md. API contract documented in Docs/API-related/Email_Processing_API.md. Target PostgreSQL and deployment validation remain separate open release checks.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented opt-in scoped cursor pagination, stable date/ID ordering with NULLS LAST, frozen relative-query clock, lookahead continuation, strict malformed/scope/overflow input checks, and unchanged offset behavior. Real SQLite/API/property regressions and Bandit pass. No live Gmail, personal email or model requests.
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
