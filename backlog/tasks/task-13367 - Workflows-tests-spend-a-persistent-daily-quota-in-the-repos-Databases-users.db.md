---
id: TASK-13367
title: >-
  Workflows tests spend a persistent daily quota in the repo's
  Databases/users.db
status: To Do
assignee: []
created_date: '2026-09-23 21:09'
labels:
  - bug
  - tests
  - workflows
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workflows run tests (test_events_cursor_pagination, test_malformed_cursor_400, ...) go through _enforce_workflows_daily_cap -> rg_governor, whose ResourceDailyLedger writes to the AuthNZ DB pool. Locally that is Databases/users.db in the checkout, so usage for user 1 accumulates across test runs; after enough runs in a day every workflow run returns 429 'Daily quota exceeded' and the tests fail. CI is unaffected (fresh DB per run), which is why it's invisible there. Observed 2026-09-23: test_events_cursor_pagination.py and test_malformed_cursor_400.py both 429 on an unmodified tree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Workflows tests do not read or write the checkout's Databases/users.db ledger (per-test DB or RG disabled for them)
- [ ] #2 Running the Workflows suite twice in a row passes the second time
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
