---
id: TASK-13367
title: >-
  Workflows tests spend a persistent daily quota in the repo's
  Databases/users.db
status: Done
assignee: []
created_date: '2026-09-23 21:09'
updated_date: '2026-09-23 23:10'
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
- [x] #1 Workflows tests do not read or write the checkout's Databases/users.db ledger (per-test DB or RG disabled for them)
- [x] #2 Running the Workflows suite twice in a row passes the second time
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
837b04144e: autouse fixture in tests/Workflows/conftest.py sets WORKFLOWS_DISABLE_QUOTAS=1 and stubs daily_ledger.get_workflows_daily_ledger -> None, so neither the cap check nor record_workflow_run touches users.db. RED on old code: with today's ledger exhausted (1000 units inserted), test_events_cursor_pagination + test_malformed_cursor_400 -> 2 failed (429 == 200). GREEN: same exhausted ledger -> 7 passed; ledger row count unchanged (3 -> 3). Regression test test_workflows_config_defaults.py::test_workflows_tests_do_not_touch_authnz_daily_ledger fails on old conftest (quotas not disabled), passes now. Full tests/Workflows twice in a row with exhausted ledger: 1598 passed, 6 skipped both runs, 0 failures; no workflows_runs rows added. Put the test in an existing file so check_shard_coverage stays OK. Bandit: no findings. Docs: none (WORKFLOWS_DISABLE_QUOTAS already documented in Docs/Design/Workflows.md as the test switch). Observed side finding: tests/Evaluations/unit also writes 'evaluations' rows to the checkout users.db ledger (same class, not fixed here).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Workflows tests no longer spend the persistent AuthNZ daily ledger: an autouse conftest fixture disables the workflows quota and the ledger for the suite; suite passes twice in a row even with today's quota exhausted.
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
