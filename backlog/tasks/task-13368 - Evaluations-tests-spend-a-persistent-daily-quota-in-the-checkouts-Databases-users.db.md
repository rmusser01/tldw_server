---
id: TASK-13368
title: >-
  Evaluations tests spend a persistent daily quota in the checkout's
  Databases/users.db
status: Done
assignee: []
created_date: '2026-09-23 23:10'
updated_date: '2026-09-23 23:44'
labels:
  - bug
  - tests
  - evaluations
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Same class as TASK-13367. Running tests/Evaluations/unit locally writes 'evaluations' category rows (op_id evals:u-eval:/api/evals/run:...) plus an evals-legacy backfill row into the AuthNZ ResourceDailyLedger in the checkout's Databases/users.db (observed 2026-09-23: 11 rows over two runs). Enough runs in a day would exhaust the evaluations daily cap and produce 429s locally; CI is unaffected (fresh DB). Fix like TASK-13367 (conftest fixture isolating the ledger).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Evaluations tests do not read or write the checkout's Databases/users.db ledger
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in bd01ae1513 (tests/Evaluations/conftest.py autouse fixture, session temp AuthNZ DB, conditional pool/settings/ledger reset). Ledger count 1879 -> 1879 over a full RUN_EVALUATIONS=1 run (before: +71/run). Failure set = HEAD's 18 plus one xdist-ordering flake that passes alone. Root cause for all suites is tests/conftest.py:51 defaulting DATABASE_URL to the checkout's users.db; changing that globally needs a full-suite validation and is left as a follow-up. Bandit: test-only change.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Evaluations tests no longer spend the checkout's real daily quota.
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
