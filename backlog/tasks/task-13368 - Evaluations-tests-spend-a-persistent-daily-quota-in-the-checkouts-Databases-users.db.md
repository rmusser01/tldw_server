---
id: TASK-13368
title: >-
  Evaluations tests spend a persistent daily quota in the checkout's
  Databases/users.db
status: To Do
assignee: []
created_date: '2026-09-23 23:10'
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
- [ ] #1 Evaluations tests do not read or write the checkout's Databases/users.db ledger
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
