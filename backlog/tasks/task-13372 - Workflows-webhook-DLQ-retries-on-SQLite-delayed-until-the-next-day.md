---
id: TASK-13372
title: Workflows webhook DLQ retries on SQLite delayed until the next day
status: Done
assignee: []
created_date: '2026-09-24 00:17'
updated_date: '2026-09-24 00:17'
labels:
  - bug
  - workflows
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
workflows_webhook_dlq_service writes next_attempt_at via isoformat() ('T' separator); Workflows_DB.list_webhook_dlq_due on SQLite compared it as text to datetime('now') (space separator), so same-day retries were never due until the date rolled over. Found during TASK-13324.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Due query normalises stored timestamps
- [x] #2 Regression test with an isoformat next_attempt_at in the past
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in 29f3e3bf24: SQLite due query uses datetime(next_attempt_at). Test red on HEAD, green now; tests/Workflows 1599 passed. Postgres column is TIMESTAMPTZ, unaffected. Bandit: SQL literal change only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
SQLite DLQ retries are picked up when due.
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
