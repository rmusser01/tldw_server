---
id: TASK-2421
title: Harden Reminders module review findings
status: Done
updated_date: 2026-06-24 04:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Reminders module review findings: skip disabled queued reminders, replace stale snooze tasks, honor reminder notification preferences, avoid unbounded snoozed-list scans, trust explicit snooze task links, and clean up unused RemindersService CRUD wrappers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Reminders review fixes: disabled queued reminder jobs now skip without creating notifications, reminder notification preferences suppress due notifications, one-time completion patching is shared for success/skips, repeat snoozes delete the prior active snooze task before linking the new task, explicit snooze_task_id links are trusted even after reminder title/body edits, current-format snoozed listing uses direct snooze_task_id lookup before legacy fallback, and unused RemindersService CRUD pass-throughs were removed. Added an index and CollectionsDatabase helpers for direct snooze lookup. Also narrowed the global character-chat test reset fixture so unrelated notification tests no longer import the Research/RAG/sklearn stack during setup.
Moved fixes into worktree .worktrees/reminders-review-hardening-2421 on branch codex/reminders-review-hardening-2421 for PR preparation against dev. New-worktree verification: direct script exercised disabled task skip, reminder preference suppression, resnooze replacement, explicit snooze link matching after task edits, and direct snoozed listing; git diff --check passed; py_compile passed for touched Python files; Bandit on Reminders plus Collections_DB.py reported 0 findings/errors. The pytest focused suite was attempted in the new worktree but interrupted after hanging in pytest cleanup; the same 19-test notification/reminder suite passed before the move in the original tree.
PR review follow-up on 2026-06-23: rebased branch codex/reminders-review-hardening-2421 onto latest origin/dev and dropped an unrelated Claims_Extraction commit that had leaked into the PR. Addressed Qodo review comments by adding docstrings to new DB/helper functions, adding type hints and docstrings to new reminder tests, treating jobs for deleted queued snooze tasks as skipped task_missing runs instead of worker failures, and narrowing current-format snoozed notification lookup to notification-linked active task IDs before legacy fallback. Verification after review follow-up: py_compile passed for touched Python files; focused review-specific pytest selection passed 6 tests; affected notification test files passed 12 tests; git diff --check passed; Bandit on touched production Reminders/Collections code reported 0 findings/errors. A wider Bandit run including pytest files produced expected LOW B101 assert findings and pre-existing LOW B106 literals in tests/conftest.py, with no production-code findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR branch codex/reminders-review-hardening-2421 onto latest origin/dev, removed the unrelated Claims_Extraction commit from the PR, and addressed all actionable Qodo review comments. Current fixes include missing docstrings/type hints, deleted queued snooze tasks being marked skipped instead of failed, and direct snoozed notification lookup limited to notification-linked active tasks before legacy fallback. Verification passed: py_compile on touched Python files, focused review-specific pytest selection (6 passed), affected notification test files (12 passed), git diff --check, and Bandit production-code scan (0 findings/errors).
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
