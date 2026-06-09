---
id: TASK-2329
title: Rebase PR 2325 and address scheduled tasks review feedback
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 04:46'
labels:
  - scheduled-tasks
  - webui
  - ux
  - phase-2b
  - pr-feedback
  - rebase
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2325'
  - TASK-2328
  - TASK-2327
  - TASK-2326
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased onto the latest origin/dev without unresolved conflicts.
- [x] #2 All actionable PR review comments and check failures are inspected and either addressed or documented with technical rationale.
- [x] #3 Focused ScheduledTasks verification is rerun after any changes.
- [x] #4 Backlog task records final status, verification, and known skips.
- [x] #5 Branch is force-pushed safely after rebase and fixes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased branch onto origin/dev without conflicts. Addressed actionable PR feedback in ScheduledTasks capability helpers and Create panel: immutable default capability map, safe missing source-family fallback, distinct unknown notification metadata copy, adapter-only availability explanation, non-Watchlists capability metadata rendering, non-secret-looking redaction fixtures, stale Backlog checklist metadata, and slow recurring form test timeouts. Verification: focused ScheduledTasks batch passed with 69 tests; route-state regression passed with 8 tests; git diff --check exited 0; touched product placeholder scan had no matches. Bandit skipped because no Python files are touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2325 onto latest origin/dev and force-pushed the updated branch. Addressed all actionable review feedback in b6997470: sourceFamily fallback, non-Watchlists capability metadata rendering, adapter-only Limited availability guidance, unknown notification metadata copy, frozen default capability map, neutral redaction test fixtures, stale Backlog checklists, and recurring form timeout reliability. Replied to all inline review threads and added a PR-level summary for the non-inline Qodo item. Verification: focused ScheduledTasks batch passed with 69 tests; route-state regression passed with 8 tests; git diff --check exited 0; touched product placeholder scan had no matches. Bandit skipped because no Python files were touched. GitHub checks were pending on the refreshed head when reviewed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Worktree is clean after final commit/push.
- [x] #2 PR is updated on GitHub.
- [x] #3 No unresolved actionable review comments remain unaddressed without rationale.
- [x] #4 Verification results are recorded.
<!-- DOD:END -->
