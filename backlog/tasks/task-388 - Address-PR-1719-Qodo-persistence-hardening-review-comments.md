---
id: TASK-388
title: Address PR 1719 Qodo persistence hardening review comments
status: Done
assignee: []
created_date: '2026-05-15 19:27'
updated_date: '2026-05-15 19:27'
labels:
  - prototype-workspaces
  - review-fix
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the live Qodo follow-up review comments on PR #1719. Scope: add missing type annotations/docstrings, add cleanup-supporting migration indexes and tests, evaluate the raw-SQL comment against existing AuthNZ repository patterns, run focused verification, and resolve review threads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transaction context managers and helper functions have return type annotations/docstrings where requested
- [x] #2 Cleanup retention predicates have supporting indexes covered by migration tests
- [x] #3 Raw-SQL review comment is either fixed or answered with codebase-pattern evidence
- [x] #4 Focused pytest, git diff --check, and Bandit on touched backend code are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect live Qodo comments and current dirty worktree edits. 2. Preserve existing local edits for annotations, docstring, and cleanup indexes; patch gaps if verification shows any. 3. Verify the raw-SQL finding against existing AuthNZ repository conventions and avoid a broad architecture move if inconsistent. 4. Run focused PrototypeWorkspaces tests, git diff --check, and Bandit on touched backend files. 5. Commit, push, and resolve/reply to review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed valid Qodo follow-ups by adding AsyncIterator return annotations for transaction context managers, adding a _result_row_count docstring, and adding cleanup-supporting migration indexes for archived workspaces, expired/revoked actors and sessions, and inactive revoked preview handles. Verified raw-SQL comment against AuthNZ repo patterns: AuthNZ repos are the persistence abstraction for this database and existing repo modules use db_pool.execute/fetch* throughout; cleanup SQL remains parameterized and transaction-bound, so moving it to DB_Management would be inconsistent and broader than the PR.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1719 Qodo follow-up review comments. Implemented type/docstring fixes and cleanup indexes with migration test coverage. The raw-SQL comment is handled with codebase-pattern evidence rather than a broad architecture move: AuthNZ repos already centralize AuthNZ DB persistence with parameterized db_pool calls. Verification passed: test_prototype_repo.py 21 passed, git diff --check passed, and Bandit on migrations.py plus prototype_workspaces_repo.py reported zero findings.
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
