---
id: TASK-376
title: Address PR 1719 cleanup_retained_state review comment
status: Done
assignee: []
created_date: '2026-05-15 06:01'
updated_date: '2026-05-15 06:05'
labels:
  - prototype-workspaces
  - review-fix
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the live PR #1719 review finding that cleanup_retained_state fails when invoked on a transaction-bound PrototypeWorkspacesRepo. Scope is limited to the repository cleanup transaction handling and focused regression tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 cleanup_retained_state can run on a transaction-bound repository without opening a nested transaction
- [x] #2 Focused prototype workspace repo tests cover the transaction-bound cleanup path
- [x] #3 Focused pytest, git diff --check, and Bandit on touched backend code are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing regression that calls cleanup_retained_state from inside repo.transaction(). 2. Extract or route the cleanup SQL so transaction-bound repos reuse the existing bound connection instead of opening a nested transaction. 3. Run focused prototype repo tests, git diff --check, and Bandit on the touched repository file. 4. Commit and resolve the PR review thread.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red-green verification: added test_cleanup_retained_state_runs_inside_existing_transaction, confirmed it failed before implementation with RuntimeError from _TransactionBoundPool lacking transaction(), then added _cleanup_transaction() so transaction-bound repos reuse self while top-level repos still open an atomic transaction. Verification passed: focused red/green regression, full test_prototype_repo.py 21 passed, git diff --check passed, Bandit JSON at /tmp/bandit_prototype_repo_1719_review_fix.json reported errors=0 and results=0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1719 review feedback for cleanup_retained_state on transaction-bound PrototypeWorkspacesRepo instances. The cleanup path now uses an internal async context manager that yields the already-bound repo when no transaction factory is present, and otherwise opens the normal transaction. Added regression coverage for calling cleanup_retained_state inside repo.transaction(). Verification: full prototype repo tests passed, git diff --check passed, and Bandit on prototype_workspaces_repo.py reported zero findings.
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
