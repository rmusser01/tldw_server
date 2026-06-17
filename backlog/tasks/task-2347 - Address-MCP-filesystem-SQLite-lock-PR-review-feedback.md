---
id: TASK-2347
title: Address MCP filesystem SQLite lock PR review feedback
status: Done
labels:
- mcp
- filesystem
- review-fix
priority: medium
modified_files:
- mcp_unified/filesystem_locks/sqlite.py
- tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py
- backlog/tasks/task-2347 - Address-MCP-filesystem-SQLite-lock-PR-review-feedback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address still-valid PR review comments for the MCP filesystem SQLite lock backend. Current actionable item: batch SQLite expired lease cleanup instead of deleting selected rows one by one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Still-valid PR review comments are verified against current code before changes.
- [x] #2 SQLite expired lease cleanup deletes selected expired rows in a batch rather than per row.
- [x] #3 Focused lock manager tests and security/diff hygiene validation are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Verified PR #2340 review threads against current code. The still-valid actionable item was Gemini's `mcp_unified/filesystem_locks/sqlite.py` comment that `_maybe_cleanup_expired()` selected expired rows and deleted them one row at a time.
- Added a regression test that fails when opportunistic SQLite expired-row cleanup calls `_delete_key()` per selected row.
- Updated `_maybe_cleanup_expired()` to issue one bounded SQLAlchemy Core `DELETE` over the selected expired `(workspace_key, path)` keys, while preserving the `expires_at_epoch_us <= now_us` guard.
- Validation:
  - Red check: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py::test_sqlite_lock_manager_batches_expired_cleanup -q` failed before the implementation change on per-row `_delete_key()`.
  - Green check: the same test passed after the implementation change.
  - `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q` -> 118 passed, 4 warnings.
  - `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q` -> 33 passed, 5 warnings.
  - `python -m pytest -c mcp_unified/pytest-artifact-gate.ini .github/tests/test_mcp_unified_artifact_gate.py -q` -> 4 passed.
  - `python -m py_compile mcp_unified/filesystem_locks/sqlite.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py` -> passed.
  - `python -m bandit -r mcp_unified/filesystem_locks/sqlite.py -f json -o /tmp/bandit_mcp_fs_sqlite_cleanup_fix.json` -> 0 findings.
  - `git diff --check` -> passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the still-valid PR review item by batching SQLite expired-lease cleanup into one SQLAlchemy Core delete instead of issuing individual delete statements for each selected expired row.

Known skips/blockers: CodeRabbit and Qodo were still processing or had only placeholder comments when reviewed; no additional actionable file-level threads were present.
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
