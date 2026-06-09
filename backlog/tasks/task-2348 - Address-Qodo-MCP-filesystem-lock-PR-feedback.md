---
id: TASK-2348
title: Address Qodo MCP filesystem lock PR feedback
status: Done
labels:
- mcp
- filesystem
- review-fix
- qodo
priority: medium
modified_files:
- mcp_unified/filesystem_locks/__init__.py
- mcp_unified/filesystem_locks/memory.py
- mcp_unified/filesystem_locks/sqlite.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py
- tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- backlog/tasks/task-2348 - Address-Qodo-MCP-filesystem-lock-PR-feedback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address Qodo review feedback on PR #2340 for filesystem lock package lint, path normalization, and lazy optional exports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Still-valid Qodo findings are verified against current code.
- [x] #2 Compatibility shim time export is lint-safe or documented without breaking existing monkeypatch compatibility.
- [x] #3 Memory backend signature is wrapped to project style.
- [x] #4 SQLite backend and factory normalize whitespace in lock database paths.
- [x] #5 Star import from mcp_unified.filesystem_locks does not force optional SQLite/SQLAlchemy import.
- [x] #6 Focused tests and security/diff hygiene validation are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Verified Qodo PR #2340 findings against current code before editing:
  - The compatibility shim intentionally exposed `time` for older monkeypatch callers but needed explicit documentation/lint suppression.
  - `InMemoryFilesystemLockManager.__init__` was a single long signature line.
  - `SQLiteFilesystemLockManager.__init__` validated a stripped path but constructed `Path(path)` from the unstripped value, and the factory forwarded the unnormalized string.
  - `SQLiteFilesystemLockManager` in `mcp_unified.filesystem_locks.__all__` caused star imports to load the optional SQLite backend and SQLAlchemy.
- Kept the compatibility `time` export, added a compatibility comment and `# noqa: F401`, and included it in the shim `__all__`.
- Wrapped the in-memory manager constructor signature.
- Normalized SQLite paths in both direct construction and factory construction.
- Removed `SQLiteFilesystemLockManager` from package `__all__` while preserving explicit lazy imports through `__getattr__`.
- Added regression tests for whitespace path normalization and star-import lazy behavior.
- Validation:
  - Red checks for the two behavior regressions failed before fixes.
  - Green checks for both regressions passed after fixes.
  - `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q` -> 119 passed, 4 warnings.
  - `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q` -> 34 passed, 5 warnings.
  - `python -m pytest -c mcp_unified/pytest-artifact-gate.ini .github/tests/test_mcp_unified_artifact_gate.py -q` -> 4 passed.
  - `python -m py_compile mcp_unified/filesystem_locks/__init__.py mcp_unified/filesystem_locks/memory.py mcp_unified/filesystem_locks/sqlite.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py` -> passed.
  - `python -m bandit -r mcp_unified/filesystem_locks tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py -f json -o /tmp/bandit_mcp_fs_qodo_fixes.json` -> 0 findings.
  - `git diff --check` -> passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed Qodo's actionable MCP filesystem lock feedback: lint-safe compatibility time export, wrapped memory backend signature, normalized SQLite lock DB paths, and prevented package star imports from forcing the optional SQLite backend.

Known skips/blockers: none.
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
