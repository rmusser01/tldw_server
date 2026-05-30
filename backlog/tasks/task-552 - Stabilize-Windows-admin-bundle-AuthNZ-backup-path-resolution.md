---
id: TASK-552
title: Stabilize Windows admin bundle AuthNZ backup path resolution
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-30 06:45'
labels:
  - ci
  - tests
  - admin
  - pr-2133
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix PR #2133 Windows full-suite failure where Admin bundle export returns export_error for the AuthNZ-only bundle test because Windows sqlite DATABASE_URL paths with backslashes can be resolved with a leading slash and fail backup path existence checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Windows-style sqlite DATABASE_URL paths like sqlite:///C:\\... resolve to a valid filesystem path for AuthNZ backups.
- [x] #2 The Admin authnz-only bundle test still passes locally.
- [ ] #3 PR #2133 CI no longer fails the Admin full-suite module for this path.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Windows full-suite Admin artifact for PR #2133 showed `test_create_bundle_authnz_only` returning `export_error`. The failing path shape was a Windows sqlite URL with a backslash drive path that parsed as `/C:\...`; the existing normalizer only stripped `/C:/...`. Updated `_resolve_dataset_db_path()` to strip the leading slash for both slash and backslash drive separators, and added a focused regression test for `sqlite:///C:\...` AuthNZ DATABASE_URL resolution.

Local verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/Admin/test_bundle_ops.py::test_authnz_backup_path_normalizes_windows_sqlite_url tldw_Server_API/tests/Admin/test_bundle_ops.py::test_create_bundle_authnz_only` passed: 2 tests.
- `git diff --check` passed.
- Bandit ran on touched Admin/Audio files; remaining findings are low-severity test assert usage only, with no B106 or medium/high findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
