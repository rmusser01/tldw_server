## Stage 1: Regression Coverage
**Goal**: Add focused tests for the Storage review findings before implementation.
**Success Criteria**: Tests fail for sibling-prefix path escapes, fail-closed quota behavior, generated-file quota preflight, and partial-write cleanup.
**Tests**: Targeted pytest runs for `tests/Storage/` and admin quota tests touched by policy changes.
**Status**: Complete

## Stage 2: Storage Safety Fixes
**Goal**: Harden filesystem path validation, atomic writes, generated-file preflight checks, and quota error handling.
**Success Criteria**: Regression tests pass; existing Storage tests remain green.
**Tests**: `tests/Storage/test_filesystem_storage.py`, `tests/Storage/test_generated_file_helpers.py`, `tests/Admin/test_admin_storage_quotas.py`.
**Status**: Complete

## Stage 3: Backup Schedule Module Split
**Goal**: Move backup schedule job helpers out of `core/Storage` while preserving compatibility imports.
**Success Criteria**: Backup scheduler and worker imports continue to work; direct compatibility shim tests pass if present.
**Tests**: `tests/Admin/test_admin_backup_jobs.py`, `tests/Admin/test_admin_backup_scheduler.py`.
**Status**: Complete

## Stage 4: Verification And Closeout
**Goal**: Run targeted tests, compile touched production files, run Bandit, and update Backlog records.
**Success Criteria**: Verification results are recorded in `TASK-12009`; known skips or blockers are documented.
**Tests**: Focused pytest, `py_compile`, Bandit on touched production scope.
**Status**: Complete
