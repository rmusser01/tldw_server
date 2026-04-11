## Stage 1: Add Focused Regression Tests
**Goal**: Add a dedicated pytest file covering fail-closed bootstrap behavior for schema normalization and RBAC seeding.
**Success Criteria**: New tests fail against the current implementation because bootstrap errors are logged and suppressed.
**Tests**: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py -q`
**Status**: In Progress

## Stage 2: Make Bootstrap Fail Closed
**Goal**: Update `UserDatabase_v2` so required schema normalization and required RBAC bootstrap validation raise `UserDatabaseError`.
**Success Criteria**: `_ensure_core_columns()` raises on required normalization failures; `_seed_default_data()` raises when required roles, permissions, or required role-permission links are still missing after seeding.
**Tests**: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py -q`
**Status**: Not Started

## Stage 3: Verify and Self-Review
**Goal**: Run the requested regression suite, Bandit, and diff checks on the touched scope.
**Success Criteria**: Required verification commands exit successfully and the diff stays limited to the intended files.
**Tests**: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py tldw_Server_API/tests/test_authnz_backends_improved.py -q`
**Status**: Not Started
