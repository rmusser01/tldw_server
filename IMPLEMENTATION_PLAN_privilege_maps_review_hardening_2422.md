# PrivilegeMaps Review Hardening Implementation Plan

## Stage 1: Transaction Portability
**Goal**: Make snapshot and trend writes safe on PostgreSQL-backed AuthNZ pools.
**Success Criteria**: Transaction-scoped SQL uses backend-appropriate placeholders; focused tests prove no raw `?` placeholders reach asyncpg-style connections.
**Tests**: Add unit coverage in `tldw_Server_API/tests/Privileges/test_privilege_snapshot_store.py` and `tldw_Server_API/tests/Privileges/test_privilege_trends.py` with fake PostgreSQL transaction connections.
**Status**: Complete

## Stage 2: Effective Permission Semantics
**Goal**: Align PrivilegeMaps user hydration with AuthNZ RBAC semantics.
**Success Criteria**: Expired roles are ignored, expired direct permission overrides are ignored, explicit deny overrides subtract role-derived permissions, and multi-user fetch failures fail closed instead of synthesizing admin data.
**Tests**: Extend `tldw_Server_API/tests/Privileges/test_privilege_service_sqlite.py` for expired/denied permissions and multi-user failure behavior.
**Status**: Complete

## Stage 3: Scope, Snapshot, and Detail Safety
**Goal**: Fix privilege-map scoping and bounded output behavior.
**Success Criteria**: Inactive org/team memberships are excluded, sync snapshot IDs are collision-resistant, org trend snapshots carry org scope, detail generation stops before unbounded matrix materialization, and unused role helper code is removed.
**Tests**: Extend `tldw_Server_API/tests/Privileges/test_privilege_service_sqlite.py`, `tldw_Server_API/tests/Privileges/test_privilege_endpoints.py`, and service unit coverage for pagination caps and org trend metadata.
**Status**: Complete

## Stage 4: Verification and Tracking
**Goal**: Prove the fixes and record closeout.
**Success Criteria**: Focused tests pass, Bandit runs on touched Python files, Backlog task `TASK-2422` records verification and final summary.
**Tests**: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Privileges -v`; targeted tests first during TDD; Bandit on touched Python files.
**Status**: Complete
