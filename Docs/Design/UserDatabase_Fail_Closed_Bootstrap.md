# UserDatabase Fail-Closed Bootstrap Design

## Goal

`UserDatabase_v2` bootstrap must fail closed when required schema normalization or RBAC seed validation cannot be completed. A partially initialized AuthNZ database can grant or deny permissions incorrectly, so required bootstrap failures should raise a `UserDatabaseError` instead of being logged and suppressed.

## Required Behavior

- Required core-column normalization failures raise during bootstrap.
- Required default roles, permissions, and role-permission links are verified after seeding.
- Optional or backward-compatible cleanup remains best-effort only when callers can safely continue.
- Tests live with the AuthNZ database coverage and reproduce the failure before implementation.

## Failure Scenarios

- A required column cannot be added or normalized because the SQLite schema is locked or malformed.
- Default RBAC roles or permissions fail to insert during bootstrap.
- Required role-permission links are missing after seeding, even if the insert attempt did not raise.

## Validation Criteria

- `_ensure_core_columns()` raises `UserDatabaseError` for required schema normalization failures.
- `_seed_default_data()` raises `UserDatabaseError` when required RBAC rows are still absent after seeding.
- Optional cleanup paths may log and continue only when the database remains usable and authorization semantics are unchanged.

## Implementation Plan

The staged implementation is tracked in [IMPLEMENTATION_PLAN_userdatabase_fail_closed_bootstrap.md](../Plans/IMPLEMENTATION_PLAN_userdatabase_fail_closed_bootstrap.md).
