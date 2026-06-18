# Workspace Active Context Eligibility Plan

## Stage 1: Contract Tests
**Goal**: Capture the active-context eligibility matrix before implementation.
**Success Criteria**: Tests describe visibility behavior, active workspace gates, resource membership gates, runtime gates, and stable denial reasons.
**Tests**:
- `tldw_Server_API/tests/Workspaces/test_workspace_eligibility.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_eligibility_api.py`
**Status**: Complete

## Stage 2: Eligibility Core
**Goal**: Add a shared Workspace eligibility helper that answers whether an operation can use a resource in the active workspace context.
**Success Criteria**: The helper accepts active workspace, resource, operation, runtime state, and permission state; it returns stable reason codes and recovery actions without changing browse/search/open/edit visibility semantics.
**Tests**:
- Unit tests for visibility operations, no active workspace, archived workspace, unlinked resource, cross-workspace resource, unsupported resource type, missing runtime, and permission denial.
**Status**: Complete

## Stage 3: API Surface
**Goal**: Expose the eligibility helper through a small contract endpoint for frontend and integration callers.
**Success Criteria**: `POST /api/v1/workspace-eligibility/check` returns the shared response shape and is registered with the workspace routes.
**Tests**:
- API tests for allowed visibility checks, linked resource checks, and denial payloads.
**Status**: Complete

## Stage 4: Documentation
**Goal**: Document the operation matrix, reason codes, and recovery guidance in the Workspace core contract.
**Success Criteria**: Workspace docs explain the difference between global visibility and active-context eligibility, and point future resource adapters to issue #2378.
**Tests**:
- Documentation review during self-review.
**Status**: Complete

## Stage 5: Verification
**Goal**: Verify focused behavior and security checks before opening a PR.
**Success Criteria**: Focused Workspace tests pass, Bandit runs on touched backend paths, Backlog task is updated, changes are committed and pushed, and a PR is opened against `dev`.
**Tests**:
- Focused pytest command for Workspace eligibility and memberships.
- Bandit against touched backend modules.
**Status**: Complete
