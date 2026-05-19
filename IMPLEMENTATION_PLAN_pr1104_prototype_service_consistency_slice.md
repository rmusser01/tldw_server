## Stage 1: Service Consistency Regressions
**Goal**: Capture partial-write failure modes in prototype workspace service orchestration.
**Success Criteria**: Tests fail against the current implementation for workspace seed failures, session snapshot state failures, and promotion request failures after preview grant issuance.
**Tests**: `test_promotion_service.py`
**Status**: Complete

## Stage 2: Compensation Fixes
**Goal**: Add explicit repository cleanup helpers and service compensation paths for the reviewed multi-step writes.
**Success Criteria**: Failed service writes leave archived/deleted/revoked or reverted state instead of usable partial state.
**Tests**: Focused promotion/service tests plus full PrototypeWorkspaces suite.
**Status**: Complete

## Stage 3: Verification And Thread Closeout
**Goal**: Run lint/security/diff checks, commit, push, and resolve the service consistency PR threads.
**Success Criteria**: Fresh verification is recorded and the addressed review threads are closed.
**Tests**: Ruff, Bandit touched scope, diff checks.
**Status**: In Progress
