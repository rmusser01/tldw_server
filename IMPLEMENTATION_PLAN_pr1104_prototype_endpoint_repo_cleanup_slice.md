## Stage 1: Cleanup Regressions
**Goal**: Capture the preview-renew endpoint and row-conversion review comments with focused tests.
**Success Criteria**: Tests fail against the existing implementation for loose renew-body validation, string-based renewal error mapping, and silent row-conversion failures.
**Tests**: `test_prototype_endpoints.py`, `test_prototype_repo.py`
**Status**: Complete

## Stage 2: Endpoint And Repo Fixes
**Goal**: Make preview-renew request handling strict and typed, and make row-conversion failures debuggable without broad exception catches.
**Success Criteria**: Focused tests pass with narrow production changes.
**Tests**: Focused endpoint/repo tests plus full PrototypeWorkspaces suite.
**Status**: Complete

## Stage 3: Verification And Thread Closeout
**Goal**: Run lint/security/diff checks, commit, push, and close the addressed PR threads.
**Success Criteria**: Verification evidence is fresh and PR threads are replied to and resolved.
**Tests**: Ruff, Bandit touched scope, diff checks.
**Status**: In Progress
