# Sharing Review Fixes Implementation Plan

## Stage 1: Token Creation And Redemption

**Goal**: Enforce resource ownership for share-token creation and make public imports consume token uses atomically.
**Success Criteria**: Workspace tokens call the workspace ownership guard, chatbook ownership is checked before token creation, public import cannot exceed `max_uses`, and password-protected imports have a usable verification path.
**Tests**: Focused Sharing endpoint tests for workspace/chatbook ownership rejection, max-use claiming, and password inline verification/import.
**Status**: Complete

## Stage 2: Workspace Deletion Cleanup

**Goal**: Wire the Sharing cleanup hook into workspace deletion.
**Success Criteria**: Deleting a workspace invokes `on_workspace_deleted` after the local workspace delete succeeds and revokes active workspace shares/tokens.
**Tests**: Workspace endpoint test covering hook invocation and existing hook tests.
**Status**: Complete

## Stage 3: Clone Data Integrity

**Goal**: Preserve media chunks during clone and report successful copy counts.
**Success Criteria**: Cloned media receives source unvectorized chunks when available, skipped/failed sources/notes/artifacts are not counted as copied, and partial clone behavior is explicit.
**Tests**: Clone service tests for chunk copy and accurate counts on failures.
**Status**: Complete

## Stage 4: Audit Failure Behavior

**Goal**: Prevent post-mutation audit failures from returning misleading failed user operations.
**Success Criteria**: Endpoint mutations that already completed do not return 500 solely because audit logging failed; failures remain logged without leaking sensitive details.
**Tests**: Sharing endpoint test that simulates audit failure after share creation.
**Status**: Complete

## Stage 5: Verification And Task Closeout

**Goal**: Verify focused behavior and record task evidence.
**Success Criteria**: Focused pytest suite passes, compile check passes, Bandit touched-scope scan is recorded, and Backlog task acceptance criteria/DoD/final summary are updated.
**Tests**: `python -m pytest` on touched Sharing/workspace tests, `python -m compileall`, and `python -m bandit` on touched code.
**Status**: Complete
