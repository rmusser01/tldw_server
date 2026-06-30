# ACP Schedules Triggers And Background Runs Implementation Plan

## Stage 1: Root Cause And Scope
**Goal**: Stabilize the #1474 schedule routing baseline before expanding behavior.
**Success Criteria**: The current failing schedule routing tests have a traced root cause and a narrow compatibility contract.
**Tests**: Targeted failing `test_acp_schedules.py` cases.
**Status**: Complete

## Stage 2: Schedule Routing Compatibility
**Goal**: Ensure `_load_all()` and `_rescan_once()` can discover schedules from both modern DB handles and older/test handles.
**Success Criteria**: ACP schedules route to `_add_acp_job`, workflow schedules route to `_add_job`, and owner IDs are preserved.
**Tests**: Add a focused `_list_registered_schedules()` fallback test, then rerun schedule routing tests.
**Status**: Complete

## Stage 3: Background Run State And Concurrency
**Goal**: Harden scheduled ACP run status, retry/failure visibility, and explicit concurrency semantics.
**Success Criteria**: ACP schedule execution records pending, queued, skipped, and error states in a way operators can inspect.
**Tests**: Focused schedule execution tests for submit success, submit failure, disabled/missing schedules, and concurrency metadata.
**Status**: Complete

## Stage 4: Trigger Security And Docs
**Goal**: Verify webhook trigger security boundaries and document operational behavior for schedules/triggers.
**Success Criteria**: Trigger secret handling, replay/signature failures, and sanitized webhook errors remain covered; ACP docs/readiness matrix describe ownership, concurrency, failure, and security boundaries.
**Tests**: Trigger endpoint/core tests, schedule tests, Bandit, `git diff --check`.
**Status**: Complete
