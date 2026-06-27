## Stage 1: Trace Settings Auth Readiness
**Goal**: Confirm why Settings can report success while Research Workspace receives authenticated 401 responses.
**Success Criteria**: Identify the specific validation or propagation gap and record it against TASK-12020.12.
**Tests**: Existing focused settings/client tests reviewed for the right regression target.
**Status**: Complete

## Stage 2: Add Authenticated Probe Regression
**Goal**: Add a failing test proving Settings must not pass on public health alone when a single-user API key is supplied.
**Success Criteria**: The test fails before implementation because `/api/v1/llm/models/metadata` is not checked.
**Tests**: `server-health-probe.test.ts`.
**Status**: Complete

## Stage 3: Implement Minimal Settings Probe Fix
**Goal**: Make the connection probe verify a workspace-critical authenticated endpoint with the same credential before returning success.
**Success Criteria**: Invalid keys return a credential-specific failure and valid keys still report success.
**Tests**: Focused settings probe tests.
**Status**: Complete

## Stage 4: Verify Propagation and Close Out
**Goal**: Run focused regression tests, update TASK-12020.12 notes, and document any remaining browser/runtime blockers.
**Success Criteria**: Tests pass or blockers are documented with evidence.
**Tests**: Focused Vitest targets and `git diff --check`.
**Status**: Complete
