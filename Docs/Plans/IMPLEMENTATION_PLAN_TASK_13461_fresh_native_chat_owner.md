# TASK-13461: Fresh Native Chat Owner Repair

ADR assessment: no new ADR. Restore the owner-bound persistence contract in
`Docs/ADR/049-chat-history-selection-ownership.md`; do not change backend
schemas or permit arbitrary native result metadata.

## Stage 1: Reproduce Ownership Failure
**Goal**: Reproduce a connected plain first send choosing local history.
**Success Criteria**: Regression fails before repair; existing local path stays covered.
**Tests**: Mounted normal mode and actual useChatActions first-send regressions.
**Status**: Complete

## Stage 2: Restore Native Ownership
**Goal**: Create and verify the scoped native owner before admission or dispatch.
**Success Criteria**: Exactly one chat and one user/assistant pair; cancelled or stale
loads cannot adopt an owner; reasoning text persists without client-clock metadata.
**Tests**: Creation/load cancellation, account/navigation fences, tools, reasoning,
reopen, and unchanged explicit-selection cancellation.
**Status**: Complete

## Stage 3: Verify and Integrate
**Goal**: Review the generic fix against latest dev and submit upstream.
**Success Criteria**: Focused and sibling tests and frontend typecheck pass;
baseline failures are recorded rather than concealed; review comments addressed.
**Tests**: Vitest ownership/sibling suites, production frontend tsc, diff checks.
**Status**: In Progress
