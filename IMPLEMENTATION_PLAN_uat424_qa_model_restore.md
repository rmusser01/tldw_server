# UAT424 Knowledge QA model restoration

Task: TASK-13260.278.19. PR #2979. Full/native UAT remains paused.

## Stage 1: Reproduce lost generation settings
**Goal**: Trace canonical and local snapshots from the failed CI thread.
**Success Criteria**: Regressions fail when provider/model are dropped during write or restore; null and malformed values have explicit controls.
**Tests**: Existing KnowledgeQAProvider persistence/history suites.
**Status**: Complete

## Stage 2: Preserve owned choices
**Goal**: Carry typed string/null generation choices through the existing account-scoped snapshots.
**Success Criteria**: Restored follow-up dispatches the selected model/provider; default and legacy behavior stay intact.
**Tests**: Persistence/history and adjacent authority/streaming suites; strict journey request assertion.
**Status**: In Progress

## Stage 3: Validate and publish
**Goal**: Verify lint/types and first-attempt CI before accepting the integrated path.
**Success Criteria**: Evidence and limits recorded in Backlog/tracker/PR; generated artifacts excluded.
**Tests**: Frontend checks and remote strict journey. No Python security scope in this repair.
**Status**: In Progress
