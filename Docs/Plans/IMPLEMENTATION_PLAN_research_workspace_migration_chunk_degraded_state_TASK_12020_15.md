## Stage 1: Trace Migration Failure Path
**Goal**: Locate the migration chunk upload caller, request transport, and status aggregation that marks Research Workspace unreachable.
**Success Criteria**: Identify whether the status-0 failure is caused by transport, endpoint behavior, or overly broad UI status handling.
**Tests**: Read focused migration/status tests and capture the smallest regression target.
**Status**: Complete

## Stage 2: Add Focused Regression
**Goal**: Reproduce the contradictory `Connected` plus `Can't reach your tldw server` state without requiring a live browser.
**Success Criteria**: A test fails before implementation for migration chunk status-0 handling.
**Tests**: Focused workspace store/status or component test.
**Status**: Complete

## Stage 3: Implement Scoped Recovery
**Goal**: Keep migration chunk failures scoped and actionable without globally degrading an otherwise connected workspace.
**Success Criteria**: Core connected state remains accurate; migration failure copy points to migration recovery.
**Tests**: Focused regression passes.
**Status**: Complete

## Stage 4: Verify and Document
**Goal**: Run focused tests, browser/CDP recheck when possible, and update Backlog/UAT matrix.
**Success Criteria**: Verification evidence is recorded and remaining environment limits are explicit.
**Tests**: Focused Vitest plus `git diff --check`.
**Status**: Complete

Verification notes:
- `apps/packages/ui`: focused Vitest passed for connection store, background proxy, Settings health probe, and chat model cache coverage (5 files, 72 tests).
- `apps/tldw-frontend`: focused Vitest passed for WebLayout backend-unreachable behavior (2 files, 10 tests).
- `git diff --check` passed for the touched frontend, docs, and plan files.
- Live browser recheck was attempted after starting the local Next.js preview server. The in-app browser handle was unavailable after the Node browser runtime reset, and standalone Chromium failed to launch in the macOS sandbox with `bootstrap_check_in org.chromium.Chromium.MachPortRendezvousServer... Permission denied (1100)`. The preview server also reported Watchpack `EMFILE: too many open files, watch` warnings. No post-fix CDP/browser assertion was possible in this environment.
