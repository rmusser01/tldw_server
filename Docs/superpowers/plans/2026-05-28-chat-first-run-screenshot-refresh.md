## Stage 1: Backend Startup Diagnosis
**Goal**: Determine why the previous evidence backend exited before screenshot capture.
**Success Criteria**: Startup either succeeds on a clean port or the failure is recorded with concrete log evidence.
**Tests**: Health check against the selected backend port.
**Status**: Complete

## Stage 2: Live Screenshot Capture
**Goal**: Capture the current first-time `/chat` state after TASK-536.
**Success Criteria**: `first-time-unseeded.png` shows `/chat` without the global assistant setup overlay.
**Tests**: Browser automation checks for `/chat`, no `first-run-gate-overlay`, and visible chat surface copy.
**Status**: Complete

## Stage 3: Evidence Update
**Goal**: Update structured evidence and review notes to treat the new screenshot as current.
**Success Criteria**: Evidence JSON parses, README/review no longer mark the first-time screenshot as pre-TASK-536, and TASK-537 records verification.
**Tests**: JSON parse check; `git diff --check`.
**Status**: Complete

## Stage 4: Closeout
**Goal**: Stop local evidence servers and complete TASK-537.
**Success Criteria**: No evidence ports remain listening; focused tests pass; Backlog AC/DoD are checked.
**Tests**: Focused Vitest; port checks; `git diff --check`.
**Status**: Complete
