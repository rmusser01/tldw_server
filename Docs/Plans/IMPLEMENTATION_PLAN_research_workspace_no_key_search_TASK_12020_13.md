## Stage 1: Confirm Search Ownership
**Goal**: Keep Research Workspace responsible for its own search on `/research-workspace`, while the app command palette remains disabled there.
**Success Criteria**: Existing command-palette route behavior remains unchanged.
**Tests**: `CommandPaletteHost.test.tsx`.
**Status**: Complete

## Stage 2: Add No-Key Search Regression
**Goal**: Prove fresh/no-key Research Workspace exposes a visible search affordance and opens search with Cmd/Ctrl+K.
**Success Criteria**: Focused Research Workspace tests fail before implementation for the missing affordance/shortcut.
**Tests**: `ResearchWorkspace.shortcuts.test.tsx`, `ResearchWorkspace.stage3.test.tsx`, `WorkspaceHeader.test.tsx`.
**Status**: Complete

## Stage 3: Implement Search Affordance
**Goal**: Add a header search button that opens the workspace search modal and advertises Cmd/Ctrl+K.
**Success Criteria**: Desktop and mobile header states expose search without requiring credentials or sources.
**Tests**: Focused workspace tests pass.
**Status**: Complete

## Stage 4: Verify and Document
**Goal**: Run focused frontend tests, update the UAT matrix and Backlog task, and document any browser environment limits.
**Success Criteria**: Test output and documentation state exactly what was verified.
**Tests**: Focused Vitest plus `git diff --check`.
**Status**: Complete
