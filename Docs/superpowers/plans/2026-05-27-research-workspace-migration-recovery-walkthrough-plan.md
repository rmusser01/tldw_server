## Stage 1: Contract And Recovery Criteria
**Goal**: Define the migration recovery walkthrough expectations for TASK-478.25 from the current Research Workspace code, migration protocol, and UAT matrix.
**Success Criteria**: Backlog acceptance criteria capture resumable failure, server-ineligible, blocked-inventory, true-move deletion, import/export recovery, and no `/workspace-playground` route compatibility.
**Tests**: Read-only inspection plus focused RED tests for the behavior gaps.
**Status**: Complete

## Stage 2: Migration Runner Recovery State
**Goal**: Preserve enough client-side migration plan state when API work fails so the UI can diagnose and retry without deleting local content.
**Success Criteria**: Failed runs include migration id, manifest hash, local inventory eligibility, and do not delete or acknowledge local content.
**Tests**: `workspace-migration.test.ts`.
**Status**: Complete

## Stage 3: Guided Recovery UI
**Goal**: Replace the non-actionable "Review recovery details" status text with a compact details action in the existing footer status bar, without adding another banner.
**Success Criteria**: Users can open a modal showing migration id, manifest hash, server status, client-delete eligibility, deleted surfaces, retained/unknown surfaces, and a retry action for failed or retained states.
**Tests**: `WorkspaceStatusBar.test.tsx` and `ResearchWorkspace.stage3.test.tsx`.
**Status**: Complete

## Stage 4: Import/Export Recovery Hardening
**Goal**: Ensure current imports work for current bundles and intentionally supported legacy workspace export bundles without exposing old route labels in current UI.
**Success Criteria**: Import accepts legacy-format bundle payloads as one-time recovery input while export continues to emit the current `tldw.research-workspace.bundle` format.
**Tests**: `workspace.test.ts`.
**Status**: Complete

## Stage 5: Live UAT Evidence
**Goal**: Validate the guided recovery walkthrough against a live backend and WebUI using Playwright/CDP, then update RW-UAT-025 only as far as the evidence supports.
**Success Criteria**: Live run records import/export, failed/retryable or retained migration state, and no `/workspace-playground` redirect/alias.
**Tests**: Focused Vitest, backend migration API tests if backend code changes, live Playwright/CDP walkthrough, and Bandit for touched backend paths or documented skip.
**Status**: Complete
