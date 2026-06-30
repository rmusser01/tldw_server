# Research Workspace Missing Server Workspace Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover Research Workspace when a local browser session points at a server workspace ID that does not exist after authentication succeeds.

**Architecture:** Keep recovery in the existing workspace store/service boundary. Treat `404 Workspace not found` from workspace context or sources as a workspace reconciliation problem, not as a global backend outage when health is reachable. Preserve local workspace state and bind it to a newly created or discovered server workspace.

**Tech Stack:** Next.js, React, Zustand, TypeScript, Vitest, Testing Library, in-app browser CDP.

---

## Stage 1: Trace Existing Failure Path
**Goal**: Identify where local workspace IDs are persisted, loaded, and mapped to server workspace APIs.
**Success Criteria**: The failing context/sources request path and global backend modal trigger are tied to concrete files and state transitions.
**Tests**: No code tests; record findings in TASK-12020.16 notes.
**Status**: Complete

- [x] Inspect workspace store, API service, source slice, and backend-unavailable modal classification.
- [x] Confirm which existing test files cover workspace loading and backend error classification.
- [ ] Record the CDP failure evidence and root-cause notes in Backlog.

## Stage 2: Add Failing Tests
**Goal**: Lock the regression around authenticated recovery and misleading unreachable messaging.
**Success Criteria**: Tests fail for the current behavior because a missing server workspace is not reconciled and is classified as a backend outage.
**Tests**: Focused Vitest cases in the existing workspace store/service and layout/modal test suites.
**Status**: Complete

- [x] Add a stale-local-workspace recovery test that starts with a persisted server workspace ID, simulates a `404 Workspace not found`, and expects recovery to a usable server workspace.
- [x] Add or extend modal classification coverage so workspace 404 does not open the global server-unreachable modal when health is reachable.
- [x] Run the focused tests and capture the expected red failures.

## Stage 3: Implement Minimal Recovery
**Goal**: Reconcile a missing server workspace without data loss and without generic backend-unreachable UI.
**Success Criteria**: On authenticated load, a missing server workspace ID is cleared, recreated, rebound, or replaced through existing workspace APIs; local sources, notes, and metadata remain available.
**Tests**: The Stage 2 tests pass.
**Status**: Complete

- [x] Add a small helper or store branch that detects workspace-not-found responses from context/sources.
- [x] Reuse existing workspace creation/binding APIs instead of adding a new server endpoint.
- [x] Keep user-visible recovery feedback scoped to workspace sync status.
- [x] Avoid destructive clearing of local workspace data.

## Stage 4: Browser Verification and Evidence
**Goal**: Re-run the invalid/no-key-auth-to-valid-key workflow in the in-app browser and update the UAT matrix.
**Success Criteria**: Settings shows a valid connection, Research Workspace reaches a usable connected state, and the global server-unreachable modal does not appear for missing-workspace recovery.
**Tests**: Focused Vitest suite, `git diff --check`, and in-app browser screenshots/logs.
**Status**: Complete

- [x] Run focused Vitest suites for workspace recovery and backend modal classification.
- [x] Run `git diff --check`.
- [x] Repeat the clean browser persona path: invalid/no-key-auth Research Workspace load, save valid API key, return to Research Workspace, retry if needed.
- [x] Update `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md` and TASK-12020.16 with results.
