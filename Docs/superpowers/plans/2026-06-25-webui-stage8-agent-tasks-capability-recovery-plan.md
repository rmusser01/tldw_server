# WebUI Stage 8 Agent Tasks Capability Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show shared, user-language capability recovery states in Agent Tasks for top-level unsupported, setup, workspace, and project-load failures while preserving the existing project/task workflow when agent orchestration is available.

**Architecture:** Reuse `RecoveryCallout`, `StatePanel`, and `buildCapabilityState`. Keep task-detail and modal action errors out of this slice; those remain workflow-specific and can be handled separately. Add structured diagnostics only for request failures where method/path/status/server URL/raw message are available.

**Tech Stack:** React, TypeScript, existing ACP/orchestration fetch calls, shared WebUI state primitives, Vitest, React Testing Library.

---

## Stage 1: Failing Coverage
**Goal**: Capture the current Agent Tasks local-alert recovery gap.
**Success Criteria**: Focused tests fail because top-level unsupported/setup/project-load states do not render through shared state primitives with diagnostics.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx`
**Status**: Complete

- [x] Add a failing assertion that unsupported orchestration renders shared recovery with `/api/v1/agent-orchestration/projects` diagnostics.
- [x] Add a failing assertion that ACP setup gaps render a shared setup state with the existing setup actions.
- [x] Add a failing assertion that workspace setup gaps render a shared setup state with the existing workspace actions.
- [x] Add a failing assertion that project-load failure renders retryable shared recovery with method/path/status/server URL/raw message diagnostics.
- [x] Run the focused Agent Tasks test file and confirm the new assertions fail for the current local-alert implementation.

## Stage 2: Agent Tasks Recovery UI
**Goal**: Replace top-level local alerts with shared state primitives without changing successful route behavior.
**Success Criteria**: Unsupported orchestration, ACP setup gaps, workspace setup gaps, and project-load failures use shared state primitives; project/task rendering remains unchanged on success.
**Tests**: Focused Agent Tasks connection test.
**Status**: Complete

- [x] Track project-list request failure metadata separately from the user-facing error string.
- [x] Build capability states for unsupported orchestration and project-list load failures.
- [x] Render setup/workspace warnings through `StatePanel` with existing setup body/actions preserved.
- [x] Render retryable project-list `RecoveryCallout` with diagnostics.
- [x] Keep workspace filtering, project/task list rendering, task diagnostics, and canonical connection tests intact.

## Stage 3: Verification And Closeout
**Goal**: Prove the slice works and record the result.
**Success Criteria**: Focused tests pass, lint/whitespace checks are clean for touched files, and Backlog reflects verification and known skips.
**Tests**: Focused Vitest, ESLint touched files, `git diff --check`.
**Status**: Complete

- [x] Run the focused Agent Tasks test file.
- [x] Run ESLint on touched TS/TSX files.
- [x] Run `git diff --check`.
- [x] Record Bandit as not applicable for TS/TSX/docs-only changes.
- [x] Update `TASK-12037` acceptance criteria, notes, touched files, and final summary.
- [x] Commit the Stage 8 slice.
