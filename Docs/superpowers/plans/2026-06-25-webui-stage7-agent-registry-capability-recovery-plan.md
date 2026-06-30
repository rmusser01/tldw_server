# WebUI Stage 7 Agent Registry Capability Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show shared, user-language capability recovery states in Agent Registry when ACP health, admin execution-health, or agent-list requests fail, while preserving the existing registry cards and execution-health summary when requests succeed.

**Architecture:** Reuse the existing `buildCapabilityState`, `RecoveryCallout`, and `StatePanel` primitives. Keep optional admin execution-health failure non-blocking, because that endpoint can be permission-gated, but expose retryable diagnostics instead of local alert-only copy. Treat the agent-list request as the route's primary capability gate.

**Tech Stack:** React, TypeScript, existing ACP REST client, shared WebUI state primitives, Vitest, React Testing Library.

---

## Stage 1: Failing Coverage
**Goal**: Capture the current Agent Registry local-alert recovery gap.
**Success Criteria**: Focused tests fail because unavailable health, unavailable admin summary, and failed agent-list states are not rendered through shared recovery primitives with diagnostics.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx`
**Status**: Complete

- [x] Add a failing assertion that ACP health failure renders a shared recovery callout or state panel with `/api/v1/acp/health` diagnostics.
- [x] Add a failing assertion that admin execution-health failure keeps the registry usable and shows non-blocking diagnostics for `/api/v1/admin/acp/execution-health/summary`.
- [x] Add a failing assertion that agent-list load failure shows user-language recovery with retry/dismiss action and does not expose raw error text as the primary state.
- [x] Run the focused Agent Registry test file and confirm the new assertions fail for the current local-alert implementation.

## Stage 2: Agent Registry Recovery UI
**Goal**: Replace local unavailable/error alerts with shared state primitives without changing successful route behavior.
**Success Criteria**: Failed health and admin summary requests show shared, diagnostic recovery states; failed agent-list requests show a retryable route capability state; successful registry cards and execution-health summary remain unchanged.
**Tests**: Focused Agent Registry connection test.
**Status**: Complete

- [x] Track ACP health and execution-health failure metadata separately from their normalized successful payloads.
- [x] Build capability states for ACP health, admin execution-health, and agent-list failures using method, request path, status, server URL, and raw message.
- [x] Render non-blocking `RecoveryCallout` or `StatePanel` sections for health/admin summary failures.
- [x] Render a retryable primary recovery state for failed agent-list loading while preserving existing card rendering on success.
- [x] Keep refresh, canonical connection config, compatibility labels, and execution-health summary behavior intact.

## Stage 3: Verification And Closeout
**Goal**: Prove the slice works and record the result.
**Success Criteria**: Focused tests pass, lint/whitespace checks are clean for touched files, and Backlog reflects verification and known skips.
**Tests**: Focused Vitest, ESLint touched files, `git diff --check`.
**Status**: Complete

- [x] Run the focused Agent Registry test file.
- [x] Run ESLint on touched TS/TSX files.
- [x] Run `git diff --check`.
- [x] Record Bandit as not applicable for TS/TSX/docs-only changes.
- [x] Update `TASK-12036` acceptance criteria, notes, touched files, and final summary.
- [x] Commit the Stage 7 slice.
