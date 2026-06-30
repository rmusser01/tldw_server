# WebUI Stage 5 MCP Hub Capability Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a clear shared recovery state on `/mcp-hub` when the connected server does not expose the MCP Hub capability, while preserving the existing workflows when it does.

**Architecture:** Reuse the existing MCP Hub service contract and shared capability-state primitives. The standalone page performs one lightweight readiness query, maps failures through `buildCapabilityState`, and renders `RecoveryCallout` before tab-specific failures cascade.

**Tech Stack:** React, TanStack Query, Vitest, React Testing Library, shared WebUI state primitives.

---

## Stage 1: Failing Coverage
**Goal**: Capture the missing top-level MCP Hub recovery behavior.
**Success Criteria**: A test fails because `McpHubPage` does not yet call the readiness service or render `RecoveryCallout`.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
**Status**: Complete

- [x] Add a mocked `getToolRegistrySummary` readiness service to `McpHubPage.test.tsx`.
- [x] Add a failing unavailable-capability test that expects a shared recovery state, diagnostics, and retry action.
- [x] Update the existing workflow tests so successful readiness preserves current behavior.
- [x] Run the focused test and confirm the new test fails for the missing behavior.

## Stage 2: MCP Hub Recovery UI
**Goal**: Add the minimal readiness query and recovery panel to the MCP Hub page.
**Success Criteria**: Failed readiness renders user-language recovery copy with diagnostics; successful readiness keeps the workflow summary, tabs, and route state intact.
**Tests**: Focused MCP Hub test file from Stage 1.
**Status**: Complete

- [x] Import `useQuery`, `RecoveryCallout`, `buildCapabilityState`, and `getToolRegistrySummary`.
- [x] Add a `["mcp-hub", "capability-readiness"]` query with a short stale time and no retry spam.
- [x] Build a capability state for `/api/v1/mcp/hub/tool-registry/summary` failures.
- [x] Render `RecoveryCallout` with a retry action when the query fails without data.
- [x] Keep the current MCP Hub UI for loading and successful query states.

## Stage 3: Verification And Closeout
**Goal**: Prove the slice works and record the result.
**Success Criteria**: Focused tests pass, lint/whitespace checks are clean for touched files, and Backlog reflects verification and known browser-smoke limits.
**Tests**: Focused Vitest, ESLint touched files, `git diff --check`.
**Status**: Complete

- [x] Run the focused MCP Hub test file.
- [x] Run ESLint on touched TS/TSX files.
- [x] Run `git diff --check`.
- [x] Record Bandit as not applicable for TS/TSX/docs-only changes.
- [x] Update `TASK-12034` acceptance criteria, notes, touched files, and final summary.
- [x] Commit the Stage 5 slice.
