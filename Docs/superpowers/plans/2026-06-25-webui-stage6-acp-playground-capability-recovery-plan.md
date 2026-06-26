# WebUI Stage 6 ACP Playground Capability Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a clear shared recovery state on `/acp-playground` when ACP health is unavailable, while preserving the existing session/chat/tools workspace when ACP is healthy or degraded.

**Architecture:** Reuse the existing ACP health query and shared capability-state primitives. The page keeps the current loading, healthy, and degraded layout, but maps definitive unavailable health into a `RecoveryCallout` with retry and non-secret diagnostics.

**Tech Stack:** React, TanStack Query, Vitest, React Testing Library, shared WebUI state primitives.

---

## Stage 1: Failing Coverage
**Goal**: Capture the missing top-level ACP Playground recovery behavior.
**Success Criteria**: A test fails because the ACP Playground recovery component does not yet exist.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/ACPPlayground/__tests__/ACPPlaygroundRecovery.test.tsx`
**Status**: Complete

- [x] Add a failing unavailable-health test to `ACPPlaygroundRecovery.test.tsx`.
- [x] Mock ACP health as an unavailable snapshot and assert user-language recovery copy.
- [x] Assert diagnostics include `GET`, `/api/v1/acp/health`, status, server URL, and raw message.
- [x] Assert the retry button calls the supplied retry handler.
- [x] Run the focused test and confirm the new test fails for the missing component.

## Stage 2: ACP Recovery UI
**Goal**: Add minimal health diagnostics and recovery rendering.
**Success Criteria**: Unavailable ACP health renders a shared recovery panel; healthy/degraded health keeps the current layout, session hydration, and deep-link behavior.
**Tests**: Focused ACP Playground connection test.
**Status**: Complete

- [x] Import shared recovery helpers through a focused ACP recovery component.
- [x] Add a small ACP health response shape that preserves `overall`, `status`, and `rawMessage`.
- [x] Build a capability state for `/api/v1/acp/health` failures using the existing connection server URL.
- [x] Render `RecoveryCallout` with a retry action only when health is definitively unavailable and the query is not loading.
- [x] Keep existing desktop and mobile layouts for healthy/degraded/loading states.

## Stage 3: Verification And Closeout
**Goal**: Prove the slice works and record the result.
**Success Criteria**: Focused tests pass, lint/whitespace checks are clean for touched files, and Backlog reflects verification and known skips.
**Tests**: Focused Vitest, ESLint touched files, `git diff --check`.
**Status**: Complete

- [x] Run the focused ACP Playground recovery test file.
- [x] Run ESLint on touched TS/TSX files.
- [x] Run `git diff --check`.
- [x] Record Bandit as not applicable for TS/TSX/docs-only changes.
- [x] Update `TASK-12035` acceptance criteria, notes, touched files, and final summary.
- [x] Commit the Stage 6 slice.
