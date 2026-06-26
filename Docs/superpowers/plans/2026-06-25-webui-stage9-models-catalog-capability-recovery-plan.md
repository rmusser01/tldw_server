# WebUI Stage 9 Models Catalog Capability Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show shared, user-language capability recovery in the Models full catalog when model metadata loading fails, while preserving successful model catalog rendering, abort-as-empty behavior, and model/default-provider workflows.

**Architecture:** Reuse `RecoveryCallout` and `buildCapabilityState` in `AvailableModelsList`. Keep route-level defaults/readiness/OpenAI OAuth handling out of this slice unless tests show a direct regression. Diagnostics should remain non-secret and scoped to metadata request path/status/raw message.

**Tech Stack:** React, TypeScript, TanStack Query, shared WebUI state primitives, Vitest, React Testing Library.

---

## Stage 1: Failing Coverage
**Goal**: Capture the current Models catalog local-alert recovery gap.
**Success Criteria**: Focused tests fail because catalog metadata errors do not render through shared recovery diagnostics.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/Models/__tests__/AvailableModelsList.test.tsx ../packages/ui/src/components/Option/Models/__tests__/ModelsBody.test.tsx`
**Status**: Complete

- [x] Add a failing assertion that metadata load failures render a shared `RecoveryCallout`.
- [x] Add a failing assertion that diagnostics include request method/path and raw message/status when available.
- [x] Add a failing assertion that the retry action remains available.
- [x] Preserve existing assertions that abort-like request failures render the empty state.
- [x] Run the focused Models tests and confirm the new assertions fail for the current local-alert implementation.

## Stage 2: Models Catalog Recovery UI
**Goal**: Replace the full catalog load alert with shared recovery UI without changing successful catalog behavior.
**Success Criteria**: Catalog metadata failures use `RecoveryCallout` with diagnostics; success and abort-empty states remain unchanged.
**Tests**: Focused Models component tests.
**Status**: Complete

- [x] Import and use `RecoveryCallout` plus `buildCapabilityState` in `AvailableModelsList`.
- [x] Derive safe diagnostics from error status/message without exposing secrets.
- [x] Render the existing title/body copy through the shared recovery primitive.
- [x] Wire the primary retry action to the existing `refetch` behavior.
- [x] Keep successful catalog cards and empty provider state behavior intact.

## Stage 3: Verification And Closeout
**Goal**: Prove the slice works and record the result.
**Success Criteria**: Focused tests pass, lint/whitespace checks are clean for touched files, and Backlog reflects verification and known skips.
**Tests**: Focused Vitest, ESLint touched files, `git diff --check`.
**Status**: Complete

- [x] Run the focused Models tests.
- [x] Run ESLint on touched TS/TSX files.
- [x] Run `git diff --check`.
- [x] Record Bandit as not applicable for TS/TSX/docs-only changes.
- [x] Update `TASK-12038` acceptance criteria, notes, touched files, and final summary.
- [x] Commit the Stage 9 slice.
