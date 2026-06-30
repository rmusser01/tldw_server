# WebUI Stage 3 Setup And Health UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make first-run setup and health diagnostics guide self-hosted users through server URL and API key setup without mislabeling missing credentials as server outages.

**Architecture:** Add a focused self-host connection panel to `/setup` that writes through the existing shared connection store and keeps the existing unified setup wizard as the backend first-run path. Update health presentation helpers and page copy so connection state from the shared store controls missing URL, missing API key, invalid key, unreachable server, and degraded feature messaging. Keep changes small and testable in the shared UI package.

**Tech Stack:** React, React Router, Ant Design, Zustand connection store, Vitest, Testing Library.

---

## Stage 1: Setup Route Connection Panel

**Goal:** `/setup` gives new self-host users direct URL/API key setup, connection testing, key-location help, and a skip/explore path.

**Success Criteria:** The route renders a self-host setup panel even when backend first-run wizard state is complete, keeps operator recovery secondary, and uses the shared connection actions for save/test.

**Tests:** `bun run test:run ../packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx`

**Status:** Complete

- [x] Add a failing test that renders `/setup` with completed backend setup state and expects "Connect your tldw server", "Server URL", "API Key", "Test connection", "Where do I find my key?", and "Skip and explore UI".
- [x] Run the focused test and confirm it fails because the self-host panel is missing.
- [x] Implement the minimal setup panel in `apps/packages/ui/src/routes/option-setup.tsx` using `useConnectionActions().setConfigPartial` and `testConnectionFromOnboarding`.
- [x] Run the focused setup route test and confirm it passes.
- [x] Add and pass a skip regression proving "Skip and explore UI" writes `assistant_setup_dismissed=true` before navigating to `/chat`.

## Stage 2: Health Diagnostics Copy And Redaction

**Goal:** Missing credentials and setup issues do not read as generic core endpoint outages, and copied diagnostics do not leak secrets.

**Success Criteria:** Health UI distinguishes missing URL, missing API key, invalid key, unreachable server, and degraded checks, with diagnostics copy redacting secret-shaped fields.

**Tests:** `bun run test:run ../packages/ui/src/components/Option/Settings/__tests__/health-status.design-system.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/tldw-connection-status.test.ts`

**Status:** Complete

- [x] Add failing tests for `getCoreIssueLabel` labels: missing URL, missing API key, invalid API key, unreachable, and degraded feature checks.
- [x] Add a failing `HealthStatus` render test where the shared connection UX state is `configuring_auth`, the local core probe fails, and the page shows API-key setup copy instead of "Unable to reach server core health endpoint."
- [x] Add a failing copied-diagnostics test proving API keys/tokens in result details are replaced with `[redacted]`.
- [x] Implement a small diagnostic redaction helper and update health copy branches.
- [x] Run focused health tests and confirm they pass.

## Stage 3: Beginner Overlay Friction

**Goal:** The generic first-run overlay remains skippable and does not reappear after a user chooses to explore.

**Success Criteria:** Existing `assistant_setup_dismissed` behavior remains covered while setup route offers the new explore affordance.

**Tests:** `bun run test:run ../packages/ui/src/components/PersonaGarden/__tests__/FirstRunGate.test.tsx`

**Status:** Complete

- [x] Keep the existing regression that clicking "Skip for now" stores `assistant_setup_dismissed=true` and reveals page content.
- [x] Run the focused first-run gate test.
- [x] Add setup-route skip persistence so explore navigation does not immediately trigger the generic overlay.

## Stage 4: Verification And Task Finalization

**Goal:** Complete TASK-12032 with focused verification and a clean commit.

**Success Criteria:** Focused tests pass, lint/diff checks are clean for touched files, Backlog acceptance criteria and notes are updated, and changes are committed.

**Tests:** setup route test, health status tests, connection-status tests, first-run gate test, direct lint on touched TS/TSX files, `git diff --check`.

**Status:** Complete

- [x] Run focused Vitest commands from `apps/tldw-frontend`.
- [x] Run lint on touched frontend files.
- [x] Run `git diff --check`.
- [x] Document Bandit as not applicable because only TypeScript/TSX, docs, and Backlog task files changed.
- [x] Update TASK-12032 acceptance criteria, notes, touched files, and final summary.
- [x] Stage and commit the Stage 3 work.
