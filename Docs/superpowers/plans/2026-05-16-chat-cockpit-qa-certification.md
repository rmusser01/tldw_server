# Main Chat Cockpit QA Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Certify the main WebUI `/chat` cockpit maturity work with durable focused tests, real-server proof coverage, and a merge checklist tied to evidence.

**Architecture:** This is a QA and certification slice, not a product-surface rewrite. It keeps implementation scoped to the existing `Playground` cockpit tests, the existing real-server Playwright workflow, and a small certification artifact that maps the roadmap acceptance items to concrete evidence.

**Tech Stack:** Next.js WebUI, shared `apps/packages/ui` React components, Vitest + Testing Library, Playwright against the running tldw server, Backlog.md task tracking.

---

Roadmap: `Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md`
Backlog: TASK-403
Scope: Main WebUI `/chat` cockpit only. No browser extension/sidebar/sidepanel work.

## Stage 1: Coverage Inventory and Guardrail
**Goal**: Confirm the current cockpit tests already cover the P0 proof paths and identify the smallest durable gap.
**Success Criteria**: Existing focused unit and real-server E2E files are mapped to PR8 acceptance items; no sidepanel/sidebar files are included.
**Tests**: Read-only inventory of `Playground.cockpit-*`, `playground-cockpit-summaries.test.ts`, and `chat-cockpit.real-server.spec.ts`.
**Status**: Complete

- [x] Review focused cockpit unit tests for summaries, rails, state labels, assistant selection, MCP states, readiness, and mobile panel behavior.
- [x] Review real-server Playwright coverage for prompt, persona, character, model settings, MCP settings, conversation send, focus mode, and mobile screenshots.
- [x] Decide PR8 implementation should add keyboard-specific cockpit control coverage plus a certification artifact, not duplicate existing send/persona/model flows.

## Stage 2: Keyboard and Focus Proof
**Goal**: Add a failing unit test first for cockpit controls that are reachable by keyboard, then implement only if the test exposes a product gap.
**Success Criteria**: The layout mode toggle, independent rail visibility controls, and mobile cockpit tabs can be activated with keyboard events in focused tests.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`
**Status**: Complete

- [x] Add a keyboard-focused test to `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`.
- [x] Run the focused test and verify the new assertion fails or proves already-working keyboard behavior.
- [x] If needed, minimally update `PlaygroundCockpitShell.tsx` to satisfy the keyboard/focus contract.
- [x] Re-run the focused test until it passes.

## Stage 3: Merge Certification Artifact
**Goal**: Make the PR8 closeout auditable without requiring reviewers to reverse-engineer the test suite.
**Success Criteria**: A certification doc maps PR8 and merge-critical cockpit criteria to evidence: unit tests, real-server Playwright tests, screenshot outputs, and known skips.
**Tests**: Documentation review plus `git diff --check`.
**Status**: Complete

- [x] Create `Docs/superpowers/specs/2026-05-16-main-chat-cockpit-merge-certification.md`.
- [x] Include the exact real-server command expected for final proof and note that it does not mock backend routes.
- [x] Map prompt, persona, character, model settings, MCP states, assistant matrix, mobile/focus, screenshots, and sidepanel/sidebar exclusion to concrete files/tests.

## Stage 4: Verification and Closeout
**Goal**: Run focused checks, update task tracking, and commit the PR8 slice.
**Success Criteria**: Focused Vitest, real-server Playwright, targeted lint, design-system guard, diff check, and applicable security checks are recorded with results; TASK-403 is completed only after evidence is fresh.
**Tests**: Focused cockpit Vitest suite; real-server `chat-cockpit.real-server.spec.ts`; targeted ESLint; `bun run verify:design-system-state`; `git diff --check`; Bandit skip documented if touched scope remains frontend/docs.
**Status**: Complete

- [x] Run focused cockpit Vitest tests.
- [x] Run real-server Playwright against the already-running server with the `.env` API key.
- [x] Run targeted ESLint on touched TS/TSX files.
- [x] Run design-system state verification if cockpit UI files changed.
- [x] Run `git diff --check`.
- [x] Update TASK-403 acceptance criteria, implementation notes, final summary, and Definition of Done.
- [x] Commit the PR8 certification slice.
