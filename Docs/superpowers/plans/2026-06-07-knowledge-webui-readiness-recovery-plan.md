# Knowledge WebUI Readiness Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the WebUI `/knowledge` readiness failure so users never see a blank route after backend health timeout.

**Architecture:** Keep the global WebUI readiness gate, but make timeout and failure states visible, actionable, and compatible with route-level Knowledge QA recovery. Route-level recovery should explain Knowledge-specific setup and offline states after the global gate stops waiting.

**Tech Stack:** Next.js, React, shared Knowledge QA UI, Vitest, Playwright.

**Backlog Task:** TASK-528.2

---

## Boundaries

- This plan fixes WebUI readiness and route recovery only.
- Do not redesign the full Knowledge QA workspace in this phase.
- Do not add flashcard behavior to `/knowledge`.

## Files

- Modify: `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx`
- Modify: `apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx`
- Create or modify: `apps/tldw-frontend/e2e/ux-audit/knowledge-readiness-recovery.spec.ts`
- Verify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx`

## Task 1: Write Failing WebUI Timeout Test

- [x] Add a test where `/api/v1/health` never resolves before the readiness timeout.
- [x] Assert that the rendered page contains a visible recovery heading, backend URL or health target, Retry action, diagnostics/setup action, and no blank main area.
- [x] Run:

```bash
bunx vitest run components/networking/__tests__/ServerReadinessGate.test.tsx
```

Result: failed before implementation because no readiness recovery heading/actions rendered and stalled health never reached recovery.

## Task 2: Expose Actionable Readiness State

- [x] Update `ServerReadinessGate` to distinguish checking, retrying, timeout, and explicit failed/stalled health outcomes.
- [x] Preserve existing behavior for healthy backend.
- [x] On timeout, render an explicit visible gate recovery panel while keeping route children mounted for route-level recovery.
- [x] Include Retry, Health and diagnostics, and server settings actions.
- [x] Ensure dark and light theme text uses existing design-system surface/text tokens.

## Task 3: Allow KnowledgeQA Route Recovery

- [x] Inspect `_app.tsx` ordering for `ServerReadinessGate` and `FirstRunGate`.
- [x] Ensure `/knowledge` can reach `KnowledgeQA` recovery states after timeout by keeping route children mounted beneath the global recovery panel.
- [x] Avoid new shared context/props because the existing route-level Knowledge QA recovery remains available.
- [x] Add test coverage that route-level content remains mounted when backend health fails.

## Task 4: Verify Browser Behavior

- [x] Add Playwright coverage for WebUI `/knowledge` with stalled health and failed health.
- [x] Assert the route exposes visible recovery within the configured timeout.
- [x] Assert route content remains mounted and the recovery panel is non-empty.
- [x] Run:

```bash
npx playwright test e2e/ux-audit/knowledge-readiness-recovery.spec.ts --reporter=line
```

Result: passed, 2 tests. A stale existing dev server had to be restarted so Playwright loaded the updated gate bundle.

## Task 5: Close Verification

- [x] Run relevant frontend unit tests:

```bash
bunx vitest run components/networking/__tests__/ServerReadinessGate.test.tsx
bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx
```

- [x] Run the Playwright readiness recovery test.
- [x] Run the prior WebUI Knowledge QA deterministic route-state spec as a regression check.
- [x] Record Bandit as not applicable because no Python backend files were touched.
- [x] Update TASK-528.2 with verification results and known skips.
