# Knowledge QA State Fixtures And Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every audited `/knowledge` state reproducible in WebUI and extension tests before UI remediation begins.

**Architecture:** Add deterministic frontend fixtures and route-level tests around the shared Knowledge QA component, the WebUI readiness gate, and the extension options route. Prefer mocked API responses and explicit state builders over live backend dependencies so the QA states can be tested in CI and during local review.

**Tech Stack:** React, Vitest, Testing Library, Playwright, Next.js WebUI, WXT/browser extension options route, shared `@tldw/ui` package.

**Backlog Task:** TASK-528.1

---

## Boundaries

- `/knowledge` remains a Knowledge QA workflow for searching a personal library and reviewing grounded answers with citations.
- Do not add flashcard, deck, spaced repetition, study-set, or card review behavior to `/knowledge`.
- This phase should capture current failures and build test access. It should not redesign the UI except for test-only seams.

## Files

- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx`
- Create or modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/knowledgeQaStateFixtures.ts`
- Create or modify: `apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx`
- Create or modify: `apps/tldw-frontend/e2e/ux-audit/knowledge-qa-states.spec.ts`
- Create or modify: `apps/extension/tests/e2e/knowledge-qa-states.spec.ts`
- Modify: `apps/extension/tests/e2e/setup/build-extension.ts`

## Task 1: Inventory Existing Test Helpers

- [x] Inspect existing Knowledge QA tests and identify reusable render helpers, mocked API clients, and connection-state builders.
- [x] Inspect existing WebUI e2e fixtures under `apps/tldw-frontend/e2e` and extension fixtures under `apps/extension/tests/e2e`.
- [x] Record missing helper seams through this plan and TASK-528.1 closeout.

## Task 2: Add State Fixture Builder

- [x] Write a failing unit test that renders each audited state from a named fixture:
  - backend offline
  - setup required
  - no indexed sources
  - no selected sources
  - ready search
  - results with citations
  - no results
  - settings drawer
  - export dialog
- [x] Add `knowledgeQaStateFixtures.ts` with small, typed builders for server state, capabilities, source lists, RAG result payloads, and provider lists.
- [x] Keep fixture names Knowledge QA-specific. Do not use flashcard terminology.
- [x] Run:

```bash
bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx
```

Result: passed, 18 tests.

## Task 3: Capture WebUI Readiness Baseline

- [x] Write a WebUI readiness baseline test that simulates failed `/api/v1/health` readiness.
- [x] Assert the current failure state is observable: route children render without recovery UI after timeout.
- [x] Leave the recovery panel implementation to TASK-528.2 rather than changing behavior in this baseline task.
- [x] Run:

```bash
bunx vitest run components/networking/__tests__/ServerReadinessGate.test.tsx
```

Result: passed, 5 tests.

## Task 4: Add WebUI And Extension Route Fixtures

- [x] Add Playwright route mocks for health, capabilities, source listing, providers, thread setup, and RAG search.
- [x] Add WebUI e2e cases for ready search and cited results without a live backend.
- [x] Add extension e2e cases for setup required and connected ready search without live backend data.
- [x] Add `TLDW_E2E_SKIP_EXTENSION_BUILD` to extension global setup so existing valid builds can be used when WXT rebuilds are not part of the test.
- [x] Run:

```bash
npx playwright test e2e/ux-audit/knowledge-qa-states.spec.ts --reporter=line
TLDW_E2E_SKIP_EXTENSION_BUILD=1 TLDW_E2E_EXTENSION_HEADLESS=1 npx playwright test tests/e2e/knowledge-qa-states.spec.ts --reporter=line
```

Result: WebUI passed, 2 tests. Extension passed, 2 tests, when launched outside the sandbox because Chromium extension startup is blocked by sandboxed macOS Crashpad permissions.

## Task 5: Close Verification

- [x] Update TASK-528.1 with created fixture files, route coverage, and known extension launch requirement.
- [x] No Python code was touched; Bandit is not applicable for this test-fixture-only phase.
- [x] Confirm all fixture copy and test names preserve the Knowledge QA-only scope.
