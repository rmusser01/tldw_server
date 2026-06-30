# Knowledge First-Run Empty Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make beginner recovery paths on `/knowledge` clear for first-run, no indexed sources, no selected sources, offline backend, and failed search states.

**Architecture:** Split state detection from copy and actions. The Knowledge QA empty-state layer should classify why the user cannot search, then render one primary recovery path plus relevant secondary actions. Source creation remains in existing source owner surfaces such as Quick Ingest, Media, Notes, or setup routes.

**Tech Stack:** React, shared Knowledge QA UI, Vitest, Playwright.

**Backlog Task:** TASK-528.4

---

## Boundaries

- `/knowledge` may point users to add or index sources, but it must not become a full source-management CRUD hub.
- Do not add flashcard behavior to `/knowledge`.
- Do not enable web fallback automatically.

## Files

- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/empty/KnowledgeReadyState.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/panels/NoResultsRecovery.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SearchBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/KnowledgeContextBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SearchBar.behavior.test.tsx`
- Create or modify: `apps/tldw-frontend/e2e/ux-audit/knowledge-empty-recovery.spec.ts`
- Create or modify: `apps/extension/tests/e2e/knowledge-empty-recovery.spec.ts`

## Task 1: Write Failing Empty-State Tests

- [x] Add tests for first-run with no indexed documents or notes.
- [x] Add tests for indexed sources present but none selected.
- [x] Add tests for web fallback available versus unavailable.
- [x] Add tests proving disabled Ask has visible inline explanation without hover.
- [x] Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SearchBar.behavior.test.tsx
```

Expected: tests fail before implementation.

Observed: focused Vitest run failed for the expected missing behavior: no recovery classifier was passed to the ready state, no-source blocking was tooltip-only, and unavailable web fallback was ignored.

## Task 2: Classify Beginner Recovery States

- [x] Add a small state classifier for:
  - backend unavailable
  - setup or auth missing
  - no indexed sources
  - sources indexed but none selected
  - no results after search
  - failed search
- [x] Keep classifier inputs based on existing source counts, selected sources, capabilities, and connection state.
- [x] Avoid duplicating backend health logic already owned by connection hooks.

Implemented: `empty/recoveryState.ts` classifies ready, backend unavailable, no indexed sources, no selected sources, and web-only states from `knowledgeStatus`, selected source count, web fallback setting, and server capability. Backend setup/auth/offline gates remain owned by the existing connection diagnostics chain. No-results and failed-search recovery remain in the results/error surfaces rather than the ready-state classifier.

## Task 3: Update First-Run And No-Source Copy

- [x] Update "Ask Your Library" copy to explain that the page searches selected personal-library sources and returns grounded answers with citations.
- [x] For no indexed sources, primary CTA should route to add or index sources.
- [x] For no selected sources, primary CTA should open source selection.
- [x] If web fallback is enabled, state that the search will use web results only.
- [x] If web fallback is available but off, offer it as a secondary action.

Implemented: ready-state copy now explains selected personal-library search and inspectable citations. No indexed sources routes to `/media` and `/notes`; no selected sources opens source selection. Search is visibly disabled for no selected sources without web fallback and for no indexed library sources when web fallback is unavailable/off.

## Task 4: Harden No-Results Recovery

- [x] Make no-results recovery actions conditional on available capabilities and real candidate data.
- [x] Offer broaden source scope, adjust query, tune settings, or enable web fallback.
- [x] Do not show nearest matches unless the backend returned candidates or the UI can prove they exist.

Implemented: no-results recovery only shows web search actions when web search is available, and only shows nearest matches when `searchDetails.alsoConsidered` is populated.

## Task 5: Verify Beginner Flow

- [x] Run unit tests.
- [x] Add WebUI and extension route tests:

```bash
npx playwright test apps/tldw-frontend/e2e/ux-audit/knowledge-empty-recovery.spec.ts
npx playwright test --config apps/extension/playwright.config.ts apps/extension/tests/e2e/knowledge-empty-recovery.spec.ts
```

- [x] Attempt WebUI and extension route tests.
- [x] Record Bandit as not applicable unless Python backend files were touched.
- [x] Update TASK-528.4 with verification and known skips.

Verification:

- Passing: `bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/SearchBar.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx` from `apps/packages/ui` passed 48 tests.
- Passing: `bun run compile` from `apps/extension` passed TypeScript compile.
- Added route specs:
  - `apps/tldw-frontend/e2e/ux-audit/knowledge-empty-recovery.spec.ts`
  - `apps/extension/tests/e2e/knowledge-empty-recovery.spec.ts`
- WebUI route runtime blocked: after manually starting `bun run dev -- -p 8080`, `TLDW_WEB_AUTOSTART=false npx playwright test e2e/ux-audit/knowledge-empty-recovery.spec.ts --reporter=line` failed before test execution because Chromium could not launch in this sandbox: `bootstrap_check_in org.chromium.Chromium.MachPortRendezvousServer... Permission denied (1100)`.
- Extension route runtime blocked: `npx playwright test --config apps/extension/playwright.config.ts apps/extension/tests/e2e/knowledge-empty-recovery.spec.ts --reporter=line` hung in WXT production build after duplicated-import warnings; terminated the verification process chain. This matches existing TASK-306 WXT build-hang blocker.
- Scope check: `rg "flashcard|deck|spaced repetition|study set" apps/packages/ui/src/components/Option/KnowledgeQA apps/tldw-frontend/e2e/ux-audit/knowledge-empty-recovery.spec.ts apps/extension/tests/e2e/knowledge-empty-recovery.spec.ts -n` returned no matches.
- Bandit: not applicable; no Python backend files were touched.
