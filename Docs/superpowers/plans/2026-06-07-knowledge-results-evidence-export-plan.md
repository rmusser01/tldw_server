# Knowledge Results Evidence Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and harden `/knowledge` results, citations, evidence inspection, no-results recovery, follow-up search, and export.

**Architecture:** Treat answer, citations, evidence, details, and export as one trust workflow. The answer panel should never claim grounded support without visible evidence, and export should preserve enough query, source, and settings context for later review.

**Tech Stack:** React, shared Knowledge QA UI, Markdown export utilities, Vitest, Playwright.

**Backlog Task:** TASK-528.6

---

## Boundaries

- This phase validates and hardens existing Knowledge QA result workflows.
- Do not introduce unrelated Research Workspace, Chat, Notes, or Media redesigns.
- Do not add flashcard behavior to `/knowledge`.

## Files

- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/panels/AnswerWorkspace.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/evidence/EvidenceRail.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SearchDetailsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SourceList.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SourceCard.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/panels/NoResultsRecovery.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/ExportDialog.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SearchDetailsPanel.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SourceList.behavior.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/ExportDialog.a11y.test.tsx`

## Task 1: Write Failing Cited Result Tests

- [x] Add a fixture result with answer text, inline citations, source cards, and search details.
- [x] Assert each visible citation maps to a source in the evidence rail.
- [x] Assert Sources and Details tabs expose relevant metadata when available.
- [x] Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SearchDetailsPanel.test.tsx
```

Result: focused tests first failed on missing export citation/settings details and missing telemetry explanation; after implementation the focused suite passed.

## Task 2: Harden Answer And Evidence Coupling

- [x] Make citation count and source support badges reflect actual available evidence.
- [x] Ensure citation buttons jump or focus the matching source.
- [x] Ensure low-confidence or no-citation answers show recovery copy instead of overstating support.
- [x] Keep streaming and final answer states distinct.

Existing `AnswerPanel.states.test.tsx` coverage was retained and rerun; no production AnswerPanel changes were needed in this slice.

## Task 3: Validate Sources And Details Views

- [x] Sources view should support cited-first sorting, filters, source feedback, copy, open original, and document workspace handoff where applicable.
- [x] Details view should show query expansion, reranking, average relevance, web fallback, verification, candidates considered, and latency only when data exists.
- [x] Missing detail data should be explained without looking broken.

## Task 4: Harden No-Results And Failed Search

- [x] Add tests for empty result set, timeout, unreachable backend, and generic failed search.
- [x] Offer broaden scope, adjust query, tune settings, or enable web fallback based on available capabilities.
- [x] Do not show "nearest matches" unless real candidates are present.

## Task 5: Validate Export

- [x] Add tests for Markdown export content structure.
- [x] Verify PDF and Chatbook actions expose success and failure states or clearly document unsupported test coverage.
- [x] Ensure export includes query, answer, sources, citations, conversation history when available, and optional settings snapshot.
- [x] Verify Save to Notes and share-link actions surface errors.

## Task 6: Close Verification

- [x] Run targeted Vitest files for answer, evidence, source list, no-results, and export.
- [x] Run WebUI route fixture e2e for ready, cited result, empty, and no-source recovery states.
- [x] Record route fixture coverage limits: export and failed-search behavior are covered by focused Vitest; extension E2E was attempted but blocked by a WXT production build hang before tests executed.
- [x] Record Bandit as not applicable unless Python backend files were touched.
- [x] Update TASK-528.6 with verification results.

## Verification Results

- `bunx vitest run src/components/Option/KnowledgeQA/__tests__/ExportDialog.a11y.test.tsx src/components/Option/KnowledgeQA/__tests__/SearchDetailsPanel.test.tsx` - passed, 30 tests.
- `bunx vitest run src/components/Option/KnowledgeQA/__tests__/NoResultsRecovery.test.tsx` - passed, 2 tests.
- `bunx vitest run src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx src/components/Option/KnowledgeQA/__tests__/SearchDetailsPanel.test.tsx src/components/Option/KnowledgeQA/__tests__/SourceList.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/NoResultsRecovery.test.tsx src/components/Option/KnowledgeQA/__tests__/ExportDialog.a11y.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/errorMessages.test.ts` - passed, 97 tests.
- `npx playwright test e2e/ux-audit/knowledge-qa-states.spec.ts e2e/ux-audit/knowledge-empty-recovery.spec.ts --project=chromium --reporter=line` - passed, 4 tests after strict locator hardening in the WebUI empty-recovery spec.
- `CI=1 npx playwright test tests/e2e/knowledge-qa-states.spec.ts tests/e2e/knowledge-empty-recovery.spec.ts --project=chromium-extension --reporter=line` - attempted; blocked because the extension WXT production build hung after the initial build warnings and had to be terminated. No extension tests executed.
- `bunx tsc --noEmit --pretty false` in `apps/packages/ui` - failed on existing repo-wide baseline TypeScript errors outside this slice; touched Knowledge QA paths compiled through Vitest.
- `git diff --check -- <touched files>` - passed.
- `rg -n "flashcard|deck|spaced repetition|study set" <touched Knowledge QA files>` - no matches.
- Bandit - not applicable; no Python backend files touched.
