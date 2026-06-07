# Knowledge UAT Regression Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `/knowledge` audit UAT scripts into repeatable release checks, docs, and regression guardrails for WebUI and extension.

**Architecture:** Consolidate the UAT scripts into page-specific QA documentation and automated smoke coverage. Keep human UAT steps concrete enough for release review while preserving automated route-state tests for CI.

**Tech Stack:** Markdown documentation, Playwright, Vitest, optional pytest and Bandit if backend code is touched.

**Backlog Task:** TASK-528.8

---

## Boundaries

- This phase documents and verifies the completed remediation series.
- Do not use this phase to add new product scope.
- Do not add flashcard behavior to `/knowledge`.

## Files

- Create or modify: `Docs/User_Guides/WebUI_Extension/Knowledge_QA_Guide.md`
- Create or modify: `Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md`
- Modify: `apps/tldw-frontend/e2e/ux-audit/knowledge-qa-states.spec.ts`
- Modify: `apps/extension/tests/e2e/knowledge-qa-states.spec.ts`
- Modify: relevant Knowledge QA Vitest files touched by TASK-528.1 through TASK-528.7
- Modify: Backlog tasks TASK-528 and TASK-528.1 through TASK-528.8

## Task 1: Write UAT Checklist Document

- [x] Create a UAT checklist covering:
  - backend unavailable recovery
  - first-run no-source setup
  - first successful grounded search
  - no-results recovery
  - power-user scoped document/note search
  - advanced settings and evidence review
  - export
- [x] Include WebUI and extension notes for each script.
- [x] Include pass/fail criteria and known setup requirements.

Result: Added `Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md`.

## Task 2: Add Page-Specific User Help

- [x] Add or update a Knowledge QA user guide.
- [x] Explain what `/knowledge` searches, how source scope works, how citations map to evidence, what web fallback does, and how exports work.
- [x] Explain relationship to Research Workspace, Chat, Notes, and Media.
- [x] State that flashcards are handled by the separate flashcards route and are not part of `/knowledge`.

Result: Added `Docs/User_Guides/WebUI_Extension/Knowledge_QA_Guide.md`.

## Task 3: Consolidate Regression Commands

- [x] Record the final command set for unit tests, WebUI e2e, extension e2e, and backend checks if backend code was touched.
- [x] Recommended frontend commands:

```bash
bunx vitest run apps/packages/ui/src/components/Option/KnowledgeQA
npx playwright test apps/tldw-frontend/e2e/ux-audit/knowledge-qa-states.spec.ts
npx playwright test --config apps/extension/playwright.config.ts apps/extension/tests/e2e/knowledge-qa-states.spec.ts
```

- [x] If Python backend files were touched in earlier phases, run:

```bash
source .venv/bin/activate
python -m pytest <touched backend test paths> -v
python -m bandit -r <touched backend paths> -f json -o /tmp/bandit_knowledge_qa.json
```

Result: Regression commands are recorded in the UAT checklist and user guide. Bandit is recorded as not applicable for documentation-only TASK-528.8 work.

## Task 4: Run Final UAT Pass

- [x] Run automated WebUI route-state regression checks for UAT-critical states.
- [x] Record extension options route UAT as blocked when the WXT build stalls before browser launch.
- [x] Record screenshots or traces for any failed or skipped state.
- [x] Update each child Backlog task with final verification status or blockers.

Result: Automated WebUI route-state checks passed. Extension runtime UAT remains blocked by the previously recorded WXT production build stall before browser launch, so no extension screenshots or traces were produced in this closeout pass.

## Task 5: Close Parent Program Task

- [x] Update TASK-528 with links to all plans, final UAT checklist, user guide, and verification results.
- [x] Confirm no `/knowledge` plan, test, or UI change introduced flashcard workflows.
- [x] Mark completed child tasks Done only when their implementation acceptance criteria and verification are complete.

Result: TASK-528 records the linked remediation plans, UAT checklist, user guide, verification commands, and known extension E2E blocker.
