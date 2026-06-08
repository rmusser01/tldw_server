# Knowledge QA UAT And Regression Checklist

Use this checklist when validating `/knowledge` for a release or focused Knowledge QA change. The page is a Knowledge QA workflow for searching a personal library and reviewing grounded answers with citations. Flashcards are handled by the separate flashcards route and are out of scope for `/knowledge`.

## Setup

Preconditions common to all scripts:

- WebUI can open `/knowledge`.
- Extension options page can open Knowledge QA.
- Test user has a configured tldw server URL and credentials unless the script explicitly tests setup failure.
- At least one fixture or real indexed document/note is available for successful-search scripts.
- At least one known query has a cited answer, and at least one known query has no matching local result.

Record for each run:

- Surface: WebUI or extension.
- Browser and viewport.
- Server URL/auth mode.
- Backend state: healthy, offline, empty, or mocked fixture.
- Pass/fail result.
- Screenshots, traces, or console/network notes for failures and skips.

## Script 1: Backend Unavailable Recovery

Preconditions:

- Backend health check is unavailable, stalled, or mocked as failed.
- No offline bypass should make the page look ready.

Steps:

1. Open `/knowledge` in WebUI.
2. Open Knowledge QA in the extension options route.
3. Wait for the readiness or setup gate to settle.
4. Inspect the visible recovery state.
5. Activate the visible retry or setup action when present.

Expected result:

- The page does not stay blank.
- The user sees that Knowledge QA cannot reach the backend.
- WebUI offers backend recovery or retry.
- Extension distinguishes setup problems from backend reachability when possible.

Pass/fail criteria:

- Pass if the blocked state is visible, actionable, and does not present the ready search state as tested.
- Fail if the page is blank, search is enabled without a usable backend, or the error gives no recovery path.

WebUI versus extension notes:

- WebUI failure is usually a server readiness or health issue.
- Extension failure can also involve missing server URL, API key, host permission, or allowlist.

Automated guardrails:

```bash
cd apps/tldw-frontend
npx playwright test e2e/ux-audit/knowledge-readiness-recovery.spec.ts --project=chromium

cd apps/extension
npx playwright test tests/e2e/knowledge-qa-setup-diagnostics.spec.ts --project=chromium-extension
```

## Script 2: First-Run Or No-Source Recovery

Preconditions:

- Backend is reachable.
- Knowledge QA reports no indexed sources, or indexed sources exist but no source categories are selected.

Steps:

1. Open Knowledge QA.
2. Confirm the first visible state explains that the library has no indexed searchable sources, or that no source categories are selected.
3. Try entering a question.
4. Confirm `Ask` remains disabled when no local source and no web fallback can be used.
5. Use the visible recovery action for adding/indexing sources or selecting categories.

Expected result:

- The page explains what is missing before the user asks a question.
- The primary recovery action leads to source setup or source selection.
- The search box does not imply a successful Knowledge QA search can run when no source can be searched.

Pass/fail criteria:

- Pass if recovery copy is specific and search is blocked until at least one valid source path exists.
- Fail if the user can submit a dead-end search, or if the page suggests data was searched when no data was available.

WebUI versus extension notes:

- Extension options may show compact source controls sooner because of width.
- The source-selection contract should remain the same across both surfaces.

Automated guardrails:

```bash
cd apps/tldw-frontend
npx playwright test e2e/ux-audit/knowledge-empty-recovery.spec.ts --project=chromium

cd apps/extension
npx playwright test tests/e2e/knowledge-empty-recovery.spec.ts --project=chromium-extension
```

## Script 3: First Successful Grounded Search

Preconditions:

- Backend is healthy.
- At least one indexed source is selected.
- A known question returns a generated answer and at least one source.

Steps:

1. Open Knowledge QA.
2. Confirm `Ask Your Library` and the search input are visible.
3. Ask the known question.
4. Wait for the answer.
5. Confirm the answer cites visible evidence.
6. Open the cited source or evidence panel if available.

Expected result:

- The answer appears with citations.
- The cited evidence is inspectable.
- Export is available after an answer is produced.

Pass/fail criteria:

- Pass if the user can connect the answer claim to a visible cited source.
- Fail if the answer is uncited, citations are not inspectable, or export appears before useful content exists.

WebUI versus extension notes:

- WebUI may show the evidence rail beside the answer in detailed layout.
- Extension may require compact controls or a narrower evidence view.

Automated guardrail:

```bash
cd apps/tldw-frontend
npx playwright test e2e/ux-audit/knowledge-qa-states.spec.ts --project=chromium
```

## Script 4: No-Results Recovery

Preconditions:

- Backend is healthy.
- Sources are selected.
- A known query returns no local results.

Steps:

1. Ask the known no-results query.
2. Read the recovery state.
3. Check for suggested actions such as changing wording, broadening scope, selecting sources, waiting for indexing, or using web fallback when available.
4. If nearest matches are available, open them and confirm they are labeled as weaker candidates.

Expected result:

- The UI clearly says no results were found.
- Recovery actions are relevant to source scope and indexing.
- The page does not fabricate an answer.

Pass/fail criteria:

- Pass if no-results recovery helps the user change scope or query without pretending success.
- Fail if an answer is generated without supporting evidence, or if no recovery action is visible.

Automated guardrails:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/NoResultsRecovery.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.golden-layout.test.tsx
```

## Script 5: Power-User Scoped Document/Note Search

Preconditions:

- Backend is healthy.
- At least two documents or notes are indexed so a scoped search can exclude one of them.

Steps:

1. Open source scope controls.
2. Select source categories.
3. Select specific documents or notes.
4. Save the scope as a profile.
5. Run a question that should only match the selected items.
6. Restore the saved profile and confirm the exact scope returns.

Expected result:

- Exact document/note counts are visible in compact mode when selected.
- Saved profiles restore source categories and exact selections.
- The search request is constrained to the chosen source scope.

Pass/fail criteria:

- Pass if the scoped answer and evidence only come from allowed sources.
- Fail if excluded sources appear, profile restore loses exact selections, or compact mode hides scope controls.

Automated guardrails:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.profiles.test.tsx src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx
```

## Script 6: Advanced Settings And Evidence Review

Preconditions:

- Backend is healthy.
- A successful cited search result is available.

Steps:

1. Open Knowledge QA settings.
2. Switch between Basic and Expert settings.
3. Change a safe setting and reset defaults.
4. Run a cited search.
5. Open evidence review.
6. Switch between `Sources` and `Details` views when available.
7. Confirm keyboard Escape/close controls return focus to the launching control.

Expected result:

- Settings are reversible and keyboard accessible.
- Expert settings are discoverable without overwhelming Basic mode.
- Evidence views support source inspection and retrieval detail review.

Pass/fail criteria:

- Pass if settings open/close safely, reset works, and evidence remains tied to citations.
- Fail if focus is trapped incorrectly, settings cannot be reset, or evidence views are missing for cited answers.

Automated guardrails:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/SettingsPanel.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/ExpertSettings.accessibility.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx
```

## Script 7: Export

Preconditions:

- A Knowledge QA answer with citations is visible.
- The current thread supports the export action being tested.

Steps:

1. Choose `Export`.
2. Select Markdown and export.
3. Select PDF and confirm the browser print/export flow starts when supported.
4. Select Chatbook and export when supported.
5. Save to Notes when available.
6. Create and revoke a share link when available.
7. Confirm export errors show actionable recovery.

Expected result:

- Exported content includes the Knowledge QA question, answer, citations, and source/retrieval context when available.
- Chatbook download uses the returned export job.
- Save-to-Notes and share-link actions report success or a clear error.

Pass/fail criteria:

- Pass if export succeeds or fails with specific, recoverable feedback.
- Fail if export silently fails, omits citations, leaks sensitive provider details, or leaves stale loading state.

Automated guardrails:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/ExportDialog.a11y.test.tsx src/components/Option/KnowledgeQA/__tests__/errorMessages.test.ts
```

## Consolidated Regression Commands

Shared Knowledge QA UI:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA
```

WebUI route states:

```bash
cd apps/tldw-frontend
npx playwright test e2e/ux-audit/knowledge-readiness-recovery.spec.ts e2e/ux-audit/knowledge-qa-states.spec.ts e2e/ux-audit/knowledge-empty-recovery.spec.ts --project=chromium
```

Extension route states:

```bash
cd apps/extension
npx playwright test tests/e2e/knowledge-qa-setup-diagnostics.spec.ts tests/e2e/knowledge-qa-states.spec.ts tests/e2e/knowledge-empty-recovery.spec.ts --project=chromium-extension
```

Backend checks:

```bash
source .venv/bin/activate
python -m pytest <touched backend test paths> -v
python -m bandit -r <touched backend paths> -f json -o /tmp/bandit_knowledge_qa.json
```

Only run backend checks when backend Knowledge QA/RAG files are touched. For documentation-only changes, record Bandit as not applicable.

Scope guard:

```bash
rg -n "deck|spaced repetition|study-set|study set" \
  apps/packages/ui/src/components/Option/KnowledgeQA \
  apps/tldw-frontend/e2e/ux-audit/knowledge*.spec.ts \
  apps/extension/tests/e2e/knowledge*.spec.ts
```

This guard intentionally excludes the word `flashcards` because documentation may mention the separate flashcards route when explaining what `/knowledge` does not do.

## Current Known Blockers

- Extension E2E can be blocked by the WXT production build stalling before browser tests start. When that happens, record the build stall as an environment/build blocker and do not claim extension runtime behavior was verified.
- Package-wide TypeScript checks may be blocked by existing baseline errors outside the Knowledge QA slice. Prefer targeted Knowledge QA Vitest and Playwright checks for this release gate unless the baseline is fixed.

## Release Sign-Off

Before closing a Knowledge QA release gate:

- [ ] UAT scripts 1-7 are passed or explicitly skipped with a reason.
- [ ] WebUI and extension differences are documented.
- [ ] Automated regression commands and results are recorded.
- [ ] No `/knowledge` UI or test change introduced flashcard workflows.
- [ ] Known blockers are recorded in the relevant Backlog task.
