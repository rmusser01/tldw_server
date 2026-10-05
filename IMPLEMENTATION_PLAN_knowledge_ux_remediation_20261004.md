# Knowledge UX remediation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Repair all fifteen audited issues and deliver the five approved enhancement themes with correct source continuity and reviewable outputs.

**Architecture:** Keep shared React UI and current API/workspace ownership. A validated source-scope route handoff carries media selections; existing ResearchWorkspace prefill carries canonical evidence and qualifications. Non-media evidence uses labeled existing-ingest excerpt snapshots so canonical media-backed generation remains valid.

**Tech stack:** Existing React/TypeScript, Zustand, AntD, Vitest/Testing Library, Next.js, WXT and FastAPI ingestion; no new dependencies.

**Spec:** `Docs/Design/2026-10-04-knowledge-ux-remediation.md`. Source findings: `Docs/Reviews/KNOWLEDGE_NNG_UX_REVIEW_2026_10_04.md`.

## Global Constraints

- No new package dependencies, modal framework, workspace abstraction, or public API family.
- Work only in the attached isolated checkout; do not edit the original dirty checkout.
- Preserve current auth/account/request-scope fencing. Transfers must not publish results after owner/server changes or expose credential material in URLs/storage/logs.
- Reuse existing ingestion, source status, Notes, workspace persistence and queue operations. Do not fabricate media IDs for notes/web results.
- Regression tests precede behavior changes. Test actual state/results rather than source-text mirrors or mock call counts.
- Do not push, merge or publish as part of this request.
- Each unit is associated with its named Backlog task, tested and committed with that task record. Maximum three failed attempts per issue; document and reassess before another approach.

## Stage 1: Source continuity and scalable selection

Goal: K01–K04 and shared named source-set handoff. Success criteria: presets preserve intent; successful added/reviewed items arrive exactly; older items are searchable; no invalid handoff silently broadens scope. Tests: Provider preset/route state, paginated picker, ingest/review integration, ID invariants. Status: Not Started.

### Task 1: Source continuity and scalable selection (TASK-13453.1)

**File scope (candidate modification sites; avoid editing a caller already covered by a shared fix and report why):**
- Create: `apps/packages/ui/src/utils/knowledge-scope-handoff.ts`, `apps/packages/ui/src/utils/__tests__/knowledge-scope-handoff.test.ts`.
- Modify: `components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`, `components/Option/KnowledgeQA/context/KnowledgeContextBar.tsx`, `services/tldw/domains/collections.ts`, `components/Common/QuickIngestWizardModal.tsx`, `components/Common/QuickIngest/WizardResultsStep.tsx`, `components/Review/MediaReviewBatchBar.tsx`, `components/Review/ReviewPage.tsx`, `components/Review/ViewMediaPage.tsx` (all under `apps/packages/ui/src/`).
- Test: existing KnowledgeQA provider/context tests, QuickIngest integration tests, Review batch-toolbar tests.

**Interfaces:**
- Consumes: successful `WizardResultItem.mediaId`, Review selected media IDs, existing RagSettings and tldwClient list/search methods.
- Produces: `buildKnowledgeMediaScopePath(mediaIds: number[]): string`; `parseKnowledgeMediaScope(search: string): { mediaIds: number[]; invalid: boolean } | null`. Query name `media_ids`; absent parameter returns null, present invalid/empty selection is explicit invalid state. Numeric IDs are unique positive safe integers. These helper signatures are binding for later units.
- Preserve `sources`, `include_media_ids`, `include_note_ids`, `collection_id`, `keyword_filter`, `generation_provider`, `generation_model`, and `enable_web_fallback` through preset changes.
- Research handoff reuses existing `buildKnowledgeQaWorkspacePrefill`/`queueResearchWorkspacePrefill`; do not introduce a second research model.

- [ ] Write failing regressions. Include this actual invariant and component scenarios:
```ts
it('round-trips the exact added media set', () => {
  const path = buildKnowledgeMediaScopePath([3, 7, 3]);
  expect(parseKnowledgeMediaScope(new URL(path, 'https://local.test').search))
    .toEqual({ mediaIds: [3, 7], invalid: false });
});
it('does not treat invalid transfer scope as the whole library', () => {
  expect(parseKnowledgeMediaScope('?media_ids=invalid')).toEqual({ mediaIds: [], invalid: true });
});
```
Set Provider exact note/media and chosen model, change every preset, and assert retained settings. Arrival on the same mounted route must cancel stale work, clear old output, and replace scope deliberately. Search a title only on page 5 and select it while retaining a page-1 selection. Observe added-ID and reviewed-ID continuation URLs and destination scope, not only navigation call counts.
- [ ] Run selected existing/new Vitest tests from `apps/packages/ui` using `bun run test -- <files>` or installed `vitest run`; record expected failures before production edits.
- [ ] Implement minimal state/route behavior. At `SET_PRESET`, overlay preserved intent fields after preset defaults. Serialize validated IDs with URLSearchParams; consume scope only after account/default hydration is stable. Apply `sources: ['media_db']`, exact successful IDs, empty note IDs and cleared old corpus-specific filters on explicit ingest/review handoff; reject malformed scope visibly. Await selection persistence where navigation depends on it. New ingest default is **Ask added items**; separately retain full-library navigation if offered.
- [ ] Replace first-page-only filtering with existing paginated list/search endpoints (page size 50). Extend `searchNotes` to accept existing backend limit/offset and abort options as needed. Keep per-page/search results separate from selected IDs; handle stale requests and page errors. Count known totals only, label loaded counts/Select visible correctly; show recognizable selected titles/IDs.
- [ ] Run targeted regressions and existing suites covering changed behavior, format changed files, check types covering the unit, self-review every caller, update TASK-13453.1 with evidence, and commit only this unit plus its task record. Report files/commits/tests and any integration interface changes.

## Stage 2: Research evidence and output review

Goal: K05–K06/K15 and outcome recipes/review loop. Success criteria: attached inspectable evidence survives continuation/save/reopen with qualifications; exact saved note opens; analysis is outcome-first. Tests: prefill normalization, partial import, Notes origin, analysis and recipes. Status: Not Started.

### Task 2: Research evidence and output review (TASK-13453.2)

**File scope (candidate modification sites; avoid editing a caller already covered by a shared fix and report why):**
- Modify: `utils/research-workspace-prefill.ts`, `components/Option/KnowledgeQA/AnswerPanel.tsx`, `components/Option/KnowledgeQA/ExportDialog.tsx`, `components/Option/KnowledgeQA/empty/KnowledgeReadyState.tsx` only for recipe controls, `components/Option/ResearchWorkspace/index.tsx`, `types/workspace.ts` only for backward-compatible provenance fields, `components/Notes/hooks/useNotesEditorState.tsx`, `components/Media/AnalysisModal.tsx`, `components/Review/MediaReviewReadingPane.tsx` (under `apps/packages/ui/src/`). Add a focused import helper near `research-workspace-prefill.ts` only if needed to keep import/error logic testable.
- Test: `utils/__tests__/research-workspace-prefill.test.ts`, `KnowledgeQA/__tests__/AnswerPanel.workspace-handoff.test.tsx`, export tests, ResearchWorkspace prefill/store tests, Notes provenance tests, AnalysisModal tests.

**Interfaces:**
- Consumes: existing ResearchWorkspacePrefill and RagResult metadata/content, original source IDs, selected scope, answerTrustState/reason/evidenceOrigin, current owner operation, media ingestion and workspace addSources.
- Produces: backward-compatible prefill retaining typed original source ID, excerpt, link, citations, scope, trust and owner binding. Media sources retain valid original media IDs; note/web sources attach labeled retrieved-excerpt snapshots using `tldwClient.uploadMedia` with request-scope/operation guards. No sentinel/fake media IDs. Successful snapshot IDs are retained for retry and source metadata stores original identity/link. Partial failures retain pending source descriptors and expose Retry unfinished imports.
- Preserve existing numeric media-backed workspace source/generation contract and split-key persistence; do not widen mediaId across the entire workspace model or add a new API.

- [ ] Write failing behavior regressions. Extend the real prefill helper tests with a note UUID and content:
```ts
const payload = buildKnowledgeQaWorkspacePrefill({
  threadId: 'thread-1', query: 'What supports this claim?', answer: 'Draft', citations: [],
  results: [{ id: 'note-uuid', content: 'Supporting excerpt', metadata: { source_type: 'notes', title: 'Field note', note_id: 'note-uuid' } }],
  answerTrustState: 'uncited_degraded_answer', answerEvidenceOrigin: 'local_library',
  scope: { sources: ['notes'], include_media_ids: [], include_note_ids: ['note-uuid'] }
});
expect(buildKnowledgeQaSeedNote(payload)).toContain('Supporting excerpt');
expect(buildKnowledgeQaSeedNote(payload)).toContain('note-uuid');
```
Tests must cover two source chunks from the same original, a web URL, a failed import then retry, owner/workspace change during an async import, saving/reopening the exact exported note, and editing its title without losing original provenance.
- [ ] Run regressions and record expected failures before implementation.
- [ ] Extend the existing builder/receiver. Label Continue in Research Workspace. Ensure handoff storage errors are surfaced rather than silently navigating empty. Retain/retry unconsumed imports transactionally. Attach native media directly; upload a text File containing canonical original reference and retrieved excerpts for non-media evidence through existing ingestion, without analysis and with search preparation. Label snapshot limitations and display attached/pending/failed counts. Guard async commits by owner/workspace; do not remove payload until useful content is retained.
- [ ] Retain returned note ID and add Open saved note. Derive Knowledge QA origin from persisted metadata independently of editProvenance, preserving trust text and session reference. Add Summary/Key claims/Custom analysis choices, keep prompts in Advanced, rename Save prompt defaults, and reveal the generated saved analysis version. Empty multi-review Analysis links to generation in Media.
- [ ] Add four visible recipes using current question input: Compare these papers; Extract claims with evidence; Summarize this interview; Save a sourced brief. Each populates an editable evidence-aware question, preserves scope and waits for the user's Ask. Do not add a new output destination or auto-run a query.
- [ ] Run prefill/export/Notes/analysis/research suites, format and check the touched type scope, inspect persistence and error paths, update TASK-13453.2 and commit the unit with its task record. Report native snapshots versus live references accurately.

## Stage 3: Accessible interaction and compact layout

Goal: K07–K08/K10/K13–K14 and minor accessibility observations. Success criteria: dialog focus is truthful, source controls are unclipped, initial action/scope remain visible, bottom actions do not collide, shortcut help is accurate. Tests: keyboard/pointer behavior, suggestions, layout/browser geometry. Status: Not Started.

### Task 3: Accessible interaction and compact layout (TASK-13453.3)

**File scope (candidate modification sites; avoid editing a caller already covered by a shared fix and report why):**
- Modify: `layout/KnowledgeQALayout.tsx`, `SourceViewerModal.tsx`, `context/KnowledgeContextBar.tsx`, `context/CompactToolbar.tsx`, `evidence/EvidenceRail.tsx`, `FollowUpInput.tsx`, `SearchBar.tsx`, `empty/KnowledgeReadyState.tsx` (all under `apps/packages/ui/src/components/Option/KnowledgeQA/`).
- Test: layout/source viewer/context/search/follow-up/evidence/ready-state accessibility and behavior suites.

**Interfaces:**
- Consumes: paginated picker and selected-title summary from Task 1, recipes and source/trust actions from Task 2.
- Produces: established accessible dialog behavior on scope/preview/evidence, one bounded exact-list region, compact ready-state with primary actions ahead of optional guidance and visible mobile exact scope. Keep Task 1 loader/selection API and Task 2 recipe behavior unchanged.

- [ ] Write regressions for actual focus entry/Tab/Shift+Tab/close return. A minimal behavior shape:
```ts
await user.click(screen.getByRole('button', { name: /sources/i }));
const dialog = screen.getByRole('dialog');
await user.tab({ shift: true });
expect(dialog.contains(document.activeElement)).toBe(true);
await user.keyboard('{Escape}');
expect(screen.getByRole('button', { name: /sources/i })).toHaveFocus();
```
Use existing test fixtures and actual accessible dialog components. Test filter accessible name, active suggestion announcements, correct Cmd/Ctrl+K help, and distinct Evidence/New Topic/Send actions; do not assert CSS class strings as a substitute for browser layout.
- [ ] Run regressions and record expected failures before changing behavior.
- [ ] Reuse established AntD/shared modal primitives for scope and source preview; make the Evidence drawer a named dialog with focus entry/return. Put exact selection directly in the scope panel or use an installed portal/collision-aware popover, preserving selection and using one bounded list scroll. Name the filter Find documents or notes; avoid duplicate labels/controls.
- [ ] Compact the intro, place Add/Ask before optional guide/examples, retain recipes from Task 2, keep exact-source summary visible at 390px, and preserve comfortable focus/touch controls. Share/reserve bottom layout space for Evidence rather than independently overlaying New Topic. Titles remain recognizable; account for safe area and keyboard.
- [ ] Remove Knowledge's conflicting Cmd/Ctrl+K handler; keep global command palette ownership and correct help. Preserve search-focus/submit/cancel shortcuts that actually work. Add suggestion roles/active descendant IDs and keyboard announcements without inventing another navigation layer.
- [ ] Run relevant suites, format/check touched types, validate geometry at 1280×720 and 390×844 when runtime is available, update TASK-13453.3 and commit the unit/task record. Report any final runtime-dependent checks for the controller's comprehensive pass.

## Stage 4: Readiness, recovery and cross-surface validation

Goal: K09/K11–K12, readiness summaries and extension vocabulary. Success criteria: actual empty readiness, eligible subset retries, accurate recovery actions and owner links, surface parity. Tests: mixed queues/retries, empty states, no-result actions, native/WebUI handoffs. Status: Not Started.

### Task 4: Readiness, recovery and cross-surface continuity (TASK-13453.4)

**File scope (candidate modification sites; avoid editing a caller already covered by a shared fix and report why):**
- Modify: `sourceHealth.ts`, `empty/recoveryState.ts`, `empty/KnowledgeReadyState.tsx`, `layout/KnowledgeQALayout.tsx`, `panels/NoResultsRecovery.tsx` (under `apps/packages/ui/src/components/Option/KnowledgeQA/`), `components/Common/QuickIngestWizardModal.tsx`, `components/Common/QuickIngest/WizardResultsStep.tsx`, existing queue/session helpers only as needed (under `apps/packages/ui/src/`). Shared extension ingest/continuation labels should use these same components.
- Test: source-health/activation/recovery suites, existing QuickIngest wizard/session/results/retry integration suites; add focused critical workflow tests under existing frontend/extension e2e folders only where current fixtures support them.

**Interfaces:**
- Consumes: exact handoffs from Task 1, output continuity from Task 2, compact/accessibility layout from Task 3, existing QuickIngest queue/session operations.
- Produces: separately named available services and searchable personal-item counts, first-add action for empty content, known readiness per result, eligible failed-item/all-item retry, accurately named scope/depth actions and explicit owner destinations.

- [ ] Write failing regressions for connected server with zero media/notes, completed new ingest refresh, mixed successful/retryable/permanent/cancelled results, missing queued file reattachment and retry settings retention. A behavior test must inspect actual retained results/new attempted subset:
```ts
expect(successfulMediaIdsAfterRetry).toEqual([1, 3]);
expect(retriedItemIds).toEqual(['failed-item']);
expect(reopenedScope.include_media_ids).toEqual([1, 3]);
```
Derive these from real queue/wizard state in the integration fixture, not fabricated local arrays. Test Change included sources opens selection and Search more results changes retrieval depth without altering scope.
- [ ] Run regressions and record expected failures before production edits.
- [ ] Readiness uses actual media/notes content counts/readiness with loading/unknown/error states, independent of category availability. Do not infer vector readiness from storage or guess counts. Show Add your first source for empty personal content and refresh after ingest through existing query invalidation/status pathways. Use known state labels Stored/Searchable/Analysis available/Needs attention and actionable owner links.
- [ ] Wire Results callbacks to existing eligible requeue/retry functions; preserve original settings/File requirements and successful records. Retry item/all eligible enters the established processing lifecycle. Permanent/cancelled/missing-file results get distinct actions. Keep Task 1 added-item handoff based on accumulated successful IDs.
- [ ] Separate source-scope and retrieval-depth recovery actions. Excluded items open Sources; media/note indexing issues navigate to their existing owner; relevant connection errors retain diagnostics. Align extension/shared continuation names with Ask added items and Continue in Research Workspace.
- [ ] Run targeted suites and relevant combined regressions, format/check changed scope, update TASK-13453.4 and commit. Controller then performs full affected-unit tests, both builds/type checks, isolated live WebUI/native extension workflows, whole-branch review, Bandit scope record, documentation update and cleanup.

## Completion gates

- [ ] All four task reviews approve specification and code quality; address important findings before dependent work. Record rulings where a skill's five-round loop conflicts with the repository's maximum three attempts.
- [ ] Run affected shared UI regression directories and project type/build checks for WebUI/extension; correct regressions introduced in this branch.
- [ ] Run deterministic provider browser workflows, exact source ID/distractor checks, note/web snapshot research continuation and save/reopen, one/mixed ingest retry, keyboard/short/narrow interaction, and native extension capture/handoff.
- [ ] Run applicable formatting/lint/pre-commit checks and Bandit on changed Python if any; record frontend-only non-applicable security scope accurately.
- [ ] Update audit K01–K15 resolution/verification and five enhancement outcomes, Backlog parent/children and final evidence. Preserve governing ADR decisions.
- [ ] Stop only owned runtime services, remove owned generated test artifacts/links, verify committed changes and final checkout status, then remove only this completed plan file as instructed by AGENTS.md. Do not push or merge.
