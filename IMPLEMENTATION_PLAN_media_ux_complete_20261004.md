# Complete Media UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Address every issue and enhancement from TASK-13450 in the shared Media and ingestion workflows.

**Architecture:** Reuse shared components, owner-fenced session/runtime and existing Media/job APIs. Correct queue and results contracts first, then review/Inspector interactions, then bounded recent-import metadata and complete platform verification.

**Tech Stack:** React, TypeScript, Zustand, existing browser storage, Ant Design, Vitest, Playwright, Next.js and WXT; no new dependency.

**Spec:** Docs/Design/2026-10-04-media-ux-complete-design.md

**Tracking:** TASK-13500 and TASK-13500.1–5. Approved audit: .impeccable/critique/2026-10-04T23-52-07Z__i-src-components-review-viewmediapage-tsx-a9a598d1.md.

## Global Constraints

- Shared behavior in WebUI and packaged extension; extension-only capture never queries tabs in WebUI.
- Preserve verified server/principal authority, stale-operation guards, cancellation and recovery.
- No new dependency, ingestion UI, backend/history service or unrelated refactor.
- User-facing text uses accurate i18n keys/defaults and corrected shipped translations.
- Replacement permission is independent of processing preset.
- At most 30 simultaneously fetched/rendered reading items; bulk selection may exceed 30.
- At most 10 owner-fenced recent-import metadata records in existing session storage; no credentials, File objects or full extracted content.
- Knowledge readiness requires affirmative server/result evidence, never requested settings alone.
- Use backlog-py with the project venv. ADR required: no new ADR; existing ADR-008, ADR-022 and ADR-059 govern.
- Add meaningful behavior tests first; observe expected failures before production changes. Commit each reviewed task with its Backlog record. Do not stage the temporary node_modules symlink.

---

## Task 1: Source handoff and eligible queue
**Goal:** TASK-13500.1 implements spec section 1.
**Success Criteria:** Both empty-state URLs and extension capture enter the active queue; parsing/dedupe/counts are reliable; presets never grant replacement implicitly.
**Tests:** Ordinary/playlist opening, draft preservation, restricted capture, comma URL edge cases, duplicate files/URLs, invalid exclusions, preset switching, current-run settings.
**Status:** Not Started

**Files:**
- Modify: apps/packages/ui/src/utils/quick-ingest-open.ts; components/Media/ResultsList.tsx; components/Review/MediaReviewResultsList.tsx; components/Common/QuickIngest/AddContentStep.tsx, IngestWizardContext.tsx, ReviewStep.tsx, WizardConfigureStep.tsx, presets.ts, result-actions.ts; active wizard/session seed caller as needed.
- Add only if existing helpers cannot carry shared pure parsing/eligibility: components/Common/QuickIngest/queue-items.ts.
- Test: utils/__tests__/quick-ingest-open.test.ts; existing Common/QuickIngest and Media/ResultsList test families plus focused queue/capture cases.
- Update: affected English and existing conflicting locale entries.

**Interfaces:** Consume QuickIngestOpenDetail `{source, url, action}` and WizardQueueItem. Produce ordinary URL seeds through `createQuickIngestSessionSeedFromOpenDetail`; one eligible queue projection consumed by Configure, Review and processing. Preserve playlist preflight and existing context method names. Persist duplicate process-again choice with its queue item if needed.

- [ ] **Step 1: Add failing tests.** Start with the existing opening utility test:
```ts
const seed = createQuickIngestSessionSeedFromOpenDetail({ source: 'manual', url: 'https://example.com/article' })
expect(seed?.openDetail.url).toBe('https://example.com/article')
```
Add mounted caller→wizard tests, preserved-draft tests and table-driven parser cases for two comma-separated URLs, one comma-containing URL, newline input, invalid input and duplicates. Assert request-eligible items, not mocks alone. Test Deep and preset switches with overwrite false/true.
- [ ] **Step 2: Run RED.** From apps/packages/ui: `node node_modules/vitest/vitest.mjs run src/utils/__tests__/quick-ingest-open.test.ts src/components/Common/QuickIngest/__tests__ src/components/Media/__tests__/ResultsList.ftux.test.tsx --maxWorkers=1 --no-file-parallelism`. Save expected assertion failures.
- [ ] **Step 3: Implement the shared contract.** Callers emit `{source:'manual',url:trimmed}`; ordinary URL seeds queue once without resetting drafts. Centralize eligible item decisions; split only unambiguous URL boundaries; explicitly exclude/allow duplicates. Set Deep overwrite false and preserve the current explicit choice on preset switches. Update misleading copy and extension capture through existing browser helpers.
- [ ] **Step 4: Verify GREEN.** Run the focused new tests and the existing Quick Ingest queue/context/preset/opening family; run changed-file formatting/type diagnostics. Record exact commands and results.
- [ ] **Step 5: Self-review and commit.** Finalize TASK-13500.1 with tests and touched paths, commit only this task's source/tests/locales and task record. Write task report with RED/GREEN evidence. Controller runs task review before Task 2.

## Task 2: Recovery and saved-batch continuation
**Goal:** TASK-13500.2 implements spec section 2.
**Success Criteria:** Failed-only retry preserves sources/options/successes; result accounting reconciles; saved batches open directly; readiness claims are truthful.
**Tests:** Mixed retry, nonretryable correction, missing File reattach, ownership invalidation, unique saved-ID handoff, count accounting, readiness evidence.
**Status:** Not Started

**Files:**
- Modify: components/Common/QuickIngestWizardModal.tsx; Common/QuickIngest/WizardResultsStep.tsx, result-actions.ts, ProcessingStep.tsx and context/reducer only for retry state; store/quick-ingest-session.ts or existing runtime where required for safe state merging.
- Reuse: hooks/useIngestResults.tsx; services/tldw/conference-collections.ts; existing Media review selection setting and saved-item identity helpers.
- Test: active wizard session/integration tests, WizardResultsStep/result-actions tests and affected session tests.
- Update: result/status/review locale entries.

**Interfaces:** Consume eligible queue from Task 1 and existing `onRetryItems(itemIds, retryItems?)`. Produce saved media-ID review handoff via existing selection setting and `/media-multi`, plus truthful result summary. Keep session authority and persisted result schema compatible. Pure saved-ID/readiness helpers belong in existing result-actions if needed.

- [ ] **Step 1: Add failing tests.** Mount active wizard with one success and one retryable failure; invoke Retry, assert only failed source is submitted and success remains openable. Add File reload/reattach and account-change cases. In Results, activate Review saved items and assert unique successful stored IDs, excluding unsaved/failed inputs. Assert requested chunking alone does not render Ready for Knowledge.
- [ ] **Step 2: Run RED.** `node node_modules/vitest/vitest.mjs run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx src/components/Common/QuickIngest/__tests__/WizardResultsStep.test.tsx --maxWorkers=1 --no-file-parallelism`, adjusting filters to actual focused tests created alongside existing files.
- [ ] **Step 3: Implement retry and continuation.** Wire existing callbacks. Retry only requested eligible failures; merge updated outcomes by source ID, preserving successes and options. Provide explicit reattach/configuration recovery. Build Review saved items from authoritative stored IDs, not arbitrary result success count. Present a compact reconciled summary and confirmed status vocabulary.
- [ ] **Step 4: Verify GREEN.** Run focused result/retry/session tests, related quick-ingest authority/runtime tests and changed-file diagnostics. Record exact failures avoided and successful outcomes retained.
- [ ] **Step 5: Self-review and commit.** Finalize TASK-13500.2, commit scoped files/task and write report. Controller reviews before Task 3.

## Task 3: Multi-review semantics and bounded reading
**Goal:** TASK-13500.3 implements spec section 3.
**Success Criteria:** Preview/select/keyboard/navigation agree across pages; larger metadata sets retain a bounded reading window; controls are grouped.
**Tests:** Pointer/Enter equivalence, accessible checkbox/Shift range, selected-content position, mobile Content handoff, 40 selected with at most 30 content loads, bounded-window navigation and full-set batch actions.
**Status:** Not Started

**Files:**
- Modify: components/Review/MediaReviewResultsList.tsx, MediaReviewReadingPane.tsx, MediaReviewPage.tsx, MediaReviewBatchBar.tsx, media-review-types.ts; hooks/useMediaReviewState.ts, useMediaReviewActions.tsx, useMediaReviewKeyboard.ts.
- Test: existing Review selection/view-mode/batch/responsive/keyboard families; new mounted interaction and reading-window tests.
- Update: shipped preview/stack/selection/layout instructions in locales.

**Interfaces:** Keep ordered `selectedIds` as full bulk selection. Keep `viewerItems` as the active bounded reading window, and distinguish active preview from selected context. Derive navigation against the active context, never only `allResults`. Existing batch actions consume full selected IDs, while detail loading is on demand. Reading-window state is local to the review surface; selected IDs remain persisted under the existing setting.

- [ ] **Step 1: Add failing tests.** Use existing mounted Review harness to select IDs on page one, change pages and preview another item; assert heading and position describe displayed content. Activate the same row with pointer and Enter and compare state. Select 40 IDs, assert all remain selected while viewer/detail requests stay at most 30; switch the reading window and assert retained selection and correct position.
- [ ] **Step 2: Run RED.** `node node_modules/vitest/vitest.mjs run src/components/Review/__tests__/MediaReviewPage.stage1.selectionLimit.test.tsx src/components/Review/__tests__/MediaReviewPage.stage5.batch-toolbar.test.tsx` plus new focused interaction/window tests, `--maxWorkers=1 --no-file-parallelism`. Update old cap expectations only to the explicitly approved reading-window behavior; retain regression assertions.
- [ ] **Step 3: Implement consistent contexts.** Pointer/Enter preview; named tabbable checkboxes select. Make preview content visible independently of selected reading content. Selected navigation uses ordered selected IDs; reading windows fetch/render at most 30. Batch actions retain full selected scope. Group controls under existing Reading/Layout/Selection actions affordances and switch mobile to Content.
- [ ] **Step 4: Verify GREEN.** Run affected Review selection/navigation/view-mode/keyboard/batch/responsive tests, type diagnostics and changed-file format checks. Check no unbounded detail prefetch was introduced.
- [ ] **Step 5: Self-review and commit.** Finalize TASK-13500.3, scoped commit/report, controller review before Task 4.

## Task 4: Inspector bulk scope, recovery and accessibility
**Goal:** TASK-13500.4 implements spec section 4.
**Success Criteria:** Cross-page selection persists with visible action scope; mobile reading/actions are accessible; trash is recoverable; Add media and contextual guidance are discoverable.
**Tests:** Four selections across two pages, complete-set exports/opening, account-change reset, cancel/confirm/partial-trash recovery, labeled controls, mobile view handoff and contextual guidance.
**Status:** Not Started

**Files:**
- Modify: components/Review/ViewMediaPage.tsx, MediaBulkToolbar.tsx; Review/hooks/useMediaSelection.ts; components/Media/ResultsList.tsx and relevant search controls; reuse existing responsive hooks/hint/trash patterns.
- Test: existing ViewMediaPage and useMediaSelection tests; new cross-page/recovery/accessible-controls tests; frontend Media workflow E2E for responsive behavior.
- Update: Media/Inspector locale entries and contextual hint text.

**Interfaces:** Inspector bulk selected IDs are independent of visible results and owner-fenced. Cache selected row metadata or fetch on action so existing export/open/collection paths consume all selected IDs. Preserve Media/Notes mixed-kind semantics. Mobile view state switches Results/Content without clearing search/page/selection. Task 3's multi-review consumes the full saved-ID handoff.

- [ ] **Step 1: Add failing tests.** Select two page-one plus two page-two rows and assert count four, full export/open-selection IDs and cleared selection on owner change. Confirm Cancel sends no trash request; partial confirmed trash retains failed selections and offers recovery. Assert Sort/export/select controls have accessible names and opening mobile content exposes a named Back to results.
- [ ] **Step 2: Run RED.** `node node_modules/vitest/vitest.mjs run src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx src/components/Review/__tests__` with focused new Inspector tests, `--maxWorkers=1 --no-file-parallelism`.
- [ ] **Step 3: Implement using existing patterns.** Remove visible-page pruning; retain complete selection metadata and owner fences. Move bulk toolbar outside shrinking filters; expose explicit mobile Results/Content with automatic content opening and preserved Back navigation. Add labels/hit areas, confirmation/Trash recovery, visible Add media and contextual batch guidance.
- [ ] **Step 4: Verify GREEN.** Run focused selection/action/Inspector suites and responsive browser cases at 390×844 and desktop; verify mixed Notes actions and existing permalink/history behavior.
- [ ] **Step 5: Self-review and commit.** Finalize TASK-13500.4, scoped commit/report, controller review before Task 5.

## Task 5: Recent imports, documentation and integration
**Goal:** TASK-13500.5 implements spec section 5 and verifies every accepted item.
**Success Criteria:** Recognizable imports appear from submission and survive reload safely; historical batch review requires no raw IDs; complete platform checks and final review resolve defects.
**Tests:** Bounded history, active submission/resume/reload, server/account transitions, stale polling, saved-ID history review, mixed import/retry and cross-page mobile journeys in WebUI and packaged extension.
**Status:** Not Started

**Files:**
- Modify: store/quick-ingest-session.ts; components/Media/MediaIngestJobsPanel.tsx; relevant session/tracking adapters only where necessary.
- Test: session ownership/history tests; MediaIngestJobsPanel tests; apps/tldw-frontend/e2e/workflows/media-ux-complete.spec.ts; packaged extension Media test/fixtures if needed.
- Document: Docs/Reviews/2026-10-04-media-ux-complete-verification.md; user workflow documentation in Docs/Product/WebUI/Media/ using an existing appropriate path if present.

**Interfaces:** Recent metadata extends the existing session store and storage key. At most 10 records; filter/read/action by verified authority. Store only session ID, source/count/status/times, known batch/job IDs and successful saved IDs. Current owned session drives resume/live status; known batch IDs feed the existing required-batch jobs API. No unfiltered job API or new persistence service.

- [ ] **Step 1: Add failing tests.** Complete/replace more than ten sessions and assert bounded ordered metadata without credentials/File/content. Switch authority A→B and assert A imports/action results do not appear. Start processing, close/reopen, and assert recognizable current import and existing resume action. Activate a historical saved review and assert its exact IDs.
- [ ] **Step 2: Run RED.** `node node_modules/vitest/vitest.mjs run src/store/__tests__/quick-ingest-session.test.ts src/store/__tests__/quick-ingest-session.authority.test.ts` plus new history/panel tests, `--maxWorkers=1 --no-file-parallelism`.
- [ ] **Step 3: Implement bounded history and panel.** Add metadata summaries through existing store transitions and authority sanitization. Show Recent imports from submission; resume current owned session, refresh known batches and review their saved sets. Hide manual IDs in optional diagnostics. Release timers and reject stale owner responses.
- [ ] **Step 4: Verify integrated GREEN.** Run relevant Media/Review/Quick Ingest/session families once; changed-scope lint/format/type checks; Next/WXT builds; backend/task-format security checks where applicable. Use existing Playwright/native browser and isolated safe API fixtures, labeling simulated outcomes. Exercise all spec coverage rows in one desktop/mobile/extension pass; repair observed defects together and confirm once. Record limitations on unavailable real ML/backend checks.
- [ ] **Step 5: Self-review, docs, commit and final review.** Write exact coverage/outcomes in verification doc; complete TASK-13500.5/coordinator only when all required work is done. Commit scoped files and tasks. Controller obtains whole-branch source review, resolves load-bearing findings and completes branch handoff. Remove only this implementation plan after completion as required by repo instructions; preserve design and verification docs and task history.
