# Flashcards UX Remediation Roadmap And Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this roadmap phase-by-phase. Each phase must have its own Backlog.md task before code or documentation edits begin. Do not reopen `TASK-479` for implementation; it is the historical planning task for the earlier phase map.

**Goal:** Turn the `/flashcards` WebUI and directly connected extension flashcards audit findings into a complete, outcome-based remediation roadmap. The roadmap must support first-time users who need to create/import/generate/select cards and complete a review session, and power users who need fast deck selection, management, repeated study, keyboard flow, and recovery from mistakes.

**Architecture:** Keep changes scoped to the existing Flashcards WebUI package, `/flashcards` route, direct Quiz handoff, Study Pack source selection, and direct extension flashcard routes. Start with evidence, reliability, and first-use comprehension. Then improve review completion, power-user deck management, responsive/accessibility behavior, and direct handoffs. Preserve current backend contracts unless a phase proves a small API addition is necessary for an observable UX state.

**Tech Stack:** Next.js WebUI, shared `apps/packages/ui` React components, Ant Design, React Query flashcard hooks, Vitest/Testing Library, Playwright e2e, WXT extension routes, FastAPI flashcards API only when needed for user-visible state.

---

## Source Inputs

- Current live audit request: `/flashcards` WebUI and directly connected WebUI/extension flashcard workflows.
- Existing master fix list: `Flashcards-UX-Fix-List.md`
- Existing planning task: `TASK-479`
- Current planning refresh task: `TASK-535`
- Primary route: `/flashcards`
- Direct WebUI handoffs: `/quiz` when launched from flashcards, Study Pack creation from flashcards source controls.
- Direct extension handoffs:
  - `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`
  - `apps/packages/ui/src/routes/option-flashcards.tsx`
  - `apps/tldw-frontend/extension/routes/option-flashcards.tsx`
- Product context:
  - `Docs/Design/Flashcards.md`
  - `Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
  - `Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`

## Current Audit Evidence

Observed or inspected during the current audit cycle:

- `/flashcards` in the running WebUI.
- `/flashcards` empty, create/import/generate/setup areas.
- Imported temporary TSV deck and reviewed two cards, then removed temporary decks through the API.
- Invalid import path where the UI reported success with zero imported cards and left the Import button in a loading/disabled state.
- Review setup, reveal, rating, completion, progress labels, and saved state.
- Narrow/mobile layout behavior where tabs, metrics, and card/review surfaces clipped core workflow controls.
- Quiz handoff from flashcards that routed to `/quiz?tab=take&source=flashcards` without preserving the reviewed deck/card context.
- Source inspection for extension flashcards routes and Study Pack creation controls.

Primary files inspected:

- `apps/tldw-frontend/pages/flashcards.tsx`
- `apps/packages/ui/src/components/Flashcards/FlashcardsWorkspace.tsx`
- `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/ImportPanel.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/ExportPanel.tsx`
- `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
- `apps/packages/ui/src/components/Flashcards/components/ReviewProgress.tsx`
- `apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx`
- `apps/packages/ui/src/components/Flashcards/components/KeyboardShortcutsModal.tsx`
- `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`
- `apps/packages/ui/src/routes/option-flashcards.tsx`
- `apps/tldw-frontend/extension/routes/option-flashcards.tsx`
- `apps/tldw-frontend/extension/routes/route-registry.tsx`

Assumptions and limits:

- Browser evidence used a local single-user API setup.
- A fully unconfigured first-run account was source-verified where browser state could not be reset without affecting the local app.
- Quiz was assessed only as a direct flashcards handoff, not as a broad Quiz product audit.
- Extension review stayed limited to routes directly connected to flashcard capture/open behavior.

## Planning Principles

- Fix trust and recoverability before layout polish.
- Keep `/flashcards` scoped to flashcard setup, management, review, history, scheduler, import/generate/export, Study Pack source selection, Quiz handoff, and extension flashcard handoffs.
- Preserve stable tab keys (`review`, `cards`, `importExport`) unless a phase explicitly updates route/query handling and tests.
- Follow the Flashcards UI convention in `Docs/Design/Flashcards.md`: top-level labels should read as `Study`, `Manage`, and `Transfer` or a task-first successor, not implementation labels.
- Pair every visible UX change with focused component tests. Add Playwright coverage for first-time setup, invalid import recovery, review completion, responsive smoke, and direct handoffs.
- Do not leak raw IDs, raw `scope_key` values, or unresolved template placeholders in primary user-facing copy.
- Use existing Ant Design and shared token conventions. Do not introduce a new UI kit.
- Do not redesign unrelated Study, Quiz, Workspaces, or backend scheduler surfaces.

## Roadmap Spine

The roadmap is organized by user outcomes instead of the earlier legacy finding buckets:

1. Can users trust the flashcards workflow when something fails?
2. Can first-time users understand how to create/import/generate/select cards and start review?
3. Can users recover from import/create/generate mistakes?
4. Can users complete a review session and understand progress?
5. Can power users quickly pick decks, manage cards, and repeat study?
6. Does the workflow hold up on mobile, narrow layouts, keyboard, and assistive technology paths?
7. Do direct handoffs to Quiz, Study Packs, and the extension preserve flashcard context?
8. Do docs, tests, and release gates match the stabilized workflow?

## Outcome Phase Mapping

| Roadmap Phase | Current audit findings covered | Legacy `Flashcards-UX-Fix-List.md` coverage |
| --- | --- | --- |
| Phase 0: Evidence And Harness Refresh | Refresh browser evidence for invalid import, create drawer, review flow, mobile clipping, and quiz handoff. | F05, F20, plus regression proof for F01, F02, F04, F17 |
| Phase 1: Trust, Empty State, And First-Time Setup | Empty state starts in dense transfer tooling; Study Pack creation asks for raw Source ID; Scheduler and setup concepts are hard to discover. | F04, F06, F14, F15, F18 |
| Phase 2: Import, Generate, And Create Recovery | Invalid import reports success and gets stuck; import success lacks "Review imported deck"; create drawer has confusing nested deck creation. | F01, F03, F05, F06 |
| Phase 3: Review Session Comprehension And Recovery | Progress language conflicts; completion/next steps/saved state are unclear; undo/re-rate and shortcut parity need visible support. | F02, F07, F08, F09, F10, F16, F19 |
| Phase 4: Power-User Deck And Card Management | Existing deck selection, filtering, repeat review, and management are too slow for experienced users. | F11, F16, F19 |
| Phase 5: Responsive Layout And Accessibility Hardening | Narrow/mobile layout clips core workflow; icon buttons lack names; shortcut controls conflict. | F14, F16, F20 |
| Phase 6: Direct Handoffs: Quiz, Study Packs, Extension | Quiz loses flashcard context; extension needs explicit capture/open behavior; Study Packs need source picker rather than raw ID entry. | F12, F17 |
| Phase 7: Documentation, Release Gate, And Follow-On Backlog | User docs and extension docs must match the stabilized workflow. | F13 and all release acceptance checks |

## Candidate Backlog Task Splits

Create one Backlog.md task per phase before implementing that phase:

- `Flashcards UX Phase 0: evidence and harness refresh`
- `Flashcards UX Phase 1: trust, empty state, and first-time setup`
- `Flashcards UX Phase 2: creation import generation recovery`
- `Flashcards UX Phase 3: review comprehension and recovery`
- `Flashcards UX Phase 4: power-user deck and card management`
- `Flashcards UX Phase 5: responsive layout and accessibility hardening`
- `Flashcards UX Phase 6: quiz study-pack and extension handoffs`
- `Flashcards UX Phase 7: docs release gate and follow-on backlog`

Do not bundle all phases into one PR unless the user explicitly asks for a large integrated change. Each phase should be independently reviewable, testable, and reversible.

## Dependency Map

1. Phase 0 runs first because later phases depend on trustworthy e2e selectors and confirmed reproduction of invalid import, create drawer, review, responsive, and quiz handoff states.
2. Phase 1 runs after Phase 0 so first-time route defaults and labels stabilize before deeper setup UI work.
3. Phase 2 runs after Phase 1 because import/create/generate recovery depends on the updated setup IA and task names.
4. Phase 3 can begin after Phase 0 but should merge after Phase 1 so Study labels and route defaults are stable.
5. Phase 4 depends on Phase 3 when deck dashboard actions reuse review completion/session labels.
6. Phase 5 can run in parallel with late Phase 3 or Phase 4 if selectors are stable, but it should not ship before the core first-time and review flows are stable.
7. Phase 6 should wait until the destination context contracts are known: selected deck/card for Quiz, selected source for Study Packs, and create/generate drafts for extension.
8. Phase 7 runs last because docs and release gates should describe final behavior, not intermediate churn.

## File Responsibility Map

- `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`: top-level tab defaults, tab labels, query param handling, global action state, quiz CTA state, manager-level create signal.
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`: deck selection, review mode, review queue copy, assistant placement, visible undo/re-rate, completion CTAs, mode switching, keyboard parity, responsive review layout.
- `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`: no-card management chrome, deck/card filters, power-user card controls, create drawer launch, create/import/generate empty-state actions.
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`: transfer/task IA, summary band, import/generate/export/image-occlusion grouping, limit copy.
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/*.tsx`: import, export, generation, Study Pack, and transfer-panel result states.
- `apps/packages/ui/src/components/Flashcards/tabs/ImageOcclusionTransferPanel.tsx`: image occlusion result state and transfer task placement.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`: manual create feedback, nested deck creation simplification, initial draft/prefill support, mutation success/error states.
- `apps/packages/ui/src/components/Flashcards/components/ReviewProgress.tsx`: due/new/learning/available-now labels and compact mobile layout.
- `apps/packages/ui/src/components/Flashcards/components/ReviewAnalyticsSummary.tsx`: completion/progress labels and responsive summary behavior.
- `apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx`: session history labels, resume/snapshot wording, reviewed counts, deck/mode names.
- `apps/packages/ui/src/components/Flashcards/components/KeyboardShortcutsModal.tsx`: visible-control parity and shortcut conflict copy.
- `apps/packages/ui/src/components/Flashcards/components/DeckStudyDashboard.tsx`: create only if Phase 4 proves current data supports a deck-first dashboard without broad backend work.
- `apps/packages/ui/src/services/flashcards.ts`: client result types only if UI-level result normalization cannot stay local to panels.
- `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`: extension capture/open entry point.
- `apps/packages/ui/src/routes/option-flashcards.tsx`: extension options route wrapper around the shared workspace.
- `apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts`: stable selectors for first-run, import/create, review, completion, mobile, and handoff tests.
- `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`: backend-backed WebUI e2e coverage.
- `apps/extension/tests/e2e/flashcards-ux.spec.ts`: extension capture/open behavior if extension harness supports it.
- `apps/packages/ui/src/components/Flashcards/**/__tests__/*.test.tsx`: component-level regression tests.
- `Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`: user-facing guide after UI changes.
- `Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`: published guide mirror if still used.
- `apps/extension/docs/features/flashcards.md`: extension behavior documentation.

---

## Phase 0: Evidence And Harness Refresh

**Goal:** Establish reliable test and browser evidence for the current highest-risk flashcards flows before changing UX.

**Findings covered:** F05, F20, and reproduction harnesses for invalid import, mobile clipping, quiz handoff, first-time empty state, and review completion.

**Files:**

- Modify: `apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`
- Modify or add: `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx`
- Modify or add: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx`
- Modify or add: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx`

**Acceptance Criteria:**

- E2E can create or seed a deck/card, open `/flashcards`, and start a review session.
- E2E captures invalid import behavior and asserts it does not silently pass once fixed in Phase 2.
- E2E or component coverage exercises manual create success and failed create feedback.
- Keyboard-only review skeleton covers reveal, rating, completion, and return to deck/manage state.
- Mobile/narrow viewport smoke identifies whether tabs, metrics, review card, import panels, and action rows remain reachable.
- Quiz handoff test records current behavior and later asserts either context preservation or disabled state.

**Implementation Notes:**

- Add selectors for create front/back fields, Create, Create and Add Another, import textarea/file controls, import submit, import result alert, rating buttons, completion state, Cram mode, progress labels, recent sessions, and quiz handoff.
- Keep this phase mostly test/harness focused. Only fix product code here if the create drawer is completely blocked and no later phase can be tested.
- Record exact reproduction evidence in the phase Backlog task before implementation phases begin.

**Focused Verification:**

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards
cd apps/tldw-frontend && TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts --grep "flashcards"
```

---

## Phase 1: Trust, Empty State, And First-Time Setup

**Goal:** Make first entry to `/flashcards` understandable before users enter dense transfer, management, or scheduler tooling.

**Findings covered:** F04, F06, F14, F15, F18, plus raw Study Pack Source ID discoverability.

**Files:**

- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/ImportPanel.tsx`
- Modify or add tests under `apps/packages/ui/src/components/Flashcards/**/__tests__`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`

**Acceptance Criteria:**

- Empty accounts land on `Study`/home, not dense transfer tooling.
- First screen says what flashcards are for in this product and offers clear actions: Create manually, Import deck, Generate from source, or choose an existing deck when present.
- Manage no-card state suppresses expert filters, sort, density, and shortcut chips until cards exist or filters are active.
- Scheduler is discoverable before a deck exists through disabled tab copy or Study empty-state scheduling preview.
- Transfer/create/import label no longer implies normal import/export is LLM-only.
- Study Pack setup does not require users to guess a raw Source ID. If a real picker is not feasible in this phase, add clear copy and defer source-picker implementation to Phase 6.

**Implementation Notes:**

- Remove any effect that sends no-card users directly to the transfer/import tab solely because no decks exist.
- Preserve generate/study-pack deep-link behavior: explicit generate intents may still open the appropriate setup area.
- Use existing manager-level create signal rather than adding a second create flow.
- Do not build the full deck dashboard in this phase; Phase 4 owns power-user deck management.

**Focused Verification:**

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.decomposition.test.tsx
cd apps/tldw-frontend && TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts --grep "first-time|empty|tabs"
```

---

## Phase 2: Import, Generate, And Create Recovery

**Goal:** Ensure creation/setup workflows never masquerade as success when they fail, partially fail, or need user recovery.

**Findings covered:** F01, F03, F05, F06, plus current invalid import stuck-state, missing import next-step CTA, and confusing create-drawer deck creation.

**Files:**

- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/ImportPanel.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/ExportPanel.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImageOcclusionTransferPanel.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
- Modify: `apps/packages/ui/src/services/flashcards.ts` only if response normalization cannot stay in UI code.
- Modify or add tests under `apps/packages/ui/src/components/Flashcards/tabs/__tests__`
- Modify or add `apps/packages/ui/src/services/__tests__/flashcards-structured-import.test.ts`

**Acceptance Criteria:**

- Invalid import produces an error or validation state, never "Imported 0 cards" as success.
- Import button exits loading state after success, partial success, validation failure, and network/API failure.
- Partial import/generate reports created, skipped, and failed counts.
- Failed import/generate preserves input and offers retry.
- Successful import/generate offers next actions: Review imported deck, Manage imported cards, Import another, or Open deck.
- Transfer summary never renders unresolved `{{cards}}` or `{{bytes}}` placeholders.
- Create drawer has one clear deck-selection/creation model. It avoids duplicate "Create" and duplicate "New deck name" controls.
- Failed create shows a visible error, keeps user input, and exits loading state.

**Implementation Notes:**

- Add a local `TransferResultSummary` model if no equivalent exists:

```ts
export type TransferResultStatus = "success" | "partial" | "error";

export type TransferResultSummary = {
  status: TransferResultStatus;
  title: string;
  detail: string;
  createdCount?: number;
  skippedCount?: number;
  failedCount?: number;
};
```

- Prefer local UI normalization of existing API responses before changing backend schemas.
- Keep LLM provider/setup gating local to Generate and Study assistant areas, not the top-level tab label.
- Do not broaden into unrelated import/export formats unless needed to fix the observed failure state.

**Focused Verification:**

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.llm-gating.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.deck-creation.test.tsx src/services/__tests__/flashcards-structured-import.test.ts
cd apps/tldw-frontend && TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts --grep "import|create"
```

---

## Phase 3: Review Session Comprehension And Recovery

**Goal:** Make review feel recall-first, recoverable, and understandable from card reveal through completion and saved history.

**Findings covered:** F02, F07, F08, F09, F10, F16, F19.

**Files:**

- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/FlashcardStudyAssistantPanel.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/ReviewProgress.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/ReviewAnalyticsSummary.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/KeyboardShortcutsModal.tsx`
- Modify related component tests.

**Acceptance Criteria:**

- Review prioritizes prompt -> reveal -> rate. Study assistant is collapsed or secondary until the user asks for help, preferably after reveal.
- After rating, Undo/Re-rate is visible in the review or completion area for the supported undo window.
- Completion offers clear next actions: Practice again/Cram this deck, Review mistakes if supported, Create card, Manage deck, Open scheduler when applicable.
- Progress labels distinguish `Due` from `Available now` or `Study queue` when new cards are reviewable.
- Recent session rows show user-facing deck name where available, mode label, reviewed count, completed time, and clear actions such as Continue session, View completed session, or Review same deck again.
- Shortcut modal only advertises actions that have visible equivalent controls or explicitly labels them as accelerators.

**Implementation Notes:**

- Extract undo/re-rate rendering so it can appear both after answer rating and in completion.
- Use existing review queue state to compute display-only `availableNow` without changing scheduler math:

```ts
const availableNow = (dueCounts?.due ?? 0) + (dueCounts?.new ?? 0) + (dueCounts?.learning ?? 0);
```

- Join recent sessions with loaded deck names in the UI if backend session rows do not include names.
- Do not build advanced analytics in this phase; Phase 4 owns deck-level power-user dashboard work.

**Focused Verification:**

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.assistant.test.tsx src/components/Flashcards/components/__tests__/KeyboardShortcutsModal.rating-scale.test.tsx src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx
cd apps/tldw-frontend && TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts --grep "review|completion|keyboard"
```

---

## Phase 4: Power-User Deck And Card Management

**Goal:** Let experienced users quickly identify the right deck, choose the right review mode, manage cards, and repeat study without walking through first-time setup paths.

**Findings covered:** F11, F16, F19, plus current power-user bottlenecks around deck selection, sorting/filtering, repeat review, resume, and batching.

**Files:**

- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx`
- Create if data supports it: `apps/packages/ui/src/components/Flashcards/components/DeckStudyDashboard.tsx`
- Create if needed: `apps/packages/ui/src/components/Flashcards/components/__tests__/DeckStudyDashboard.test.tsx`
- Modify: `apps/packages/ui/src/services/flashcards.ts` only if existing types need client-side fields.

**Acceptance Criteria:**

- Study or Manage presents a deck-first overview when multiple decks exist.
- Deck rows can expose due/new/learning/mature counts, last studied, next due, and direct Review/Cram/Manage/Scheduler/Export actions if existing data supports it.
- If existing data cannot support a deck dashboard without N+1 requests or backend changes, the phase stops and creates a follow-up API/data design task.
- Search/filter/sort/card-management paths remain available for experienced users after setup.
- Repeat study is efficient: Cram/review again is reachable from deck rows and completion states.
- Batch actions are included only if existing Manage primitives already support them; otherwise record as follow-up.

**Implementation Notes:**

- Before building `DeckStudyDashboard`, inspect whether current hooks can provide per-deck due/new/learning/mature counts efficiently.
- Preferred V1 dashboard shape:

```ts
type DeckStudyDashboardProps = {
  decks: Deck[];
  dueCountsByDeck?: Record<number, DeckDueCounts>;
  onReviewDeck: (deckId: number) => void;
  onCramDeck: (deckId: number) => void;
  onManageDeck: (deckId: number) => void;
  onOpenScheduler?: (deckId: number) => void;
};
```

- Keep V1 compact. Avoid nested cards. Prefer a dense table/list band with row actions.

**Focused Verification:**

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/components/__tests__/DeckStudyDashboard.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx
cd apps/tldw-frontend && TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts --grep "deck|manage|cram|resume"
```

---

## Phase 5: Responsive Layout And Accessibility Hardening

**Goal:** Make the core flashcards workflow usable at narrow/mobile widths and through keyboard/screen-reader paths.

**Findings covered:** mobile/narrow clipping, F14, F16, F20, and icon-button accessible-name gaps.

**Files:**

- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/ReviewProgress.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/ReviewAnalyticsSummary.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/KeyboardShortcutsModal.tsx`
- Modify or add responsive/a11y tests where existing test utilities support them.

**Acceptance Criteria:**

- At narrow/mobile widths, users can reach tabs, deck selector, review prompt, Show Answer, rating buttons, progress, import submit, and completion CTAs without horizontal clipping.
- Metric rows wrap or collapse without overlapping core actions.
- Icon-only buttons have accessible names and tooltips where useful.
- Focus order follows the visible workflow: deck/setup -> prompt -> reveal -> rating -> recovery/completion.
- Shortcut controls do not conflict with a global shortcuts button or hide required recovery actions.
- Keyboard-only review e2e passes for reveal, rating, undo/re-rate, completion, and navigation back to Manage.

**Implementation Notes:**

- Favor responsive constraints, wrapping, and compact controls over hiding core actions.
- Do not introduce viewport-scaled font sizes.
- Do not use cards inside cards to solve spacing.
- Use Ant Design responsive primitives and existing flashcards layout guardrails where possible.

**Focused Verification:**

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards
cd apps/tldw-frontend && TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts --grep "mobile|keyboard|accessibility|responsive"
```

Manual browser verification is required for this phase:

- `/flashcards` desktop width.
- `/flashcards` narrow/mobile width.
- Review before reveal, after reveal, and completion.
- Import/generate after success, partial success, and failure.

---

## Phase 6: Direct Handoffs: Quiz, Study Packs, Extension

**Goal:** Ensure flashcards context is preserved or honestly blocked when users move into directly connected workflows.

**Findings covered:** F12, F17, current quiz context loss, raw Study Pack Source ID entry, and extension capture/open gaps.

**Files:**

- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/ImportPanel.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`
- Modify: `apps/packages/ui/src/routes/option-flashcards.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-flashcards.tsx`
- Modify route registry and extension tests.

**Acceptance Criteria:**

- `Test with Quiz` appears only when Quiz can receive a meaningful selected deck/card/study context, or it is disabled with explanatory copy.
- If a deck-to-quiz route is supported, the handoff includes the deck/card context and Quiz consumes it in tests.
- If the route is not supported, do not invent untested Quiz behavior in flashcards. Disable the CTA and create a separate Quiz contract task.
- Study Pack creation uses a source selector/search affordance if current data supports it. If not, the raw Source ID field is clearly labeled as advanced and a follow-up picker task is created.
- Extension sidepanel does not surprise users by auto-opening the full flashcards page.
- Extension offers explicit actions: Create flashcard, Generate from page/selection if available, and Open full Flashcards.
- Extension capture/open route is not emitted until WebUI consumes the matching route params.

**Implementation Notes:**

- Implement WebUI route parsing before extension buttons emit new params:

```text
/flashcards?tab=cards&create=1&source_url=...&source_title=...
/flashcards?generate=1&generate_text=...
```

- `FlashcardsManager` should own query parsing and route to existing manager-level create/open behavior.
- `FlashcardCreateDrawer` may accept an initial draft, but must not overwrite user edits after the drawer opens.
- Keep Quiz scope limited to direct handoff contract. Do not redesign Quiz setup or scoring here.

**Focused Verification:**

```bash
cd apps/packages/ui && bunx vitest run src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx
cd apps/tldw-frontend && TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts e2e/workflows/tier-2-features/quiz.spec.ts --grep "flashcards|quiz"
cd apps/extension && bunx playwright test tests/e2e/flashcards-ux.spec.ts
```

---

## Phase 7: Documentation, Release Gate, And Follow-On Backlog

**Goal:** Make the stabilized flashcards workflow documented, releasable, and separated from larger product/design improvements.

**Findings covered:** F13 and release readiness for all phases.

**Files:**

- Modify: `Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
- Modify: `Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
- Modify: `apps/extension/docs/features/flashcards.md`
- Modify or add final e2e release-check documentation if the repo has a standard location.

**Acceptance Criteria:**

- Docs explain first-time setup, create/import/generate, review, rating, undo/re-rate, completion, progress labels, session history, scheduler basics, Study Pack source selection, Quiz handoff, and extension capture/open behavior.
- Docs distinguish WebUI behavior from extension behavior.
- Release gate records test commands, manual browser checks, baseline failures, and remaining follow-up tasks.
- Larger non-blocking improvements are filed as backlog tasks rather than left as ambiguous TODOs.

**Focused Verification:**

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards
cd apps/tldw-frontend && TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts
```

---

## Cross-Phase Verification Rules

Run at the end of each frontend phase that touches Flashcards UI:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards
```

Run backend-backed flow coverage before final integration:

```bash
cd apps/tldw-frontend && TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts
```

Run extension coverage for extension phases:

```bash
cd apps/extension && bunx playwright test tests/e2e/flashcards-ux.spec.ts
```

Run Bandit only for phases that touch backend Python:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/flashcards.py tldw_Server_API/app/core/Flashcards -f json -o /tmp/bandit_flashcards_ux.json
```

For frontend-only or docs-only phases, record Bandit as not applicable in the Backlog task final summary.

Every phase final summary must record:

- Backlog task ID.
- Files changed.
- Tests or verification commands run.
- Known baseline failures separated from new failures.
- Manual browser notes for responsive/accessibility phases.
- Follow-up items moved to a later phase or separate Backlog task.

## Release Acceptance Criteria

- A first-time user opening `/flashcards` understands what flashcards are for in this product and can choose create, import, generate, or existing deck review without landing in dense transfer tooling first.
- A user can create or import/generate cards, start review, reveal answers, rate cards, undo/re-rate a mistake, finish a session, and understand what changed.
- Failed or partial import/generate/create flows never masquerade as full success and always offer a recovery path.
- Progress distinguishes due cards from available study queue/new cards.
- An experienced user can identify the right deck, review due work, cram intentionally, manage/filter/edit cards, read useful recent session history, and repeat review with minimal friction.
- Mobile/narrow viewport and keyboard-only workflows remain usable for setup, review, recovery, and completion.
- Quiz, Study Pack, and extension handoffs either preserve flashcard context or clearly explain why the action is unavailable.
- Docs match the implemented WebUI and extension behavior.

## Ideal Target Workflow

### First-Time User

1. User opens `/flashcards` and lands on Study/Home.
2. Empty state explains the product in one sentence and offers Create manually, Import deck, Generate from source, and any available sample/existing deck route.
3. Manual create opens a focused drawer with deck, front, back, card type, and optional metadata. Deck creation is clear and not duplicated.
4. Import/generate flows confirm created, skipped, and failed counts and show next actions.
5. After first cards exist, user sees a clear Study queue count and starts review.
6. Review shows front, then answer, then rating controls. Assistant help is available but secondary.
7. After rating, user can undo/re-rate immediately.
8. Completion shows what changed, when to come back, and the next useful actions.
9. Reloading `/flashcards` preserves progress and points to the next due or practice action.

### Experienced Power User

1. User opens `/flashcards` and sees deck-level status or can quickly select/filter decks.
2. Each deck exposes direct actions: Review due, Cram, Manage, Scheduler, Export where supported.
3. Manage supports search/filter/sort/edit without hiding expert controls after cards exist.
4. Review supports fast keyboard flow: reveal, rate, undo/re-rate, edit, and continue.
5. Completion supports repeat workflows: review mistakes, cram again, continue another due deck.
6. History shows meaningful sessions and can reopen completed snapshots or resume interrupted work.
7. Extension capture lets the user preserve selected web/page context into a chosen deck or full WebUI create/generate flow.

## Follow-On Backlog Candidates

These are not required for the core remediation unless a phase proves they are necessary:

- Rich deck health analytics: overdue, weak, recently failed, never studied, generated/imported review needed.
- Advanced spaced-repetition controls beyond display and discoverability fixes.
- Full Anki parity for import/export/editing.
- Broad Quiz redesign beyond the direct flashcards handoff.
- Broad Workspaces study-material IA beyond source selection needed for flashcards.
- Multi-user sharing/collaboration behavior for decks.

## Non-Goals

- Do not redesign unrelated Study or Quiz surfaces except the direct flashcards-to-Quiz handoff.
- Do not change scheduler algorithms unless a display fix proves impossible without backend support.
- Do not introduce a new design system, router, state manager, or backend job architecture.
- Do not attempt full Anki parity in this remediation program.
- Do not expand into broad Workspaces IA beyond preserving source/deck/workspace filters already present in flashcards.
- Do not treat extension flashcards as a full extension redesign; only direct capture/open handoffs are in scope.
