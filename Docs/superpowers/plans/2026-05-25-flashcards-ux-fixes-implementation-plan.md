# Flashcards UX Fixes Implementation Plan

Source audit and fix list: [Flashcards-UX-Fix-List.md](../../../Flashcards-UX-Fix-List.md)

## Stage 3B: Deck Study Dashboard
**Goal**: Prove existing client/API data supports a deck-level study dashboard, then add a compact deck-first launch surface on `/flashcards` Study without broad backend work.

**Success Criteria**:
- Existing analytics data provides deck id, deck name, total, new, learning, due, and mature counts.
- Study renders a deck-level dashboard only when no active review card is displayed.
- Dashboard rows provide direct Review, Cram, Edit, Scheduler, and Export actions.
- Cross-tab actions preselect the target deck and do not leave stale one-shot handoffs after normal Study deck changes.
- Focused component and tab tests cover dashboard rendering, action callbacks, analytics coexistence, and manager handoffs.

**Tests**:
- `bunx vitest run src/components/Flashcards/components/__tests__/DeckStudyDashboard.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.analytics-summary.test.tsx src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx`
- `bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab*.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.scope-change.guard.test.ts`
- `bun run verify:design-system-state`
- `git diff --check`
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`

**Status**: Complete

## Stage 3B Notes
- Data proof: `FlashcardAnalyticsSummary.decks` already exposes the required deck-level counts, so no backend schema or endpoint change is needed for this slice.
- Typecheck note: the full package typecheck still fails on existing repo-wide baseline errors outside this slice; the failure list does not include the new dashboard or changed Study/Manager handoff files.

## Supplemental Closeout Slices

The master checklist was later finished through PR-sized closeout branches so each remaining UX cluster could be reviewed and verified independently.

| Slice | Task | Commit | Goal | Status |
| --- | --- | --- | --- | --- |
| PR0: Evidence and harness refresh | TASK-2401 | `b536cd7de4` | Refresh Flashcards page-object/e2e helpers and component guards before behavior edits. | Complete |
| PR1: First-time setup and IA | TASK-2402 | `c0042977ec` | Improve first-run Study defaults, Create & Import task separation, transfer-limit copy, and empty states. | Complete |
| PR2: Create/import/generate reliability | TASK-2403 | `6ca740bb6a` | Clarify create/import/generate recovery, large import confirmation, structured draft selection, and deck setup behavior. | Complete |
| PR3: Review comprehension and recovery | TASK-2404 | `b1cd0400ca` | Clarify review progress, rating semantics, assistant disclosure, completion actions, and shortcut copy. | Complete |
| PR4: Errors, empty states, and feedback | TASK-2405 | `b3af85be0e` | Tighten Manage bulk recovery feedback, selection reset behavior, conflict guidance, and undo coverage. | Complete |
| PR5: Responsive layout and accessibility hardening | TASK-2406 | `8e8aadfa9f` | Add focused responsive contracts for tab actions, review progress, long deck labels, and first-run review CTAs. | Complete |
| PR6: Final checklist consolidation | TASK-9926 | this PR | Align this plan and `Flashcards-UX-Fix-List.md` with the completed PR0-PR5 closeout slices and remaining deferred audit work. | Complete |
