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
