# Flashcards

## UI Conventions (H4 Consistency Standards)

This note defines Flashcards UI consistency rules for tabs, create-entry routing, and drawer behavior.

### Navigation Labels

- Use `Study`, `Manage`, `Transfer` for Flashcards top-level tabs.
- Keep tab keys stable (`review`, `cards`, `importExport`) to avoid breaking keyboard-shortcut routing and deep state references.

Reference: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`

### Primary Create Entry Point

- All secondary create actions (for example, Study empty state or top CTA) should route through the shared manager-level create entry point.
- The manager emits `openCreateSignal` to `ManageTab`, which opens `FlashcardCreateDrawer`.
- Avoid adding isolated create flows that bypass this routed path.

Reference:
- `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`

### Drawer Standards

- Use shared drawer width token `FLASHCARDS_DRAWER_WIDTH_PX` for Create/Edit/Move drawers.
- Keep action order aligned as: `Cancel` -> secondary action -> primary action.
- Use consistent section rhythm (`mb-6`) and section headers for Organization/Content blocks.

Reference:
- `apps/packages/ui/src/components/Flashcards/constants/drawer-tokens.ts`
- `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
- `apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`

### Study Pack Chapter Sets

The Study tab opens **Browse cards** for the selected deck. **All cards** includes every active card in the deck. The same drawer lists generated chapter sets and their current counts. Browsing leaves the review session and ratings intact.

Each generated passage carries the media ID, section plan revision, and section index in its saved citation. Every pass of a chapter shares this identity. The picker uses the saved heading as the chapter label. Counts follow the cards when they move between decks.

### Guardrails

- Tab naming and create-entry routing:  
  `apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx`
- Shared drawer width token usage:  
  `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.cloze-help.test.tsx`  
  `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.scheduling-metadata.test.tsx`  
  `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx`

## H8 Layout Guardrails (Aesthetic + Minimalist)

### State Priority Mapping

State maps are defined in code and must include all four review states:
`empty`, `active`, `success`, `error`.

- Canonical mapping source:
  `apps/packages/ui/src/components/Flashcards/constants/layout-guardrails.ts`
- Mapping integrity tests:
  `apps/packages/ui/src/components/Flashcards/constants/__tests__/layout-guardrails.test.ts`

### Primary Action Placement by Tab

- `Study` (`review`):
  Primary actions live in card content states. The top bar may show at most one primary CTA only in `empty`.
- `Manage` (`cards`):
  Primary create action is floating/empty-state scoped, not top-bar scoped.
- `Transfer` (`importExport`):
  Primary actions belong to per-panel action rows, not summary/top-level chrome.

### Top-Bar CTA Budgets

- `review`: `active=0`, `success=0`, `empty<=1`, `error<=1`
- `manage`: `0` for all states
- `transfer`: `0` for all states

Enforced by:
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx`

### Baseline Snapshot Coverage

The baseline suite tracks core visual states for regression:

- Review active and caught-up completion states:
  `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx`
- Cards selection state:
  `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx`
- Import result + transfer summary state:
  `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx`

## H10 Help and Docs Sync

- User-facing Flashcards help source-of-truth:
  `Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
- In-app links to guide sections are centralized in:
  `apps/packages/ui/src/components/Flashcards/constants/help-links.ts`
- Link/anchor integrity tests:
  `apps/packages/ui/src/components/Flashcards/constants/__tests__/help-links.test.ts`
- UI surfaces using guide links:
  `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
  `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`

## Background References

- https://www.kaggle.com/code/thomasanderson1962/public-synthetic-dataset-generation-w-internvl2
- https://www.youtube.com/watch?v=8zaKVFC9Eu4
- https://github.com/kerrickstaley/genanki
- https://darigovresearch.github.io/genanki/build/html/index.html
- https://github.com/patarapolw/anki-export
