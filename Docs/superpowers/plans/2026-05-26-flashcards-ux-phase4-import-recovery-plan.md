# Flashcards UX Phase 4 Import Recovery Plan

## Stage 4A: Generated Save Recovery State
**Goal**: Make generated-card save outcomes persist in the Create & Import screen so users can recover from partial or failed saves without relying on transient toast copy.
**Success Criteria**:
- Generated-card save success, partial success, and full failure produce an inline status in the Generate panel.
- Partial success clearly says successful drafts were saved and only failed drafts remain editable.
- Full failure clearly says all drafts are still available and can be retried.
- A visible retry action reuses the existing save path for remaining drafts.
- Focused regression tests cover partial and failed generated-card save recovery.
**Tests**:
- `bunx vitest run src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx`
- `bun run verify:design-system-state`
- `git diff --check`
**Status**: Complete

## Stage 4B: Transfer Summary Consistency Check
**Goal**: Ensure the top-level transfer summary reflects generated save recovery without adding more primary CTAs to the already dense Create & Import surface.
**Success Criteria**:
- Last action remains synchronized with generated save success, warning, and error outcomes.
- Transfer summary stays free of top-level primary CTAs.
- Snapshot or focused assertions cover the summary state.
**Tests**:
- Covered by `ImportExportTab.import-results.test.tsx`.
**Status**: Complete

## Non-Goals
- Do not split Create & Import into new route-level tabs in this slice.
- Do not change backend import/generate APIs unless a current client contract is proven insufficient.
- Do not add extension capture or documentation updates in this slice.
