---
id: TASK-512
title: Flashcards UX Phase 4 import recovery states
status: Done
labels:
- ux
- flashcards
- phase-4
- frontend
modified_files:
- Docs/superpowers/plans/2026-05-26-flashcards-ux-phase4-import-recovery-plan.md
- backlog/tasks/task-512 - Flashcards-UX-Phase-4-import-recovery-states.md
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next scoped flashcards UX remediation slice after Phase 3B: improve Create & Import result recovery states so import/generate/save flows clearly distinguish full success, partial success, and failure, with retry/recovery affordances and focused regression coverage. Scope is /flashcards Create & Import workflows only; defer extension capture, docs, and broad tab restructuring unless directly needed for recovery copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generated-card save success, partial success, and full failure render a persistent inline status in GeneratePanel.
- [x] #2 Partial save status explains that saved drafts were removed and failed drafts remain editable.
- [x] #3 Full save failure status explains that all drafts remain available and can be retried.
- [x] #4 A visible retry action reuses the existing save path for remaining generated drafts.
- [x] #5 Focused regression tests cover partial and full generated-card save recovery plus transfer summary consistency.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-26-flashcards-ux-phase4-import-recovery-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a persistent GeneratePanel save status for generated-card save outcomes, including success, partial success, and full failure.
- Partial generated saves now keep the existing failed-draft-only behavior while explaining that saved drafts were removed and failed drafts remain editable below.
- Full generated-save failures now keep all drafts visible and show an inline retry action.
- Migrated the existing GeneratePanel AntD Alert instances to the shared design-system Alert and removed the stale product-state baseline exceptions for that file.
- Addressed PR review feedback by catching fatal deck-resolution/save-path errors, surfacing them inline and in the transfer summary, clearing retryable status when remaining drafts are manually edited/removed, and explicitly disabling retry while a save is in progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 4 generated-save recovery is implemented. GeneratePanel now shows durable design-system status messaging for saved, partially saved, and failed generated-card saves, with retry for recoverable partial/full failures. The top-level transfer summary stays synchronized with the generated save result, existing generated save behavior still removes saved drafts while preserving failed drafts for editing, and review follow-ups now cover fatal save-path errors plus stale retry states after draft removal/editing. Verification: the new tests failed red first on the missing review fixes, then ImportExportTab import-results passed 26/26. The broader ImportExportTab suite, design-system product-state guard, and git diff --check were rerun after the review fixes. Bandit skipped because this slice only touched frontend TypeScript/TSX tests, JSON baseline, docs, and Backlog files.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
