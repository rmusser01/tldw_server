---
id: TASK-540
title: Migrate Flashcards ManageTab bulk delete warning to design-system Alert
status: Done
labels:
- flashcards
- design-system
- webui
priority: medium
modified_files:
- apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the remaining AntD Alert in the Flashcards ManageTab large bulk-delete confirmation workflow with the shared design-system Alert primitive, preserving copy and destructive-action behavior while removing the migrated product-state baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused ManageTab test proves the large bulk-delete warning renders through data-ds-component="Alert".
- [x] #2 ManageTab bulk-delete warning copy, icon semantics, and confirmation flow remain unchanged for users.
- [x] #3 The ManageTab Alert baseline exception is removed and the design-system product-state verifier reports no stale ManageTab Alert exception.
- [x] #4 Focused Flashcards ManageTab verification and git diff hygiene are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current ManageTab bulk-delete warning and nearby ManageTab tests on fresh origin/dev. 2. Add a focused failing assertion for the large bulk-delete warning design-system Alert marker. 3. Replace the AntD Alert callout with the design-system Alert primitive without changing visible copy or modal flow. 4. Remove the ManageTab Alert baseline exception. 5. Run focused tests, product-state verification, and diff checks; document results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red/green completed. Added focused coverage for the large bulk-delete path that selects all matching cards through the ManageTab select-all-across affordance, opens the type-to-confirm delete modal, and asserts the warning copy has a `data-ds-component="Alert"` ancestor. The RED run failed because the AntD Alert had no design-system marker. Replaced only that warning callout with the shared design-system Alert primitive using `variant="warning"`, preserving the translated title copy and modal flow. Removed the Flashcards ManageTab Alert baseline exception. Verification: focused ManageTab undo-stage3 Vitest passes 8/8; `git diff --check` passes; exact baseline grep shows no Flashcards ManageTab Alert exception remains. `bun run verify:design-system-state` still exits 1 on unrelated Integrations/Writing/Notes/Research product-state findings and stale Integrations baseline entries, with no Flashcards ManageTab finding. Bandit skipped because this slice only touches frontend TypeScript/TSX, baseline JSON, and Backlog markdown.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the Flashcards ManageTab large bulk-delete warning from AntD Alert to the shared design-system Alert primitive. Focused TDD coverage now verifies the >100-card bulk-delete modal warning renders with `data-ds-component="Alert"` through the select-all-across flow. Removed the Flashcards ManageTab Alert baseline exception. Verification: focused ManageTab undo-stage3 Vitest passes 8/8; `git diff --check` passes; product-state verifier still fails on unrelated Integrations/Writing/Notes/Research baseline findings, with no Flashcards ManageTab finding. Bandit skipped because no Python files changed.
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
