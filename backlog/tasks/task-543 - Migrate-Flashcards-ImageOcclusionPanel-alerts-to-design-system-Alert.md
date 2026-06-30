---
id: TASK-543
title: Migrate Flashcards ImageOcclusionPanel alerts to design-system Alert
status: Done
labels:
- flashcards
- design-system
- webui
priority: medium
modified_files:
- apps/packages/ui/src/components/Flashcards/tabs/ImageOcclusionPanel.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImageOcclusionPanel.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the remaining AntD informational Alert notices in the Flashcards ImageOcclusionPanel authoring workflow with the shared design-system Alert primitive, preserving the upload/region authoring behavior and removing the migrated baseline exceptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused ImageOcclusionPanel test proves the empty image and empty regions notices render through data-ds-component="Alert".
- [x] #2 Existing image upload, region drawing, label editing, and selection behavior remains covered.
- [x] #3 ImageOcclusionPanel Alert baseline exceptions are removed.
- [x] #4 Focused Flashcards ImageOcclusionPanel verification and known product-state verifier caveats are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the existing ImageOcclusionPanel test baseline on current dev. 2. Add focused failing assertions that both empty image/empty region notices render with data-ds-component="Alert". 3. Replace the AntD informational alerts with the shared design-system Alert primitive while preserving translated copy and panel behavior. 4. Remove the two migrated ImageOcclusionPanel baseline exceptions. 5. Run focused tests, product-state checks, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red/green completed. Added focused coverage for the two empty authoring notices: the initial "Choose an image" state and the uploaded-image/no-regions state. The RED run failed because the existing AntD Alert did not expose `data-ds-component="Alert"`. Replaced only those two informational notices with the shared design-system Alert primitive using `variant="info"`, preserving translated title copy and authoring flow. Removed the two Flashcards ImageOcclusionPanel Alert baseline exceptions. PR review follow-up: switched the design-system Alert import/usages from the temporary `DsAlert` alias to direct `Alert` now that AntD Alert is no longer imported in this file. Verification: baseline ImageOcclusionPanel Vitest passed before implementation; focused RED failed for the missing design-system marker; focused GREEN passed; full ImageOcclusionPanel Vitest passes 2/2; exact baseline grep shows no ImageOcclusionPanel Alert exception remains. `bun run verify:design-system-state` still exits 1 on unrelated Integrations/Writing/Notes/Research product-state findings and stale Integrations baseline entries, while Flashcards exceptions dropped from 32 to 30 and ImageOcclusionPanel no longer appears. Bandit skipped because this slice only touches frontend TSX, baseline JSON, and Backlog markdown.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the Flashcards ImageOcclusionPanel empty-state informational notices from AntD Alert to the shared design-system Alert primitive using the direct `Alert` import requested in PR review. Added focused test coverage for both notices and removed the migrated ImageOcclusionPanel Alert product-state baseline exceptions. Verification: ImageOcclusionPanel Vitest passes 2/2; exact baseline grep has no ImageOcclusionPanel Alert match; broader product-state verifier still fails on unrelated baseline findings outside this Flashcards slice. Bandit skipped because no Python files changed.
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
