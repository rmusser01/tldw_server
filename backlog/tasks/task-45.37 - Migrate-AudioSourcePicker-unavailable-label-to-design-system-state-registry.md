---
id: TASK-45.37
title: Migrate AudioSourcePicker unavailable label to design-system state registry
status: Done
assignee:
  - Codex
created_date: '2026-05-10 01:37'
updated_date: '2026-05-10 01:40'
labels:
  - design-system
  - ui
  - product-state
  - audio
dependencies: []
references:
  - apps/packages/ui/src/components/Common/AudioSourcePicker.tsx
  - apps/packages/ui/src/components/Common/__tests__/AudioSourcePicker.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Route AudioSourcePicker's remembered missing-device unavailable label through the canonical design-system state registry instead of leaving the hardcoded product-state label as a baseline exception. Preserve the existing missing-device option text and source fallback message behavior while removing the matching product-state baseline entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AudioSourcePicker uses the design-system state registry for the unavailable fallback label in remembered missing-device options while preserving existing option text structure and i18n fallback behavior.
- [x] #2 Focused AudioSourcePicker tests prove the unavailable label comes from the design-system registry and still show the source fallback message.
- [x] #3 The AudioSourcePicker canonical-state-label baseline entry is removed and the design-system product-state verifier passes.
- [x] #4 Verification notes include focused Vitest coverage, product-state guard coverage, product-state verifier, diff check, TypeScript touched-file status, and Bandit applicability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused AudioSourcePicker regression test by partially mocking @/design-system so the unavailable state label is distinct from the hardcoded string, then verify the test fails before production changes. 2. Update AudioSourcePicker to use getDesignSystemState("unavailable").label as the i18n fallback for the remembered missing-device unavailable label. 3. Remove the matching AudioSourcePicker canonical-state-label baseline exception. 4. Run focused AudioSourcePicker tests, product-state guard tests, bun run verify:design-system-state, git diff --check, and a touched-file TypeScript check; record Bandit as skipped if the touched scope remains UI-only TypeScript/JSON/Backlog.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation notes: Added RED coverage by partially mocking @/design-system so getDesignSystemState("unavailable") returns a distinct label. The focused AudioSourcePicker test first failed because the missing-device option still rendered the hardcoded Unavailable label. Updated AudioSourcePicker to use getDesignSystemState("unavailable").label as the i18n fallback for remembered missing-device unavailable labels, preserving option structure and source fallback messaging. Removed canonical-state-label:src/components/Common/AudioSourcePicker.tsx:Unavailable from the product-state baseline. Verification: focused AudioSourcePicker Vitest passed 6 tests; product-state guard Vitest passed 52 tests; bun run verify:design-system-state exited 0 with 509 baseline exceptions; git diff --check passed; repo-wide bunx tsc --noEmit --pretty false exited 2 on existing unrelated UI TypeScript debt, and rg found no touched-file/design-system matches in the tsc output. Bandit skipped because the touched scope is UI TypeScript, JSON, and Backlog markdown only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated AudioSourcePicker's remembered missing-device unavailable fallback label to the design-system state registry and removed the corresponding baseline exception. Added focused test coverage proving the unavailable label comes from the registry while the source fallback message remains visible. Verification passed for focused component coverage, product-state guard coverage, the design-system state verifier, and diff whitespace; repo-wide TypeScript still has unrelated existing failures with no touched-file matches. Bandit was not applicable to this UI-only TS/JSON/markdown slice.
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
