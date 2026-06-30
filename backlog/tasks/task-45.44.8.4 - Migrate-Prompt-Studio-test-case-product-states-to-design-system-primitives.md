---
id: TASK-45.44.8.4
title: Migrate Prompt Studio test case product states to design-system primitives
status: Done
labels:
- design-system
- webui
- product-state
- prompt-studio
parent_task_id: TASK-45.44.8
references:
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- backlog/tasks/task-45.44.8 - Migrate-design-system-product-state-Prompt-and-Prompt-Studio.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate Prompt Studio TestCases product-state exceptions in TestCaseBulkPanel, TestCaseGenerateModal, and TestCaseRunModal from AntD Alert/Tag to design-system Alert/Badge primitives, remove the matching baseline entries, and add focused coverage proving the migrated design-system markers render.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `TestCaseBulkPanel` uses the design-system `Alert` primitive for import/export guidance instead of AntD `Alert`.
- [x] #2 `TestCaseGenerateModal` and `TestCaseRunModal` use the design-system `Alert` primitive for guidance instead of AntD `Alert`.
- [x] #3 `TestCaseRunModal` uses design-system `Badge` primitives for run summaries and row statuses instead of AntD `Tag`.
- [x] #4 Focused Prompt Studio test-case tests assert the migrated design-system markers render.
- [x] #5 The product-state baseline removes only the migrated Prompt Studio test-case exceptions, and `verify:design-system-state` passes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `TestCaseDesignSystem.test.tsx` with red/green coverage for bulk import/export guidance, AI generation guidance, run guidance, and test-run summary/row status labels.
- Replaced AntD `Alert` usage in `TestCaseBulkPanel`, `TestCaseGenerateModal`, and `TestCaseRunModal` with the design-system `Alert` primitive.
- Replaced `TestCaseRunModal` AntD `Tag` status labels with design-system `Badge` variants while preserving existing pass/fail/error/run labels and icons.
- Removed five baseline entries for the migrated Prompt Studio test-case product-state exceptions.
- TDD red run reached the intended assertions after dependency install: `TestCaseDesignSystem.test.tsx` failed because content was still rendered under AntD `Alert`/`Tag` instead of DS markers.
- Review fix: tightened `TestCaseDesignSystem` badge assertions so summary badges are matched exactly and row-status badges are scoped to the rendered results table, preventing `Pass`/`Fail` checks from being satisfied by `passed`/`failed` summary text.
- Verification after rebase on latest `origin/dev`: focused test-case DS test passed; test-case + evaluation + optimization Prompt Studio DS tests passed; `bun run verify:design-system-state` passed with total baseline exceptions at 97 and Prompt/Prompt Studio at 11; `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed.
- Bandit not applicable: touched code is frontend TypeScript/TSX, JSON baseline, and Backlog task metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Prompt Studio test-case guidance and run-result state labels from AntD product-state primitives to design-system primitives. Added focused render coverage and removed the matching five baseline exceptions.
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
