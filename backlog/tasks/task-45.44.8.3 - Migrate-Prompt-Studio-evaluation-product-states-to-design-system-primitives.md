---
id: TASK-45.44.8.3
title: Migrate Prompt Studio evaluation product states to design-system primitives
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
Migrate the remaining Prompt Studio evaluation product-state exceptions in CreateEvaluationWizard and EvaluationDetailPanel from AntD Alert/Tag to design-system Alert/Badge primitives, remove the matching baseline entries, and add focused coverage proving the migrated design-system markers render.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `CreateEvaluationWizard` uses the design-system `Alert` primitive for wizard guidance/review state instead of AntD `Alert`.
- [x] #2 `EvaluationDetailPanel` uses the design-system `Badge` primitive for evaluation status labels instead of AntD `Tag`.
- [x] #3 Focused Prompt Studio evaluation tests assert the migrated design-system markers render.
- [x] #4 The product-state baseline removes only the migrated Prompt Studio evaluation exceptions, and `verify:design-system-state` passes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `EvaluationDesignSystem.test.tsx` with red/green coverage for evaluation wizard guidance alerts and detail-panel status badges.
- Replaced four AntD evaluation wizard `Alert` instances with the design-system `Alert` primitive.
- Replaced `EvaluationDetailPanel` status `Tag` rendering with design-system `Badge` variants while preserving existing status labels and icons.
- Removed five baseline entries for the migrated Prompt Studio evaluation product-state exceptions.
- TDD red run reached the intended assertions after dependency install: `EvaluationDesignSystem.test.tsx` failed because content was still rendered under AntD `Alert`/`Tag` instead of DS markers.
- Verification before final rebase: focused evaluation test passed; evaluation + optimization Prompt Studio DS tests passed; `bun run verify:design-system-state` passed with this slice reducing Prompt/Prompt Studio exceptions from 21 to 16; `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed.
- Verification after rebasing onto latest `origin/dev`: evaluation + optimization Prompt Studio DS tests passed; `bun run verify:design-system-state` passed with total baseline exceptions at 105 and Prompt/Prompt Studio at 16; `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed.
- Bandit not applicable: touched code is frontend TypeScript/TSX, JSON baseline, and Backlog task metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the Prompt Studio evaluation wizard and evaluation detail status labels from AntD product-state primitives to design-system primitives. Added focused render coverage and removed the matching five baseline exceptions.
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
