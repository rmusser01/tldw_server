---
id: TASK-45.44.5.1
title: Migrate RecipesTab alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
priority: medium
parent_task_id: TASK-45.44.5
references:
- https://github.com/rmusser01/tldw_server/issues/1662
- https://github.com/rmusser01/tldw_server/pull/1825
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
modified_files:
- apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the Evaluations RecipesTab product-state AntD Alert usage with the shared design-system Alert primitive, keep recipe copy/actions intact, and remove the matching product-state guard baseline entries for this file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] RecipesTab product-state alerts render through the shared design-system Alert primitive instead of AntD Alert.
- [x] Product-state guard baseline entries for RecipesTab are removed without increasing other baseline counts.
- [x] Focused RecipesTab tests and design-system product-state verification pass, with broader TypeScript debt checked for touched-path regressions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Migrated all RecipesTab AntD Alert product-state messages to `Alert` from `@/components/ui/primitives/Alert`.
- Added focused regression coverage that asserts the recipe-load failure state is rendered by the shared design-system Alert primitive.
- Removed 14 RecipesTab entries from the product-state baseline, reducing the total baseline count from 399 to 385.
- PR review follow-up: changed the launch-readiness reuse hint child rendering to idiomatic `&&` conditional rendering.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Evaluations RecipesTab Alert migration in PR #1825 and removed its product-state baseline exceptions. Verification: RecipesTab focused Vitest passed (24 tests); product-state guard Vitest passed (52 tests); `bun run verify:design-system-state` passed with 385 remaining baseline exceptions and zero RecipesTab entries; `git diff --check` passed; baseline JSON parses. `bunx tsc --noEmit --pretty false` still fails on existing repo-wide TypeScript debt, but the post-fix scan has no matches for the touched RecipesTab files or baseline/task metadata. Bandit is not applicable because this slice touches TypeScript, JSON, and Backlog metadata only.
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
