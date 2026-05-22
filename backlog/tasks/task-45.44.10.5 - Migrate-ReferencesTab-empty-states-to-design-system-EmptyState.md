---
id: TASK-45.44.10.5
title: Migrate ReferencesTab empty states to design-system EmptyState
status: Done
labels:
- design-system
- webui
- product-state
- document-workspace
priority: medium
parent_task_id: TASK-45.44.10
references:
- https://github.com/rmusser01/tldw_server/pull/1959
- apps/packages/ui/src/components/DocumentWorkspace/LeftSidebar/ReferencesTab.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
modified_files:
- apps/packages/ui/src/components/DocumentWorkspace/LeftSidebar/ReferencesTab.tsx
- apps/packages/ui/src/components/DocumentWorkspace/LeftSidebar/__tests__/ReferencesTab.design-system-empty.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Document and Workspace product-state design-system migration by replacing the remaining AntD Empty product-state surfaces in ReferencesTab with the shared design-system EmptyState primitive. Keep scope limited to the two ReferencesTab empty states, focused tests, and removal of matching baseline entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ReferencesTab no longer renders AntD Empty for the server-unavailable product state.
- [x] #2 ReferencesTab no longer renders AntD Empty for the reference-load error product state.
- [x] #3 Both migrated states render through the shared design-system EmptyState primitive.
- [x] #4 The two matching ReferencesTab product-state baseline rows are removed without introducing unbaselined findings.
- [x] #5 Focused ReferencesTab tests and product-state verifier checks are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing focused tests that assert ReferencesTab server-unavailable and generic error empty states render through the shared design-system EmptyState marker.
2. Replace the ReferencesTab AntD Empty product-state usages with the shared design-system EmptyState primitive while preserving copy.
3. Remove the two migrated ReferencesTab Empty rows from the product-state baseline.
4. Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red test evidence: `bunx vitest run src/components/DocumentWorkspace/LeftSidebar/__tests__/ReferencesTab.design-system-empty.test.tsx --reporter=dot` failed on missing `data-ds-component="EmptyState"` ancestors for both states before the component migration.
- Migrated only the guarded ReferencesTab server-unavailable and reference-load error Empty surfaces to the shared design-system EmptyState primitive. The no-references and filtered-empty states remain out of scope for this product-state baseline slice.
- Removed two baseline rows for `ReferencesTab.tsx`, reducing total product-state baseline exceptions from 294 to 292 and Document/Workspace exceptions from 3 to 1.
- Verification:
  - `bunx vitest run src/components/DocumentWorkspace/LeftSidebar/__tests__/ReferencesTab.design-system-empty.test.tsx --reporter=dot` passed: 2 tests.
  - `bunx vitest run src/components/DocumentWorkspace/LeftSidebar/__tests__/ReferencesTab.test.tsx --reporter=dot` passed: 3 tests.
  - `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed: 54 tests.
  - `node -e "JSON.parse(require('fs').readFileSync('apps/packages/ui/scripts/design-system-product-state-baseline.json','utf8')); console.log('baseline json ok')"` passed.
  - `bun run verify:design-system-state` passed with 292 baseline exceptions total and 1 remaining Document/Workspace exception.
  - `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on existing repo-wide UI type debt outside the touched ReferencesTab files; `/tmp/tldw_refs_tab_tsc.log` contains no `ReferencesTab` or `design-system-empty` matches.
- Bandit is not applicable to this UI-only TypeScript/React slice; no Python files were touched.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the ReferencesTab server-unavailable and reference-load error states from AntD Empty to the shared design-system EmptyState primitive, added focused regression coverage for both states, and removed the two matching baseline entries. PR: https://github.com/rmusser01/tldw_server/pull/1959.

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
