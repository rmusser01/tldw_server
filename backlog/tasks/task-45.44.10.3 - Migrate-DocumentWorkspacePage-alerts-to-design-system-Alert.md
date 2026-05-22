---
id: TASK-45.44.10.3
title: Migrate DocumentWorkspacePage alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
- document-workspace
priority: medium
parent_task_id: TASK-45.44.10
references:
- apps/packages/ui/src/components/DocumentWorkspace/DocumentWorkspacePage.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
modified_files:
- apps/packages/ui/src/components/DocumentWorkspace/DocumentWorkspacePage.tsx
- apps/packages/ui/src/components/DocumentWorkspace/__tests__/DocumentWorkspacePageAlerts.design-system.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Document and Workspace product-state design-system migration by replacing the remaining AntD Alert product-state surfaces in DocumentWorkspacePage with the shared design-system Alert primitive. Keep scope limited to the page-level Alert states, focused tests, and removal of matching baseline entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DocumentWorkspacePage no longer imports or renders AntD Alert for the page-level loading document state.
- [x] #2 DocumentWorkspacePage no longer imports or renders AntD Alert for the page-level workspace storage health state.
- [x] #3 Loading and health states render through the shared design-system Alert primitive.
- [x] #4 The two matching DocumentWorkspacePage product-state baseline rows are removed without introducing unbaselined findings.
- [x] #5 Focused page tests and product-state verifier checks are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing focused tests that assert the page-level loading and health states render through `data-ds-component="Alert"`.
2. Replace the DocumentWorkspacePage AntD Alert usages with the shared design-system Alert primitive while preserving layout and copy.
3. Remove the two migrated DocumentWorkspacePage Alert rows from the product-state baseline.
4. Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red test evidence: `bunx vitest run src/components/DocumentWorkspace/__tests__/DocumentWorkspacePageAlerts.design-system.test.tsx --reporter=dot` failed on missing `data-ds-component="Alert"` ancestors for both page-level states.
- Migrated only the loading document and workspace storage health Alert surfaces to the shared design-system Alert primitive. Layout, copy, auto-open mechanics, and health issue construction were left unchanged.
- Removed two baseline rows for `DocumentWorkspacePage.tsx`, reducing the Document/Workspace product-state baseline count from 5 to 3.
- Verification:
  - `bunx vitest run src/components/DocumentWorkspace/__tests__/DocumentWorkspacePageAlerts.design-system.test.tsx --reporter=dot` passed: 2 tests.
  - `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed: 54 tests.
  - `node -e "JSON.parse(require('fs').readFileSync('apps/packages/ui/scripts/design-system-product-state-baseline.json','utf8')); console.log('baseline json ok')"` passed.
  - `bun run verify:design-system-state` passed with 294 baseline exceptions total and 3 remaining Document/Workspace exceptions.
  - `git diff --check` passed.
  - `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on existing repo-wide UI type debt outside the current page Alert migration.
- Bandit is not applicable to this UI-only TypeScript/React slice; no Python files were touched.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the DocumentWorkspacePage loading and workspace storage health Alert states from AntD Alert to the shared design-system Alert primitive, added focused regression coverage for both states, and removed the two matching baseline entries.
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
