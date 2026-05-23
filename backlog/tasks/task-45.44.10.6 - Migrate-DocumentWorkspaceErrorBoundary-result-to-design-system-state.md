---
id: TASK-45.44.10.6
title: Migrate DocumentWorkspaceErrorBoundary result to design-system state
status: In Progress
labels:
- design-system
- webui
- product-state
- document-workspace
priority: medium
parent_task_id: TASK-45.44.10
references:
- apps/packages/ui/src/components/DocumentWorkspace/DocumentWorkspaceErrorBoundary.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
modified_files:
- apps/packages/ui/src/components/DocumentWorkspace/DocumentWorkspaceErrorBoundary.tsx
- apps/packages/ui/src/components/DocumentWorkspace/__tests__/DocumentWorkspaceErrorBoundary.design-system.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Document and Workspace product-state design-system migration by replacing the final AntD Result product-state surface in DocumentWorkspaceErrorBoundary with the shared design-system EmptyState primitive. Keep scope limited to the error-boundary fallback, focused tests, and removal of the matching baseline entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DocumentWorkspaceErrorBoundary no longer imports or renders AntD Result for the default recovery fallback.
- [x] #2 The default recovery fallback renders through the shared design-system EmptyState primitive.
- [x] #3 The fallback preserves title, description, development error details, warning icon tone, and Try again behavior.
- [x] #4 The matching DocumentWorkspaceErrorBoundary product-state baseline row is removed without introducing unbaselined findings.
- [x] #5 Focused error-boundary tests and product-state verifier checks are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing test that trips DocumentWorkspaceErrorBoundary and asserts the fallback renders through a design-system product-state primitive instead of AntD Result.
2. Replace the AntD Result fallback with the shared design-system EmptyState primitive while preserving title, description, development details, warning icon tone, and retry action.
3. Remove the migrated DocumentWorkspaceErrorBoundary Result row from the product-state baseline.
4. Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red test evidence: `bunx vitest run src/components/DocumentWorkspace/__tests__/DocumentWorkspaceErrorBoundary.design-system.test.tsx --reporter=dot` failed on missing `data-ds-component="EmptyState"` for the default fallback before the component migration.
- Migrated only the default DocumentWorkspaceErrorBoundary fallback from AntD Result to the shared design-system EmptyState primitive. The custom `fallback` prop path remains unchanged.
- Removed one baseline row for `DocumentWorkspaceErrorBoundary.tsx`, targeting the final current Document/Workspace product-state baseline exception.
- Verification:
  - `bunx vitest run src/components/DocumentWorkspace/__tests__/DocumentWorkspaceErrorBoundary.design-system.test.tsx --reporter=dot` passed: 1 test.
  - `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed: 54 tests.
  - `node -e "JSON.parse(require('fs').readFileSync('apps/packages/ui/scripts/design-system-product-state-baseline.json','utf8')); console.log('baseline json ok')"` passed.
  - `bun run verify:design-system-state` passed with 291 baseline exceptions total and no remaining Document/Workspace product-area bucket.
  - `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on existing repo-wide UI type debt outside the touched DocumentWorkspaceErrorBoundary files; `/tmp/tldw_doc_workspace_error_boundary_tsc.log` contains no `DocumentWorkspaceErrorBoundary` or `ErrorBoundary.design-system` matches.
  - `git diff --check` passed.
- Bandit is not applicable to this UI-only TypeScript/React slice; no Python files were touched.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
