---
id: TASK-45.44.10.2
title: Migrate DocumentViewer PDF and EPUB alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
- document-workspace
priority: medium
parent_task_id: TASK-45.44.10
references:
- https://github.com/rmusser01/tldw_server/pull/1952
- apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/EpubViewer/index.tsx
- apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/PdfViewer/PdfDocument.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
modified_files:
- apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/EpubViewer/index.tsx
- apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/PdfViewer/PdfDocument.tsx
- apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/__tests__/DocumentViewerAlerts.design-system.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Document and Workspace product-state design-system migration by replacing the remaining AntD Alert product-state surfaces in EpubViewer and PdfDocument with the shared design-system Alert primitive while preserving no-document-url and load-error behavior. Keep scope limited to viewer alerts, focused tests, and removal of matching baseline entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 EpubViewer no longer imports or renders AntD Alert for product-state no-url or load-error states.
- [x] #2 PdfDocument no longer imports or renders AntD Alert for product-state no-url or load-error states.
- [x] #3 EPUB no-document-url and load-error states render through the shared design-system Alert primitive.
- [x] #4 PDF no-document-url and load-error states render through the shared design-system Alert primitive.
- [x] #5 The four matching product-state baseline rows are removed without introducing unbaselined findings.
- [x] #6 Focused viewer tests and product-state verifier checks are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing focused tests that assert EPUB/PDF no-url and load-error states render with `data-ds-component="Alert"`.
2. Replace the EpubViewer and PdfDocument AntD Alert usages with the shared design-system Alert primitive while keeping viewer mechanics unchanged.
3. Remove the four migrated viewer Alert rows from the product-state baseline.
4. Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red test evidence: `bunx vitest run src/components/DocumentWorkspace/DocumentViewer/__tests__/DocumentViewerAlerts.design-system.test.tsx --reporter=dot` failed on the missing `data-ds-component="Alert"` ancestor for EPUB/PDF no-url and load-error states.
- Migrated only the DocumentViewer PDF/EPUB Alert surfaces to the shared design-system Alert primitive. Spin/loading behavior, PDF rendering, EPUB setup, search, and selection mechanics were left unchanged.
- Removed four baseline rows for `EpubViewer/index.tsx` and `PdfViewer/PdfDocument.tsx`, reducing the Document/Workspace product-state baseline count from 9 to 5.
- Verification:
  - `bunx vitest run src/components/DocumentWorkspace/DocumentViewer/__tests__/DocumentViewerAlerts.design-system.test.tsx --reporter=dot` passed: 4 tests.
  - `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed: 54 tests.
  - `node -e "JSON.parse(require('fs').readFileSync('apps/packages/ui/scripts/design-system-product-state-baseline.json','utf8')); console.log('baseline json ok')"` passed.
  - `bun run verify:design-system-state` passed with 296 baseline exceptions total and 5 remaining Document/Workspace exceptions.
  - `git diff --check` passed.
  - `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on existing repo-wide UI type debt outside the touched DocumentViewer files.
- Bandit is not applicable to this UI-only TypeScript/React slice; no Python files were touched.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the DocumentViewer EPUB/PDF warning and load-error Alert surfaces from AntD Alert to the shared design-system Alert primitive, added focused regression coverage for all four states, and removed the four matching baseline entries. PR: https://github.com/rmusser01/tldw_server/pull/1952.
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
