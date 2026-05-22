---
id: TASK-45.44.10.4
title: Fix PdfDocument design-system Alert translation binding
status: Done
labels:
- design-system
- webui
- typescript
- document-workspace
priority: medium
parent_task_id: TASK-45.44.10
references:
- https://github.com/rmusser01/tldw_server/pull/1955
- apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/PdfViewer/PdfDocument.tsx
- apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/__tests__/DocumentViewerAlerts.design-system.test.tsx
modified_files:
- apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/PdfViewer/PdfDocument.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the missing translation function binding in PdfDocument introduced by the previous design-system Alert migration so the merged dev branch no longer has a direct cannot-find-name TypeScript error in the DocumentViewer alert code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PdfDocument binds the existing `useTranslation` hook before calling `t(...)` in design-system Alert content.
- [x] #2 Focused DocumentViewer alert regression coverage still passes.
- [x] #3 TypeScript check no longer reports the direct `PdfDocument.tsx` cannot-find-name `t` errors.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Use the existing react-i18next import by binding `t` inside PdfDocument before rendered Alert states call it.
2. Run the focused DocumentViewer alert regression test and TypeScript check evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red evidence: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` reported `src/components/DocumentWorkspace/DocumentViewer/PdfViewer/PdfDocument.tsx(416,12): error TS2304: Cannot find name 't'.` and the same error at line 455.
- Bound `const { t } = useTranslation(["option"])` inside `PdfDocument`, using the existing import.
- Verification:
  - `bunx vitest run src/components/DocumentWorkspace/DocumentViewer/__tests__/DocumentViewerAlerts.design-system.test.tsx --reporter=dot` passed: 4 tests.
  - `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on existing repo-wide UI type debt, but the direct `PdfDocument.tsx` missing-`t` errors are no longer present.
- Bandit is not applicable to this UI-only TypeScript/React fix; no Python files were touched.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the merged PdfDocument design-system Alert translation binding by calling `useTranslation` in the component before using `t(...)`. PR: https://github.com/rmusser01/tldw_server/pull/1955.
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
