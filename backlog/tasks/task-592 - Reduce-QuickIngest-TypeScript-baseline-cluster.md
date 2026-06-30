---
id: TASK-592
title: Reduce QuickIngest TypeScript baseline cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 20:54'
labels:
  - typescript
  - webui
  - quick-ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce the rebased package-wide TypeScript baseline by fixing the QuickIngest FileDropZone acceptance test props and QuickIngest open-detail narrowing diagnostics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Package TypeScript diagnostics for QuickIngest FileDropZone disabled props are removed.
- [x] #2 QuickIngest open-detail url/firstSource/firstSourceKind diagnostics are removed without weakening runtime behavior.
- [x] #3 Relevant targeted checks or full package typecheck are run and recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed FileDropZone acceptance coverage to use the current running prop for disabled behavior. Narrowed playlist preflight seeds through isQuickIngestPlaylistPreflightDetail. Guarded first-source retry fallback with isFirstSourceQuickIngestKind. Full package tsc dropped from 20 src diagnostics after rebase to 12, with no QuickIngest diagnostics remaining. Bandit is not applicable for this JS/TS-only touched scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced the QuickIngest TypeScript baseline cluster by aligning stale test props with the current FileDropZone API and tightening QuickIngest open-detail narrowing. Verification: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exits with the remaining non-QuickIngest 12-diagnostic baseline; `bunx vitest run src/components/Common/QuickIngest/__tests__/FileDropZone.acceptance.test.tsx`; `bunx vitest run src/utils/__tests__/quick-ingest-open.test.ts`.
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
