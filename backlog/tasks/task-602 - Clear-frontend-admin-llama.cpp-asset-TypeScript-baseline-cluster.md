---
id: TASK-602
title: Clear frontend admin llama.cpp asset TypeScript baseline cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-03 01:16'
labels:
  - typescript
  - frontend
  - tsc-baseline
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the remaining frontend standalone tsc diagnostics in the admin llama.cpp E2E spec by typing mocked asset responses with the shared llama.cpp admin contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend standalone tsc exits clean after the admin llama.cpp asset mock cluster is fixed.
- [x] #2 Admin llama.cpp mocked assets use the shared UI llama.cpp admin asset/response types instead of ad hoc inferred unions.
- [x] #3 Bandit applicability is documented for the TypeScript-only test change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification recorded for this tsc slice:
- RED: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/tldw-frontend reported 5 diagnostics, all in admin-llamacpp.spec.ts asset metadata/response inference.
- GREEN: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/tldw-frontend exits 0.
- The mocked llama.cpp admin assets and mutable assets response now use the shared LlamacppAsset and LlamacppAssetsResponse types consumed by the UI.
- git diff --check exits 0.
- Bandit not applicable: touched file is a TypeScript Playwright spec only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cleared the remaining frontend standalone TypeScript baseline by typing admin llama.cpp mocked assets with the shared UI llama.cpp admin contracts. The full apps/tldw-frontend tsc check now exits clean with NODE_OPTIONS=--max-old-space-size=8192.
<!-- SECTION:FINAL_SUMMARY:END -->

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
