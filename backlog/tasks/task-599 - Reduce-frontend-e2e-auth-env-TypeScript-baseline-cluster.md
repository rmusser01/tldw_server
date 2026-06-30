---
id: TASK-599
title: Reduce frontend e2e auth env TypeScript baseline cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-03 01:12'
labels:
  - typescript
  - frontend
  - tsc-baseline
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the frontend standalone tsc diagnostics caused by e2e auth helper environment typing while preserving explicit API key resolution behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend standalone tsc no longer reports the e2e/utils/e2e-auth.ts ProcessEnv diagnostics targeted by this task.
- [x] #2 Focused e2e auth behavior coverage passes if present, or the compiler check documents coverage for the narrow helper change.
- [x] #3 Bandit applicability is documented for the TypeScript-only change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification recorded for this tsc slice:
- RED: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/tldw-frontend reported 9 diagnostics, including 2 e2e/utils/e2e-auth.ts ProcessEnv assignability diagnostics.
- GREEN: bunx vitest run __tests__/e2e/e2e-auth.test.ts from apps/tldw-frontend passes 4 tests.
- GREEN: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/tldw-frontend now reports 7 remaining diagnostics, all in unrelated chat-cockpit/agent-tasks/admin-llamacpp clusters.
- git diff --check exits 0.
- Bandit not applicable: touched file is a TypeScript Playwright auth helper only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced the frontend standalone tsc baseline by removing the e2e-auth ProcessEnv cluster. The auth helper now accepts a process.env-compatible string map while preserving explicit API key lookup behavior, confirmed by focused Vitest coverage.
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
