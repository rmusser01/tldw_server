---
id: TASK-601
title: Reduce frontend agent tasks fixture TypeScript baseline cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-03 01:15'
labels:
  - typescript
  - frontend
  - tsc-baseline
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the frontend standalone tsc diagnostic in the agent tasks workflow by preserving discriminated-union narrowing across async cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend standalone tsc no longer reports the agent-tasks.spec.ts fixture.reason diagnostic targeted by this task.
- [x] #2 The skipped-fixture path still cleans up before calling test.skip with the original reason.
- [x] #3 Bandit applicability is documented for the TypeScript-only test change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification recorded for this tsc slice:
- RED: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/tldw-frontend reported 6 diagnostics, including the agent-tasks.spec.ts fixture.reason diagnostic.
- First attempt with !fixture.created still failed because this compiler did not narrow the boolean discriminant for fixture.reason; changed the guard to fixture.created === false.
- GREEN: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/tldw-frontend now reports 5 remaining diagnostics, all in the unrelated admin-llamacpp cluster.
- The skipped-fixture path still stores the skip reason, runs cleanup, then calls test.skip with the original reason.
- git diff --check exits 0.
- Bandit not applicable: touched file is a TypeScript Playwright spec only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced the frontend standalone tsc baseline by removing the agent-tasks fixture.reason diagnostic. The workflow now uses an explicit false discriminant guard and snapshots the skip reason before awaited cleanup.
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
