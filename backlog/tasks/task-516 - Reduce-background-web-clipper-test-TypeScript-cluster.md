---
id: TASK-516
title: Reduce background web clipper test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:18'
labels: []
dependencies: []
references:
  - TASK-515
  - apps/packages/ui/src/entries/__tests__/background.web-clipper.test.ts
  - apps/packages/ui/src/entries/background.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/entries/__tests__/background.web-clipper.test.ts`. Current package `tsc` output reports two fixture object errors where context-menu click info includes stale `pageTitle` properties that are now sourced from the tab object.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current background web clipper compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to stale test fixture shape rather than behavior changes.
- [x] #3 The `background.web-clipper.test.ts` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused background web clipper test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task515-tsc-final.txt`: package `tsc` reported two diagnostics in `src/entries/__tests__/background.web-clipper.test.ts` because context-menu click-info fixtures included `pageTitle`, which is not part of `WebClipperContextMenuClickInfo`.
- Root cause was stale test fixture shape. `launchWebClipperFromContextMenu` now derives `pageTitle` from the tab object, while click info only contains `pageUrl` and optional `selectionText`.
- Removed the stale `pageTitle` properties from the click-info arguments and left the tab title fixtures intact, preserving the behavior under test.
- Focused verification: `bunx vitest run src/entries/__tests__/background.web-clipper.test.ts` passed: 4 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task516-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 67 in `/tmp/task515-tsc-final.txt` to 65 in `/tmp/task516-tsc-final.txt`; `rg -n 'background\.web-clipper\.test\.ts' /tmp/task516-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `background.web-clipper.test.ts` TypeScript cluster by deleting stale `pageTitle` fields from context-menu click-info fixtures while keeping titles on the tab fixture where the implementation reads them. Focused Vitest passed with 4 tests, and package `tsc` baseline dropped from 67 to 65 with no remaining background web-clipper diagnostics.
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
