---
id: TASK-529
title: Reduce prompt filter presets test TypeScript diagnostic
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 20:07'
labels: []
dependencies: []
references:
  - TASK-528
  - >-
    apps/packages/ui/src/components/Option/Prompt/__tests__/useFilterPresets.test.tsx
  - apps/packages/ui/src/components/Option/Prompt/useFilterPresets.ts
  - apps/packages/ui/src/components/Option/Prompt/prompt-workspace-types.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostic in `src/components/Option/Prompt/__tests__/useFilterPresets.test.tsx`. Current package `tsc` output reports a stale `savedView: "grid"` literal that is no longer part of `PromptSavedView`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current prompt filter presets compiler diagnostic is captured.
- [x] #2 Root cause is documented and tied to stale test saved-view fixture data rather than behavior changes.
- [x] #3 The `useFilterPresets.test.tsx` diagnostic is removed from package `tsc` output.
- [x] #4 Focused prompt filter presets test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task528-tsc-final.txt`: package `tsc` reported one diagnostic in `src/components/Option/Prompt/__tests__/useFilterPresets.test.tsx` where `savedView: "grid"` was passed to the typed `savePreset` filters object.
- Root cause was stale test fixture data only. `PromptSavedView` now accepts saved filter views such as `"all"`, while the test assertion only covers recovery and persistence behavior.
- Updated the typed save-preset fixture to use `savedView: "all"`; raw localStorage fixtures were left unchanged because they exercise normalization of persisted data.
- Focused verification: `bunx vitest run src/components/Option/Prompt/__tests__/useFilterPresets.test.tsx` passed: 3 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task529-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 40 in `/tmp/task528-tsc-final.txt` to 39 in `/tmp/task529-tsc-final.txt`; searching for `useFilterPresets.test.tsx` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `useFilterPresets.test.tsx` TypeScript diagnostic by updating the typed save-preset fixture from stale `savedView: "grid"` to the current `PromptSavedView` value `"all"`. Focused Vitest passed with 3 tests, and package `tsc` baseline dropped from 40 to 39 with no remaining prompt filter presets diagnostic.
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
