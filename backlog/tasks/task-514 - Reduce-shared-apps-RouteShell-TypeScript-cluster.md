---
id: TASK-514
title: Reduce shared apps RouteShell TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:10'
labels: []
dependencies: []
references:
  - TASK-513
  - apps/packages/ui/src/entries/shared/apps.tsx
  - apps/packages/ui/src/routes/sidepanel-route-shell.tsx
  - apps/packages/ui/src/routes/options-route-shell.tsx
  - apps/packages/ui/src/routes/app-route.tsx
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/entries/shared/apps.tsx`. Current package `tsc` output reports two errors where raw `RouteShell` calls omit the now-required route registries for sidepanel and options entrypoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current shared apps RouteShell compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to RouteShell prop requirements rather than behavior changes.
- [x] #3 The `entries/shared/apps.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused entrypoint test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task513-tsc-final.txt`: package `tsc` reported two diagnostics in `src/entries/shared/apps.tsx` because raw `RouteShell` calls for sidepanel and options omitted the required `routes` prop.
- Root cause was a RouteShell API mismatch. Existing `SidepanelRouteShell` and `OptionsRouteShell` wrappers already provide the correct route registries, and `OptionsRouteShell` preserves deferred unmatched-route handling.
- Replaced the raw `RouteShell kind="sidepanel"` and `RouteShell kind="options"` usages with `SidepanelRouteShell` and `OptionsRouteShell`, matching the newer split entrypoint files.
- Focused test search found no dedicated `entries/shared/apps.tsx` test; package `tsc` was used for verification.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task514-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 71 in `/tmp/task513-tsc-final.txt` to 69 in `/tmp/task514-tsc-final.txt`; `rg -n 'entries/shared/apps\.tsx' /tmp/task514-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `entries/shared/apps.tsx` TypeScript cluster by using the existing `SidepanelRouteShell` and `OptionsRouteShell` wrappers instead of raw `RouteShell` calls without route registries. Package `tsc` baseline dropped from 71 to 69 with no remaining shared apps diagnostics. No focused entrypoint test exists; package `tsc` was used for verification.
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
