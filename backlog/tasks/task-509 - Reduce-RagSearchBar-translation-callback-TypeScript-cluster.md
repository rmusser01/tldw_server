---
id: TASK-509
title: Reduce RagSearchBar translation callback TypeScript cluster
status: Done
references:
- TASK-508
- apps/packages/ui/src/components/Sidepanel/Chat/RagSearchBar.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/hooks/useRagSearchState.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/hooks/useRagResultsDisplay.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/hooks/useRagFilterPanel.tsx
- apps/packages/ui/tsconfig.json
modified_files:
- apps/packages/ui/src/components/Sidepanel/Chat/RagSearchBar.tsx
- backlog/tasks/task-509 - Reduce-RagSearchBar-translation-callback-TypeScript-cluster.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained RagSearchBar translation callback typing cluster. Current package `tsc` output reports three errors in `src/components/Sidepanel/Chat/RagSearchBar.tsx` because the i18next `TFunction` overload type is passed directly to helpers that expect a simple `(key, fallback?) => string` callback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current RagSearchBar compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to helper callback typing rather than translation behavior.
- [x] #3 The `RagSearchBar.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused behavior test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red compiler evidence came from `/tmp/task508-tsc-final.txt`, which contained three `RagSearchBar.tsx` diagnostics where the overloaded i18next `TFunction` was passed directly into helper APIs typed as `(key: string, fallback?: string) => string`.
- Root cause was helper callback type mismatch, not translation behavior. `RagSearchBar` can still use `t` directly for JSX translations.
- Added a local `translateFallback` adapter that delegates to `t(key, fallback ?? key)` and passed that narrow function to `getRagSourceOptions`, `useRagSearchState`, `useRagResultsDisplay`, and `useRagFilterPanel`.
- Focused behavior test blocker: no `RagSearchBar` component or hook-focused test currently exists under `apps/packages/ui/src/components/Sidepanel/Chat`; package `tsc` was used as the targeted verification for this type-only adapter.
- Package compiler capture: `bunx tsc --noEmit --pretty false > /tmp/task509-tsc-final.txt 2>&1` from `apps/packages/ui` still exits 2 for the known baseline, but `error TS` lines reduced from 85 to 82 and `rg -n 'RagSearchBar' /tmp/task509-tsc-final.txt` returned no matches.
- Bandit skipped: this is a TypeScript-only change and Bandit is a Python security scanner; no Python touched scope exists for this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the three-error `RagSearchBar.tsx` package `tsc` cluster by adapting the overloaded i18next translator before passing it to helper APIs with a narrow fallback callback type. The shared UI baseline is now 82 `error TS` lines after this slice.
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
