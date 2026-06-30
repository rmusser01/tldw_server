---
id: TASK-502
title: Reduce OnboardingConnectForm TypeScript design-system test cluster
status: Done
references:
- TASK-501
- apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx
- apps/packages/ui/tsconfig.json
modified_files:
- apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx
- backlog/tasks/task-502 - Reduce-OnboardingConnectForm-TypeScript-design-system-test-cluster.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained OnboardingConnectForm design-system test cluster. Current package `tsc` output reports six errors in `src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx`, all around unknown mock prop inspection values.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current OnboardingConnectForm compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test mock prop typing rather than production behavior.
- [x] #3 The `OnboardingConnectForm.design-system.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused behavior test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the OnboardingConnectForm test and current compiler diagnostics to identify the mock prop typing root cause.
2. Use current package `tsc` output as red evidence for the six-error cluster.
3. Make the smallest test-only typing changes needed to preserve behavior and remove the cluster.
4. Run the focused OnboardingConnectForm design-system test, then package `bunx tsc --noEmit --pretty false` and record remaining baseline counts.
5. Record Bandit decision and final evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red compiler evidence came from `/tmp/task501-tsc-after-composer-queue.txt`, which contained six `OnboardingConnectForm.design-system.test.tsx` diagnostics around `unknown` props being assigned to intrinsic input/select values and invoked as callbacks.
- Root cause was test-only AntD mock typing: `Button`, `Input`, `Input.Password`, and `Select` accepted `[key: string]: unknown` and then spread the rest object into native JSX elements. Production `OnboardingConnectForm` behavior was not changed.
- Added explicit React DOM attribute-based mock prop aliases and destructured AntD-only props before spreading the remaining DOM-safe props into native elements.
- Focused test: `bunx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx` from `apps/packages/ui` passed 4/4.
- Package compiler capture: `bunx tsc --noEmit --pretty false > /tmp/task502-tsc-final.txt 2>&1` from `apps/packages/ui` still exits 2 for the known baseline, but `error TS` lines reduced from 110 to 104 and `rg -n 'OnboardingConnectForm' /tmp/task502-tsc-final.txt` returned no matches.
- Bandit skipped: this is a TypeScript test-only change and Bandit is a Python security scanner; no Python touched scope exists for this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the six-error `OnboardingConnectForm.design-system.test.tsx` package `tsc` cluster by narrowing local AntD mock prop types instead of spreading `unknown` values into intrinsic JSX elements. The shared UI baseline is now 104 `error TS` lines after this slice.
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
