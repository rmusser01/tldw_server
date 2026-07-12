---
id: TASK-12107
title: Fix stale onboarding validation mocks
status: Done
priority: Medium
modified_files:
- apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update OnboardingConnectForm test mocks to provide the exported connectivity classifier introduced by the onboarding validation refactor, restoring the focused baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The onboarding design-system state wiring test no longer errors on a missing isConnectivityErrorKind export.
- [x] #2 The matching success-screen mock stays compatible with the validation module contract.
- [x] #3 Focused onboarding tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Confirmed the validation module gained an exported isConnectivityErrorKind helper while two full-module mocks retained the old export shape. Converted both mocks to partial mocks using vi.importActual so pure validation helpers remain available and only external-auth functions are replaced.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored onboarding test compatibility with the validation module contract. Verification: focused Vitest run passed 8/8 tests. Bandit is not applicable because this prerequisite changes test-only TypeScript.
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
