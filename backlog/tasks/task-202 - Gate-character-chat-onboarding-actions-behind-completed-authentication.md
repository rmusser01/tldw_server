---
id: TASK-202
title: Gate character-chat onboarding actions behind completed authentication
status: Done
assignee: []
created_date: '2026-05-09 22:56'
updated_date: '2026-05-09 23:02'
labels:
  - webui
  - ux
  - character-chat
  - auth
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure route-aware character-chat onboarding actions are only visible after the user has entered credentials and the onboarding connection flow has completed authentication and authorization. The first-run credential/setup screen may remain route-aware in title/copy, but it must not expose character-chat actions before auth succeeds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A first-run user with character-chat route intent does not see the character-chat onboarding action lane before completing credentials and connection validation.
- [x] #2 After the onboarding connection flow reaches its authenticated success state with character-chat route intent, the character-chat action lane remains visible and actionable.
- [x] #3 The route intent return path still sends character-chat users to the intended destination after completing onboarding.
- [x] #4 Focused frontend tests cover the pre-auth negative case and the post-auth success case.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation: removed the disabled character-chat action lane from the unauthenticated first-run OptionIndex shell; kept route-aware title/copy; left the active lane in the OnboardingConnectForm authenticated success screen. Also fixed returnTo route recognition so /characters?... query paths preserve their existing source context when adding create/import action flags.

Verification: bunx vitest run src/routes/__tests__/core-route-identity.test.tsx src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.tsx src/utils/__tests__/onboarding-route-intent.test.ts --maxWorkers=1 --no-file-parallelism passed 15/15; git diff --check passed; Chromium browser check of first-run character-chat route after readiness timeout reported laneCount=0 headingCount=1 credentialFields=2 and saved output/playwright/character-chat-first-run-pre-auth-no-lane-dismissed.png.

Bandit: skipped because touched implementation/test files are TypeScript frontend files, not Python.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Gated the character-chat onboarding actions behind completed onboarding auth by removing the pre-auth disabled lane from the first-run route shell while preserving route-aware setup copy. Added regression coverage for the pre-auth negative case, preserved the post-auth success action lane, and fixed character return-path parsing so /characters query routes continue to their original context after auth.
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
