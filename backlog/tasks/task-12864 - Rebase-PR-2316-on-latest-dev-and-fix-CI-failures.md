---
id: TASK-12864
title: Rebase PR 2316 on latest dev and fix CI failures
status: Done
references:
- https://github.com/rmusser01/tldw_server/pull/2316
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase GitHub PR #2316 onto the latest dev branch, investigate failed GitHub Actions checks, determine whether failures are already addressed on dev, and apply targeted fixes if needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Rebased PR #2316 branch onto origin/dev at 6eb33300f2c5cba62b5c9d31904e860f385941bd; merge-base now equals origin/dev.', 'Backend CI failures inspected from Actions logs were addressed by dev/rebase; targeted persona aliasing, chatbook manifest/integration, and Fish S2 tests pass locally.', 'Remaining UX Smoke failure reproduced locally as a Next runtime overlay in Knowledge QA. Fixed by allowing Knowledge QA chat creation without a default character, routing createChat/ragSearch through current WebUI config, and preserving stored single-user keys as runtime overrides before bootstrap scrubbing.', 'Onboarding E2E first-source desktop/mobile focused spec passes locally after rebase/fixes.']
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2316 branch onto latest origin/dev, resolved the tracked-character conflict, committed frontend fixes, and force-pushed with lease. Backend CI failures reviewed from the previous run were already addressed on dev and targeted backend tests passed locally. Remaining local UX Smoke failure was fixed by allowing Knowledge QA to create plain chats without a default character, routing createChat/ragSearch through the current WebUI config, and preserving stored single-user auth as an in-memory runtime override before scrubbing persisted config. Local targeted Vitest, Playwright smoke/onboarding gates, diff check, and Bandit frontend touched-scope check pass. Post-push GitHub checks were queued/pending with no failures when last checked.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
