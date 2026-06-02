---
id: TASK-512
title: 'Task 6: Cover onboarding setup to first chat UAT paths'
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-02 04:51
labels:
- onboarding-uat
- playwright
- test
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add hosted and local OpenAI-compatible setup-to-first-chat Playwright UAT specs that run through the real WebUI, real backend, and repo mock OpenAI server without route-mocking provider behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hosted setup-to-first-chat spec uses firstRunPage, real backend setup, mock OpenAI response assertion, and desktop/mobile coverage
- [x] #2 Local OpenAI-compatible setup-to-first-chat path is first-class when supported and explicitly recorded when current UI cannot expose it yet
- [x] #3 Specs avoid page.route provider mocks, seedAuth, setup-completion storage flags, and waitForTimeout
- [x] #4 Runner-focused verification or current product blocker evidence is recorded
- [x] #5 Vitest guard/lint/static checks and Bandit skip for non-Python touched scope are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented setup-to-first-chat UAT coverage with real WebUI/backend/mock OpenAI server. Hosted path captures setup-open, setup-connected, and first-chat-success screenshots plus JSON step artifacts. Local OpenAI-compatible path remains explicitly skipped unless TLDW_ONBOARDING_UAT_LOCAL_SUPPORTED=1 because the current UI does not expose that peer path yet. Fixed harness blockers found during real UAT: UI package antd dev dependency resolution, runner log write race by waiting for child close, setup helper false-positive completion, route-aware post-connect readiness, avoiding redundant /chat reloads, and narrow onboarding diagnostic filtering for route-transition model fetch aborts plus expected chat-settings 404. Verification: real desktop UAT passed at apps/tldw-frontend/test-results/onboarding-uat/2026-06-02T06-31-09-167Z-8iep7n and real mobile UAT passed at apps/tldw-frontend/test-results/onboarding-uat/2026-06-02T06-41-50-103Z-t6w3gu; both summaries reported status passed and response text 'onboarding UAT ready. The mock provider returned a deterministic success response.' Also ran bunx vitest run __tests__/e2e-harness-readiness.guard.test.ts, bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts, bunx eslint e2e/onboarding-uat __tests__/e2e-harness-readiness.guard.test.ts scripts/__tests__/onboarding-uat-runner.test.ts, bunx tsc --noEmit over onboarding UAT files, and git diff --check. Bandit skipped because this task touched JS/TS/package metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added hosted setup-to-first-chat UAT coverage against the real WebUI, real backend, and repo mock OpenAI server. Passed desktop and mobile runs capture setup-open, setup-connected, and first-chat-success screenshots/JSON plus backend/frontend/mock/browser logs. The local OpenAI-compatible path is represented as an explicit gated skip until the current UI exposes the peer setup path.
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
