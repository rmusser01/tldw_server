---
id: TASK-45.44.6.3
title: Migrate IntegrationPolicyPanel alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-30 07:30'
labels:
  - design-system
  - webui
  - product-state
  - settings
dependencies: []
references:
  - TASK-45.44.6
  - 'https://github.com/rmusser01/tldw_server/issues/1663'
  - >-
    apps/packages/ui/src/components/Option/Integrations/IntegrationPolicyPanel.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the four IntegrationPolicyPanel AntD Alert product-state callouts to the shared design-system Alert primitive while preserving pairing policy, workspace policy, success, warning, and error copy/visibility behavior. Remove the matching baseline exceptions and verify the Settings/account-security count moves down by this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace policy error and unavailable callouts render with the shared design-system Alert primitive.
- [x] #2 Telegram policy error and pairing-code success callouts render with the shared design-system Alert primitive.
- [x] #3 IntegrationPolicyPanel has no remaining AntD Alert product-state baseline exceptions.
- [x] #4 Focused integrations tests and scoped product-state guard verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect IntegrationPolicyPanel and existing tests to identify the alert states and available render harness.
2. Add focused failing test assertions that the workspace and Telegram policy alert copy renders inside the design-system Alert container.
3. Replace AntD Alert with the shared Alert primitive while preserving existing title/body copy and conditional rendering.
4. Remove the four matching IntegrationPolicyPanel baseline entries and run focused tests, scoped product-state guard, TypeScript or relevant package check, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the four IntegrationPolicyPanel AntD Alert callouts with the shared design-system Alert primitive while preserving workspace policy error/warning copy and Telegram error/success pairing copy.
- Added focused design-system assertions for workspace policy error/unavailable states and Telegram error/pairing-code states.
- Removed the four IntegrationPolicyPanel baseline exceptions; the path now has zero product-state baseline entries.
- While running the existing integrations page suite, fixed the Telegram linked-actors dependency failure to render as a partial/degraded workspace-integrations state, matching the existing test and UX intent that the rest of the page remains usable.
- Verification: initial design-system alert test run failed as expected against AntD Alert containers, then passed after migration.
- Verification passed: bun run test src/components/Option/Integrations/__tests__/IntegrationPolicyPanel.design-system-alert.test.tsx --maxWorkers=1 --no-file-parallelism.
- Verification passed: bun run test src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx --maxWorkers=1 --no-file-parallelism.
- Verification passed: scoped product-state guard for IntegrationPolicyPanel.tsx and IntegrationManagementPage.tsx with baseline filtered to those paths reported no product-state guard issues.
- Verification passed: env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false.
- Verification passed: git diff --check.
- Known skip/blocker: package-wide bun run verify:design-system-state still fails on unrelated existing drift outside this slice, including WritingPlayground/WritingActionBar.tsx, Notes unavailable labels, and ResearchWorkspace canonical-state labels. It did not report IntegrationPolicyPanel after this migration.
- Bandit was not run because the touched implementation scope is TypeScript/JSON/Backlog only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation complete locally. IntegrationPolicyPanel no longer imports AntD Alert, its policy/pairing callouts render through the shared DS Alert primitive, and the component-specific baseline exceptions were removed. The adjacent Telegram linked-actors failure path now classifies as a degraded partial workspace-integrations state instead of a generic error.
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
