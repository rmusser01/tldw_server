---
id: TASK-12119
title: Rebase PR 2589 onto latest dev
status: Done
labels:
- pr-rebase
references:
- https://github.com/rmusser01/tldw_server/pull/2589
modified_files:
- apps/tldw-frontend/e2e/smoke/stage6-interaction-stage1.spec.ts
- backlog/tasks/task-12117 - Fix-PR-2571-release-CI-failures.md
- backlog/tasks/task-12119 - Rebase-PR-2589-onto-latest-dev.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve merge conflicts on PR #2589 (`fix/ux-smoke-theme-toggle-route`) and rebase it onto the latest `dev` branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2589 is rebased onto latest `dev`. Conflict resolution kept `dev`'s existing runtime-auth shell fix/unit coverage and preserved the smoke test change that checks the theme toggle on `/` instead of `/chat`. Verification: `bunx vitest run __tests__/app/app-layout.test.tsx` passed 17/17, and `npx playwright test e2e/smoke/stage6-interaction-stage1.spec.ts --reporter=line --grep "home route exposes"` passed 1/1. Bandit skipped: touched code is frontend TypeScript/test markdown only, no Python scope.
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
