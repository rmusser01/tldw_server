---
id: TASK-12943
title: Fix stale webapp session redirect after relaunch
status: Done
assignee: []
created_date: '2026-07-10 01:41'
updated_date: '2026-07-10 03:32'
labels:
  - frontend
  - auth
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the webapp relaunch path where a stale multi-user session can bypass /login and render the protected homepage without header/sidebar chrome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stale multi-user access tokens redirect protected routes to /login after validation fails.
- [x] #2 Transient non-auth validation failures preserve persisted authenticated shell state.
- [x] #3 Hosted multi-user sessions validate without requiring a locally persisted access token.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression coverage for stale token, transient validation failure, and hosted tokenless session cases.
2. Update app shell auth resolution to distinguish auth failures from transient validation errors.
3. Verify focused frontend auth/layout tests and lint touched files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification:
- bunx vitest run __tests__/app/app-layout.test.tsx (red before production fix: 3 expected failures; green after fix: 21/21 passed)
- ./node_modules/.bin/eslint pages/_app.tsx lib/configured-auth-state.ts __tests__/app/app-layout.test.tsx
- ./node_modules/.bin/tsc --noEmit --pretty false
- git diff --check -- touched files

Bandit: skipped because the touched implementation and tests are TypeScript/React frontend files only; no Python code changed.

PR review follow-up: replaced silent logout rejection handling with logged try/catch, classified plain-object Not authenticated errors, and added coverage for optional logout, logout rejection, and hosted tokenless transient failures. Verification: app-layout Vitest 25/25, touched frontend ESLint, TypeScript noEmit, and git diff --check.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed multi-user app-shell auth resolution after relaunch. Explicit auth validation failures now clear stale auth and redirect protected routes to /login, while transient validation failures preserve the persisted authenticated shell. Hosted deployments now validate sessions without requiring a locally stored access token.

PR review feedback is addressed with explicit logout diagnostics and broader auth-error compatibility.
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
