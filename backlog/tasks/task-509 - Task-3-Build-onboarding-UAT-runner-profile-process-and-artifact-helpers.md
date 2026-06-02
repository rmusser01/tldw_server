---
id: TASK-509
title: 'Task 3: Build onboarding UAT runner profile process and artifact helpers'
status: Done
labels:
- onboarding-uat
- runner
- test
priority: medium
modified_files:
- apps/tldw-frontend/scripts/onboarding-uat/ports.mjs
- apps/tldw-frontend/scripts/onboarding-uat/processes.mjs
- apps/tldw-frontend/scripts/onboarding-uat/profile.mjs
- apps/tldw-frontend/scripts/onboarding-uat/artifacts.mjs
- apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the onboarding UAT helper layer for port reservation, process management, isolated runtime profiles, artifact paths, redaction, cleanup, and focused helper tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ports.mjs reserves distinct loopback ports with a named map helper
- [x] #2 artifacts.mjs creates run artifact paths, redacts synthetic secrets, scans artifacts for leaks, and supports cleanup
- [x] #3 profile.mjs creates an isolated temp runtime profile from repo config and builds backend env without referencing developer .env
- [x] #4 processes.mjs starts logged child processes, waits for HTTP readiness, and stops process trees with redacted logs
- [x] #5 Focused Vitest tests cover redaction, artifact setup/cleanup, profile/env generation, port reservation, and command/process helper behavior
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented onboarding UAT runner helper modules and extended the focused Vitest suite. Quality-review follow-up hardened backend env allowlisting to avoid passing developer provider secrets, added local-ingest fixture roots to config and env, guarded artifact cleanup with an onboarding marker/root check, expanded artifact secret leak scanning for assignment, JSON-shaped object/header-array diagnostics, raw token, and private-key forms, and made process cleanup wait for exit after SIGTERM/SIGKILL races. Verification: `bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts` from `apps/tldw-frontend` passed with 19 tests; `git diff --check` from repo root passed. Bandit skipped: no Python touched by Task 3.
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
