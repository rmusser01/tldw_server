---
id: TASK-12076
title: Add WebUI AUTH_MODE to local env example
status: Done
labels:
- webui
- auth
- config
priority: High
modified_files:
- apps/tldw-frontend/.env.local.example
- apps/tldw-frontend/README.md
- backlog/tasks/task-12076 - Add-WebUI-AUTH-MODE-to-local-env-example.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persist the local WebUI single-user runtime auth setting in the checked-in .env.local.example so new local dev users do not hit runtime-config auth-mode failures when connecting to a single-user backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added AUTH_MODE=single_user to apps/tldw-frontend/.env.local.example with comments tying it to /api/_tldw-webui/runtime-config, and added AUTH_MODE to the WebUI README key variables list so local single-user WebUI dev setup docs match the template. Verification: git diff --check passed; rg confirmed AUTH_MODE=single_user in the template and AUTH_MODE in the README. Bandit skipped for the env template and README because they are non-code documentation/config changes. PR: https://github.com/rmusser01/tldw_server/pull/2561.
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
