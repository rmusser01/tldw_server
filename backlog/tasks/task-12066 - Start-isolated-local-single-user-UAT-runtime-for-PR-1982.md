---
id: TASK-12066
title: Start isolated local single-user UAT runtime for PR 1982
status: Done
labels:
- uat
- release
- pr-1982
references:
- https://github.com/rmusser01/tldw_server/pull/1982
modified_files:
- Docs/Product/WebUI/evidence/pre_main_uat/pre-main-uat-20260629054510/local-single-user.md
- apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts
- apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts
- /tmp/tldw-pre-main-uat/pre-main-uat-20260629054510/local-runtime/*
- /tmp/tldw-pre-main-uat/pre-main-uat-20260629054510/uat.env
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare run-scoped local config/database paths, start the FastAPI backend and Next.js WebUI for the pre-main UAT run, and record local runtime health evidence without using existing user data or repo config files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Started an isolated single-user UAT backend/WebUI runtime for PR 1982, verified loopback API/WebUI availability, and documented the evidence. UAT found a real quickstart runtime-auth bootstrap bug: the WebUI runtime-config route rejected loopback forwarding metadata emitted by Next dev, leaving the browser unauthenticated with 401s. Fixed the route to allow only loopback-only Forwarded/x-forwarded-for metadata, ignore x-forwarded-host for the exposure decision, and continue rejecting external/empty forwarded client IP values and x-real-ip. Verified with focused Vitest coverage and a browser check showing first-time setup without auth/readiness console errors. Bandit not applicable because the product fix is TypeScript-only.
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
