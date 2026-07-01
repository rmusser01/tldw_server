---
id: TASK-12080
title: Fix Explainer endpoint auth dependency boundary
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-01 00:08'
labels:
  - ci
  - auth
  - explainer
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The full-suite core-utils-tooling CI shard fails because the Explainer endpoint imports common auth dependency symbols directly from core.AuthNZ instead of the API dependency boundary module. Route the endpoint through the approved auth_deps exports so the lint boundary guard passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Explainer endpoint imports common auth dependency symbols from tldw_Server_API.app.api.v1.API_Deps.auth_deps instead of core.AuthNZ.User_DB_Handling.
- [x] #2 The endpoint auth dependency boundary lint test passes locally.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the Explainer endpoint and API auth dependency exports. - Complete
2. Replace direct core AuthNZ imports with the approved auth_deps import path. - Complete
3. Run the focused lint test and relevant sanity checks. - Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the Explainer endpoint to import `User` and `get_request_user` from `tldw_Server_API.app.api.v1.API_Deps.auth_deps` instead of directly from `core.AuthNZ.User_DB_Handling`, matching the endpoint auth dependency boundary used elsewhere. Verified with the focused lint test, Explainer endpoint tests, and Bandit on the touched endpoint.
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
