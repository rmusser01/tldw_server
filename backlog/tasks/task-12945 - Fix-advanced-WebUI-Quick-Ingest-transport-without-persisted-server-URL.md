---
id: TASK-12945
title: Fix advanced WebUI Quick Ingest transport without persisted server URL
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-10 04:59'
labels:
  - frontend
  - quick-ingest
  - bug
  - uat
dependencies: []
references:
  - 'PR #2702 release UAT'
  - 'https://github.com/rmusser01/tldw_server/pull/2702'
priority: high
ordinal: 12105
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT on release 0.1.40 found that Quick Ingest direct uploads fail before reaching the backend when advanced WebUI mode is configured via NEXT_PUBLIC_API_URL but tldwConfig.serverUrl is unset. Fix the shared request guard, correct the misleading configuration error classification, and add focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Direct browser upload requests use the resolved advanced NEXT_PUBLIC_API_URL origin when persisted serverUrl is unset.
- [x] #2 Advanced requests still fail closed when neither persisted nor runtime API origin resolves to a valid HTTP origin.
- [x] #3 A missing server configuration is not presented as an unsupported file format.
- [x] #4 Focused frontend tests and typecheck pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Full-stack CDP UAT completed after the patched WebUI and isolated backend were launched on ports 18180 and 18000. With localStorage.tldwConfig removed immediately before submission, Quick preset sent an authenticated POST to http://127.0.0.1:18000/api/v1/media/ingest/jobs and received batch 05dd45df-9788-4618-91e3-63328861aade / job 1. The WebUI's own polling completed with 1 succeeded, 0 failed in 15 seconds. SQLite verification found media ID 1, title 'Me at the zoo', exact YouTube URL, type video, and 210 content characters. Screenshot: /tmp/quick-ingest-advanced-transport-uat.png. The initial attempt to remove every connection-related storage key was intentionally discarded because it destabilized unrelated connection state; the authoritative regression check removes only tldwConfig, which is the config source evaluated by the request guard, while retaining runtime/bootstrap mirrors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed advanced WebUI requests so a valid resolved NEXT_PUBLIC_API_URL transport is accepted when tldwConfig.serverUrl is absent, while retaining fail-closed behavior for invalid or missing origins. Added focused request-core and Quick Ingest error-classification regression tests, and classified missing server configuration as an auth/configuration issue instead of an unsupported file format. Verified with focused Vitest, frontend typecheck, touched-file ESLint, git diff checks, and full-stack CDP UAT of a real YouTube Quick Ingest that stored 'Me at the zoo' with transcript content.
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
