---
id: TASK-12945
title: Fix advanced WebUI Quick Ingest transport without persisted server URL
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-10 05:20'
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
Full-stack CDP UAT remains valid: with localStorage.tldwConfig removed immediately before submission, Quick preset sent an authenticated POST to the runtime API origin; batch 05dd45df-9788-4618-91e3-63328861aade / job 1 completed with 1 succeeded and 0 failed, and SQLite stored media ID 1 (Me at the zoo) with 210 content characters. PR #2703 review follow-up addressed all three inline findings: narrowly matched client configuration errors, deduplicated test fixtures, and centralized foreground/background advanced transport validation. Verification: 35 request/classification tests and 63 networking/background-proxy tests passed; frontend typecheck passed; touched-file ESLint had 0 errors with existing warnings only; git diff --check passed. Bandit remains not applicable because only TypeScript, tests, and Backlog metadata changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed advanced WebUI requests so a valid resolved NEXT_PUBLIC_API_URL transport is accepted when tldwConfig.serverUrl is absent while invalid or missing origins remain fail-closed. Configuration errors are narrowly classified without capturing unrelated backend messages. Review follow-up centralized advanced transport origin/validation for request-core and background-proxy and reduced test fixture duplication. Focused tests, background-proxy suites, typecheck, lint, diff checks, and full-stack YouTube ingestion UAT passed.
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
