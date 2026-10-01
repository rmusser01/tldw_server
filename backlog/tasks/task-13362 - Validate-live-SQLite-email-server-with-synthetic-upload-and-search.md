---
id: TASK-13362
title: Validate live SQLite email server with synthetic upload and search
status: Done
assignee: []
created_date: '2026-09-25 18:45'
updated_date: '2026-09-25 18:56'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the full local HTTP server and real startup/shutdown against isolated SQLite AuthNZ and per-user Media storage. Verify configured core email flags, synthetic upload, search/detail and media search over a loopback socket without Gmail, personal mail or model egress. Record exact results and residual limits before PostgreSQL validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A loopback HTTP server starts and stops cleanly with isolated SQLite storage and synthetic credentials.
- [x] #2 Authenticated synthetic email upload, search, detail and media search return scoped expected results with Gmail disabled.
- [x] #3 Evidence records flags, backend, failure modes and remaining production/scale limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Full FastAPI app via Uvicorn loopback, TEST_MODE off; isolated AuthNZ and per-user Media SQLite, synthetic users/orgs and quota. Final probe: health/ready 200, unauth 401, upload 200, Alice email/search/detail and media search one result, Bob search empty/detail 404, outbound/model attempts zero; server shutdown exit 0. Raw probe/log in /tmp. First 403 was probe omission of required org context; macOS /var path alias was probe-only. No Python code touched, so Bandit not applicable. Missing email_native_persist_total warning tracked as TASK-13363. Production/proxy/TLS/scale and PostgreSQL remain open.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated live full-app SQLite startup, authenticated synthetic email ingestion and owner-scoped search/detail/media search over loopback; documented exact flags, results and limits in Docs/Operations/Email_Live_SQLite_Validation_2026-09-25.md and release checklist. No production code changed. Missing persistence metric is TASK-13363.
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
