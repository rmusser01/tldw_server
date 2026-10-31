---
id: TASK-13364
title: Validate live PostgreSQL email server with synthetic upload and search
status: Done
assignee: []
created_date: '2026-09-25 19:02'
updated_date: '2026-09-25 19:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After SQLite validation and metric fix, run the full local HTTP server with isolated PostgreSQL AuthNZ and content storage. Use synthetic users/orgs and EML, verify startup, authenticated upload, email operator search/detail, media search, tenant isolation, and clean shutdown. Keep Gmail/model egress disabled. Record exact backend and RLS evidence plus limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Loopback full app starts and stops with isolated PostgreSQL AuthNZ/content databases and non-test settings
- [x] #2 Synthetic authenticated upload/search/detail/media search succeeds and cross-tenant access is denied with effective PostgreSQL RLS
- [x] #3 Evidence records configuration, tests, failure modes, and remaining release/scale limits
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Using existing local postgres:18 test service at loopback port 5434, with separate temporary AuthNZ/content databases and a non-superuser role. Full app runs outside test mode; a /tmp probe blocks non-loopback/model calls. Plan: IMPLEMENTATION_PLAN_email_live_postgres_13364.md.

Full Uvicorn loopback probe outside test mode passed with separate PostgreSQL AuthNZ/content databases, synthetic two-org upload/search/detail/media search, forced RLS Alice 1/Bob 0, zero outbound/model attempts, clean shutdown. Report: Docs/Operations/Email_Live_PostgreSQL_Validation_2026-09-25.md. 27 focused unit tests passed; Bandit 0 findings; fatal Ruff clean. Limits: one worker/message; scale, TLS/reverse proxy, production parity open.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Full Uvicorn loopback probe outside test mode passed with separate PostgreSQL AuthNZ/content databases, synthetic two-org upload/search/detail/media search, forced RLS Alice 1/Bob 0, zero outbound/model attempts, clean shutdown. Report: Docs/Operations/Email_Live_PostgreSQL_Validation_2026-09-25.md. 27 focused unit tests passed; Bandit 0 findings; fatal Ruff clean. Limits: one worker/message; scale, TLS/reverse proxy, production parity open.
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
