---
id: TASK-13255
title: Validate authenticated email search and per-user SQLite isolation
status: Done
created_date: 2026-09-13 19:39
documentation:
- Docs/Operations/Email_Core_Validation_2026-09-13.md
updated_date: 2026-09-13 19:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue core release validation after TASK-13251. Exercise production authentication and per-user DB dependencies with synthetic credentials/data in isolated SQLite storage; keep Gmail and model calls disabled. Test auth rejection, own-message search/detail, cross-user isolation and cursor scope. Preserve deployment/full-scale/native PST/PostgreSQL gates as unverified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unauthenticated and invalid-credential requests cannot read email search or message detail
- [x] #2 Two synthetic authenticated users see only their own persisted messages and cannot reuse each other's cursors
- [x] #3 Email feature flags and media-search compatibility are validated through production route dependencies where feasible
- [x] #4 Tests intercept outbound/model calls and docs accurately state remaining startup/deployment limitations
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect authentication/SQLite fixture patterns and production route registration. 2. Add synthetic integration coverage using real credential validation and real per-user database dependency; investigate/fix reproduced scoped defects. 3. Run related regressions, Bandit, independent review, update evidence/task, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ten real-authentication integration cases. No auth/Media DB dependency overrides, genuine AuthNZ user/key creation, per-user temporary SQLite, missing/invalid/revoked keys, single-user configured key, concurrent/interleaved users, foreign IDs/cursors, feature flags, media-search bridge. Main-app case exercises real route registration and test-mode middleware without lifespan. Combined65 passed/8 existing warnings/zero outbound attempts. Ruff check/format pass; Bandit0 with B101 excluded. Independent review no blockers, nine focused cases passed. Main app test mode omits security headers/HTTP metrics/usage/access logging; deployment/startup/JWT/upload quotas/PostgreSQL/full scale remain unverified. Evidence: Docs/Operations/Email_Authenticated_Validation_2026-09-13.md. Completed local plan removed per repo policy.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added and verified real API-key authentication/per-user SQLite isolation coverage for email core. Confirmed authorization failures, isolated retrieval and cursor handling, revocation, single-user access, flags and compatibility search. Updated PRD/release checklist/runbook with evidence and precise middleware/startup limits. No production behavior changes, personal mailbox access, live Gmail, or external model requests.
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
