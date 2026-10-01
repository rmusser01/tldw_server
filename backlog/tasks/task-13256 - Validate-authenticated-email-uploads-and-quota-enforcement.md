---
id: TASK-13256
title: Validate authenticated email uploads and quota enforcement
status: Done
assignee: []
created_date: '2026-09-13 20:03'
updated_date: '2026-09-25 16:59'
labels: []
dependencies: []
documentation:
  - Docs/Operations/Email_Authenticated_Upload_Validation_2026-09-25.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-13255 with synthetic uploads through the production media/add router and real AuthNZ/per-user SQLite dependencies. Exercise denied credentials, permissions, expected-user checks and organization quota behavior, fix proven defects, and record exact validation limits. No Gmail or model access.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real authenticated email upload persists and searches only in the owning user database
- [x] #2 Denied credentials, permissions, expected-user mismatches and exhausted organization storage prevent persistence
- [x] #3 Proven defects have regression coverage; tests, lint, Bandit and review pass
- [x] #4 Evidence records billing behavior and remaining deployment and scale limits
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Staged validation plan executed; final evidence is in Docs/Operations/Email_Authenticated_Upload_Validation_2026-09-25.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Synthetic offline ASGI/SQLite validation: 124 affected tests passed with zero outbound attempts (combined_final3 log); 27 child/persistence tests passed; two-org team assertion passed. Ruff clean on touched modules except 24 pre-existing persistence diagnostics versus 25 at base. Bandit zero findings on touched production and test code. Independent reviewer found no remaining actionable issues. PostgreSQL, deployed startup, JWT login, PST/OST, million-message scale and live Gmail remain outside this local validation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated authenticated synthetic email upload, organization quota and billing selection, scoped API keys, active-org JWT behavior, two-org dedupe/search isolation, team-scope clearing and sanitized 503 quota outages. Fixed production defects with regression coverage and documented deployment limits.
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
