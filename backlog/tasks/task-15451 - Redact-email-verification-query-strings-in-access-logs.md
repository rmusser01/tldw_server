---
id: TASK-15451
title: Redact email verification query strings in access logs
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Protect the verification token in the Uvicorn request target while retaining the route, HTTP version and status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The verification query is redacted with and without a trailing slash.
- [x] #2 Neighboring routes and existing redaction targets retain their behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Extends the existing access-log redaction contract. Plan: reproduce the formatted Uvicorn record, extend the sensitive-query matcher and run adjacent tests plus Bandit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Uvicorn request targets for email verification now drop the full query. Both route variants exposed the synthetic token before the fix. All 8 logging tests, Ruff and diff checks passed. Bandit reported zero runtime findings. No public API or configuration changes.
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
