---
id: TASK-15650
title: 'Session cap: second-review minors'
status: Done
labels:
- authnz
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nine minor issues a second review found in the merged per-user session cap (#3615).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 M1 cleanup removes logged-out rows about a day after access expiry without touching cap-row retention
- [x] #2 M2 ADR states the inline blacklist limit and its effect
- [x] #3 M3 ADR records the listing expires_at change and the unchanged admin/monitoring counters
- [x] #4 M4 a failed sign-in step revokes the placeholder session row
- [x] #5 M5 token update refuses a non-active row on both backends and the caller surfaces SessionError
- [x] #6 M6 session-replaced notice is recorded only when invalidation applied, and cleared on logout
- [x] #7 M7 refresh error details are sanitized and session_replaced still reaches its reader
- [x] #8 M8 docs say a cookie browser signs in again on the next page load
- [x] #9 M9 create_session_record_with_cap pins READ COMMITTED on PostgreSQL
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
M1 to M9 implemented with tests; see the PR description. M8 changes wording only (cookie browsers sign in again on the next page load); bootstrap untouched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
