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
M1 to M9 implemented with tests. M8 changes wording only (cookie browsers sign in again on the next page load); bootstrap untouched. A review follow-up moved update_user_last_login inside the revoke guard (token-issuing helper and magic-link handler) and made logout clear the session-replaced notice after the remote call.
Known limits, left on purpose: login's update_user_last_login is outside the revoke guard (its call is shared with the single-user branch). A cancelled request skips the placeholder revoke. In a same-install race the placeholder's revoke_reason changes from session_cap to sign_in_failed. The background-proxy.ts notice path has no negative test. The notice is written after the config-updated dispatch, so a login page that mounts first could miss it. auth-refresh-error.ts and background-proxy.ts import each other.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the nine second-review minors on the per-user session cap. Cleanup (M1), the token-update guard and READ COMMITTED on PostgreSQL (M5, M9) are in sessions_repo and the session manager. A failed multi-user sign-in step now revokes the placeholder session (M4); a review follow-up moved update_user_last_login inside the revoke guard in the token-issuing helper and the magic-link handler. ADR-099 and the docs state the blacklist limit, the listing expires_at change and the cookie wording (M2, M3, M8). The web UI records the session-replaced notice only when invalidation applied, sanitizes refresh error details (M6, M7), and logout now clears the notice after the remote call. Bandit was not run.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
