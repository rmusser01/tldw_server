---
id: TASK-15649
title: New-sign-in alert and Active sessions page
status: In Progress
labels:
- authnz
- security
- webui
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 2 of the session cap, alert and lock design (Docs/Design/2026-11-23-session-cap-alert-lock-design.md section 2): new_sign_in notification, is_current and device_label on /auth/sessions, keep_current on revoke-all, interactive-session requirement for ending sessions and silencing the alert, Active sessions settings page and banner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A session created while the user has another live session writes one new_sign_in notification (warning, title New sign-in to your account) that stays until read or dismissed
- [ ] #2 GET /auth/sessions reports is_current and device_label; POST /auth/sessions/revoke-all accepts keep_current
- [ ] #3 Revoke-one, revoke-all and logout (all_devices, or a token with no session id) refuse API keys, virtual keys and service tokens with 403 interactive_session_required
- [ ] #4 A new_sign_in notification can be marked read, dismissed or snoozed only by a browser session older than its source session, or by any browser session once 24 hours old
- [ ] #5 The WebUI shows a banner for an unread new_sign_in alert from another session and an Active sessions page at /settings/sessions
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no ADR in this PR; the rules are recorded in AuthNZ-API-Guide.md and restated by the lock ADR in PR 3. No schema change, so no migration.
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
