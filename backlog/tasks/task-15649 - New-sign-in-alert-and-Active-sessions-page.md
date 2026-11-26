---
id: TASK-15649
title: New-sign-in alert and Active sessions page
status: Done
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
- [x] #1 A session created while the user has another live session writes one new_sign_in notification (warning, title New sign-in to your account) that stays until read or dismissed
- [x] #2 GET /auth/sessions reports is_current and device_label; POST /auth/sessions/revoke-all accepts keep_current
- [x] #3 Revoke-one, revoke-all and logout (all_devices, or a token with no session id) refuse API keys, virtual keys and service tokens with 403 interactive_session_required
- [x] #4 A new_sign_in notification can be marked read, dismissed or snoozed only by a browser session older than its source session, or by any browser session once 24 hours old
- [x] #5 The WebUI shows a banner for an unread new_sign_in alert from another session and an Active sessions page at /settings/sessions
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no ADR in this PR; the rules are recorded in AuthNZ-API-Guide.md and restated by the lock ADR in PR 3. No schema change, so no migration.
Task 10 verification (2026-11-25), left In Progress: backend targeted 551+510+624 passed (12 skipped, all sandbox host-gated, 0 Postgres skips) and AuthNZ_SQLite+Notifications 996 passed; tests/lint 414 passed (test_runtime_package_discovery ignored, setuptools missing); tests/CI 6 failed = the known local-only six, 1277 passed. Frontend: ui suite 3 failed (2 Layout.shell-overrides Joyride tests pass alone = load flake; chat-document-processing fails on dev too), tldw-frontend 42 failed in 19 files identical on origin/dev, extension 1 passed. Open: ruff I001 on tldw_Server_API/app/core/AuthNZ/session_manager.py:65 (unsorted sign_in_alert import, dev clean) needs a fix; Bandit not run. Known limits: see plan.
Fix wave (2026-11-25): ruff I001 fixed; tests and asserts added after the final review. Gates passed: tests/lint (ignoring test_runtime_package_discovery) + CI test-quality ratchet + tests/Docs 669 passed; touched backend files 138 passed (the PostgreSQL sessions test ran, not skipped); touched frontend files passed; OpenAPI fingerprint OK. Bandit is not installed locally and was not run (CI only). Known skips: test_user_endpoints.py keeps its interactive-session override (a real-token rewrite is not small); the 44 ruff findings on changed files all exist unchanged at the branch base. Known limits are in the AuthNZ guide Caveats, including that cancelling a snooze is not covered by the silence rule.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 2 of the session cap, alert and lock design. A sign-in that finds another live session (or evicts one through the cap) writes a warning new_sign_in notification off the request path. /auth/sessions reports is_current, same_install and device_label, and revoke-all accepts keep_current. Ending sessions and logout all_devices need an interactive browser session (API keys, virtual keys and service tokens get 403 interactive_session_required). A new_sign_in alert can be marked read, dismissed or snoozed only by an older interactive session, or by any interactive session after 24 hours. The WebUI gets a sign-in banner and an Active sessions page at /settings/sessions, and the extension skips the OS notification for the browser's own sign-in. Bandit was not run locally (not installed); it runs in CI.
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
