---
id: TASK-15640
title: 'Email-change hardening: password on self-service change, previous address
  kept, notice to old address'
status: Done
labels:
- authnz
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Self-service email changes require the current password; the previous address and change time are stored and the previous address is notified. Prerequisite for the account lock (design: Docs/Design/2026-11-23-session-cap-alert-lock-design.md section 0).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PUT /api/v1/users/me and PATCH /api/v1 and /api/v2 users/me/profile refuse a changed email without the correct current password (403 current_password_required or current_password_invalid) in multi-user mode
- [x] #2 An account with no stored password gets 403 email_change_requires_admin
- [x] #3 A self-service change stores users.previous_email and users.email_changed_at on SQLite and PostgreSQL; a second change within 7 days keeps the original previous address
- [x] #4 The previous address gets an email-changed notice after the change commits; a failed send does not fail the change
- [x] #5 Admin changes of another user's email are unchanged
- [x] #6 Chatbook account restore no longer changes the account email
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no ADR in this PR. It applies the approved design (section 0); the ADR for the session cap and account lock rules is written with PR 1 as the design states.
Verification (2026-11-24). Wide run: 162 test files selected by the plan's git grep (non-.py json hit dropped; tests/lint/test_runtime_package_discovery.py not in the list, setuptools missing from the venv), one pytest invocation against a running Dockerized Postgres (tldw_postgres_test). Branch: 1 failed, 2053 passed, 0 skipped, 0 errors (47m). The failure, AuthNZ_SQLite/test_mfa_login_sqlite.py::test_mfa_login_http_sqlite (enable_mfa returns False after a PostgreSQL transaction RuntimeError leaked from an earlier file), passes alone on this branch and fails identically in the same combined order on the base commit 518bda3da3 (161 files, 2 failed, 2013 passed, 0 skipped): order-dependent on base too, not caused by this branch. The base run also failed Admin/test_admin_e2e_support_api.py::test_admin_e2e_reset_rejects_unsafe_backup_paths (200 != 500) in a temporary detached worktree under TMPDIR, and again when run alone there; it passes on the branch both alone and in the combined run, so it is not a branch regression (likely tied to the worktree location, unproven). No Postgres test was skipped. Lint: tests/lint ratchets covered by the wide run where selected; new files pass ruff; one touched file, tests/AuthNZ/integration/test_user_endpoints_integration.py, has fewer ruff findings than base (6 vs 7, none new); OpenAPI fingerprint and paths check exit 0. bandit: environment skip (not installed in the venv). No docs edited (no mirrored doc touched). Known facts: (1) a non-owner PostgreSQL deployment without the new users columns refuses to serve at pool initialisation with a DatabaseError (not AUTHNZ_CORE_SCHEMA_NOT_READY) until the owner migration (AuthNZ initialize or owner-role start) has run; (2) v2 PATCH /users/me/profile error responses that carry skipped entries now serialize correctly (they returned 500 before); (3) recipient addresses are masked in the mail send path's log lines; (4) test addresses use @example.com because EmailStr rejects .test. Known limits and follow-up: no per-user limit on wrong passwords or on successful email changes; filed as TASK-15641.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Self-service email changes (PUT /users/me, PATCH v1/v2 /users/me/profile) now require the current password in multi-user mode (403 current_password_required, current_password_invalid, email_change_requires_admin). A change stores users.previous_email and users.email_changed_at (SQLite and PostgreSQL migrations; a second change within 7 days keeps the original previous address) and mails a notice to the previous address after commit, best effort. Chatbook account restore no longer changes the sign-in email. Admin changes of other users are unchanged. Recipient addresses are masked in mail send logs. Rate limits are deferred to TASK-15641.
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
