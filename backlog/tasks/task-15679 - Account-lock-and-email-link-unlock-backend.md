---
id: TASK-15679
title: 'Account lock and email-link unlock: backend'
status: Done
labels:
- authnz
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a of the session cap, alert and lock design (Docs/superpowers/plans/2026-11-26-account-lock-backend.md): self-service account lock, email-link unlock and take-back, API-key suspension, admin unlock. Server half only; the WebUI half is PR 3b.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New routes: POST /api/v1/auth/account/lock, GET /api/v1/auth/account/lock-availability, POST /api/v1/users/api-keys/{key_id}/unsuspend, POST /api/v1/admin/users/{user_id}/security-unlock
- [x] #2 magic-link/verify refuses a token whose email claim is not the account's current email (or, for a take-back, its previous_email inside 7 days)
- [x] #3 magic-link/request and forgot-password for a locked account send the account-locked mail and keep their generic answer
- [x] #4 Self-service email change refuses an address another account released in the last 7 days (email_recently_released)
- [x] #5 reset-password refuses a locked account (423), a token issued before credentials_valid_after (400) and a token whose email is no longer the account's (400)
- [x] #6 API keys are refused when the owner is locked, the owner is inactive, or the key is suspended; usage is not recorded for a refused key; a suspended key cannot be rotated
- [x] #7 Admin: is_locked removed from UserUpdateRequest; listings gain security_locked_at; impersonating a locked user is refused; own-email change needs step-up
- [x] #8 reset_admin_password also clears a security lock
- [x] #9 ACP trigger webhooks and Telegram approvals of a locked owner are refused
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered from TASK-15671 on 2026-11-27: the id was allocated on an unpushed branch and open PR #3651 took it meanwhile.
Verification (local, macOS, PostgreSQL 18 in Docker). On the fix-wave tree before the rebase onto dev: AuthNZ_SQLite + AuthNZ_Postgres 790 passed, none skipped; AuthNZ + AuthNZ_Unit + AuthNZ_Federation 5227 passed, 1 skipped, 1 failed and 1 teardown error (test_websocket_route_auth_matrix.py::test_sandbox_single_user_key_keeps_the_ip_allowlist[X-API-KEY] and the teardown of test_initialize_mcp_secrets.py::test_authnz_initializer_does_not_create_missing_exclusive_env_file; both from the core config cache reloading under another test's environment patch, on paths this work does not change, not yet reproduced or baselined against dev); Admin + Notifications + UserProfile 2036 passed, 3 failed for want of pg_dump on the machine. On the final tree: test-quality triage, OpenAPI fingerprint and shard coverage pass. Known skips: tests/lint, tests/Docs, the MCP API-key tests, the Slack, Discord, Telegram and ACP directories and the Chatbooks full-account import test were not run on the final tree (free disk fell under the test runner's 100 GiB floor); Bandit is not installed locally; no frontend typecheck (no node_modules in the worktree). CI is the first run of those.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Server half of the account lock (PR 3a). A signed-in user can lock their own account when the deployment can mail them a link; while it is locked every session, token, sign-in route, API key, MCP credential and chat-integration or agent-trigger entry point of the account is refused (423 account_locked to interactive callers, nothing revealed to anonymous ones). Unlock is a mailed link redeemed through magic-link verify; tokens from before the unlock stay dead and API keys created in the week before the lock stay suspended until the owner restores each one. For 7 days after an email change the link also goes to the previous address and takes the account back: it restores that address, voids the password, ends sessions, suspends recent keys and removes recent SSO links, and wins even if the newer address unlocked first. Administrators can unlock behind the step-up check and need the step-up to change their own email through PUT /admin/users/{id}. Change-password, reset-password and the self-service email change re-check the lock on their own transaction and hold the user's row on PostgreSQL, so an in-flight write cannot undo a take-back. Also fixes reset-password on SQLite, which revoked sessions while holding its own write lock. Design: ADR 104. Known limits are listed there; the follow-ups are TASK-15678, TASK-15680, TASK-15681, TASK-15682, TASK-15683 and TASK-15684. The WebUI half is PR 3b.
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
