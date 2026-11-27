---
id: TASK-15671
title: 'Account lock and email-link unlock: backend'
status: In Progress
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
- [ ] #1 New routes: POST /api/v1/auth/account/lock, GET /api/v1/auth/account/lock-availability, POST /api/v1/users/api-keys/{key_id}/unsuspend, POST /api/v1/admin/users/{user_id}/security-unlock
- [ ] #2 magic-link/verify refuses a token whose email claim is not the account's current email (or, for a take-back, its previous_email inside 7 days)
- [ ] #3 magic-link/request and forgot-password for a locked account send the account-locked mail and keep their generic answer
- [ ] #4 Self-service email change refuses an address another account released in the last 7 days (email_recently_released)
- [ ] #5 reset-password refuses a locked account (423), a token issued before credentials_valid_after (400) and a token whose email is no longer the account's (400)
- [ ] #6 API keys are refused when the owner is locked, the owner is inactive, or the key is suspended; usage is not recorded for a refused key; a suspended key cannot be rotated
- [ ] #7 Admin: is_locked removed from UserUpdateRequest; listings gain security_locked_at; impersonating a locked user is refused; own-email change needs step-up
- [ ] #8 reset_admin_password also clears a security lock
- [ ] #9 ACP trigger webhooks and Telegram approvals of a locked owner are refused
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
