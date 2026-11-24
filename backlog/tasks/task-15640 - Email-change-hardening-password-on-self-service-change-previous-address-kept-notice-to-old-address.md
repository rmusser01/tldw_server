---
id: TASK-15640
title: 'Email-change hardening: password on self-service change, previous address
  kept, notice to old address'
status: In Progress
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
- [ ] #1 PUT /api/v1/users/me and PATCH /api/v1 and /api/v2 users/me/profile refuse a changed email without the correct current password (403 current_password_required or current_password_invalid) in multi-user mode
- [ ] #2 An account with no stored password gets 403 email_change_requires_admin
- [ ] #3 A self-service change stores users.previous_email and users.email_changed_at on SQLite and PostgreSQL; a second change within 7 days keeps the original previous address
- [ ] #4 The previous address gets an email-changed notice after the change commits; a failed send does not fail the change
- [ ] #5 Admin changes of another user's email are unchanged
- [ ] #6 Chatbook account restore no longer changes the account email
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no ADR in this PR. It applies the approved design (section 0); the ADR for the session cap and account lock rules is written with PR 1 as the design states.
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
