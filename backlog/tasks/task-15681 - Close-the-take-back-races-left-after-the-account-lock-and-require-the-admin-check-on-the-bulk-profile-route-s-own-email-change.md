---
id: TASK-15681
title: Close the take-back races left after the account lock, and require the admin
  check on the bulk profile route's own-email change
status: To Do
labels:
- authnz
- security
dependencies:
- TASK-15679
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Left open by the account-lock backend (TASK-15679) after its whole-branch review; each is listed as a known limit in ADR 104. (1) Sign-in rehash: endpoints/auth.py rehashes an out-of-date password hash after a successful sign-in with an unconditional UPDATE (services/auth_service.py update_user_password_hash). If a take-back voids the password between the verify and the rehash, the rehash puts the intruder's password back. One request wide, once per account. (2) A password sign-in that verified just before a take-back can create its session just after the take-back's last sweep. (3) POST /api/v1/admin/users/profile/bulk lets an administrator change their own email without the password or the admin step-up: services/admin_profiles_service.py bulk_update_user_profiles calls update_service.apply_updates directly and never reaches command_service._own_email_change_refusal. On dev since the email-change hardening PR; the owner's rule is that an admin changing their own email passes the step-up check. (4) Two behaviours of the fix wave have no test that fails without them. Land before the account-lock WebUI (PR 3b).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The sign-in rehash writes only when the stored hash is still the one the sign-in verified; a test voids the password between verify and rehash and the void survives
- [ ] #2 A password sign-in that verified before a take-back cannot leave a working session after it, proven by a test that orders the two
- [ ] #3 An administrator changing their own email through POST /admin/users/profile/bulk is refused without the password or the admin step-up, as on PUT /admin/users/{id}, proven by a test
- [ ] #4 A test fails if reset-password stops revoking sessions inside the request on PostgreSQL
- [ ] #5 A PostgreSQL test fails if change-password, reset-password or the self-service email change reads the credential before the lock guard
- [ ] #6 ADR 104 known limits are updated to match
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
