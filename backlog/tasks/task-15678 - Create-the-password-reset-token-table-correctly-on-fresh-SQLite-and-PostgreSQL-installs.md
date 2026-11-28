---
id: TASK-15678
title: Create the password-reset token table correctly on fresh SQLite and PostgreSQL
  installs
status: To Do
labels:
- authnz
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a fresh install forgot-password cannot store a token. The packaged SQLite schema creates password_reset_tokens without the token_hash column the code reads and writes, and no PostgreSQL DDL creates the table at all. Found while building the account lock: a take-back replaces the password with one nobody knows, so the owner then needs forgot-password to set a new one, and until this lands only an administrator can give a taken-back account a password. The account-lock SQLite tests carry a test-only helper (repair_reset_tokens_table in tests/AuthNZ_SQLite/_account_lock_fixtures.py) that adds the column so they can exercise the reset path. Land before the account-lock WebUI (PR 3b).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a fresh SQLite install, forgot-password stores a token and reset-password redeems it, proven by an integration test that does not repair the table first
- [ ] #2 On a fresh PostgreSQL install the same holds, proven by a PostgreSQL integration test
- [ ] #3 An existing install is upgraded by a migration on both backends
- [ ] #4 The test-only repair_reset_tokens_table helper is deleted and the account-lock tests pass without it
- [ ] #5 After a take-back the owner sets a new password through forgot-password, proven by a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
SQLite detail: the packaged schema creates the table with a token column, and migration 011's ADD COLUMN token_hash TEXT UNIQUE fails silently because SQLite cannot add a UNIQUE column. The test helper rebuilds the table as migration 011 meant it.
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
