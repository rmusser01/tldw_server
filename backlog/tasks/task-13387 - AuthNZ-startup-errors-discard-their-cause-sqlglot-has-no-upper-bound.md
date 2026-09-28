---
id: TASK-13387
title: AuthNZ startup errors discard their cause; sqlglot has no upper bound
status: To Do
assignee: []
created_date: '2026-09-27 18:57'
updated_date: '2026-09-28 17:50'
labels:
  - authnz
  - observability
  - dependencies
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On 2026-09-27, sqlglot 30.20.0 (released 16:35Z) broke every SQLite AuthNZ startup in CI. Fixed in #3030: the users-bootstrap guard compared a standalone AUTOINCREMENT render, which the new release emits as ''. Two things made that outage slow to diagnose and easy to trigger. Both are still open.

1. Swallowed causes. Users_DB._create_tables raises DatabaseError('Failed to create users table') from None, and AuthNZ database._transaction_context raises TransactionError('SQLite transaction') from None. The real exception (ProfileUserWriteRejected) was invisible. _log_storage_failure binds exception_type as a loguru extra, but the CI log format does not print extras. So 97 CI failures showed only the generic message. The cause was found by walking __context__ locally.

2. No upper bound. pyproject.toml pins sqlglot>=25.0.0. The profile-user write guard depends on sqlglot's parse tree and rendering, so any minor release can change what the guard accepts. #3030 added a positive test for the canonical SQLite DDL, which now catches a change like this one. It does not stop a future release from failing CI across every PR at once.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The underlying exception type and message reach the CI log when the users bootstrap fails, without logging SQL parameters or secrets
- [x] #2 A test proves a guard rejection during _create_tables surfaces its cause
- [ ] #3 Owner decision recorded on sqlglot: an upper bound (e.g. <31) with deliberate upgrades, or no bound, relying on the canonical-DDL tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC #1 and AC #2 done on fix/authnz-surface-startup-causes: new app/core/exceptions.exception_type_chain() walks __cause__/__context__ (sees through 'from None'), types only, bounded and cycle-safe. Users_DB._log_storage_failure and AuthNZ database._transaction_context now put the chain in the message text; the bound extras stayed invisible in CI. The outage shape now logs 'cause=TransactionError <- ProfileUserWriteRejected'. Messages are deliberately not logged: a PostgreSQL unique-violation detail carries the email. Tests: tests/AuthNZ/unit/test_users_db_startup_failure_cause.py (4); the startup test was red before the log change. AC #3 (sqlglot upper bound) stays open as an owner decision.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
