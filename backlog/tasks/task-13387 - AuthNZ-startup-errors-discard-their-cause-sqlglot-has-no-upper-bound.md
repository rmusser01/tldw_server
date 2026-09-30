---
id: TASK-13387
title: AuthNZ startup errors discard their cause; sqlglot has no upper bound
status: Done
assignee: []
created_date: '2026-09-27 18:57'
updated_date: '2026-09-29 15:12'
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
- [x] #1 A test proves a guard rejection during _create_tables surfaces its cause
- [x] #2 Owner decision recorded on sqlglot: an upper bound (e.g. <31) with deliberate upgrades, or no bound, relying on the canonical-DDL tests
- [x] #3 The underlying exception type chain reaches the CI log (message text, not only bound extras) when the users bootstrap fails, on SQLite and PostgreSQL, without logging exception messages, SQL parameters or secrets
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC #1 and AC #2 done on fix/authnz-surface-startup-causes: new app/core/exceptions.exception_type_chain() walks __cause__/__context__ (sees through 'from None'), types only, bounded and cycle-safe. Users_DB._log_storage_failure and AuthNZ database._transaction_context now put the chain in the message text; the bound extras stayed invisible in CI. The outage shape now logs 'cause=TransactionError <- ProfileUserWriteRejected'. Messages are deliberately not logged: a PostgreSQL unique-violation detail carries the email. Tests: tests/AuthNZ/unit/test_users_db_startup_failure_cause.py (4); the startup test was red before the log change. AC #3 (sqlglot upper bound) stays open as an owner decision.

AC amended 2026-09-28 (Qodo on #3047): the original wording asked for type *and message*. Messages are deliberately not logged, because a PostgreSQL unique-violation detail carries the email address; the type chain alone diagnosed the 2026-09-27 outage. Also per Qodo: the PostgreSQL path raises its TransactionError outside any except block, so the chain cannot be recovered downstream; its own log line now carries 'cause=<type chain>', pinned by test_postgres_transaction_execute_failure_log_omits_raw_exception (probed red without the change) alongside the existing no-leak assertions.

2026-09-29, owner decision (AC #2): leave sqlglot uncapped ("leave sqlglots version uncapped"). pyproject keeps sqlglot>=25.0.0. The protection is the canonical-DDL test from #3030 (_bootstrap_simple_constraint_is_canonical plus its positive test), which fails loudly on a release that changes what the profile-user write guard accepts, and the type-chain logging from #3047, which names the cause in CI. Accepted risk: a breaking sqlglot minor still fails every PR at once until fixed, as on 2026-09-27; the fix is a code change like #3030, not a pin. DoD: tests recorded in the notes above (#3030, #3047); no docs change (the decision lives here); bandit N/A, no code in this closure; no skips or blockers. Final summary: causes now reach the CI log on SQLite and PostgreSQL, and sqlglot stays unbounded by owner decision.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
