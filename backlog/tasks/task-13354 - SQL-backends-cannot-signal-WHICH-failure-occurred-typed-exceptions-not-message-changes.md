---
id: TASK-13354
title: >-
  SQL backends cannot signal WHICH failure occurred; typed exceptions, not
  message changes
status: Done
assignee: []
created_date: '2026-09-23 02:35'
updated_date: '2026-09-23 04:58'
labels:
  - db
  - diagnostics
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CORRECTED TWICE. Both of this task's earlier framings were wrong; what remains is a narrower and more interesting question.

WHAT IS ACTUALLY THERE. Both SQL backends wrap driver errors as a fixed string:
    sqlite_backend.execute       -> DatabaseError("SQLite query execution failed")
    postgresql_backend.execute   -> DatabaseError("PostgreSQL query execution failed")
                                 -> UniqueConstraintError(...) when unique_failure

This is NOT careless. The raise sits OUTSIDE the except block behind a `redacted_failure`
flag precisely so the driver exception is never chained: __cause__ is None and
__suppress_context__ is True, so the original message -- which can carry query text and
bound parameters -- cannot reach a traceback. Verified: raising a constraint violation
with a sensitive parameter value leaks neither the value nor the SQL.

SO THE ORIGINAL PROPOSAL WAS WRONG. "raise ... from exc" would reintroduce exactly what
the pattern exists to prevent.

AND THE SECOND PROPOSAL WAS ALSO WRONG. Appending the driver exception CLASS -- e.g.
"SQLite query execution failed (IntegrityError)", which leaks no data -- was implemented
and then reverted, because the exact message is a TESTED CONTRACT:
    tests/DB_Management/test_media_postgres_support.py:328,356
        pytest.raises(BackendDatabaseError, match="^PostgreSQL query execution failed$")
    tests/DB_Management/test_postgres_unique_conflict.py:94,160
        assert str(error) == "PostgreSQL query execution failed"
Those anchors and equality checks are deliberate: they pin that NOTHING is appended. Any
message change is a change to a security contract someone wrote tests for, and should be
decided rather than slipped in.

THE REAL QUESTION, which is narrow: callers currently cannot distinguish a constraint
violation from a disk error, a locked database, or a type error, without re-running the
query by hand. That cost real time twice today --
test_cleanup_candidate_schema_rejects_path_hash_identity_drift had to be rewritten to
assert the schema and an accept/reject pair because no assertion on the exception could
identify the constraint (see e47ac1c97a).

The safe mechanism already exists in this codebase and does not touch any message: TYPED
exceptions. postgresql_backend already raises UniqueConstraintError for unique_failure
while keeping the same string. Extending that -- e.g. an IntegrityError/constraint
subclass, raised from both backends, message unchanged -- gives callers what they need,
leaks nothing, and leaves every anchored assertion passing.

Compare core/Sync/v2/profile.py:552, which maps any unrecognised SyncStoreError to
"personal_context_snapshot_unavailable" and hid a real bug for weeks (TASK-13351). That
one has no redaction justification and is the stronger candidate for change.

Source: found while fixing TASK-13352/13344 follow-ups; corrected twice while attempting it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Callers can distinguish a constraint violation from other backend failures without re-running the query
- [x] #2 Done via typed exceptions; the redacted message strings and their anchored tests are unchanged
- [x] #3 profile.py:552's blanket remap to personal_context_snapshot_unavailable is assessed separately
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC1 and AC2 DONE in bcf6e2b5ed. AC3 (profile.py:552) is left open deliberately -- see below.

WHAT SHIPPED. ConstraintViolationError sits between DatabaseError and UniqueConstraintError in backends/base.py, so:
    except DatabaseError            catches everything it did before
    except UniqueConstraintError    catches exactly what it did before
    except ConstraintViolationError is new, and catches both
SQLite maps sqlite3.IntegrityError (CHECK, NOT NULL, FOREIGN KEY, UNIQUE). PostgreSQL maps SQLSTATE class 23, with 23505 still preferring UniqueConstraintError. No message anywhere was changed.

TWO EARLIER ATTEMPTS WERE WRONG AND WERE REVERTED, not shipped:
1. 'raise ... from exc', this task's original proposal, would reintroduce precisely the leak the redaction pattern exists to prevent. The raise sits OUTSIDE the except block behind a redacted_failure flag on purpose.
2. Appending the driver class to the message -- 'SQLite query execution failed (IntegrityError)' -- leaks no data and was implemented and working, then reverted on discovering that test_media_postgres_support.py:328,356 and test_postgres_unique_conflict.py:94,160 pin the strings with ^anchors^ and equality. Those assertions exist to guarantee the message is bare; changing it is a security-contract decision, not a refactor.

VERIFIED, not assumed:
- a sensitive parameter value and the SQL text appear nowhere in the raised error, and __cause__ is still None with context suppressed
- the three message-pinning files still pass (47 tests)
- the one real hazard of widening a base class, exact-type comparisons: the two 'type(x) is DatabaseError' assertions in the tree use the db_errors class rather than the backend one, and pass (116 tests)
- DB_Management 23 failed / 3059 passed, IDENTICAL failure set with and without the change (baseline taken by restoring all three files from HEAD and re-running just the 23). Those 23 are pre-existing.

AC3 LEFT OPEN. core/Sync/v2/profile.py:552 maps any unrecognised SyncStoreError to 'personal_context_snapshot_unavailable'. It has no redaction justification -- SyncStoreError messages are the product's own -- and it hid two real bugs for weeks (TASK-13351). It is the stronger candidate for change, but it is a Sync-layer decision rather than a DB-backend one, so it should not ride along on this commit.

AC3 ASSESSED -- NO CODE CHANGE, and the premise I wrote for it was wrong.

I claimed profile.py:552 'has no redaction justification and hid two real bugs for weeks', making it the stronger candidate for change. Checking it rather than acting on my own note:

  profile.py:552-554  raise PersonalContextBootstrapError('personal_context_snapshot_unavailable') from exc

The 'from exc' is already there. Verified at runtime: __cause__ is populated and the original message -- 'Notes suggestion authority requires the owned default dataset' -- is fully recoverable, and it renders in tracebacks. That is literally how both bugs in TASK-13351 were found earlier today; the pytest output showed the SyncStoreError and the PersonalContextBootstrapError together.

So nothing is destroyed here, and this is NOT the same failure mode as the DB backends, where the raise sits outside the except specifically so no chaining occurs. What actually hid the cause historically was a swallow that TASK-13306 already fixed -- TASK-13344's own description says so: 'Real cause, now visible because TASK-13306 made the swallow log it'. I misread that as the remap being at fault.

What remains is much weaker and is probably deliberate: the CODE reported to callers is generic, which is reasonable for an API-facing error where the internal reason should not leak, given the reason is preserved on __cause__ for diagnosis. The one mild smell is that lines 547-550 whitelist specific error strings to pass through, so a new meaningful code needs an edit there -- but the authority error is a sentence, not a code, and would not belong in that set anyway.

Recording this rather than changing working code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Constraint violations are now distinguishable by exception type, using the mechanism the module already had for uniqueness, with every redacted message and its anchored tests untouched. Two more obvious fixes were implemented and reverted first: chaining reintroduces the leak the pattern prevents, and appending the driver class breaks assertions that exist to guarantee the message is bare.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
