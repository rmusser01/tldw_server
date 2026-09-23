---
id: TASK-13354
title: >-
  SQL backends cannot signal WHICH failure occurred; typed exceptions, not
  message changes
status: To Do
assignee: []
created_date: '2026-09-23 02:35'
updated_date: '2026-09-23 02:55'
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
- [ ] #1 Callers can distinguish a constraint violation from other backend failures without re-running the query
- [ ] #2 Done via typed exceptions; the redacted message strings and their anchored tests are unchanged
- [ ] #3 profile.py:552's blanket remap to personal_context_snapshot_unavailable is assessed separately
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CORRECTION to the description: one sentence lost its content to shell backtick substitution when this task was filed. It should read:

Fix is small: raise the wrapper with a 'raise ... from exc' chain so the original is preserved, and ideally include the driver message in the wrapper text. Callers that deliberately hide SQL from logs can still do so -- chaining does not force anything into a log line, it just stops the information being destroyed.
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
