# Qodo #2972 Notes repository focus fixes (TASK-13263.1)

Review read using `gh api repos/rmusser01/tldw_server/issues/comments/5752629050 --jq .body`.
Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42
No commit, branch switch, release-doc edit, or task edit.

## Findings and changes

1. Owner filtering: consistency issue, not a demonstrated current shared-owner exclusion bug. `_selected_owner_filter(self._db.client_id, alias)` currently produces the same PostgreSQL predicate as the hard-coded clauses, and the database constructor requires a nonempty client ID. Notes FTS/ILIKE search, keyword-filtered search, and keyword-filtered counts now obtain both SQL and parameters from that existing abstraction. Both note and keyword owners remain restricted. No new shared/legacy owner access semantics were invented. Six regression variants exercise FTS, fallback, keyword text/no-text and keyword count text/no-text, ensuring the predicate's selected-owner bindings flow through each path.

2. Error misclassification: confirmed defect. A typed/sanitized backend UniqueConstraintError raised by graph projection was caught across the entire transaction and mislabeled a duplicate note ID. PostgreSQL insertion now uses `ON CONFLICT (id) DO NOTHING` and interprets only zero inserted rows as the note-ID conflict; unrelated unique failures propagate as CharactersRAGDBError. SQLite exact notes-ID / parent foreign-key translation is restricted to the INSERT itself. Projection and outer transaction failures are no longer reclassified. Both PostgreSQL backend QueryResult and BackendCursorWrapper propagate cursor.rowcount (ChaChaNotes_DB.py lines 543/557; postgresql_backend.py execute result), so the exact-conflict result is observable.

## Files

- tldw_Server_API/app/core/DB_Management/chacha/note_store.py
- tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py
- tldw_Server_API/tests/DB_Management/test_chacha_postgres_fts.py
- tldw_Server_API/tests/DB_Management/test_chacha_postgres_note_read_lifecycle.py

## Verification

All Python commands activated `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate` and ran in the stated worktree.

Red: `python -m pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_fts.py -k 'projection_unique_failure or insert_duplicate or selected_owner_predicates'` — 8 failed as intended, 1 positive duplicate-control test passed. `/tmp/qodo-notes-red.log`.

Final: `python -m pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_fts.py` — 47 passed, 4 warnings, 21.36s. `/tmp/qodo-notes-final.log`. Projection failure tests verify real SQLite rollback/no saved note; a portable real SQLite transaction exercises the PostgreSQL INSERT branch's exact conflict result. Selected-owner assembly tests use the established PostgreSQL query-boundary stub pattern.

Live PostgreSQL: `python -m pytest -q tldw_Server_API/tests/DB_Management/test_chacha_postgres_note_read_lifecycle.py -k note_duplicate_and_projection` — 1 skipped by existing official pg_notes/pg_database_config fixtures, 24 deselected, 31.40s. `/tmp/qodo-notes-pg.log`. No custom database setup and no live PostgreSQL pass claimed.

Ruff on note_store.py: all checks passed. Black applied only changed line ranges. `git diff --check`: clean.

Bandit on touched source: `python -m bandit -r tldw_Server_API/app/core/DB_Management/chacha/note_store.py -f json -o /tmp/bandit_qodo_notes.json` — exit 0, zero findings/errors. Existing nosec-comment parser warnings remain elsewhere in this large file. Predicate formatting interpolates only trusted helper SQL; all values remain bound parameters.
