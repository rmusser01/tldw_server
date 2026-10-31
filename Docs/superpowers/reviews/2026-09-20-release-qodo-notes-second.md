# Fresh Qodo Notes review dispositions

Task: TASK-13263.1; reviewed comment https://github.com/rmusser01/tldw_server/pull/2972#issuecomment-5752629050 against starting HEAD 10b11f6acb. Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42. Parent concurrently merged recovery changes; no branch changes/commits performed here.

## Dispositions

All three fresh findings are **verified no_change for runtime code**; added canaries pin existing guarantees. Tests are initially green because the proposed failure is already prevented; no deliberately broken runtime change was introduced to manufacture red tests.

1. **Keyword-link authorization (including reverse lookup): already enforced.** ChaChaNotes_DB.py `_selected_keyword_link_filter` (~32626) emits two independent EXISTS predicates, matching the note's client_id and the mapped keyword row's client_id to the selected owner. It does not depend on ownership of the link row. NoteStore.get_keywords_for_note, get_keywords_for_notes and get_notes_for_keyword all call this helper. Foreign note + own keyword and own note + foreign keyword are excluded. Portable tests execute actual NoteStore query generation and the real PostgreSQL-selected owner helper over SQLite tables; they preserve PostgreSQL table mapping and use no stand-in owner filter. Live PostgreSQL malformed-link tests also now explicitly ask for the foreign note and foreign keyword, rather than only querying owned identifiers.

2. **Keyword retrieval `deleted = 0`: already translated.** CharactersRAGDB.execute_query -> BackendCursorWrapper.execute -> PostgreSQLBackend.execute; both the facade preparation and the backend's `_prepare_query` use the shared SQL transformation. query_utils._replace_boolean_comparisons rewrites qualified `k.deleted = 0` to `k.deleted = FALSE` before psycopg. Bound values are unchanged.

3. **Note update/delete/restore `deleted = 0/1`: already translated.** BackendConnectionWrapper.execute -> BackendCursorWrapper.execute -> PostgreSQLBackend.execute performs the same transformation for transaction commands, including SET assignments. SQL-literal soft-delete/restore become TRUE/FALSE; delete_note already supplies its SET deleted value as a bound Python boolean and its literal WHERE check is normalized. Added driver-boundary tests call the actual backend, including the real ChaCha transaction wrapper, with only the psycopg connection mocked. A positive live PostgreSQL lifecycle test exercises update, both soft-delete entry points, restore and keyword reads through the official fixture.

No production behavior or tenant visibility was changed. PostgreSQL unavailability limits live-server verification; portable ownership query tests validate SQL selection semantics and mocked-driver tests validate actual SQL reaching the driver, not PostgreSQL execution itself.

## Files changed

- tldw_Server_API/tests/DB_Management/test_chacha_postgres_fts.py: 8 parameterized driver-boundary boolean cases, 5 portable actual-query ownership cases.
- tldw_Server_API/tests/DB_Management/test_chacha_postgres_note_read_lifecycle.py: 2 positive live PostgreSQL note lifecycle cases using existing pg_notes fixture.
- tldw_Server_API/tests/DB_Management/test_note_shared_owner_contract.py: 2 additional malformed-link boundary cases using existing official PostgreSQL-backed note_owners fixture.

## Verification

Every Python command activated `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate` first.

- `python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_postgres_fts.py tldw_Server_API/tests/DB_Management/test_backend_utils.py -q`: **46 passed**, 4 warnings, 0.78s. Log `/tmp/qodo-notes-second-unit.log`.
- `python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_postgres_note_read_lifecycle.py tldw_Server_API/tests/DB_Management/test_note_shared_owner_contract.py -q -rs -k 'owned_note_mutations_and_keyword_reads_use_postgres_booleans or malformed_note_keyword_links_do_not_cross_owners'`: **11 skipped, 176 deselected**, 31.35s. Official fixture reports **Postgres not reachable**; no custom provisioning. Log `/tmp/qodo-notes-second-pg.log`. Do not claim live PostgreSQL passed.
- Bandit against note_store.py and all three changed test files, skipping B101 solely because test assertions are intentional: **0 findings, 0 errors**, `/tmp/bandit_qodo_notes_second.json`. Command: `python -m bandit <four paths> -s B101 -f json -o /tmp/bandit_qodo_notes_second.json`.
- Ruff: **0 findings on changed lines**. Full-file run finds 10 existing typing modernization findings in test_chacha_postgres_fts.py lines 6–219; none introduced here. Other two changed files fully pass.
- `git diff --check -- <three changed test paths>`: pass. Global diff check temporarily saw parent's in-progress Backlog merge conflict; parent notified, file left untouched.

No parent release docs/task records changed. No commits, merges, publication, or delegation performed by this agent.
