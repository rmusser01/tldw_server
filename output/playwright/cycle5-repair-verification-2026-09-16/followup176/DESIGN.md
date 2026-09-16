# UAT176 / TASK-13260.112 — world-book initializer transaction ownership

## Cause and approved bounded scope

The real character-conversation factory calls WorldBookService before its creation transaction. Its initializer already selects PostgreSQL DDL but enters the raw BackendConnectionWrapper as a context manager, which is unsupported. Both real PostgreSQL global and workspace routes return500 after the now-correct UAT173 quota check. This is distinct from provider execution.

Use the existing CharactersRAGDB.transaction() context in `_init_tables` and remove its explicit `conn.commit()`. The standard context owns outer commit/rollback and joins a caller-owned transaction on both backends. Keep all DDL, indexes, backend branching, request-scoped caches and error wrapping unchanged. Do not change the global connection wrapper or unrelated world-book CRUD methods.

## Evidence and tests

- Original real-router RED remains in ../uat173-repair-20260916/quota-green-followup-failure.log; the expanded ordinary-chat control shows14PASS/2characterFAIL with no skips.
- Add real official pg_database_config and temporary SQLite tests: initialize twice and read all tables; successful outer transaction keeps caller data; failing outer transaction rolls back caller data even after initialization. The latter catches the old SQLite initializer's premature explicit commit too.
- Preserve unmocked character-create router tests; combine UAT173/UAT176 required-PG suite for final GREEN. Existing legacy mock fixture needs to expose the existing database transaction context in addition to the connection context; no actual PG/factory path is mocked.
- Run existing SQLite world-book controls, Ruff baseline comparison, Bandit and independent review. Root owns native/runtime/tracker/commit work.

## Owned paths

world_book_manager.py; new focused test_world_book_initialization_backends.py; only necessary legacy mock-fixture compatibility in test_world_book_manager_legacy.py. No UAT171/UAT174 shared DB helper edits.
