# UAT176 / TASK-13260.112 — world-book initialization transactions

## Minimal repair

The real character-conversation factory unconditionally constructs WorldBookService before its create transaction. The initializer had PostgreSQL DDL but attempted to enter BackendConnectionWrapper as a raw context manager. This wrapper intentionally is not the transaction owner.

The initializer now uses the existing CharactersRAGDB.transaction() context and leaves commit/rollback to it by removing the explicit conn.commit(). DDL, indexes, backend branching, cache construction and error wrapping are unchanged. No global wrapper or other CRUD method was changed. The existing legacy mock fixture now exposes transaction() returning its normal test connection context so its established initializer coverage follows the real database API.

## Test-first evidence

- Real router regression from UAT173 exposed PG global/workspace create500 after quota repair (retained in ../uat173-repair-20260916/quota-green-followup-failure.log).
- New direct initialization RED:4 failed /2 passed /0 skipped. All3 PG cases hit unsupported wrapper context; SQLite nested-rollback case proved the old manual commit prematurely committed caller data. Receipt:red.log. This independently establishes why both the context change and removal of explicit commit are necessary.
- After the repair, combined required-PG/SQLite suites:22 passed /0 skipped in23.56s (6 initializer cases +16 quota/actual-router cases). Receipt:combined-green.log. Actual PG global/workspace character creation and existing quota guards pass without mocking the factory or initializer.
- Direct initializer controls:repeat initialization and later rollback leave tables available; nested success permits outer commit; nested failure rolls back the caller's real conversation write.
- Existing world-book regressions: 94 passed / 0 skipped (4 warnings), 38.25 seconds. Receipt: existing-worldbook-regressions.log.

Official pg_database_config -> pg_temp_db and temporary SQLite fixtures own all test data. The private runner uses required PostgreSQL with no Docker autostart on the existing owned55475 cluster, never AuthNZ administrative-DSN reset fixtures. No live application runtime, provider inference or browser is involved.

## Static/security checks

Ruff on production plus touched legacy fixture:17 baseline diagnostics and17 current, zero new; new test clean. Bandit production:zero findings/zero errors. Diff check:PASS. Exact snapshots/hash manifest/patch retained alongside this report. Existing source-wide style diagnostics were not changed as part of this two-line production repair.

Commands (activate .venv first):

    TLDW_UAT_EVIDENCE_LABEL=uat176-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_world_book_initialization_backends.py -q --tb=short
    TLDW_UAT_EVIDENCE_LABEL=uat173-176-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_conversation_quota_count_backends.py tldw_Server_API/tests/DB_Management/test_world_book_initialization_backends.py -q --tb=short
    python -m pytest tldw_Server_API/tests/Character_Chat/test_world_book_manager_legacy.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_world_book_manager.py -q --tb=short

## Limits

This repair initializes world-book tables on the real character-create path; it does not claim a complete PostgreSQL world-book CRUD audit. Other CRUD connection usages were not swept or changed. Native character creation/provider Retry and independent review remain root-owned acceptance. No tracker/shared design/git/browser/runtime edits were performed. Private approved design is DESIGN.md.
