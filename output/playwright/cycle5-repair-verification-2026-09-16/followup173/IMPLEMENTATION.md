# UAT173 / TASK-13260.110 — PostgreSQL conversation quota counts

## Result and scope

Source frozen for independent review. One production-line change reads the existing COUNT(*) AS cnt column by name. SQL predicates, owner/client filtering, global/workspace selection, deleted and character filters, quota threshold and fail-closed handling are unchanged. PostgreSQL dict rows and SQLite Row both support named columns.

Native POST/chats returned503 because the quota count threw KeyError 0 before creation. This was unrelated to the planned provider fault, which never executed. Adjacent by-character and paged-search counts already contain mapping fallbacks; they were inspected without changing them. The new suite controls the real per-character count on both backends.

## RED/GREEN

- Initial official real-PostgreSQL suite:4 expected failures /8 passes /0 skips. PG empty/scoped aggregate checks raised KeyError; PG global/workspace character routes returned the exact quota-unavailable503. SQLite, adjacent per-character and controlled-storage-failure checks passed. Receipt:red.log.
- After the one-line fix,10 passed /2 failed. Both remaining character paths reached the distinct world-book context-manager failure now tracked UAT176. Receipt:quota-green-followup-failure.log. The code did not mask that failure.
- Added ordinary-chat route controls:14 passed /2 UAT176 character failures /0 skips. Receipt:ordinary-control.log. This proves the repaired count supports ordinary create201 and actual quota rejection403 while retaining the character regression.
- After separate UAT176 initialization repair, combined suites:22 passed /0 skipped in23.56s (16 UAT173 cases +6 UAT176 cases). Receipt:combined-green.log; run label uat173-176-green.
- Existing conversation scope/store/character filter and rate-limiter regressions:35 passed /0 skipped (6 warnings),20.53s. Receipt:adjacent-regressions.log.

The new fixture uses official pg_database_config -> pg_temp_db and temporary SQLite. The private runner enforces TLDW_TEST_POSTGRES_REQUIRED=1 and disables Docker autostart, reusing the owned cluster on55475. It never uses AuthNZ test_db_pool or creates an ad-hoc database. The real FastAPI router uses a test-owned user/DB dependency and a real CharacterRateLimiter with max_chats_per_user=2. Only request-frequency governance is disabled; real quota enforcement rejects at cap. Unrelated sync capture is disabled per existing endpoint test convention. Character factory and world-book initialization are real; no provider call is needed. Controlled count-storage failure remains503 and leaves no conversation. Authentication/RLS integration and native Alice/Bob acceptance remain root-owned; these tests prove the unchanged SQL client filtering, not a full auth stack.

## Checks and evidence

Ruff:6 pre-existing production diagnostics before and after, zero added; new test clean. Bandit production:zero findings/zero parsing errors. Diff check:PASS. Exact snapshots, hashes and patch are in review-snapshot/, owned-manifest.json and owned.patch. No fixes were made to unrelated baseline lint. New scope controls use literal expected counts including empty, two owners, absent owner, global default/explicit global, two workspaces, deleted/all/deleted-only and character/non-character selections.

Commands use the project virtual environment:

    TLDW_UAT_EVIDENCE_LABEL=uat173-176-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_conversation_quota_count_backends.py tldw_Server_API/tests/DB_Management/test_world_book_initialization_backends.py -q --tb=short
    python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_conversation_character_scope_filters.py tldw_Server_API/tests/ChaChaNotesDB/test_conversation_scope_db.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py tldw_Server_API/tests/unit/test_character_rate_limiter.py -q --tb=short

Root owns browser/runtime/native Retry acceptance, tracker, shared design and commit. No such action was performed by this author. Private design:DESIGN.md. Production ownership is only conversation_store.py for this task; shared ChaChaNotes_DB.py/note_store.py were not edited.
