# Post-publication Qodo follow-up

Tracking: TASK-13263.3. Four additional PR2974 findings and six PR2972 findings
were found in the final publication inventory. The candidate comments were posted
at 07:08 UTC after its approved merge. The immutable v0.1.43 source/grant is not
rewritten by this follow-up.

## Approved publication-gate disposition

PR2974 thread PRRT_kwDOL1aGf86kPXwm proposes restoring a full-suite publisher job
with an enforced 80% coverage threshold. The requester explicitly approved the
bounded contracts/startup/packaging gate after the old full suite timed out.
Normal PR CI remains intact; the repository's coverage aim is not an existing
full-suite publication threshold. The proposal would reverse that approved
decision. The thread is answered and resolved, without claiming a full-suite
coverage pass. Both releases passed their approved publication gate.

# Postrelease frontend review follow-up — TASK-13263.3

Branch: codex/release-0.1.43-review-followup
Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42
No commit, push, GitHub comments, release record, manifest, version, or lockfile edits performed.

## Review dispositions

- PR2972 discussion_r4059936977 / thread PRRT_kwDOL1aGf86kQvZE — fixed. HTTP 401/403 now clear serverChatId using the existing missing-reference reset path, after loading finishes and only while the request owns the active selection and authority. Existing visible message/history/title clearing is preserved. A late denied response cannot clear a newer chat selection or affect a changed principal.
- PR2972 discussion_r4059936955 / thread PRRT_kwDOL1aGf86kQvY2 — fixed with credential-safe diagnostics. The wizard records the failing stage (configuration/navigation) and allowlisted built-in error category using its existing console.error convention. Raw error messages, stacks, and custom names are explicitly suppressed because config/navigation errors can contain keys and identities. Tests inject secret text and a secret custom error name and verify neither is logged. No telemetry added.
- PR2972 discussion_r4059936960 / thread PRRT_kwDOL1aGf86kQvY6 — fixed. Wizard uses useTranslation(settings), onboarding.loginSettingsError, and the English fallback. Added translations in all 18 runtime settings catalogs and all 21 browser-message settings catalogs, including alias locales. The failure test uses the real Spanish runtime catalog. The other two wizard test suites initialize the real existing i18n module to honor the new dependency without warnings.
- PR2972 discussion_r4059936968 / thread PRRT_kwDOL1aGf86kQvY- — recommend dismissal as an unnecessary architecture refactor, with no production edits. _app.tsx is the web shell integration point for runtimeBootstrapReady, runtime/environment credentials, Next router behavior, and offline private-content unmounting. It already invokes the shared TldwAuth.getCurrentUser/logout stack for principal validation; moving the web policy into packages/ui would couple extension/shared routing to web bootstrap and Next semantics or require a larger injected-adapter abstraction. The 70 existing app-layout tests pass, including offline auth boundary events, queued draft preservation, reconnect redirects, late success/failure after newer logout, public route availability, non-auth network failures, and hosted cookie sessions. The reviewer identifies future refactor potential, not a demonstrated authentication correctness defect. Preserve the tested platform boundary in this bounded bug fix.

## Verification

- Red: focused wizard + loader-scope command returned 4 failures and 50 passes. Two failures show HTTP 401/403 leaves chat-a selected; two show configuration/navigation alerts stay English. Log: /tmp/postrelease-frontend-red.log.
- Green: 8 Vitest files, 215 passing tests. Files: useServerChatLoader.scope.test.tsx (22), useServerChatLoader.test.ts (30), useServerChatLoader.images.test.ts (9), useServerChatLoader.mirror.integration.test.tsx (12), UnifiedSetupWizard.test.tsx (34), UnifiedSetupWizard.model-handoff.test.tsx (37), UnifiedSetupWizard.extension-authority.test.tsx (1), app-layout.test.tsx (70). Log: /tmp/postrelease-frontend-green.log.
- Full WebUI npm run typecheck: exit 0. Log: /tmp/postrelease-frontend-typecheck.log.
- Scoped ESLint over all six touched TS/TSX files: exit 0, 0 errors, seven loader warnings. HEAD baseline run via git show piped to ESLint --stdin reproduces all seven warnings (five any types, two existing effect ref-cleanup warnings); only the final line shifts by two. Logs: /tmp/postrelease-frontend-lint.log and /tmp/postrelease-frontend-lint-baseline.log. Root invocation prints the same pre-existing Next pages-directory diagnostic in both runs; initial web-directory invocation ignored files outside base path, so it was replaced by the root invocation with explicit config.
- All 39 edited JSON catalogs parse and have a nonempty translated message.
- git diff --check -- apps: exit 0.
- No Python files touched by this work unit; Bandit scope is handled by the parent/backend work unit.
- Vitest infrastructure emits existing Node experimental localStorage/Browserslist database warnings. No i18n missing-instance warnings remain after test setup initialization.

## Exact changed frontend files

- apps/packages/ui/src/assets/locale/ar/settings.json
- apps/packages/ui/src/assets/locale/da/settings.json
- apps/packages/ui/src/assets/locale/de/settings.json
- apps/packages/ui/src/assets/locale/en/settings.json
- apps/packages/ui/src/assets/locale/es/settings.json
- apps/packages/ui/src/assets/locale/fa/settings.json
- apps/packages/ui/src/assets/locale/fr/settings.json
- apps/packages/ui/src/assets/locale/it/settings.json
- apps/packages/ui/src/assets/locale/ja-JP/settings.json
- apps/packages/ui/src/assets/locale/ko/settings.json
- apps/packages/ui/src/assets/locale/ml/settings.json
- apps/packages/ui/src/assets/locale/no/settings.json
- apps/packages/ui/src/assets/locale/pt-BR/settings.json
- apps/packages/ui/src/assets/locale/ru/settings.json
- apps/packages/ui/src/assets/locale/sv/settings.json
- apps/packages/ui/src/assets/locale/uk/settings.json
- apps/packages/ui/src/assets/locale/zh-TW/settings.json
- apps/packages/ui/src/assets/locale/zh/settings.json
- apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.extension-authority.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
- apps/packages/ui/src/hooks/__tests__/useServerChatLoader.scope.test.tsx
- apps/packages/ui/src/hooks/chat/useServerChatLoader.ts
- apps/packages/ui/src/public/_locales/ar/settings.json
- apps/packages/ui/src/public/_locales/da/settings.json
- apps/packages/ui/src/public/_locales/de/settings.json
- apps/packages/ui/src/public/_locales/en/settings.json
- apps/packages/ui/src/public/_locales/es/settings.json
- apps/packages/ui/src/public/_locales/fa/settings.json
- apps/packages/ui/src/public/_locales/fr/settings.json
- apps/packages/ui/src/public/_locales/it/settings.json
- apps/packages/ui/src/public/_locales/ja-JP/settings.json
- apps/packages/ui/src/public/_locales/ja/settings.json
- apps/packages/ui/src/public/_locales/ko/settings.json
- apps/packages/ui/src/public/_locales/ml/settings.json
- apps/packages/ui/src/public/_locales/no/settings.json
- apps/packages/ui/src/public/_locales/pt-BR/settings.json
- apps/packages/ui/src/public/_locales/ru/settings.json
- apps/packages/ui/src/public/_locales/sv/settings.json
- apps/packages/ui/src/public/_locales/uk/settings.json
- apps/packages/ui/src/public/_locales/zh-TW/settings.json
- apps/packages/ui/src/public/_locales/zh/settings.json
- apps/packages/ui/src/public/_locales/zh_CN/settings.json
- apps/packages/ui/src/public/_locales/zh_TW/settings.json


# Release 0.1.43 post-publication Qodo test follow-up

Task: TASK-13263.3. Working checkout: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42`, branch `codex/release-0.1.43-review-followup`. Edits began after parent confirmed task creation. No production code, shared AuthNZ fixture, release/tag, legal record, commit, push, or GitHub reply modified by this agent.

## Findings and disposition

- PR2974 `PRRT_kwDOL1aGf86kPXwp`, missing markers: fixed. Both isolation regression functions now have exactly one category marker, `integration` (route test retains parametrization). Before edit, `-m integration --collect-only` selected zero of four collected cases; after edit all four execute and pass.
- PR2974 `PRRT_kwDOL1aGf86kPXwq`, missing annotations: fixed in the four specified files. Isolation test arguments/returns, Prompt Studio app fixture and modified dual-backend client/nested overrides, media database/client/seed fixtures and title-update test now have explicit argument/return types. Generator fixtures use Iterator/AsyncIterator, app uses FastAPI, DB fixtures use their database classes.
- PR2974 `PRRT_kwDOL1aGf86kPXws`, SQL/private FTS helpers: private `_update_fts_media` calls and the newly added seed SELECT were removed. Both media seed fixtures now reuse existing public `sync_refresh_fts_for_entity` with recorded UUIDs. No new production abstraction was added. Direct MATCH assertions intentionally remain, with an inline explanation: these checks prove index population and removal of stale title tokens, while `MediaSearchRepository.search` falls back to LIKE for FTS query errors (around lines 522–544 and 583–600), which could hide this regression. A wholesale switch to `add_media_with_keywords` would change fixture semantics: that method unconditionally creates a DocumentVersion (media_repository.py around 749–758), whereas video/audio fixtures deliberately have no document versions. Parent approved this bounded disposition.
- PR2972 `PRRT_kwDOL1aGf86kQvY0`, shared mutable PostgreSQL schema: fixed in the target test. It now depends on the official `isolated_test_environment`, obtains its configured pool through `get_db_pool`, and verifies `current_database()` equals the unique fixture database name before writes or ALTER COLUMN. Existing finally restoration remains. Shared `test_db_pool`, common conftest and production database code are untouched.

## Verification

All Python tools ran after `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate`. Pytest command prefix: `PYTHONPATH=/tmp/pypi0142-ci-pytest-plugins:. MINIMAL_TEST_APP=1 python -m pytest -q`.

1. Pre-fix marker reproduction, isolation modules with `-m integration --collect-only`: 4 collected, 4 deselected, 0 selected. `/tmp/postrelease-tests-markers-before.log`.
2. Both Infrastructure isolation modules with `-m integration`: **4 passed**, 2 warnings, 17.60s. `/tmp/postrelease-tests-isolation.log`.
3. Complete `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py`: **43 passed**, 8 warnings, 38.27s. `/tmp/postrelease-tests-media.log`.
4. Prompt Studio app fixture exercised by `tldw_Server_API/tests/prompt_studio/unit/test_ps_create_request_id.py`: **1 passed**, 6 warnings, 21.21s. `/tmp/postrelease-tests-prompt.log`.
5. `tldw_Server_API/tests/AuthNZ/integration/test_authnz_api_keys_repo_postgres.py::test_create_virtual_key_row_persists_text_and_jsonb_lists_postgres`: **1 skipped**, 8 warnings, 49.59s. `/tmp/postrelease-tests-authnz.log`. The official fixture's PostgreSQL availability path skipped; no ad-hoc database was created and no live PostgreSQL pass is claimed. The test requires re-execution where the official fixture can reach PostgreSQL. Default pytest output did not emit detailed skip reason, but the selected test has no skip calls and official `isolated_test_environment` skips at conftest.py:653 when PostgreSQL is unavailable after its availability/start attempt.
6. Scoped Bandit JSON comparison: **241 baseline findings, 241 current findings, zero new findings**. Existing findings are predominantly test asserts. The new current-database test assertion has standard `# nosec B101`. Baseline copied before edits to `/tmp/postrelease-tests-baseline`; JSON reports `/tmp/postrelease-tests-bandit-baseline.json`, `/tmp/postrelease-tests-bandit.json`.
7. Scoped Ruff comparison: **9 baseline findings, 9 current findings, zero new findings**. Baseline linted using `--stdin-filename` with each actual repo path so per-file ignores match. Reports `/tmp/postrelease-tests-ruff-baseline.json`, `/tmp/postrelease-tests-ruff.json`. Existing unsorted imports/unused variables outside the touched definitions remain; new AuthNZ imports are sorted.
8. `git diff --check`: clean.

No new production behavior was introduced, so no production TDD cycle was needed. The marker selection change has an observed before/after failure/pass, and existing behavior regressions were executed after the test-only refactor.

## Files edited

- `tldw_Server_API/tests/Infrastructure/test_prompt_studio_collection_isolation.py`
- `tldw_Server_API/tests/Infrastructure/test_route_collection_isolation.py`
- `tldw_Server_API/tests/prompt_studio/conftest.py`
- `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_api_keys_repo_postgres.py`


# TASK-13263.3: PR2972 chat boolean comparison finding

Disposition: false positive. No production SQL change is warranted.

## Evidence

- `chacha/chat_history_queries.py:15` (`search_chat_history`) and `:55` (`get_chat_history_metadata`) emit portable `m.deleted = 0` / `conv.deleted = 0` comparisons and keep values bound.
- Their production caller is `RAG/rag_service/database_retrievers.py:3925` / `:4011`. `BaseRetriever._execute_query` at `:844` routes adapters through `execute_query`; the adapter-free fallback at `:876` is explicitly SQLite.
- `ChaChaNotes_DB.py:8234` (`CharactersRAGDB.execute_query`) prepares SQL then uses its wrapped cursor. `BackendCursorWrapper.execute` at `:533` calls the actual backend; `BackendConnectionWrapper.execute` at `:610` takes the same route for transactions.
- `backends/postgresql_backend.py:997` (`PostgreSQLBackend.execute`) always calls `_prepare_query` (`:977`), which enables `prepare_backend_statement(..., apply_default_transform=True)`.
- `backends/query_utils.py:619` calls `_replace_boolean_comparisons` (`:505`). That function rewrites boolean-column `= 0` to `= FALSE`, including the aliased `m.deleted` and `conv.deleted`, before psycopg receives SQL. Bound owner, search, metadata ID, excluded source, and limit parameters remain separate.

## Change

Only `tldw_Server_API/tests/DB_Management/test_chacha_postgres_fts.py` was edited by this agent: 51 added lines at approximately line 357.

The new parameterized test invokes the actual search and metadata helpers through three real execution routes: PostgreSQLBackend, ChaCha transaction connection wrapper, and CharactersRAGDB.execute_query. Only the external driver connection/cursor is mocked. Six cases verify both boolean filters at the driver boundary, PostgreSQL owner placeholder, unchanged bindings, and returned rows. The test fails if normalization is removed or a helper bypasses it.

## Verification

Activated `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate` before Python commands.

- `python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_postgres_fts.py -q --override-ini addopts=''`: **33 passed** (four existing dependency/config warnings).
- Mutation check in a separate Python process replaced only `query_utils._replace_boolean_comparisons` in memory with identity, then ran the six new cases: **6 failed**, each at the expected `m.deleted = FALSE` assertion. No production file was changed. Details: `/tmp/postrelease-chat-sql-mutation.log`.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/DB_Management/test_character_message_search_prose.py -q -k 'chat_query_helper or chat_metadata_query_helper' --override-ini addopts=''`: **6 passed, 3 skipped, 43 deselected**. The existing official PostgreSQL fixture cases skipped; this is driver-boundary and SQLite evidence, **not a live PostgreSQL validation claim**. Details: `/tmp/postrelease-chat-sql-integration.log`. Existing pytest temporary-directory cleanup warnings also appeared.
- Scoped Bandit over `chacha/chat_history_queries.py` and the changed test, excluding B101 for test assertions: **0 findings, 0 errors**, JSON `/tmp/bandit_postrelease_chat_sql.json`. Existing nosec comment parsing warnings appeared.
- Ruff whole test file: 10 existing UP035/UP006/UP045 findings. Programmatic baseline/current JSON comparison against `git show HEAD:<file>` found **identical findings, no new lint findings**. Unrelated lint cleanup intentionally omitted.
- `git diff --check`: passed.

## Proposed inline reply

This comparison is normalized before PostgreSQL execution. Both helpers run through the database adapter, whose PostgreSQLBackend calls `prepare_backend_statement(..., apply_default_transform=True)`; the shared transform changes `m.deleted = 0` and `conv.deleted = 0` to `FALSE` before psycopg sees the query. The transaction wrapper reaches the same backend. Added six helper-to-driver regression cases covering search and metadata through the backend, transaction wrapper, and CharactersRAGDB adapter; all pass, and disabling normalization in memory makes all six fail. No production SQL change is needed. This validation uses the real adapter with a driver double; live PostgreSQL fixture cases were skipped locally.

## Coordination

Task TASK-13263.3 existed in In Progress before editing. Backlog workflow overview was read. A read-only task_view MCP call remained pending; task content was read from the existing file instead. Parent should record this report in the shared task with the other agents' findings. No commit, push, GitHub reply, version, legal, or protected-record edit was performed.

## Final policy follow-up

Added `@pytest.mark.unit` to the six-case regression and explicit annotations to `monkeypatch`, `operation`, `path`, the test return, and nested query executor parameters/return. The file now has 52 added lines rather than 51.

Revalidation after that edit:
- `python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_postgres_fts.py -q -m unit -k chat_helpers_reach_postgres_driver --override-ini addopts=''`: **6 passed, 27 deselected**, four existing warnings. Log: `/tmp/postrelease-chat-sql-final.log`.
- Scoped Bandit rerun: **0 findings, 0 errors**, B101 excluded for test assertions.
- Ruff baseline/current comparison rerun: **the same 10 existing findings, zero new findings**.
- `git diff --check`: passed.

## Independent review of postrelease_tests five-file patch

No actionable correctness findings.

Reviewed all five changed files and the relevant production/fixture callees. The isolation module marker and annotation changes preserve behavior. Prompt Studio generator annotations match their yielded objects. Media fixtures refresh the same inserted rows using their recorded UUIDs through `sync_refresh_fts_for_entity`; its create operation fetches the active Media row and delegates to the existing FTS updater on the supplied transaction connection, preserving the intended absence of video/audio document versions. The AuthNZ test now obtains the pool only after the official per-test fixture configures and resets the database settings, verifies the selected database before writes, and retains column restoration. The old shared and new isolated fixtures define the relevant api_keys columns identically; the pool's existing event-loop compatibility handling also covers the TestClient fixture boundary.

Review limitation: the PostgreSQL path remains unavailable locally, as recorded by the implementing agent; static lifecycle/schema inspection supplements its official-fixture skip and does not constitute a live PostgreSQL pass. No other agent's files were edited.


## Parent integration verification

Combined selection across all six touched Python modules plus licensing and the
Prompt Studio HTTP fixture smoke: **94 passed, 1 skipped**, clean exit 0.
The skip reason is the official AuthNZ PostgreSQL availability fixture:
"PostgreSQL not available; attempted docker start; skipping AuthNZ integration
tests." This is not a live PostgreSQL pass. Log:
`/tmp/postrelease-combined-backend.log`.

Independent cross-review of the frontend and five fixture files found no
additional actionable issues. Parent inspected both production frontend changes
and the helper-to-driver test. `git diff --check` passes; package version and
published LICENSES/releases/0.1.43 bytes exactly match immutable v0.1.43.
