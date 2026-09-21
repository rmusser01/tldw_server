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


# Research console CI race follow-up — TASK-13263.3

Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42
Branch: codex/release-0.1.43-review-followup
Only file changed in this work unit: apps/tldw-frontend/__tests__/pages/research-run-console.test.tsx
No production code, released source record, manifest, version, lockfile, commit, or push changes.

## Evidence and root cause

The final sync CI log /tmp/sync0143-frontend-shard2.log records `lazy-loads artifacts and completed bundles` failing at `getByRole('button', { name: 'Load plan.json' })`, while the exact-base replay passes all 19 tests. The test waits for `Investigate local evidence`, which is rendered from the run-list query. The artifact controls are rendered from `selectedSnapshot.artifacts`, supplied independently by subscribeResearchRunEvents in a later effect/event. A resolved run title therefore does not establish artifact readiness. No production invariant says they must arrive simultaneously.

The same test's bundle button is always rendered and remains disabled until getResearchRun refresh returns a completed run. Waiting for the button's enabled state avoids treating the completed user-event click as proof the asynchronous refresh has completed.

## Deterministic reproduction and repair

The existing final test now installs its event subscription without immediately delivering the snapshot. It waits for the visible run title and subscription, confirms the artifact button is absent, starts the button readiness query, then explicitly delivers the snapshot using act. No wall-clock delay, timer sleep, test retry, or timeout increase is involved.

With the original synchronous getByRole at that boundary, the isolated test reliably failed with the same missing Load plan.json error as CI (1 failed, 18 skipped). Changing the readiness query to findByRole fixes that deterministic failure. The bundle click additionally waits for its existing button to be enabled after refresh. Existing assertions still verify lazy artifact fetching, rendered artifact body, bundle request, and final answer.

## Verification

- RED: `./node_modules/.bin/vitest run __tests__/pages/research-run-console.test.tsx -t 'lazy-loads artifacts and completed bundles'` — exit 1; same missing Load plan.json error. /tmp/postrelease-research-race-red.log
- GREEN: `./node_modules/.bin/vitest run __tests__/pages/research-run-console.test.tsx` — exit 0, 19 passed. /tmp/postrelease-research-race-green.log
- `./node_modules/.bin/eslint __tests__/pages/research-run-console.test.tsx` — exit 0, no warnings/errors. /tmp/postrelease-research-race-lint.log
- `npm run typecheck` — exit 0. /tmp/postrelease-research-race-typecheck.log
- `git diff --check -- apps/tldw-frontend/__tests__/pages/research-run-console.test.tsx` — exit 0.

The only test-run warning is the existing Node experimental localStorage warning. This is a test scheduling assumption, not evidence of a released runtime regression; the published source remains untouched.

Parent reviewed the delayed-snapshot test repair: no production changes, timeout increases or assertion removals. All ten original review threads are answered and resolved with links to follow-up PR #2978; fixes remain unreleased until that PR is merged.


# Second Qodo review batch on PR2978

These changes remain separate from immutable v0.1.43.

# PR2978 test and setup review follow-up

Task TASK-13263.3, shared worktree `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42`, branch `codex/release-0.1.43-review-followup`. No commits/pushes; immutable releases, versions, legal records, and manifests untouched.

## Dispositions

- `PRRT_kwDOL1aGf86kRPcW` (test categories): fixed with one module category each: DSR `integration` (public service + actual storage boundary), CodeQL `unit` (bounded workflow-expression evaluation). Existing asyncio and parametrization decorators retained. Before edits `-m 'unit or integration'` selected 0 of 66; after edits all 66 pass.
- `PRRT_kwDOL1aGf86kRPcd` (DSR fixture types): fixed. `chroma_base` annotates MonkeyPatch, Path and Path return; all modified DSR test functions and nested stat helper also have annotations.
- `PRRT_kwDOL1aGf86kRPcj` (DSR test docstring): fixed, including explicit precondition and fail-closed outcome; remaining touched tests also document their behavior.
- `PRRT_kwDOL1aGf86kRPca` (private helper tests): meaningful public-boundary refactor. Every test now invokes `preview_data_subject_request` with its existing injected `users_repo` seam, then asserts response counts/summary or documented HTTPException `(500, requester_data_unavailable)`. No direct `_count_embeddings` or `_build_summary_for_user` calls remain. The category-order test no longer monkeypatches any category counter: it seeds three real notes through `CharactersRAGDB.note_store.add_note`, leaves unselected media/audit stores absent, and asserts the public summary's canonical order and counts. The optional, lazy-import Chroma factory remains mocked to control missing-library/list/count failures. This is the only private service collaborator still patched, because the service has no public manager injection seam; adding production dependency machinery solely for this test is unwarranted. Path-policy/stat error simulation uses existing public DatabasePaths/Path boundaries. Real public preview resolution, category normalization, selection, count aggregation, and error translation remain exercised.
- `PRRT_kwDOL1aGf86kRPcI` (setup helper docstrings): fixed. Both helpers document public projection, already-verified authorization input, local-provider restrictions and unchanged secret/path redaction.
- `PRRT_kwDOL1aGf86kRPcE` (move setup projection into core): retain current architecture with evidence. Repository-wide search finds these private projections consumed only inside this HTTP endpoint module. Stored state is intentionally richer than anonymous HTTP state. `get_first_run_state` resolves the request principal and `system.configure` permission before projection; POST and skip responses retain default redaction. The public result is reconstructed and validated as `FirstRunStateResponse`. There is no demonstrated second consumer requiring a shared core service, and moving this route-specific response shaping would only relocate the existing allowlist/auth policy. Existing HTTP tests already exercise anonymous, permitted and unprivileged principals across six POSIX/Windows/UNC/relative paths, persistence, first-chat reads and skip responses. They pass, along with hosted-provider path rejection and secret-shaped ID rejection. No runtime policy change was made.

## Evidence

All Python commands first activate `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate`. Pytest prefix: `PYTHONPATH=/tmp/pypi0142-ci-pytest-plugins:. MINIMAL_TEST_APP=1 python -m pytest -q`.

- Pre-change category collection of DSR + CodeQL modules: **66 deselected / 0 selected**, `/tmp/pr2978-category-before.log`.
- DSR + CodeQL modules with `-m 'unit or integration'`: **66 passed**, 8 warnings, 4.31s. `/tmp/pr2978-dsr-codeql.log`.
- `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -k 'provider_model_identifier_roundtrips_path_like_ids or rejects_path_model_for_hosted_provider or provider_model_identifier_still_rejects_secret_values'`: **21 passed**, 118 deselected, 6 warnings, 10.27s. `/tmp/pr2978-setup-boundary.log`.
- Ruff for all three edited files: **all checks passed**, baseline zero/current zero. `/tmp/pr2978-tests-setup-ruff-baseline.json`, `/tmp/pr2978-tests-setup-ruff.json`.
- Bandit on the same scope: **12 baseline / 12 current**, no new findings. `/tmp/pr2978-tests-setup-bandit-baseline.json`, `/tmp/pr2978-tests-setup-bandit.json`. Baseline snapshots were captured before edits in `/tmp/pr2978-tests-setup-baseline`.
- `git diff --check`: clean.

No production runtime change; setup edits are docstrings. No new production test abstraction was added. The category-selection failure was observed before edits and fixed afterward. DSR tests catch returning synthetic zero/partial coverage, querying unavailable unselected stores, reordered summaries, and loss of public error translation.

## Edited files

- `tldw_Server_API/app/api/v1/endpoints/setup.py`
- `tldw_Server_API/tests/Admin/test_dsr_preview_coverage.py`
- `tldw_Server_API/tests/CI/test_codeql_cache_event_boundaries.py`


# PR2978 frontend findings — second review

TASK-13263.3; branch codex/release-0.1.43-review-followup; worktree /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42.

## Dispositions

1. ExecutionPanel null checks — PRRT_kwDOL1aGf86kRPcQ, discussion_r4060129290: addressed by documenting deliberate nullish semantics. Both `null` and `undefined` mean timestamp absent; numeric epoch 0 is a valid timestamp. Workflow store uses optional startedAt/completedAt and resets completedAt to undefined; existing component fixtures also reset timestamps to null. Loose equality with null is intentional here and has no arbitrary numeric/string coercion path. Added one explanatory comment before the effect/useMemo timer calculations; no behavior changed.

2. HealthSummary filename — PRRT_kwDOL1aGf86kRPcT, discussion_r4060129293: recommend dismissing the rename as legacy naming cleanup outside this bounded follow-up. The existing shared Settings directory contains many peer component names with the same convention (health-status.tsx, general-settings.tsx, model-settings.tsx, preferences-settings.tsx, system-settings.tsx, ui-customization.tsx, tldw.tsx). The PascalCase requirement located by the review exists in apps/extension/AGENTS.md, which is not an ancestor of apps/packages/ui/src/components/Option/Settings/health-summary.tsx; no applicable shared-package AGENTS mandates it. Repo search finds the default-exported HealthSummary definition and two historical docs references, with no live imports or runtime route discovery relying on this file name. No broken import or component discovery was demonstrated. Renaming one old file would introduce churn and change a historical documentation path without addressing a runtime defect; no edits made to this file.

3. Native extension storage probe — PRRT_kwDOL1aGf86kRPcs, discussion_r4060129325: documented intentional platform boundary; reject unconditional replacement with wxt/browser. The caller agent-task-handoff.ts implements two storage lanes: native extension session/local storage, or window.sessionStorage when extension storage is absent/fails. The WebUI aliases wxt/browser to apps/tldw-frontend/extension/shims/wxt-browser.ts; that shim always provides storage.session backed by module-level sessionMemory Map (lines 135-150), not window.sessionStorage. Importing the wrapper unconditionally would make the native capability probe succeed in a non-extension browser and replace the caller's existing tab-persistent fallback with a module-lifetime store, changing full navigation/reload behavior. The adapter also handles callback APIs and runtime.lastError alongside Promise APIs and preserves method receiver binding. The existing 13 handoff tests cover these behaviors, null tombstones preventing stale fallback resurrection, missing methods, throw/reject failures, local fallback, and expiration. Added a three-line comment above the native resolver explaining the WebUI shim exception and native callback error contract. No behavior/import changed. This rejects the proposed mechanism; it does not claim arbitrary browser namespaces are supported beyond the implemented native API compatibility.

## Exact changes

- apps/packages/ui/src/components/WorkflowEditor/ExecutionPanel.tsx — one intent comment.
- apps/packages/ui/src/services/web-clipper/extension-storage.ts — three intent-comment lines.

No tests, health-summary code, imports, release records, versions, manifests, or lockfiles changed. No commits or pushes.

## Verification

- WebUI Vitest owning suites: ExecutionPanel.design-system.test.tsx (3) + agent-task-handoff.test.ts (13), 16/16 passed. /tmp/pr2978-frontend-second-tests.log
- Scoped ESLint across all three reviewed files: exit 0, zero errors, one pre-existing unused TextArea warning in ExecutionPanel. Baseline reproduced by linting HEAD through --stdin. /tmp/pr2978-frontend-second-lint.log and /tmp/pr2978-frontend-second-lint-baseline.log. Both root invocations print the same Next pages-directory location diagnostic.
- git diff --check on the two edited files: passed.
- No redundant full TypeScript run: comments alone do not change code, imports, or type contracts.
- Expected mocked storage failure warnings appear in the tests that specifically exercise warning/fallback handling. No new runtime warning path added.


# PR2978 Docker and audio follow-up — TASK-13263.3

Working tree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42`.

## Docker profile data omission: fixed

Review thread: `PRRT_kwDOL1aGf86kRPcy`, https://github.com/rmusser01/tldw_server/pull/2978#discussion_r4060129331

The root `pyproject.toml:541–546` declares profile schema and fixture JSON data installed beneath `share/tldw_profile_core`. Production previously copied only the package's `src` directory; workers already copied the complete directory. The omission is real at the wheel build boundary, even though Python imports succeed. No claim is made that the Python profile API itself reads those JSONs: the failure is its declared distribution payload being incomplete.

Changes:
- `Dockerfiles/Dockerfile.prod`: copy all of `packages/tldw_profile_core` into the builder, matching worker images; retain package imports and additionally parse the installed `schemas/personal-context-v1.json` and `fixtures/v1/01-manifest.json` beneath `sys.prefix/share/tldw_profile_core` in the runtime build smoke check. The `/install` prefix copied to `/usr/local` carries that data location correctly.
- `tests/CI/test_worker_container_packaging.py`: extend the existing manifest-derived COPY contract to production; stop parsing at the second FROM so runtime copies cannot falsely satisfy builder requirements.

Red: before changing Dockerfile, the expanded contract produced **1 failed, 5 passed**. The failing production case listed omitted schema/fixture paths. Log: `/tmp/pr2978-docker-red.log`.

Green: all six container cases pass as part of the final combined suite.

Actual packaging experiment: copied the production first-stage COPY inputs into a temporary directory, invoked `python -m pip wheel --no-deps --no-build-isolation` there, and inspected the resulting wheel. The wheel built successfully and contains **47 profile data files**, including the required schema and fixture. Build output: `/tmp/pr2978-profile-wheel.log`. Temporary build context and wheel were cleaned afterward. This validates the packaging boundary on the host Python environment; a full Docker image/apt build was not run locally.

## Qwen Hub-shaped CWD collision: intentional security rejection retained

Review thread: `PRRT_kwDOL1aGf86kRPc4`, https://github.com/rmusser01/tldw_server/pull/2978#discussion_r4060129339

Disposition: the observed rejection is real but intentional under the current loader contract. The recommended Hub-first normalization is unsafe here; no production audio change made.

Evidence:
- Commit `28ddb8ad8c` deliberately introduced this directory-precedence check with the stated purpose of applying local-path policy before model loading. Existing tests explicitly require rejection of `organization/local-model` when it names an existing directory outside `WHISPER_MODEL_BASE_DIR`.
- `Audio_Transcription_Lib.py:788` normalizes Qwen identifiers; the public validator at `:825` adds artifact checks. `load_qwen2audio`'s normal configuration route at approximately `:2305` calls normalization and then passes its result directly to both Transformers `from_pretrained` calls around `:2375`. The planned route and `stt_provider_adapter.py:1491` use the public validator and additionally require local execution. Those are all callers.
- The installed `transformers/utils/hub.py:432–446` checks `os.path.isdir(path_or_repo_id)` and returns matching local artifacts before reaching Hub lookup. `transformers/configuration_utils.py:708` also checks local-directory precedence. Returning unchanged `organization/model` from normalization does not force remote resolution.
- A network-free experiment called the **real** Transformers `cached_file('organization/local-model', 'config.json', local_files_only=True)` with a CWD directory outside the managed root. It returned that local config. The public Qwen validator rejected the same identifier. Log: `/tmp/pr2978-audio-collision-proof.log`. An initial experiment assertion compared a macOS temporary-path alias with its resolved path; correcting both sides to resolved paths produced the documented successful proof.
- Accepting the remote ID despite a collision would need a separate remote-resolution design that passes a confined, explicit snapshot path to Transformers. Reordering the validator alone would restore arbitrary local loading. That redesign is outside this bounded review and is not required to preserve the existing security contract.

Added two unit cases to `tests/MediaIngestion_NEW/unit/test_audio_transcription.py` using the public validator and `load_qwen2audio`, covering both `organization/local-model` and explicit `./organization/local-model`. They assert rejection before runtime initialization/loading. Existing focused tests also cover ordinary valid Hub IDs, complete local models, managed relative models, and directory symlinks. All new functions have explicit annotations, category markers and docstrings.

## Verification

All Python commands activated `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate` first.

- `python -m pytest tldw_Server_API/tests/CI/test_worker_container_packaging.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_transcription.py -q -k 'worker or model_identifier or qwen_loader_rejects_shadowing' --override-ini addopts=''`: **26 passed, 45 deselected**, four existing dependency/config warnings. Existing pytest stale temporary-directory cleanup warnings followed. Log: `/tmp/pr2978-docker-audio-final.log`.
- Ruff on both changed Python tests: **0 findings**, with baseline comparison also 0.
- Bandit on both changed tests, B101 excluded for ordinary pytest assertions: **0 findings, 0 errors**. JSON: `/tmp/pr2978-docker-audio-bandit.json`. Dockerfile is not Python; production audio code was not changed.
- `git diff --check`: passed.

## Proposed review replies

Docker: “Fixed by copying the complete profile package into the production builder, matching the workers, and adding a runtime build smoke check that parses an installed schema and fixture. The existing manifest/COPY contract now covers the production builder stage and failed before this fix. A wheel built from those COPY inputs includes all 47 declared profile JSON data files.”

Audio: “This rejection preserves the existing local-model confinement policy. Transformers gives an existing CWD directory precedence over Hub lookup: a direct, offline `cached_file` reproduction selects the outside-root directory for the same Hub-shaped string. Returning that string before checking the local directory would therefore cause `from_pretrained` to load the unconfined directory. Retained the guard and added public validator/loader regressions covering implicit and explicit relative paths, with the managed-local and symlink tests passing. Supporting remote resolution despite a collision would require resolving to a confined snapshot before loading, not only changing classification.”

Only the Dockerfile and two named test modules were edited by this agent. No commit, push, GitHub message, version, legal record, tag, or manifest change performed.


# PR2978 retry warning context — TASK-13263.3

Thread PRRT_kwDOL1aGf86kRPcN is fixed in both single and bulk requeue paths.
Each schema-validation warning binds operation, normalized stage, DLQ stream and
Redis-returned entry ID. The message continues to include only the exception
class; response warnings remain static and no invalid payload or exception text
is logged. Existing stream-name validation runs before these log sites.

The existing parameterized HTTP regression now captures Loguru records, verifies
all four safe context fields, and checks that the sentinel invalid media value
appears in neither logs nor responses. Existing successful requeue, deletion and
client cleanup assertions remain. Both cases failed before the implementation
with KeyError: operation; the complete module now passes all five tests.

Verification (project virtual environment activated first):
- Red: pytest test_dlq_admin_endpoints.py -k sanitizes_schema_warning: 2 failed,
  3 deselected. /tmp/pr2978-dlq-red.log.
- Green: complete module: 5 passed, 2 existing warnings.
  /tmp/pr2978-dlq-green.log.
- Ruff: baseline 0, current 0.
- Bandit: baseline 17/current 23, all B101 ordinary test assertions; no other
  findings and no analysis errors. Six new behavioral assertions account for
  the increase. No production finding introduced.

Only embeddings_v5_production_enhanced.py and test_dlq_admin_endpoints.py changed.


Independent cross-review of Docker packaging and Qwen regression tests found no actionable issues; 26 tests passed independently. Parent reviewed the two frontend intent comments. Final readiness work and integrated verification remain pending.


# Independent review — second-batch setup/tests/Embeddings

No actionable correctness or regression findings in the reviewed changes.

Reviewed the full current diffs and `/tmp/pr2978-tests-setup-second-review.md` for:
- `app/api/v1/endpoints/setup.py`: only docstrings; described authorization/redaction contract matches the existing gates and Pydantic reconstruction.
- `tests/Admin/test_dsr_preview_coverage.py`: the injected AuthnzUsersRepo mock preserves its asynchronous method contract; tests exercise real public requester resolution/category normalization/count aggregation/error translation. The real notes store and missing unselected media/audit stores preserve meaningful category-selection coverage. Embedding manager/path failures still produce authoritative failure or confirmed-zero outcomes through the public service. No new production seam or global persistent mutation was introduced.
- `tests/CI/test_codeql_cache_event_boundaries.py`: module unit marker affects test selection only.
- `app/api/v1/endpoints/embeddings_v5_production_enhanced.py`: both retry warnings bind the normalized allowlisted stage, a fixed-format stream selected by that stage, and an entry ID returned by Redis. No payload, arbitrary exception text, request field overrides, or user content is added to the diagnostic. Type-only exception reporting and API warning strings remain intact. Single/bulk operation names correctly distinguish the paths.
- `tests/Embeddings/test_dlq_admin_endpoints.py`: log sink is removed in `finally`; assertions inspect the emitted structured record, preserve request/response and queue-mutation checks, and scan formatted messages plus extras for the secret canary.

Independent execution after activating the root virtual environment:

`PYTHONPATH=/tmp/pypi0142-ci-pytest-plugins:. MINIMAL_TEST_APP=1 python -m pytest -q tldw_Server_API/tests/Admin/test_dsr_preview_coverage.py tldw_Server_API/tests/CI/test_codeql_cache_event_boundaries.py tldw_Server_API/tests/Embeddings/test_dlq_admin_endpoints.py --override-ini addopts=''`

Result: **71 passed**, six existing warnings, 4.00s. Log: `/tmp/pr2978-independent-review-tests.log`.

No repository files edited and no commits made during this review.


# PR2978 readiness response validation

TASK-13263.3; fixes Qodo thread `PRRT_kwDOL1aGf86kRPcM`. Working checkout `.worktrees/release-main-0.1.42`, branch `codex/release-0.1.43-review-followup`. No commits/pushes; immutable release records, versions and manifests untouched.

## Change

Added versioned `health_schemas.py` with `ReadinessResponse` and its small compatibility database model. Strict known field types reject malformed readiness flags, engine/diagnostic maps, provider flags, and compatibility database values. Existing optional collector sections remain optional; dynamic engine metrics and additional already-sanitized collector diagnostics remain supported. `time` stays a string, preserving the existing ISO `+00:00` wire representation.

`api_readiness` explicitly calls `ReadinessResponse.model_validate` before returning `JSONResponse`; merely declaring the FastAPI response model would not validate that Response object. JSON-mode dumping with `exclude_unset=True` keeps omitted diagnostic sections absent. The route declares the model for both 200 and 503 OpenAPI responses. Existing status selection and no-store headers remain unchanged.

Tests cover malformed fields, rich ready/unready output including dynamic metrics and nullable policy values, sparse shutdown output, exact UTC offset string format, and the OpenAPI schema reference. No collector, public liveness route or authorization policy changed.

## Tests and checks

All Python commands activated the project root venv.

- RED: new compatibility tests before production edits: **6 failed / 5 passed**. Five invalid field shapes were accepted, and OpenAPI lacked a response model. `/tmp/pr2978-readiness-red.log`.
- GREEN: entire `tldw_Server_API/tests/Health` suite: **45 passed**, 10 warnings, 113s. `/tmp/pr2978-readiness-green.log`.
- CI dependency confirmation: readiness compatibility subset with previously prepared Pydantic2.13.5/settings2.15.0 on PYTHONPATH: **11 passed**, 11 deselected, 6 warnings. Includes final explicit `+00:00` wire-format assertion. `/tmp/pr2978-readiness-ci-tests.log`.
- Ruff: baseline **0**, current **0** for production/test scope.
- Bandit: **16 baseline -> 29 current**; all 13 new findings are B101 assertions in the pytest module. No suppressions added; **zero new production or non-assert findings**. `/tmp/pr2978-readiness-bandit-baseline.json`, `/tmp/pr2978-readiness-bandit.json`.
- `git diff --check`: clean.
- Ignored OpenAPI TypeScript artifacts regenerated with installed `openapi-typescript 7.13.0`; full WebUI `tsc --noEmit` completed without diagnostics. `/tmp/pr2978-readiness-tsc.log` is empty.
- Fresh canonical exporter `--check` matches the updated tracked fingerprint. `/tmp/pr2978-readiness-fingerprint-check.log`.

## Exact API drift control

The ordinary root virtual environment has Pydantic2.11.7 and produces one unrelated extra schema, so that candidate fingerprint was never written to the repository. The initial standard frontend generator also lacked the vendored profile package import. Reused the already-existing isolated CI dependencies from `/tmp/release-openapi-deps` and explicitly included `packages/tldw_profile_core/src`, matching the prior documented release export (`Docs/superpowers/reviews/2026-09-20-release-openapi-drift.md`). No packages installed or shared environment changed.

Canonical command:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
PYTHONPATH=/tmp/release-openapi-deps:.:packages/tldw_profile_core/src python Helper_Scripts/export_openapi_schema.py --out apps/tldw-frontend/lib/api/generated/openapi.json --fingerprint /tmp/pr2978-readiness-ci-fingerprint.json
```

The original endpoint source was captured before edits and separately exported from an in-memory module override, without reverting working tree files. Taking the new CI-dependency schema and replacing only `GET /api/v1/health/ready` with that original operation, then removing `ReadinessResponse` and `ReadinessDatabaseStatus`, reproduces the recorded pre-change fingerprint **exactly**:

- Before: `4d8718a9387567b3278a9034acb13e39bf75375714632e90c49ffc3dbda58e22`, **2097 paths / 3207 schemas**.
- After: `0ded5fa90b1485ecdd0a256450631f0631d1b77cd5cb93c33a41ff45302616ac`, **2097 paths / 3209 schemas**.

Thus the complete drift is the readiness response contract and two new models; no unrelated environment-dependent changes entered the fingerprint. Evidence: `/tmp/pr2978-readiness-drift-control.txt`. The tracked fingerprint was refreshed only after the comparison passed.

## Files

- `tldw_Server_API/app/api/v1/endpoints/health.py`
- `tldw_Server_API/app/api/v1/schemas/health_schemas.py` (new)
- `tldw_Server_API/tests/Health/test_shared_readiness_service.py`
- `apps/tldw-frontend/lib/api/openapi.fingerprint.json`
- Ignored generated OpenAPI/type artifacts refreshed under `apps/tldw-frontend/lib/api/generated/`.


# Final sync PR2971 review inventory

Nine additional threads on sync head `557b75a0ee9d43803988a3f1b8decf7b7ac265a8`
were inventoried before merge. Confirmed repairs live in PR2978; the sync carries
the immutable published source. These dispositions do not imply the fixes were
included in v0.1.43.

- `PRRT_kwDOL1aGf86kRFrZ` (frontend coverage): retain the explicit existing
  report-only coverage summaries. The repository says “Aim for >80% code
  coverage,” not that every existing frontend package already has an enforced
  80% baseline. Required package-owned unit shards, type checks and lifecycle/E2E
  gates still fail on errors. Turning the baseline report into a new global
  threshold is a separate coverage-policy change, not repair of this sync.
  No threshold or required test is weakened by this follow-up.
- `PRRT_kwDOL1aGf86kRFru` (publisher gate): duplicate of the documented, explicitly
  approved bounded publisher-gate decision. Required PR application checks precede
  the release; publisher contracts/startup/packaging are supplemental publication
  checks. Do not reverse the requester's explicit adoption or claim full-suite
  publisher coverage.
- `PRRT_kwDOL1aGf86kRFrq` (nullish timer): same documented intent fix as PR2978
  `RPcQ`; preserves both absent representations and valid epoch zero.
- `PRRT_kwDOL1aGf86kRFr4` (Docker profile data): same reproduced/fixed omission as
  PR2978 `RPcy`; full directory copy, installed-data smoke, manifest contract and
  actual 47-file wheel validation are recorded above.
- `PRRT_kwDOL1aGf86kRFr7` (Qwen metadata probe): retain the existing classifier
  deliberately. It does perform a directory metadata probe before confinement;
  this is not described as a probe-free boundary. Transformers itself selects an
  existing CWD directory before Hub resolution, demonstrated by the real offline
  experiment above. The suggested canonical-Hub-first classification would pass
  the outside-root directory to the loader and weaken the actual content-loading
  boundary. Both implicit/explicit outside-root cases are rejected before model
  loading, and existing symlink/local-artifact tests pass. Supporting remote
  resolution despite a local collision without this probe requires a distinct
  explicit remote-snapshot loader contract; this review does not introduce that
  redesign or claim arbitrary-path metadata is never consulted.
- `PRRT_kwDOL1aGf86kRFr0` (saved image detail): fixed in both successful decoding
  branches. Only string auto/high/low values persist; absent, null, unsupported
  and non-string options normalize to auto. Public request-schema validation is
  unchanged. New cases use actual small/large PNG decoding, database persistence
  and saved-content reconstruction. Red: 10 failed/12 passed; green owning
  module: 57 passed/1 official PostgreSQL-unavailable skip. No live PostgreSQL
  validation is claimed. Logs `/tmp/pr2978-image-{red,green}.log`.
- `PRRT_kwDOL1aGf86kRFre` (soak CLI logging): replaced print diagnostics with
  Loguru operation/outcome fields, safe exception class for invalid inputs, and
  visible PASS/FAIL text. Exit codes and exclusive JSON output remain intact;
  paths, payloads, credentials and exception messages are not logged.
- `PRRT_kwDOL1aGf86kRFrh` (soak request attribution): caught workload and telemetry
  exceptions now retain bounded timeout/http_error/invalid_json/response_size_limit/
  invalid_response counters. Initial, per-phase and final telemetry catches all
  record attribution. Log only the first category occurrence per phase/workload;
  repeated requests increase counts without flooding or retaining raw errors.
  Existing HTTP statuses and response-equality error counters are unchanged.
- `PRRT_kwDOL1aGf86kRFrm` (soak contracts): expanded public helper docstrings with
  arguments, return values and applicable exceptions; documented evidence fields
  and safe logging in `Docs/Development/Release_Capacity_Soak.md`.

The ten new soak cases failed before implementation (missing categories/logs).
All 40 owning tests now pass, including the existing real loopback HTTP test.
That test initially encountered the sandbox's loopback bind restriction; rerun
with the required local-network permission passed. New phase-boundary tests
observe actual gather boundaries rather than depending on a fixed sample count.
Log `/tmp/pr2978-soak-final.log`.

Soak/image validation: Ruff has three unchanged chat import-order findings and
no new findings. Bandit baseline85/current100 contains only ordinary pytest B101
assertions; no production/non-assert findings, no suppressions, no analysis
errors. New code is manually reviewed by the parent. The parallel review agents
hit account usage limits after their earlier reports; no additional independent
review is claimed for the final soak/image changes.

Parent combined second-batch verification before the soak/image additions:
112 tests passed across DSR, CodeQL, Docker packaging, DLQ, readiness and licensing.
The subsequent owning suites above verify the added production paths. Release
records and package versions remain unchanged. Remote final-head CI and final
review replies remain required.
