---
id: TASK-2401
title: Resolve PR 1982 dev-to-main conflicts with dev precedence
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-26 19:58'
labels:
  - merge-conflict
  - pr-1982
  - dev-main
dependencies: []
priority: high
modified_files:
  - apps/mcp-unified/pyproject.toml
  - Docs/Plans/2026-06-23-pr1982-dev-main-conflict-resolution.md
  - tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
  - .github/workflows/ui-watchlists-extension-e2e.yml
  - apps/packages/ui/src/components/Notes/__tests__/task-markdown.test.ts
  - apps/packages/ui/src/components/Notes/task-markdown.ts
  - .github/workflows/e2e-smoke.yml
  - tldw_Server_API/tests/CI/test_required_workflow_contracts.py
  - tldw_Server_API/tests/Helper_Scripts/test_mcp_standalone_user_guide_uat.py
  - tldw_Server_API/tests/RAG_NEW/unit/test_analytics_db_dev_reconciliation.py
  - tldw_Server_API/app/core/DB_Management/media_db/runtime/fts_ops.py
  - tldw_Server_API/tests/DB_Management/test_media_postgres_support.py
  - apps/extension/tests/e2e/utils/extension-build.test.ts
  - apps/extension/tests/e2e/utils/extension-build.ts
  - apps/extension/tests/e2e/utils/extension-id.test.ts
  - apps/extension/tests/e2e/utils/extension-id.ts
  - apps/extension/tests/e2e/utils/extension-paths.test.ts
  - apps/extension/tests/e2e/utils/extension-paths.ts
  - apps/extension/tests/e2e/utils/extension.launch.test.ts
  - apps/extension/tests/e2e/utils/extension.ts
  - .github/workflows/ci.yml
  - tldw_Server_API/app/core/Context_Integrity/inventory.py
  - tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
  - tldw_Server_API/tests/Context_Integrity/unit/test_inventory.py
  - tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v51.py
  - tldw_Server_API/tests/http_client/test_http_client_retry_after.py
  - tldw_Server_API/app/core/Audio_Studio/export.py
  - tldw_Server_API/app/core/Audio_Studio/render.py
  - tldw_Server_API/tests/Audio/test_failopen_cap_minutes.py
  - tldw_Server_API/tests/Audio/ws_test_helpers.py
  - tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_export.py
  - tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_render.py
  - tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py
  - tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py
  - tldw_Server_API/app/core/Research/jobs.py
  - tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py
  - tldw_Server_API/tests/Audio/test_ws_tts_endpoint.py
  - tldw_Server_API/tests/Chat/unit/test_streaming_structured_events.py
  - tldw_Server_API/tests/Streaming/test_ws_pings_labels_multi.py
  - tldw_Server_API/app/core/Chat/chat_service.py
  - tldw_Server_API/app/core/DB_Management/db_migration.py
  - tldw_Server_API/tests/Chat_NEW/integration/test_chat_loop_dual_emit_compat.py
  - tldw_Server_API/tests/DB_Management/test_db_migration_planning.py
  - tldw_Server_API/tests/Local_LLM/test_llamacpp_hardening.py
  - tldw_Server_API/app/core/Sandbox/orchestrator.py
  - tldw_Server_API/tests/sandbox/test_cross_runtime_cleanup_contracts.py
  - tldw_Server_API/tests/sandbox/test_orchestrator_artifact_security.py
  - tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py
  - tldw_Server_API/tests/AuthNZ/unit/test_api_keys_repo_schema_strictness.py
  - tldw_Server_API/tests/Collections/test_reading_import_export.py
  - tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py
  - tldw_Server_API/tests/Services/test_startup_worker_groups.py
  - tldw_Server_API/tests/prompt_studio/integration/test_mcts_integration.py
  - tldw_Server_API/app/api/v1/endpoints/workflows.py
  - tldw_Server_API/tests/Workflows/test_engine_scheduler.py
  - tldw_Server_API/tests/Workflows/test_adapter_path_security.py
  - tldw_Server_API/tests/Evaluations/test_synthetic_eval_service.py
  - tldw_Server_API/tests/Personalization/test_personalization_endpoints.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the open PR #1982 (`dev` -> `main`) conflict state by merging current `origin/main` into current `origin/dev` while preserving `dev` for overlapping conflicts, then verify and push the resulting merge back to `dev` if clean.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Confirm the current PR #1982 conflict surface between `origin/dev` and `origin/main`.
- [x] #2 Merge `origin/main` into the PR head with `dev` winning overlapping conflicts.
- [x] #3 Verify no unresolved merge paths or conflict markers remain before pushing.
- [x] #4 Push the verified merge back to `dev` and confirm PR #1982 merge state/checks update.
- [x] #5 Address current PR #1982 MCP Unified Internal RC package-boundary failure.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/2026-06-23-pr1982-dev-main-conflict-resolution.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `git merge-tree --write-tree origin/dev origin/main` identified a single content conflict in `README.md`.
- Merged `origin/main` into the work branch from `origin/dev` with `git merge origin/main -X ours --no-edit`; Git auto-merged `README.md` with the `dev` side winning the overlap.
- Verification before push: `git status --short --branch` showed no unresolved paths; `rg -n '<<<<<<<|=======|>>>>>>>' README.md` returned no matches; `git diff --check HEAD~1 HEAD` exited 0; `git diff --quiet HEAD:README.md origin/dev:README.md` exited 0.
- Pushed the conflict-resolution branch to `dev` at `3e9756f5f2bd7c1d577f439b5780eba386aed801`; GitHub PR #1982 moved from `DIRTY` to `UNSTABLE`, confirming conflicts were cleared and checks were running.
- The current PR #1982 `MCP Unified Internal RC` job failed because the built `mcp-unified` wheel omitted `mcp_unified.policy_grants`; gateway imports then failed with `ModuleNotFoundError: No module named 'mcp_unified.policy_grants'`.
- Added `mcp_unified.policy_grants` to the standalone setuptools package list and extended package-boundary coverage so the wheel/sdist must include that package.
- Verification for the package fix: targeted package-boundary pytest failed before the fix with the missing package assertions, passed after the fix, and `PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python make mcp-unified-rc` exited 0 with `RC status: ok`.
- Current PR #1982 CI failures also include four `rag-new-unit-core-misc` shards failing because `FakePostgresBackend` lacked `table_exists`; added that fake backend method and verified `test_analytics_db_dev_reconciliation.py` passes locally.
- `core-utils-tooling` shards failed because the UAT test still referenced the removed root-level `mcp_unified/pyproject.toml` package layout and did not pass the standalone package install args; updated the test to use `apps/mcp-unified`, assert the `src` package layout, and cover the additional tool-event UAT steps.
- `Watchlists Extension E2E (No Skips)` failed because the extension launch skipped all tests when no system Chrome extension target appeared; updated the workflow to use Playwright Chromium and minimal locales, with a workflow contract test covering that behavior.
- GitHub Advanced Security CodeQL reported checklist label metadata sanitization in `task-markdown.ts`; replaced the HTML-comment regex with a scanner that removes multiline and dangling checklist metadata comments, with Vitest regressions.
- Local verification after these fixes: `bunx vitest run src/components/Notes/__tests__/task-markdown.test.ts` passed; focused Python pytest covering UAT, RAG reconciliation, workflow contract, and MCP package-boundary checks passed with 12 tests; `git diff --check` passed; Bandit on touched Python files passed with no issues.

Old-head CI later exposed Full Suite shard integrations failing test_reconcile_content_update_updates_media_fts_versions_and_binding because SQLite external-content FTS deleted using the already-updated media_fts row. Changed _update_fts_media to delete with the supplied old title/content payload, preserving synonym expansion for old and new content, and updated the unit expectation. Local verification: focused failing integration plus FTS unit passed, full test_media_postgres_support.py passed, and test_sync_coordinator.py passed.
- New-head Watchlists Extension E2E was canceled after `bunx playwright install --with-deps chromium` spent about 43 minutes downloading browsers and exhausted the 45-minute job window before the backend could start. Reverted that job to the runner's system Chrome channel while keeping `TLDW_E2E_EXTENSION_MINIMAL_LOCALES=1`; local verification: targeted workflow contract passed, full workflow-contract test file passed (36 tests), focused prior-failure Python regression set passed (16 tests), `git diff --check` passed, and Bandit on the touched CI contract test found no issues.
- Latest Watchlists Extension E2E reached the Playwright spec but skipped all 14 tests after repeatedly logging `[E2E_DEBUG] No service worker found after waiting`; the strict no-skip gate then failed with `passed=0 skipped=14`. Added deterministic staged extension IDs via a test manifest key, resolved extension IDs from that manifest key when no MV3 service worker/background target is active, and preserved the real default locale catalog while pruning non-default locales. Local verification: the extension utility Vitest set passed (12 tests) and `bun run compile` passed; local headed/headless browser probes could not complete in this macOS sandbox before browser startup timeout, so CI remains the end-to-end confirmation for the full Watchlists workflow.
- The same push exposed `Extension Research Workspace Parity` timing out after launch under `launchWithBuiltExtension`. Scoped deterministic manifest-key injection to the Watchlists/general `launchWithExtension` path only, leaving built-extension parity staging on its prior extension ID behavior while still preserving default locale catalogs. Local verification after scoping: extension utility Vitest set passed (13 tests), `bun run compile` passed, and `git diff --check` passed.
- Follow-up parity root cause: headed Chromium hung inside `chromium.launchPersistentContext` when `launchWithBuiltExtension` staged the full default `messages.json` under `TLDW_E2E_EXTENSION_MINIMAL_LOCALES=1`; restoring the built-extension launcher to the lightweight default-locale stub while leaving the general Watchlists launcher on the real default locale catalog made the reproduced headed parity spec pass. Local verification: extension utility Vitest set passed (14 tests), headed `bunx playwright test tests/e2e/research-workspace.parity.spec.ts --reporter=line` passed (1 test), strict headed `bun run test:e2e:workspace-parity:strict` passed with `[playwright-no-skips] passed=1 skipped=0 unexpected=0 flaky=0`, `bun run compile` passed, and `git diff --check` passed. Bandit was not applicable because this follow-up only touched TypeScript test utilities.
- 2026-06-26 CI fan-out triage: PR #2520 historical `CI` run `28178992004` had 726 jobs; PR #1982 current `CI` run `28222746881` was stuck at `pending` with zero jobs because stale queued workflow runs from older PR #1982 head SHAs and merged PR #2520 were still occupying/colliding with the Actions queue. Normal cancellation cleared most queued runs; force-cancel cleared the stuck old `ci-1982` run `28219062432`, after which current `CI` moved to 8 starter jobs and then expanded to 726 jobs. Current PR #1982 now shows 777 total checks. New failure roots identified from logs: shard coverage has 22 unassigned test files; `core-audit-unified` fails on `CancelledError` escaping fixture teardown through `UnifiedAuditService.stop()`; Windows `core-utils-tooling` prompt inventory tests return no prompt assets; and `UI Watchlists Extension E2E` remains failed separately.
- 2026-06-26 follow-up CI fixes before pushing another 700+ check run: added the five missing shard path groups to `.github/workflows/ci.yml`; restored the audit test's monkeypatched `flush` before fixture teardown; added portable prompt inventory fallbacks for Windows when fd-relative traversal is unavailable; aligned Watchlists extension launch staging with the lightweight default-locale path and covered it with Vitest; and made AuthNZ migrations opt into Redis-to-file migration-lock fallback only in test mode so CI jobs that export `REDIS_URL` without starting Redis do not fail unrelated shards. Verification: shard coverage passed with `shards=721 test_files=3882 ignored=4 baseline=130 new_uncovered=0`; focused Python regression set passed 57 tests; Redis-unavailable admin conflict subset passed 9 tests; Redis-unavailable Prompt Studio startup representative passed 1 test; extension utility Vitest passed 12 tests; `bun run compile` passed; `bun run build:chrome:prod` passed after temporarily pointing the local generated `antd` symlink at the installed Bun package target, then restoring it; `git diff --check` passed; Bandit over touched Python scope produced only existing pytest `B101` assert findings, and rerun with `-s B101` produced 0 findings.

- 2026-06-26 current-run CI fixes: addressed the four failed PR #1982 checks from run `28246760439`. Root causes were PostgreSQL v51 migration splitting inside a dollar-quoted function, Windows fd-relative inventory assumptions in `Context_Integrity`, a too-tight Retry-After timing assertion, missing shard assignments/contract entries for newly added Audio_Studio and MediaIngestion_NEW tests, and Watchlists extension minimal-locale staging dropping the default Chrome locale catalog.
- Fixes added: reused `split_sql_statements` for ChaChaNotes PostgreSQL schema conversion and preserved Postgres trigger/function statements; added a regression test for the v51 dollar-quoted migration; added portable user-skill inventory fallback and skipped fd-internals tests when fd traversal is unavailable; widened the Retry-After test threshold; assigned `test_book_zip_safe_extract.py` and `test_pdf_analysis_regressions.py` across all full-suite matrices and updated the contract; restored default-locale catalog preservation for general `launchWithExtension` while keeping prior built-extension behavior untouched.
- Verification: focused backend regression set passed 4 tests; full `test_inventory.py` passed 46 tests; shard coverage passed with `shards=723 test_files=3883 ignored=4 baseline=130 new_uncovered=0`; extension launch utility Vitest passed 5 tests; `bun run compile` passed; `bun run build:chrome:prod` passed after `bun install --frozen-lockfile` refreshed local node_modules without lockfile changes; `git diff --check` passed; `py_compile` on touched Python app/test files passed; Bandit on touched Python app files wrote `/tmp/bandit_pr1982_ci_fixes.json` with 0 results. Local Postgres round-trip tests skipped because Postgres was not reachable, matching their skip guards; local browser launch smoke could not validate headed CI because macOS Chrome startup timed out and headless Chromium blocks extension pages, so Watchlists E2E remains CI-confirmed.
- 2026-06-26 follow-up after push `6fe09bb`: `e2e-smoke (macos-latest, py3.11)` failed during dependency installation before tests because `pip install -e .[dev]` pulled `locust -> gevent==25.9.1`; macOS arm64 Python 3.11 had no matching gevent wheel and the sdist build-dependency install hit `[Errno 5] Input/output error`. Narrowed `.github/workflows/e2e-smoke.yml` to install runtime dependencies plus only the pytest plugins used by that lane, and added a workflow contract test preventing `.[dev]`/`locust` from returning there. `UI Watchlists Extension E2E` still skipped all 14 tests after no MV3 service worker/background target appeared; added page-context storage seeding when no background target is visible and capped the background-target probe to 5s when a deterministic manifest key is available, avoiding 90s-per-test skip loops. Verification: the new e2e-smoke dependency contract failed before the workflow change and passed after it; full workflow contract file passed 37 tests; extension launch Vitest passed 6 tests; `bun run compile` passed; `git diff --check` passed; Bandit on the touched Python CI contract test with `B101` excluded wrote `/tmp/bandit_pr1982_e2e_smoke_watchlists.json` with 0 results. Current remote PR #1982 run still has those two known failures and the full-suite matrix is being allowed to finish before the next push.
- 2026-06-26 additional current-run failures: `Full Suite shard (Ubuntu / Python 3.13 / auth-core-unit-m-z)` failed because single-user SQLite RBAC seeding continued after the migration-lock backstop refused to run and then hit `no such table: roles`; `Full Suite shard (Ubuntu / Python 3.13 / gap-verified-3)` failed because the DataTables integration stub did not accept the worker's `timeout=` adapter keyword, causing jobs to fail and the worker helper to time out, and because the prototype link exchange test still expected post-provisioning audit failures to raise even though sharing audit logging is now best-effort. Fixes added: run the SQLite RBAC table backstop for file-backed SQLite as well as in-memory SQLite before baseline seeding, keep optional single-user role/API-key backfill non-fatal when migrations are unavailable and legacy `api_keys` schema lacks newer columns, accept passthrough adapter kwargs in the DataTables stub, and update the prototype audit test to assert a 200 response plus retained single-use claim. Verification: the focused four-test reproduction failed before the test/AuthNZ fixes and passed after with 4 passed; the explicit AuthNZ migration-skip regression then exposed the legacy API-key column issue and passed after the optional-backfill catch; DataTables job integration passed 3 tests; full prototype link exchange passed 24 tests.
- 2026-06-26 media-ingestion current-run failures: `Full Suite shard (macos-latest / Python 3.12 / media-ingestion-modification)` failed because the nested email test used `EmailMessage.add_attachment(..., maintype="message", subtype="rfc822")`, which raises `TypeError: set_message_content() got an unexpected keyword argument 'maintype'`, and because yt-dlp cookie unit tests inherited the CI workflow egress allowlist that excluded `youtu.be`/`youtube.com` despite stubbing `yt_dlp.YoutubeDL`. Fixes added: use the Python-compatible `EmailMessage` RFC822 attachment form with `subtype="rfc822"` only, and isolate the yt-dlp unit fixture with an explicit allowlist for its stubbed hosts. Verification: the exact three failed tests passed locally, then the affected media-ingestion test files passed with `32 passed, 7 skipped, 1 xpassed`; `git diff --check` passed; `py_compile` on the touched test files passed; Bandit over the touched test files reported only an existing low-severity `B311` at an untouched line in `test_add_media_endpoint.py`, and the filtered run excluding existing test-file findings wrote `/tmp/bandit_pr1982_media_ingestion_followups_filtered.json` with 0 results.
- 2026-06-26 chat-legacy current-run failures: `Full Suite shard (macos-latest / Python 3.12 / chat-legacy-integration)` exposed stale chat integration test contracts: the auto-routing test patched the removed sync skill-tool helper instead of the async endpoint import; image/streaming tests still expected literal provider `event:` frames even though the endpoint now emits normalized `data:` metadata/tool/result frames; and the disconnect smoke expected the old `stream_start` control frame. Fixes added: patch `add_skill_tool_to_tools_list_async` with an async fake, parse normalized JSON SSE payloads for conversation IDs and tool-results ordering, assert `[DONE]` as the final frame, and keep the disconnect test focused on receiving a streamed delta before closing the client context. Verification: the exact four failed tests failed before the change and passed after; the four modified chat integration files passed with `14 passed`; `git diff --check` passed; `py_compile` on touched chat tests passed; filtered Bandit wrote `/tmp/bandit_pr1982_chat_followups_filtered.json` with 0 results.
- 2026-06-26 admin/settings current-run failures: `Full Suite shard (Ubuntu / Python 3.12 / admin-s-sessions-settings)` and the matching Python 3.13 shard failed in `test_admin_smoke_roles_permissions_sqlite_and_pg` because the isolated SQLite smoke called `ensure_authnz_tables` while CI exported a dead `REDIS_URL`; the migration helper correctly permits Redis-to-file fallback only in test mode, but this specific smoke used `_fresh_client()` with `TEST_MODE=0`. Fix added: run that smoke client with `test_mode=True`, matching the existing isolated test-client pattern used elsewhere in the file without weakening production migration-lock behavior. Verification: the exact failed admin smoke passed locally with `REDIS_URL=redis://127.0.0.1:6379/0`, then the full admin smoke file passed with `3 passed`; `git diff --check` passed; `py_compile` on the touched admin test passed; filtered Bandit wrote `/tmp/bandit_pr1982_admin_smoke_followup_filtered.json` with 0 results.
- 2026-06-26 chat-unit/media-audio current-run failures: `Full Suite shard (macos-latest / Python 3.12 / chat-legacy-unit-a-l)` exposed a stale streaming auto-exec unit contract that still looked for provider `event:` frames instead of normalized `data:` JSON payloads. `Full Suite shard (windows-latest / Python 3.12 / media-audio)` exposed four clusters: legacy WebSocket tests used query-token auth without enabling the new opt-in flag; Audio Studio classified Windows drive paths like `C:/...` as URL schemes; a fail-open BYOK logging test patched the aggregate audio logger after resolution moved to the core TTS service; and a TTS missing-credentials test patched the old aggregate resolver instead of the core resolver. Fixes added: parse normalized SSE JSON in the chat-unit test, enable `AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH=1` only inside the shared WebSocket test client helper and restore the prior env afterward, treat Windows drive prefixes as local filesystem paths while still rejecting real URL storage pointers, add Windows-drive regression tests for render/export artifact paths, and retarget the stale BYOK/TTS monkeypatches to `core.Audio.tts_service`. Verification: focused Windows-drive regressions failed before the path fix and passed after; WebSocket query-token files passed `12 passed`; Audio Studio affected files passed `42 passed`; chat streaming auto-exec passed `6 passed`; fail-open file passed `18 passed`; missing-provider-credentials TTS test passed; `git diff --check` passed; `py_compile` on touched files passed; filtered Bandit wrote `/tmp/bandit_pr1982_media_audio.json` with 0 results.
- 2026-06-26 research/chatbooks-streaming follow-up failures: `research-websearch` shards exposed that pause requests during the collecting loop halted before artifact writes and phase finalization, leaving the phase as `collecting` instead of parking the next phase as paused; media-audio queue-overflow tests used empty fake queues while production overflow recovery drops an existing stale item before retrying; `chat-legacy-unit-m-z` still expected provider `event: structured_result/error` frames after streaming normalization; and `chatbooks-streaming` had an order-dependent `ws_pings_total` failure when `_STREAM_METRICS_REGISTERED` outlived a reset metrics registry. Fixes added: allow only cancel to halt mid-collecting while pauses park after finalization, seed fake overflow queues with a stale item, parse normalized structured JSON payloads, opt the multi-endpoint ping test into audio WS routing before app import, and reset stream metric registration before the ping assertion. Verification: focused research failure passed; full research worker file passed; exact audio queue failures passed and the touched audio files passed; structured streaming file passed; the multi-endpoint ping test passed alone; the combined touched set passed with `51 passed`; `git diff --check` passed; `py_compile` on touched files passed; filtered Bandit wrote `/tmp/bandit_pr1982_research_streaming_followups.json` with 0 results.

- 2026-06-26 chat-new/db-privileges follow-up failures: `chat-new-integration-property` failed because chat unified SSE treated internally generated event frames as provider control lines and dropped `event: stream_start`, `event: stream_end`, and loop compat events; `db-privileges` failed because `DatabaseMigrator` did not pass the existing Redis-to-file migration-lock fallback flag during pytest/test mode, so CI jobs exporting a dead `REDIS_URL` failed before migration-planning assertions and media DB v23 bootstrap could run. Fixes added: chat unified SSE now preserves internal control frames with `provider_control_passthru=True`, the dual-emission test explicitly exercises `STREAMS_UNIFIED=1`, and DB migrations opt into file-lock fallback only under `TEST_MODE` or explicit pytest runtime while preserving fail-closed behavior outside tests. Verification: exact chat dual-emission reproduction passed; Redis-unavailable DB migration/media bootstrap reproduction passed 6 tests; broader Redis-unavailable DB/embedding set passed 14 tests; touched chat streaming set passed 2 tests; `git diff --check` passed; `py_compile` on touched Python files passed; Bandit on touched app files had 0 findings and filtered touched-scope Bandit wrote `/tmp/bandit_pr1982_chat_db_followups_filtered.json` with 0 results.
- 2026-06-26 llm-local-backends follow-up failure: `Full Suite shard (macos-latest / Python 3.12 / llm-local-backends)` failed in `test_model_swap_rollback_on_stop_failure` because the test monkeypatched `stop_server()`, but model swapping already holds the lifecycle lock and calls `_stop_server_unlocked()` directly; the fake non-executable server then failed earlier with macOS `Permission denied` instead of exercising rollback. Fix added: patch `_stop_server_unlocked()` in the rollback test. Verification: the exact failed test passed locally, the full `tldw_Server_API/tests/LLamaCpp tldw_Server_API/tests/Local_LLM` shard passed with `101 passed, 3 skipped`, `git diff --check` passed, `py_compile` on the touched test passed, and filtered Bandit wrote `/tmp/bandit_pr1982_llm_local_followup_filtered.json` with 0 results after excluding existing test-only `B101/B105` findings.

- 2026-06-26 sandbox/platform-state-store follow-up failure: the Windows shard executed local Linux VM entry-script contracts through host `/bin/sh`, which is unavailable on Windows, and the artifact path-only fallback could create an outside file when a checked parent was swapped during `os.open`. Fixes added: skip host shell execution contracts on Windows because the scripts run inside Linux VMs, harden the path-only artifact writer by re-resolving the parent after open with exclusive create and closing/unlinking escaped files before data is written, and add a forced-fallback race regression. Verification: exact artifact security file passed 6 tests; cross-runtime contract file passed 11 tests; full sandbox state-store shard path group passed with 228 passed and 7 skipped; `git diff --check` passed; `py_compile` on touched files passed; Bandit app scope had 0 findings and filtered test scope had 0 findings.
- 2026-06-26 services/prompt/collections/LLM adapter follow-up failures: service-worker catalog tests were missing the new `writing_annotation_review_jobs_task`; the Prompt Studio MCTS evaluator toggle test enabled code eval without the required unsafe-execution acknowledgement and only exercised the heuristic path; the reading import invalid-token test held a stale exception class after module reload; and LLM adapter endpoint shards hit legacy SQLite `api_keys` tables missing virtual-key columns such as `is_virtual`. Fixes added: update the expected worker-name sets, set the Prompt Studio unsafe-eval acknowledgement via `monkeypatch`, assert reading import errors through the reloaded module reference, and make SQLite API-key lookups build a static projection with defaults for legacy optional virtual-key columns. Verification: the AuthNZ legacy regression failed before the repo fix and passed after; focused patched-failure run passed 24 tests; affected AuthNZ/LLM subset passed 11 tests after query-template cleanup; `git diff --check` passed; `py_compile` on touched files passed; Bandit app scope had 0 findings and filtered test scope had 0 findings.
- 2026-06-26 workflows/evals/persona follow-up failures: Windows workflow sync polling raised a permanent 404 when the scheduler-created run row was transiently invisible; Windows artifact path-security assertions compared `\\?\`/`//?/` long-path resolved paths against normal paths; synthetic eval workflow queue tests instantiated an owner-scoped workflow service without matching `created_by` draft rows; and the personalization consolidation test imported the removed `_resolve_user_id_to_int` helper. Fixes added: allow `_wait_for_run_completion` a bounded grace period for transient missing rows while preserving the final 404 behavior, normalize Windows long-path prefixes on both sides of artifact containment assertions, seed synthetic eval workflow queue fixtures with the workflow service owner, and update the personalization test to assert the replacement companion storage user-id helper. Verification: the new transient-run regression failed before the workflow fix and passed after; exact six failed/new regressions passed; the broader touched test set passed with 72 tests; `git diff --check` passed; `py_compile` on touched app/test files passed; Bandit app scope had 0 findings and filtered test scope had 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
