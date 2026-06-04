---
id: TASK-2234
title: Split full-suite CI checks to avoid PR timeouts
status: Done
labels:
- ci
- github-actions
- testing
priority: high
modified_files:
- .github/workflows/ci.yml
- .github/workflows/sbom.yml
- Docs/Plans/2026-06-03-ci-full-suite-sharding-implementation-plan.md
- Docs/Published/User_Guides/index.md
- Docs/User_Guides/index.md
- Dockerfiles/Dockerfile.webui
- Dockerfiles/docker-compose.webui.yml
- Makefile
- apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts
- apps/tldw-frontend/__tests__/pr-916-review-followups.test.ts
- apps/packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx
- apps/packages/ui/src/routes/option-setup.tsx
- backlog/tasks/task-2234 - Split-full-suite-CI-checks-to-avoid-PR-timeouts.md
- tldw_Server_API/app/core/AuthNZ/email_service.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py
- tldw_Server_API/app/api/v1/endpoints/admin/admin_rbac.py
- tldw_Server_API/app/api/v1/endpoints/sync.py
- tldw_Server_API/app/core/http_client.py
- tldw_Server_API/app/core/DB_Management/chacha/runtime.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/document_version_rollback_ops.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/fts_ops.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/media_item_update_ops.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/synced_document_update_ops.py
- tldw_Server_API/app/core/AuthNZ/db_config.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/tests/CI/test_required_workflow_contracts.py
- tldw_Server_API/tests/Claims/test_claims_rebuild_health_persistence.py
- tldw_Server_API/tests/Audit/test_audit_pii_overrides.py
- tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py
- tldw_Server_API/tests/AuthNZ/unit/test_email_service.py
- tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_api_keys.py
- tldw_Server_API/tests/AuthNZ_Unit/test_resource_governor_permissions_claims.py
- tldw_Server_API/tests/Collections/test_collections_close.py
- tldw_Server_API/tests/Collections/test_embedding_queue.py
- tldw_Server_API/tests/Collections/test_reminders_notifications_db.py
- tldw_Server_API/tests/Config/test_config_providers_endpoints.py
- tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py
- tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py
- tldw_Server_API/tests/Infrastructure/test_distributed_lock.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_policy_overrides.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_shared_workspace_registry.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_workspace_set_objects.py
- tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py
- tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py
- tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py
- tldw_Server_API/tests/Media/test_json_url_download.py
- tldw_Server_API/tests/Notifications/test_bridge_opt_out.py
- tldw_Server_API/tests/Notifications/test_notifications_service_lifecycle.py
- tldw_Server_API/tests/RAG/test_analytics_backend.py
- tldw_Server_API/tests/Resource_Governance/test_e2e_tokens_daily_cap.py
- tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py
- tldw_Server_API/tests/WebScraping/integration/test_websearch_cancellation.py
- tldw_Server_API/tests/conftest.py
- tldw_Server_API/tests/http_client/test_http_client_egress_metrics.py
references:
- https://github.com/rmusser01/tldw_server/pull/2258
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restructure the GitHub Actions CI full-suite jobs so PRs do not run all slow test modules serially in one runner. Keep full Linux coverage for Python 3.12 and 3.13, keep full macOS/Windows Python 3.12 coverage on PRs through shards, and run expanded macOS/Windows Python 3.13 coverage only for non-PR release/main/manual contexts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] PR full-suite checks no longer run every backend module serially in one runner for each Python/OS combination.
- [x] Python 3.12 and Python 3.13 full testing runs through Ubuntu shard jobs.
- [x] Python 3.11 runs compatibility smoke coverage.
- [x] PR macOS/Windows Python 3.12 checks are backed by full shard coverage, not smoke-only subsets.
- [x] Expanded macOS/Windows Python 3.13 shard coverage runs only for non-PR release/main/manual contexts.
- [x] macOS/Windows full shard jobs skip Postgres fixture Docker auto-start quickly when no explicit Postgres service is provided.
- [x] Workflow validation evidence is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create an isolated worktree from origin/dev. 2. Replace the serial full-suite matrix jobs with reusable shard jobs and smoke/release variants. 3. Preserve check-name compatibility via summary jobs when practical. 4. Validate workflow YAML/action syntax and commit the scoped changes. 5. Push branch and open a PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
CI investigation on 2026-06-04 found PR #2258 failing because the branch was stale against dev and the initial shard path groups exposed isolated-fixture issues. Artifacts from run 26928302943 showed docs-index failures, import collection failures in auth-db/chat-llm shards, repeated app 503/shutdown_in_progress failures, Postgres client exhaustion, and a Windows file-lock permission failure. Follow-up work rebased the branch onto current dev, restored sync compatibility paths used by tests, isolated ChaCha runtime shutdown state, fixed Media FTS refresh behavior, hardened MCP AuthNZ-token detection, fixed /setup heading semantics for UX smoke, removed hard-coded PostgreSQL DSNs from CI env export scripts, and updated the SBOM workflow for the current pyproject-aware CycloneDX contract.

2026-06-04 recheck of run 26934657519 found new full-suite shard failures after the previous push. Representative logs showed broad shards inheriting a shared Postgres `DATABASE_URL`/`TEST_DATABASE_URL`, causing tests that should use SQLite or per-test Postgres databases to hit the shared `tldw_content` database, drop it, exhaust connections, or miss AuthNZ tables. macOS auth-db also exposed a closed managed SQLite backend cached between JWT refresh tests, and integrations exposed a Python 3.12 default-event-loop assumption in `test_bridge_opt_out.py`.

2026-06-04 recheck after commit `530804c529` found additional PR #2258 failures: Resource Governor diag tests using a stale app state returned 503, notifications service lifecycle fakes no longer matched sidecar worker startup, MCP Hub tests compared authored policy documents to normalized resolved policy documents, product-module tests inherited `TLDW_TEST_MODE`, a config test poisoned global `os.getenv`, the WebUI Docker build still installed unrelated workspaces, quickstart Makefile parsing leaked env-file semantics, httpx transport exceptions bypassed network-error normalization, reminders DB tests reused cached SQLite backends across temp directory teardown, external federation fake managers did not expose the current write-flag API, and a telemetry sanitizer test patched the pre-extraction global instead of runtime dependencies.

2026-06-04 recheck after commit `7dd08107a3` found `build (webui)` failing in the container-build workflow because the Dockerfile rewrote `apps/package.json` workspaces before `bun install --frozen-lockfile`, producing a package graph that no longer matched `apps/bun.lock`.

2026-06-04 recheck after commit `dc28e1f1e9` found `Full Suite shard (macos-latest / Python 3.12 / auth-db)` failing because `test_create_permission_sanitizes_backend_error` patches the legacy `admin_rbac.is_test_mode` test hook, but the split admin RBAC module no longer exposed that compatibility wrapper.

2026-06-04 recheck after commit `3d3c796af9` found `Full Suite shard (macos-latest / Python 3.12 / media-audio)` failing because fake-client JSON download tests still hit global egress allowlist validation before streaming, and `Full Suite shard (macos-latest / Python 3.12 / product-modules)` failing because the claims rebuild health persistence test could read a stale module settings/path target and fall back to live service health instead of the inserted persisted row.

2026-06-04 final completed-check recheck before pushing found the remaining completed PR #2258 failures concentrated in full-suite shards rather than frontend/build gates. Additional root causes included single-user API key test setup not matching the stricter single-user key contract, macOS `/tmp` symlink path expectations in MCP path-scope assertions, direct mutation of the read-only RAG analytics backend type property, Audit PII settings imported before test overrides, Windows path-string and locked-file assumptions in Evaluations and Infrastructure tests, raw JSON bodies missing `application/json` in Resource Governance E2E cap tests, egress metric assertions not accepting allowlist denial wording, WebSearch cancellation tests depending on globally registered routes and media DB startup, mock email filenames containing Windows-invalid timestamp separators, Evaluation webhook tests overriding only the module source instead of direct route dependency objects, and Character Chat custom prompt preset endpoints calling missing ChaChaNotesDB persistence methods. The canceled Windows ai-retrieval shard showed a HuggingFace 429 during model predownload but no pytest failure before cancellation; the other canceled shards only reported cancellation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Follow-up CI remediation for PR #2258 after rebasing onto dev. Addressed failing full-suite shards by restoring sync compatibility helpers, adding a managed ChaCha runtime to prevent shutdown leakage between tests, fixing Media FTS update/delete refresh behavior, tightening MCP AuthNZ-token detection for revoked tokens, and correcting Collections test setup for backend runtime attributes. Addressed UX Smoke Gate by keeping /setup to a single semantic h1 and strengthening the readiness-route unit test mock. Addressed PR review comments by removing hard-coded PostgreSQL DSNs from CI env export scripts while keeping OS PR full-suite shard coverage intact. Updated the SBOM workflow to satisfy the current pyproject-aware CycloneDX contract. Verification on 2026-06-04: focused backend suites passed (Docs 82, MediaDB2 sync 15, External Sources sync coordinator 6, MCP auth paths 22, plus targeted collections/media/audio/audit/resource-governor/infrastructure tests); UX smoke focused /setup Playwright test passed; option setup unit test passed; CI workflow contract tests passed (40); compileall passed; Bandit on touched Python scope passed with no findings; git diff --check passed.

Second recheck remediation for run 26934657519: forced full-suite pytest steps back to SQLite defaults while leaving Postgres host/port/user/password available for explicit per-test Postgres fixtures; added a CI workflow contract preventing shared Postgres DSN leakage into broad shard pytest steps; evicted managed SQLite AuthNZ backends on config reset so closed pools are not reused between tests; reset the shared FastAPI app lifecycle state around tests to prevent `shutdown_in_progress` 503 leakage; and updated the notification bridge test helper to use `asyncio.run` on Python 3.12+. Verification on 2026-06-04: JWT refresh rotation and notifications bridge tests passed (7); CI contracts plus representative audit/embeddings/resource-governor 503 regressions passed (44); lifecycle/drain gate tests passed (32); representative audio/claims/MCP failures passed (3); compileall passed; app-code Bandit passed with no findings; broader touched test-scope Bandit only reported existing test assert/test-string findings; git diff --check passed.

Third recheck remediation for PR #2258: isolated Resource Governor app-state overrides, updated notification lifecycle fakes for scheduler task return values and sidecar worker paths, made MCP Hub policy tests assert authored versus normalized resolved policies explicitly, disabled both test-mode env vars in embedding queue tests, localized the quickstart `os.getenv` failure stub, constrained the WebUI Docker install to frontend/UI workspaces, moved single-user WebUI key fallback into Compose interpolation, normalized httpx transport exceptions in the shared HTTP client, closed shared SQLite backends around reminders notification DB tests, updated external federation and telemetry tests to current injected APIs, and left unrelated untracked watchlist template files unstaged. Verification on 2026-06-04: original five failing CI tests passed together (5); MCP downstream tail passed (107 passed, 2 skipped); config quickstart, Docker contract, Makefile masking, frontend quickstart Vitest, HTTP client, unit subset, product tail, reminders notification DB, and watchlist/WebSub tail checks passed from the local recheck; compileall passed; Bandit on touched production code passed; git diff --check passed.

Fourth recheck remediation for PR #2258: restored frozen-lockfile-compatible WebUI Docker workspace setup by keeping the root workspace graph intact, copying only unrelated workspace manifests and the extension prepare shim, and removing the lockfile-breaking workspace rewrite. Verification on 2026-06-04: the WebUI Docker `bun install --frozen-lockfile` step was simulated locally with the same copied files and passed; Docker hardening pytest passed; frontend Docker/networking Vitest passed (14); compileall for the updated Python contract test passed; git diff --check passed.

Fifth recheck remediation for PR #2258: restored the patchable `admin_rbac.is_test_mode()` compatibility wrapper by delegating to the shared core testing helper, keeping the admin RBAC error-mapping test hook available without changing endpoint behavior. Verification on 2026-06-04: the failing `test_create_permission_sanitizes_backend_error` was reproduced locally before the fix; the exact test passed after the fix; the full `test_admin_rbac_error_mapping.py` file passed (11); compileall passed for `admin_rbac.py`; Bandit on `admin_rbac.py` passed; git diff --check passed.

Sixth recheck remediation for PR #2258: isolated fake-client JSON URL download tests from global egress allowlist validation, and hardened claims rebuild health persistence test setup to patch settings through `monkeypatch` plus the `claims_service` module's path hook. Verification on 2026-06-04: `test_download_url_json_content_type` reproduced the CI egress denial locally before the fix; the exact claims persistence test passed alone locally before the hardening; the full Claims folder passed locally on Python 3.11 before the hardening; after the fix, `test_json_url_download.py` passed under the restrictive CI allowlist (3), the exact claims persistence test passed, and the full Claims folder passed (166 passed, 1 skipped); compileall passed for the two touched test files; Bandit on the touched tests passed with test assert rule B101 skipped; git diff --check passed.

Seventh recheck remediation for PR #2258: fixed the remaining completed full-suite failures by aligning single-user API-key test setup with the configured key, normalizing MCP `/tmp` scope-root expectations across macOS/Linux, using a real AnalyticsDatabase backend mock shape instead of assigning the read-only property, patching Audit PII module-level settings, normalizing Windows-sensitive path and file-lock test behavior, sending Resource Governance E2E payloads with `json=`, accepting allowlist wording in egress denial metrics, idempotently installing and overriding the WebSearch route dependencies for the cancellation test, making mock email output filenames platform-safe, overriding direct Evaluation webhook route dependencies, and adding ChaChaNotesDB custom prompt preset schema plus CRUD support required by Character Chat preset endpoints. Verification on 2026-06-04: focused CI failures passed locally across AuthNZ/MCP/RAG/Audit/Infrastructure/http-client (7), email/recipe (3), WebSearch/webhook (2), Character Chat prompt preview (1), Resource Governance chat/embeddings cap cases (2), full email service file (16), full webhook multi-user file (4), Character Chat preset editor plus request-preset preview (9), and adjusted recipe/MCP tests (2); compileall passed for touched Python files; production Bandit on `email_service.py` and `ChaChaNotes_DB.py` passed with zero findings; test-scope Bandit showed only existing assert/test-fixture literal findings after the new `/tmp` finding was removed; git diff --check passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
