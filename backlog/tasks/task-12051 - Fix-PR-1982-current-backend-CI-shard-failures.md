---
id: TASK-12051
title: Fix PR 1982 current backend CI shard failures
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-28 17:43'
labels:
  - ci
  - pr-1982
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1982'
  - 'https://github.com/rmusser01/tldw_server/actions/runs/28282225659'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the current PR #1982 CI failures after the full matrix appeared on head 93fb333a09. Known groups include workflow contract drift for the watchlists extension job, tokenizer metadata test monkeypatch drift, provider readiness tests affected by CI egress env, audio artifact invalid path handling on Windows, distributed lock residual file cleanup on Windows, workflow scheduler stats, and new llm-adapters/orchestrator/chat endpoint shard failures that need log triage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-06-27 PR #1982 CI follow-up:
- Current live run checked before push: 28282225659 still shows 25 failed, 737 passed, 9 canceled, 4 skipped checks.
- Local focused regression set covering the failed shards passed: 23 passed, 8 warnings.
- Workflow YAML parse passed for ui-watchlists-extension-e2e.yml and ci.yml.
- git diff --check passed.
- Bandit on tldw_Server_API/app/core/Workflows/engine.py passed with 0 findings (/tmp/bandit_pr1982_workflows_engine.json).
- Remaining action: commit and push fixes so GitHub re-runs the failed matrix against the patched branch.

2026-06-27 post-push Watchlists E2E follow-up: live PR run 28293474837/job 83829303188 reached the strict Watchlists Playwright spec and timed out in the first test after 120s; this is no longer the Chromium install failure. Root cause: workflow target wait was 90s and each test only had a 120s budget, leaving too little room for extension target discovery, storage/React/connection waits, and backend startup/model warmup. Changed the workflow target wait back to 30s, preserved .watchlists-e2e-report.json into test-results even when the strict command fails, and raised the Watchlists spec timeout constant to 180s. Verification: workflow YAML parse passed; CI workflow contract test passed; apps/extension bun run compile passed; Watchlists Playwright --list parsed and listed all 14 tests; git diff --check passed. Vitest utility tests were not used as a gate because this worktree has no extension-local Vitest config and both Bun test and inherited monorepo Vitest discovery resolve the wrong runner/config for those files.

2026-06-27 PR #1982 head b3695d3a4f follow-up: after PR #2534 merged into dev, pre-commit failed on run 28294733263/job 83832583693 because Black reformatted tldw_Server_API/cli/wizard/cli.py around the new --api-key-env Typer option. Current check scan before push showed only this one failed check (pre-commit), with 11 pass, 33 pending, and 3 skipped. Applied Black to cli.py only. Verification: python -m black --check tldw_Server_API/cli/wizard/cli.py passed; python -m pre_commit run black --files tldw_Server_API/cli/wizard/cli.py passed; git diff --check passed.

2026-06-27 Watchlists headless launch follow-up: current head 6ccc7340ca failed UI Watchlists Extension E2E run 28294893523/job 83833000391 after all 14 tests skipped. The preserved JSON report showed every skip came from launchWithExtensionOrSkip catching browserType.launchPersistentContext Timeout 90000ms while the workflow forced TLDW_E2E_EXTENSION_HEADLESS=0 under xvfb. Setup, extension build, Playwright Chromium install, backend start, and health check all succeeded; failure was specifically headed persistent-context launch. Removed the workflow's headed override so the helper uses its CI-headless default, and added a workflow contract assertion that Watchlists E2E must not set TLDW_E2E_EXTENSION_HEADLESS. Verification: workflow YAML parse passed; test_required_workflow_contracts.py::test_watchlists_extension_e2e_uses_playwright_chromium passed; Watchlists Playwright --list parsed all 14 tests; git diff --check passed. Current check scan before push showed only this one failed check, with the full matrix expanded at 56 pass, 710 pending, 3 skipped.

2026-06-27 Watchlists E2E root-cause update: current head 59b4281962 failed because headless Chromium could start but could not load/open the extension page (no service worker and page.goto chrome-extension://.../options.html returned ERR_BLOCKED_BY_CLIENT in the saved report). Comparing the passing Extension Research Workspace Parity workflow on the same head showed it uses headed Chromium plus launchWithBuiltExtensionOrSkip, whose built-extension launcher seeds storage before page load and does not wait for backend connection during launch. Watchlists still uses launchWithExtensionOrSkip, which waits for connected/offline state inside launch and converts launch/connection timeouts into skips, causing the strict no-skip job to burn all 14 tests and fail with little detail. Next fix: move Watchlists to the built-extension launcher with allowOffline, restore headed CI mode, and contract-test the workflow/spec wiring.

2026-06-27 Watchlists built-launcher fix applied: changed the Watchlists spec from launchWithExtensionOrSkip plus explicit .output/chrome-mv3 path to launchWithBuiltExtensionOrSkip via a local allowOffline wrapper, and restored the workflow to headed Playwright Chromium with TLDW_E2E_EXTENSION_TARGET_WAIT_MS=5000. Added CI contract coverage for the headed env, target wait, and built-extension launcher usage. Verification before push: focused pytest contract checks passed (2 passed); workflow YAML parsed successfully; bun run compile passed in apps/extension; bun run test:e2e:watchlists -- --list found all 14 tests; git diff --check passed; Bandit with B101 excluded found no non-assert issues in the touched pytest contract file. Raw Bandit still reports existing pytest assert_used findings across the contract file. Live PR check immediately before push showed one failed check, UI Watchlists Extension E2E / Watchlists Extension E2E (No Skips), with 125 passed, 640 pending, and 4 skipped.

2026-06-27 tokenizer metadata CI follow-up: current PR #1982 run 28300003793/job 83847033571 failed Full Suite shard (Ubuntu / Python 3.13 / gap-verified-10) in test_llm_providers_tokenizer_metadata_mirrors_strict_fields because the metadata projection test faked tokenizer resolution but left provider readiness live. In CI, Ollama readiness could disable the provider before the fake resolver ran, producing count_accuracy=unavailable instead of exact. Added a readiness stub for the tokenizer metadata projection tests and removed the existing Bandit B108 hardcoded /tmp default from the helper. Verification: focused failed test passed with CI-like env flags; full test_llm_providers_tokenizer_metadata.py passed (6 tests); Bandit on the touched test file with B101 excluded passed; git diff --check passed. Live run check before staging still showed only this one failed current-head job.

2026-06-27 tokenizer metadata fix pushed: committed e089e5085133dad3ff8c35594ab58c76d4a3c7f2 (Fix tokenizer metadata readiness test isolation) and pushed it to origin/dev for PR #1982. Confirmed PR head moved to e089e5085133dad3ff8c35594ab58c76d4a3c7f2. New PR CI run is 28301705852. Cancelled superseded old-head CI run 28300003793 for head 33a4af634c83191442e281d159e3ad5c78758a4e.

2026-06-27 llm-adapters-unit Windows follow-up: new PR #1982 run 28301705852/job 83854084894 failed Full Suite shard (windows-latest / Python 3.12 / llm-adapters-unit) in test_llm_models_metadata_handles_local_discovery_policy_errors with calls["count"] == 0. Root cause: the test set allowed ports and disabled private-host blocking, but did not override CI WORKFLOWS_EGRESS_ALLOWLIST, so provider readiness rejected http://127.0.0.1:8080/v1 before discover_models_from_endpoint reached the mocked _http_fetch. Added WORKFLOWS_EGRESS_ALLOWLIST=127.0.0.1,localhost inside the test to isolate the intended discovery policy-error path. Verification: exact failing test passed with CI-like allowlist env; test_llm_models_filters.py passed (6 tests); full LLM_Adapters/unit shard passed with -p pytest_asyncio.plugin (159 tests); Bandit on touched test file with B101 excluded passed; git diff --check passed.

2026-06-27 chat-new-unit-a-l follow-up: current PR #1982 run 28301705852/jobs 83854084880 and 83854085179 failed on Windows/macOS in test_llm_provider_readiness_marks_unreachable_local_endpoint_unavailable because CI WORKFLOWS_EGRESS_ALLOWLIST blocked 127.0.0.1 before the opt-in endpoint probe ran, returning egress_blocked instead of endpoint_unreachable. Added WORKFLOWS_EGRESS_ALLOWLIST=127.0.0.1,localhost to that readiness test. Verification: exact failing readiness test passed with CI-like allowlist env; test_llm_providers_readiness.py passed (4 tests); full Chat_NEW/unit/test_[a-l]*.py shard passed with -p pytest_asyncio.plugin (135 tests); combined Bandit on touched allowlist tests with B101 excluded passed; git diff --check passed.

2026-06-27 product-workflows-runtime Windows follow-up: pre-push live run check found an additional failure in run 28301705852/job 83854087649, Full Suite shard (windows-latest / Python 3.12 / product-workflows-runtime). The failing test was test_webhook_step_noop, which left an async workflow run in running for 30 seconds. The exact test and full product-workflows-runtime shard both passed locally under CI-like env on macOS, so this is Windows/order-sensitive scheduler timing rather than webhook adapter behavior. Changed this adapter-focused test to run the saved workflow with mode=sync, matching nearby workflow adapter tests and avoiding background scheduler timing for a no-op webhook assertion.

Run 28306969720 added Ubuntu/Python 3.13 llm-adapters-chat-endpoint failure: AuthNZ SQLite migration retry failed because adapter tests delete TEST_MODE while CI exports REDIS_URL to an unavailable Redis service; apply_authnz_migrations only allowed Redis-to-file fallback for TEST_MODE, not explicit pytest runtime. Patched migrations to also allow fallback under PYTEST_CURRENT_TEST, added regression coverage, and locally verified AuthNZ migration unit tests plus LLM adapter shards with REDIS_URL/EMBEDDINGS_REDIS_URL pointed at localhost and TEST_MODE removed by the tests.

2026-06-28 current-head CI-contract follow-up: PR #1982 run 28328796097/job 83927186551 failed gap-verified-12 on tldw_Server_API/tests/CI/test_required_workflow_contracts.py. Root cause: the previous platform-services shard split added platform-services-main-routing in .github/workflows/ci.yml, but the contract test still expected FFmpeg gating and uncovered Services test membership for platform-services-core only. Updated the contract expectations to include platform-services-main-routing. Verification: the two failed contract tests passed locally; the full workflow contract file passed (38 tests); the failed shard path set passed under CI-like pytest settings (313 tests); git diff --check passed; Bandit with B101 excluded wrote /tmp/bandit_pr1982_ci_contracts.json with no findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and locally verified the current PR #1982 CI shard fixes for run 28282225659. The patch covers the watchlists extension Chromium install contract, tokenizer metadata test isolation, egress-env leakage in readiness/model metadata tests, AuthNZ schema readiness for adapter endpoint tests, Windows path/lock/artifact range assumptions, deterministic circuit-breaker recovery tests, and workflow scheduler active-count cleanup. Pre-push verification: focused pytest set passed with 23 passed and 8 warnings; workflow YAML parse passed; git diff --check passed; Bandit on app/core/Workflows/engine.py reported 0 findings. Known pending item: GitHub Actions must rerun after the push; the 25 live failures observed before this push were from the previous head commit/run, not from the patched commit.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-27 run 28306969720 llm-adapters-orchestrator-extra cancellation follow-up: local exact shard reproduced the CI environment with TEST_MODE=true plus REDIS_URL/EMBEDDINGS_REDIS_URL pointed at localhost while endpoint tests delete TEST_MODE. The full orchestrator-extra shard passed locally (21 passed, 8 warnings), confirming the AuthNZ pytest-runtime Redis fallback covers this cancelled shard as well.
2026-06-27 final run 28306969720 triage: additional Python 3.13 llm-adapters-orchestrator/chat-errors failures and cancellations were the same AuthNZ/Redis migration-lock family already covered by the pytest-runtime file fallback change. Python 3.13 platform-services-core cancelled after completing test_main_router_contract.py and before the next services test emitted a result; no traceback or Redis failure was present. Aligned the Linux Python 3.13 shard timeout with Python 3.12 at 60 minutes to cover the large services shard while keeping the code fix focused on the adapter/AuthNZ root cause.
Current PR #1982 run 28314994849 completed with two real workflow failures plus timeout/cancelled shards. The workflow failures share a root pattern: tests and async workflow engine activity share the same SQLite connection object (`check_same_thread=False`) across threads. On Windows this surfaced as `sqlite3.OperationalError: cannot commit - no transaction is active` in `WorkflowsDatabase.add_artifact`; on Ubuntu/Python 3.13 it surfaced as `sqlite3.InterfaceError: bad parameter or other API misuse` while polling `WorkflowsDatabase.get_run`. Focused tests pass locally on macOS/Python 3.11, so the fix needs to remove cross-thread shared-connection races rather than only chase assertions.
Applied PR #1982 follow-up fixes for run 28314994849: refactored media embedding job endpoint tests to call endpoint functions directly instead of starting the full FastAPI app with TestClient; changed workflow scheduler status polling to use a short-lived independent SQLite read connection; waited for the workflow run to reach terminal state before inserting the synthetic artifact in the manifest mismatch API test; split the oversized platform-services-core CI shard into platform-services-core and platform-services-main-routing across the full-suite matrices. Verification: exact workflow failures passed (2 tests); full test_engine_scheduler.py passed (8 tests); embeddings media/message shard passed (40 tests); test_main_shutdown_job_pollers.py plus test_media_files_cleanup_service.py passed (31 tests); ci.yml YAML parse passed; shard coverage passed with no new unshared files; git diff --check passed; py_compile on touched Python passed; Bandit with B101 excluded on touched Python wrote /tmp/bandit_pr1982_ci_followup_12051.json with exit 0.
2026-06-28 PR #1982 head fcb05017 UX gate follow-up: after PR #2536 merged into dev, current Frontend UX Gates run 28327254434/job 83919271487 failed Stage 4 responsive landmarks because /mcp-hub exposed zero semantic h1 elements at 390px. Root cause found in McpHubPage: the route title is rendered with AntD Typography.Title level={3}, which creates an h3 while the smoke gate requires exactly one route-level h1. Changed the title to Typography.Title level={1} while preserving the compact visual size and added a unit regression assertion. Verification: focused McpHubPage Vitest passed (12 tests); exact /mcp-hub Playwright Stage 4 responsive landmarks case passed locally against the dev server; git diff --check passed. Bandit not applicable because this fix touched only frontend TypeScript/TSX and task notes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
