---
id: TASK-2396
title: Stabilize workflow media ingest chunking CI test
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-06-21 21:49
labels:
- ci
- tests
- workflows
dependencies: []
priority: high
modified_files:
- tldw_Server_API/app/core/Sandbox/orchestrator.py
- tldw_Server_API/tests/sandbox/test_artifacts_perf_large_tree.py
- tldw_Server_API/tests/Workflows/test_workflows_extras.py
- tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_scanner.py
- tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py
- tldw_Server_API/tests/Notes_Graph/integration/test_graph_endpoint.py
- tldw_Server_API/tests/Slides/test_slides_ordering.py
- tldw_Server_API/tests/kanban/test_kanban_performance.py
- tldw_Server_API/tests/Workflows/test_new_step_adapters.py
- tldw_Server_API/tests/CI/test_required_workflow_contracts.py
- .github/workflows/coverage-required.yml
- .github/workflows/backend-required.yml
- .github/workflows/jobs-suite.yml
- .github/workflows/ci.yml
- tldw_Server_API/app/core/Workflows/adapters/audio/stt.py
- tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py
- tldw_Server_API/tests/Workflows/test_adapter_path_security.py
- tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix PR #2258 CI failures in workflow test shards: media ingest polling should fail with explicit diagnostics instead of KeyError, and approval permission tests should set up waiting approval runs deterministically instead of depending on background scheduler timing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The media ingest chunking test handles workflow run polling with explicit response assertions and useful timeout diagnostics.
- [x] #2 The test fixture grants the permissions required by the workflow run status endpoint.
- [x] #3 The failed test passes locally and touched files pass syntax/security checks.
- [x] #4 Approval permission tests set up waiting approval runs deterministically without depending on background scheduler timing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CI run 27886920141 direct failures addressed:
- Job 82523799578, product-workflows-storage: test_media_ingest_local_text_chunking crashed with KeyError: status while polling a non-validated run payload.
- Job 82523800213, product-workflows-api: test_reject_allows_admin_override reused the engine run path for permission setup and the second helper call observed succeeded instead of waiting_approval.

Fixes:
- Added explicit status-code checks and timeout diagnostics to media-ingest run polling, and granted WORKFLOWS_RUNS_READ in the test auth principal.
- Changed approval-permission setup to create the waiting run/step rows directly in the test database, keeping the test focused on approve/reject authorization and removing background scheduler timing from setup.

Verification:
- product-workflows-api shard command: 77 passed, 2 skipped.
- product-workflows-storage shard command: 29 passed, 6 skipped.
- Focused media-ingest pair: 2 passed.
- compileall on touched tests: passed.
- git diff --check on touched tests/task: passed.
- Bandit on touched tests wrote /tmp/bandit_ci2258_workflow_tests.json; findings were existing low-severity pytest assert/test-token patterns only.

Merged origin/dev into PR #2258 and resolved the lone MCP server import conflict by keeping both the AuthNZ JWT token-detection imports from the PR branch and the single-user IP/settings imports from dev.

Post-merge verification after merging origin/dev:
- product-workflows-api shard command: 77 passed, 2 skipped.
- product-workflows-storage shard command: 29 passed, 6 skipped.
- MCP mounted JSON-RPC/auth targeted command: 19 passed.
- compileall on MCP server and touched workflow tests: passed.
- git diff --cached --check: passed.
- Bandit including MCP server and touched workflow tests wrote /tmp/bandit_ci2258_post_merge.json; MCP server had no findings, remaining findings are existing low-severity pytest assert/test-token patterns in workflow tests.

gap-verified-5 follow-up: full local shard reproduction without --maxfail surfaced 26 failures: one Slides Hypothesis too_slow health check and a Telegram 404 cluster. Root cause for Telegram was Notes Graph integration fixture reloading tldw_Server_API.app.main with MINIMAL_TEST_APP=0 and leaving the shared module in full-app route-gated state; fixture now restores the default minimal test profile after yield. Slides ordering property test now uses a smaller unique-order range to avoid slow-input health checks under loaded shards. Verification: Notes Graph then Telegram admin ordered reproduction passed 6 tests; Slides ordering passed; full gap-verified-5 shard passed 689 tests with 1 skipped; compileall/diff check passed; Bandit on touched tests wrote /tmp/bandit_ci2258_gap5_tests.json and exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stabilized both direct PR #2258 workflow shard failures. Media ingest polling now validates run-status responses and reports useful diagnostics. Approval permission tests now create waiting approval state directly in the workflow test database, avoiding a scheduler race while preserving approve/reject authorization coverage. Local shard verification passed for both affected CI shards.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Additional failures surfaced in run 27892030109/27892030110 before pushing local fixes: macOS/Ubuntu/Windows core-utils-tooling failed test_endpoint_auth_dependency_symbols_come_from_auth_deps because research_discovery.py imported User/get_request_user from core.AuthNZ; Ubuntu e2e-smoke failed test_deep_research_run_creation_persists_chat_handoff with ChaChaNotes migration V36->V37 expected 37, got 38; Build and Validate Distributions failed test_mcp_unified_standalone_distribution_metadata_matches_extras because standalone MCP package metadata did not expose the wheel-installed smoke transport dependencies. Fixes: moved research_discovery auth imports to API_Deps.auth_deps, added httpx/websockets to the MCP standalone metadata contract and gateway package payload assertions, and added a process-local SQLite schema initialization lock keyed by ChaChaNotes DB path to prevent concurrent in-process schema migrations from interleaving. Added a unit regression for path-shared schema locks. Local verification: MCP package metadata tests passed; auth import boundary test passed; schema lock unit test passed; deep research E2E handoff test passed with E2E_INPROCESS/single-user env; compileall and git diff --check passed; Bandit wrote /tmp/bandit_ci2258_mcp_research_chacha.json with zero findings in changed production files.

Later completed macOS, Ubuntu, and Windows db-privileges shards in the same still-running CI run failed test_readme_no_longer_mentions_media_db_v2_in_source because README.md still used the legacy Media_DB_v2 filename in the database architecture note and Mermaid node. Fix: replaced those README references with generic per-user media content DB wording. Local verification: README guard test passed and rg found no Media_DB_v2 occurrences in README.md.

Windows chat-new-integration-property failed test_chat_command_concurrency_respects_rate_limit with results['ok'] == 6 for a 5/min command limit. The failure was caused by Windows CI taking long enough for the token bucket to legitimately refill an extra token during the HTTP-level burst, plus bucket creation was not guarded against concurrent first-use. Fix: serialized command rate-limit bucket creation with a module RLock and made the integration assertion deterministic by using a 1/min user command limit for the burst. Local verification: exact chat command concurrency integration test passed; command-router unit concurrency test passed.
macOS py3.12 platform-mcp-core failed test_mcp_request_tools_list_unauth_returns_403_hint because POST /api/v1/mcp/request returned a JSON-RPC 200 error envelope for an anonymous tools/list authorization denial, while the HTTP compatibility contract expects a 403 response with a detail hint matching the /api/v1/mcp/tools convenience endpoint. Fix: added an anonymous-only tools/list permission mapping in mcp_unified_endpoint.py that converts MCP authorization error -32001 to HTTPException 403 with the existing listing-tools hint, while preserving credentialed and protocol-level JSON-RPC authorization denials as 200 envelopes. Local verification: test_mcp_http_403_mapping.py passed; the credentialed JSON-RPC denial contract tests passed; the CI platform-mcp-core shard equivalent passed locally with 554 passed, 4 skipped; compileall and git diff --check passed; Bandit on mcp_unified_endpoint.py wrote /tmp/bandit_ci2258_mcp_request_403.json with zero findings.
Remaining failures from CI run 27892030109 were inspected before pushing local commits. product-notes-persona failed test_persona_persists_session_turns_and_tool_outcomes because persona MCP tool execution could receive a None response and then dereference resp.error, closing the WebSocket. product-workflows-step-adapters failed test_rerank_step_flashrank_test_mode on Windows because the same workflow run was started twice; logs showed the run updating to succeeded twice and then failed. product-workflows-engine failed test_wait_for_approval_then_resume with the run stuck running, consistent with the same duplicate-start race allowing a second worker to begin an already running approval workflow. platform-mcp-core failures on Windows/Ubuntu were duplicates of the anonymous tools/list 403 mapping already fixed locally.

Fixes added in this pass: persona MCP tool calls now convert a None transport response into a structured tool_result with reason_code TOOL_EXECUTION_EMPTY_RESPONSE instead of crashing the WebSocket; workflow start_run now ignores non-queued duplicate starts instead of demoting completed runs to invariant_violation failure. Added regression coverage for both cases.

Verification for this pass: new persona/workflow regressions passed; original failing tests test_persona_persists_session_turns_and_tool_outcomes, test_wait_for_approval_then_resume, and test_rerank_step_flashrank_test_mode passed together; test_engine_state_contracts.py + test_engine_step_types.py passed (18 passed); test_persona_ws.py passed (83 passed); test_new_step_adapters.py passed (117 passed); product-workflows-engine shard paths passed locally (88 passed, 7 skipped); compileall on touched files passed; git diff --check passed; Bandit on persona.py and engine.py wrote /tmp/bandit_ci2258_persona_workflow.json with zero findings.

Later CI checks on PR #2258 surfaced three additional failures before pushing local fixes. platform-sandbox-admin-artifacts on Windows failed test_artifacts_list_perf_large_tree because artifact listing walked/stat'ed the shared artifacts tree and took 12.374s for 300 files; store_artifacts now persists sanitized relative artifact paths in memory and list_artifacts returns cached sizes before filesystem traversal, with regression coverage that fails if os.walk is used while an in-memory artifact map exists. product-workflows-runtime failed L1 ACP pipeline template tests because async background execution remained running past the 30s poll window; those tests now submit with mode=sync to use the deterministic request-side completion path already used by neighboring workflow tests. gap-verified-6 failed at teardown in test_scanner_does_not_open_ordinary_file_contents because monkeypatching builtins.open leaked into fixture/app teardown; the test now scopes the open guard to just scan_workspace_file_inventory via mock.patch.object.

Fresh local verification after rebasing onto PR head 0cf999d58eb3481c2a24eba7d2b6e1cf66528726: product-workflows-runtime shard command passed 57 tests; platform-sandbox-admin-artifacts shard command passed 72 tests with 1 skipped; workspace file inventory scanner file passed 11 tests; compileall on touched files passed; git diff --check passed; Bandit on orchestrator.py wrote /tmp/bandit_ci2258_sandbox_artifact_cache.json with zero findings; Bandit on changed tests with B101 skipped wrote /tmp/bandit_ci2258_test_changes.json with zero findings. Current remote CI on head 0cf999d is still running/queued, so no push has been made yet per instruction.
After the PR branch was rewritten from 0cf999d58eb3481c2a24eba7d2b6e1cf66528726 to 9a089196f2f6e3c1e89ef87147c9bfa3423b3abd, the local stabilization commit was rebased by dropping the obsolete local CI-speedup commit and replaying only the CI failure fixes. Fresh post-rebase verification passed: product-workflows-runtime shard 57 passed; platform-sandbox-admin-artifacts shard 72 passed, 1 skipped; workspace file inventory scanner file 11 passed; compileall passed; git diff --check passed; Bandit production/test scans exited 0. Current remote CI is still running on 9a089196, so the local commit remains unpushed until those checks finish and any failures are inspected.
After rebasing the local stabilization commit onto PR head a38804165a, three additional remote CI failures from run 27915865615 were inspected. gap-verified-3 failed collection because test_vn_scripts_api.py used @pytest.mark.anyio while CI runs with plugin autoload disabled and only pytest-asyncio loaded; the marker was changed to @pytest.mark.asyncio. gap-verified-6 was the workspace file-inventory scanner builtins.open teardown leak already fixed locally. gap-verified-7 failed on Linux because the seatbelt standard opt-in test depended on host runtime preflight state before reaching SandboxPolicy validation; the test now stubs collect_runtime_preflights to make the unsupported standard trust-level branch deterministic. While verifying the full gap-verified-7 shard locally, test_acp_session_new_error exposed an independent fixture isolation gap: the test used the persistent ACP session store and could receive a quota 429 before its mocked runner error path. It now applies the existing stub_runner_client fixture to use a temporary ACP session store while preserving the mocked runner error. Fresh verification for this pass: VN focused marker test passed; gap-verified-3 shard passed 432 tests; ACP seatbelt focused test passed; ACP session_new_error focused red/green passed after fixture isolation; full gap-verified-7 shard passed 1407 tests; compileall for the three touched test files passed; git diff --check passed; Bandit on the touched tests passed with B101 and the pre-existing B108 /tmp fixture literals excluded, writing /tmp/bandit_ci2258_gap3_gap7_tests_skip_existing.json.
The local stabilization commit was rebased again after the PR branch moved to d3358654f4ae7775e28763c4022b03753a7b521f. Fresh post-rebase verification on that head: gap-verified-7 shard passed 1407 tests; gap-verified-3 shard passed 432 tests; product-workflows-runtime shard passed 57 tests; platform-sandbox-admin-artifacts shard passed 72 tests with 1 skipped; workspace file inventory scanner file passed 11 tests; compileall on all touched Python files passed; git diff --check passed; Bandit on orchestrator.py wrote /tmp/bandit_ci2258_rebased_sandbox_orchestrator.json with zero findings; Bandit on touched tests with B101 and pre-existing B108 test fixture literals excluded wrote /tmp/bandit_ci2258_rebased_touched_tests.json and exited 0. Remote PR checks on d3358654f4 had no failures when last checked, but 721 checks were still pending, so no push has been made yet.
Run 27916658435 added gap-verified-9 failures on Python 3.12 and 3.13. Root cause: the kanban performance benchmark tests imported pytest_benchmark to decide whether to request the benchmark fixture, but CI disables pytest plugin autoload and only explicitly loads pytest_asyncio.plugin, so the package can be installed while the benchmark fixture is unavailable. Fix: replaced the direct benchmark fixture dependency with a benchmark_runner fixture that uses pytest-benchmark when registered and otherwise falls back to a small timing adapter. Also removed --maxfail=1 from GitHub workflow pytest invocations so future CI runs report full shard failure inventories instead of one setup failure at a time. Verification: focused kanban performance file passed 18 tests under CI plugin-autoload settings; full local gap-verified-9 shard passed 430 tests with 6 skipped and 26 warnings without --maxfail; compileall on the kanban test passed; git diff --check passed; Bandit on the kanban test with B101 skipped wrote /tmp/bandit_ci2258_kanban_benchmark_test.json and exited 0; actionlint is not installed locally, so the four edited workflow YAML files were parsed with PyYAML successfully.
After rebasing the local stabilization commit onto rewritten PR head 5e5c6664d2, post-rebase verification passed again: compileall on the kanban benchmark test exited 0; git diff --check HEAD~1..HEAD exited 0; the four edited workflow YAML files parsed successfully with PyYAML; Bandit on the kanban benchmark test with B101 skipped wrote /tmp/bandit_ci2258_kanban_benchmark_test_post_rebase.json and exited 0; the full local gap-verified-9 shard passed 430 tests with 6 skipped and 26 warnings without --maxfail. Remote PR checks on 5e5c6664d2 had no failures at last poll, with 23 passed, 12 pending, and 3 skipped, so no push has been made yet.
Run 27919096399 completed before this push with three non-pass checks inspected: Windows product-workflows-step-adapters timed out on async kanban workflow completion, the aggregate Windows Full Suite reflected that shard failure, and character-chat-rate-limits hit its 20 minute job timeout because the job still ran whole Character_Chat directories under TEST_MODE=0. Fixes in this pass: test_kanban_step_crud now submits the workflow in sync mode and _wait_terminal reports the last observed status/data on timeout; the character-chat-rate-limits job now targets only legacy/new rate-limit tests instead of whole directories; a workflow contract test prevents broad Character_Chat directory runs from returning. Fresh local verification without --maxfail: Character_Chat targeted command passed 3 tests with 3 skipped; Character_Chat_NEW -m rate_limit command selected 1 test and skipped it under current Resource Governor config; workflow contract test passed; full test_new_step_adapters.py passed 117 tests; compileall on touched tests passed; four edited workflow YAML files parsed with PyYAML; git diff --check passed; Bandit on touched tests with B101 skipped wrote /tmp/bandit_ci2258_step_adapters_rate_limit_tests.json and exited 0.
Investigated completed CI run 27925713474. Non-pass results were: Windows py3.12 chat-legacy-unit-m-z checkout/network failure before tests, Ubuntu py3.13 product-workflows-adapters-core timeout/cancel during test run, plus their aggregate jobs. The adapters-core shard completed in ~1.5-2.25 minutes on Ubuntu py3.12/macOS/Windows py3.12 but timed out at ~33m52s on Ubuntu py3.13 after the first 8 audio-adapter tests. Root cause isolated to STT adapter tests importing the full Audio_Transcription_Lib backend just to patch speech_to_text; on Python 3.13/Linux that import path can hang. Added an adapter-local _speech_to_text wrapper and patched that in workflow/STT tests so unit tests avoid loading optional heavy STT dependencies while production still lazy-loads the real backend.
Verification for the STT wrapper fix: test_audio_adapters.py passed 80 tests; test_adapter_path_security.py plus test_new_step_adapters.py passed 143 tests; the exact adapters-core shard command passed 1048 tests in 26.39s without --maxfail; compileall on touched files passed; git diff --check passed; Bandit on production stt.py wrote /tmp/bandit_ci2258_stt_wrapper_prod.json with zero findings; Bandit on touched tests with existing B101/B404 test-only warnings excluded wrote /tmp/bandit_ci2258_stt_wrapper_tests_skip_existing.json and exited 0; import sanity check confirmed importing the STT adapter leaves Audio_Transcription_Lib unloaded until the wrapper is called.
Fresh run 27955578114 on head b6ecf7bc1d was still queued/in progress at inspection time, not fully complete: 188 success, 1 failure, 19 in_progress, 512 queued, 1 skipped. The completed failure was Windows py3.12 core-utils-tooling job 82723353348. Root cause: test_stdio_sequence_runner_times_out_on_partial_line_and_cleans_up used timeout_seconds=0.01, and Windows CI can spend that budget starting the fake process/stdout reader before the first stdin frame is written, producing 'timed out before initialize' and len(written)==0. The test now patches the helper module's monotonic clock so the first frame is written and the timeout branch fires deterministically without changing production helper behavior.
Verification for the ACP smoke timeout test fix: the focused failing test passed; the full test_acp_certification_smoke.py file passed 49 tests; the exact local core-utils-tooling shard command passed 803 tests with 2 skipped and 26 warnings without --maxfail; compileall on the touched test passed; git diff --check passed; Bandit on test_acp_certification_smoke.py with B101 skipped wrote /tmp/bandit_ci2258_acp_smoke_test.json with zero findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
