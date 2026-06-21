---
id: TASK-2396
title: Stabilize workflow media ingest chunking CI test
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-06-21 03:16
labels:
- ci
- tests
- workflows
dependencies: []
priority: high
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
