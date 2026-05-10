---
id: TASK-223.1
title: 'PR 1: MCP Hub live discovery and chat payload correctness'
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-10 06:13'
updated_date: '2026-05-10 15:08'
labels:
  - mcp
  - webui
  - backend
  - chat
dependencies: []
references:
  - https://github.com/rmusser01/tldw_server/pull/1514
documentation:
  - Docs/superpowers/specs/2026-05-10-mcp-hub-walkthrough-remediation-design.md
  - docs/superpowers/plans/2026-05-10-mcp-hub-live-discovery-chat-plan.md
parent_task_id: TASK-223
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first PR-sized remediation slice from the MCP Hub walkthrough. This phase should remove the backend restart requirement after managed external server setup and make chat MCP selection match the actual request payload.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Managed external server create, update, and import can trigger live external tool discovery refresh without backend restart.
- [x] #2 The existing external.tools.refresh MCP tool validates arguments and no longer fails the write-tool pre-exec validator for valid calls.
- [x] #3 MCP Hub setup and catalog surfaces report refresh success, refresh failure, and runtime unavailable states clearly.
- [x] #4 Chat request construction and raw request preview use the same effective MCP tool decision and expose the reason when tools are omitted.
- [x] #5 The readiness gate allows degraded but usable health into the app while preserving blocking behavior for unreachable or unhealthy API states.
- [x] #6 Focused backend, frontend, and readiness tests cover the changed behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
PR 1 implementation plan drafted at docs/superpowers/plans/2026-05-10-mcp-hub-live-discovery-chat-plan.md.

Stages:
1. Backend runtime refresh/reconcile endpoint at POST /api/v1/mcp/hub/external-servers/refresh-discovery, live MCP singleton resolution, manager reconciliation, module registry remapping, external federation validator coverage.
2. Frontend MCP Hub refresh hooks and TanStack Query invalidation after external-server create/update/import/delete plus Tool Catalog refresh.
3. Shared chat tool eligibility resolver for pageAssistModel and normal/comparison raw preview with omission reasons outside the wire payload.
4. Server readiness gate accepts degraded HTTP 206/200 health states while preserving blocking behavior for unreachable/unhealthy states.
5. Focused pytest/Vitest/Bandit/git diff verification before PR packaging.

2026-05-10 spec-review follow-up scope: fix only Stage 1 backend failures in .worktrees/mcp-hub-pr1-live-discovery. TDD sequence: add focused tests for reconcile adapter-construction isolation and external.servers.list argument rejection through direct/module execution; run focused pytest to observe RED; minimally adjust ExternalServerManager.reconcile_servers and ExternalFederationModule.execute_tool; rerun focused pytest and git diff --check. No frontend files and no commit.

2026-05-10 Stage 1 code-quality follow-up scope: backend/tests only in .worktrees/mcp-hub-pr1-live-discovery. TDD sequence: inspect existing dirty changes; add/update failing tests for replacement preservation, attempted total_servers counting, and refresh endpoint request validation; run the user-specified focused pytest to observe RED; minimally update ExternalServerManager.reconcile_servers and MCP Hub refresh schema/endpoint; rerun focused pytest, git diff --check, and Bandit over touched production backend files; no frontend edits and no commit.

2026-05-10 Stage 2 frontend-only scope in .worktrees/mcp-hub-pr1-live-discovery: add RED Vitest coverage first for external-server create/update/import/delete calling refreshExternalServerDiscovery and preserving mutation success when refresh fails; add Tool Catalog explicit refresh test for collection refresh plus registry reload; implement service helper, ExternalServersTab refresh helper/query invalidation/messages, and ToolCatalogsTab refresh button; rerun focused UI tests, git diff --check, and self-review. Do not touch backend files and do not commit.

2026-05-10 Stage 3 frontend chat-tool payload scope in .worktrees/mcp-hub-pr1-live-discovery: TDD first for shared resolveChatToolRequest omissions/inclusion, pageAssistModel payload/header behavior, and normal/comparison raw-preview debug metadata; implement shared pure resolver in chat-tools.ts; update pageAssistModel, raw preview, and chat debug snapshot threading so omission reasons remain metadata only and wire bodies include tools/tool_choice only when effective; run focused package-local Vitest command and git diff --check; self-review; no MCP Hub tab edits and no commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started implementation planning for PR 1 only. No code changes yet. Plan will resolve spec open questions for endpoint path, external federation module-id fallback, delete/disable reconciliation coverage, and verification commands before implementation.

Plan self-review tightened Stage 1 and Stage 2 details: the refresh endpoint remains POST /api/v1/mcp/hub/external-servers/refresh-discovery but should be placed before nearby parameterized routes; ExternalFederationModule.validate_tool_arguments must validate external.tools.refresh server_id and __confirm_write booleans; frontend invalidation should include the exact ["mcp-health"] query family.

2026-05-10: Continuing option 1 with subagent-driven implementation on branch codex/mcp-hub-pr1-live-discovery in .worktrees/mcp-hub-pr1-live-discovery. Baseline backend focused pytest before edits: 1 failure / 39 passed; failure is existing stale expectation in ExternalServerManager partial failure test expecting raw exception text while manager returns external_server_discovery_failed. Frontend focused baseline could not run from root because bunx resolved temp Vitest without local React/jsdom dependencies; will re-establish correct workspace test command during frontend stage.

Stage 1 backend-only implementation completed in .worktrees/mcp-hub-pr1-live-discovery. Added RED tests first for ExternalServerManager.reconcile_servers add/remove/replace/partial-failure isolation, MCP Hub refresh-discovery endpoint permission/success/runtime-unavailable behavior, and ExternalFederationModule argument validation. Baseline stale manager test was corrected to the stable external_server_discovery_failed reason-code contract after reproducing the old failure. Verification: focused pytest passed 49 tests; git diff --check passed; Bandit over touched backend production files passed with 0 findings and output at /tmp/bandit_mcp_hub_stage1.json.

2026-05-10 spec-review follow-up completed in .worktrees/mcp-hub-pr1-live-discovery without frontend edits. RED focused pytest after adding tests failed as expected: adapter construction failure escaped reconcile; external.servers.list execution did not reject args; ext.* execution did not reject invalid __confirm_write. Implemented per-server adapter-construction error isolation in ExternalServerManager.reconcile_servers and execute-time sanitized validation in ExternalFederationModule.execute_tool. Verification: focused pytest 52 passed, git diff --check passed, Bandit on touched production files passed with 0 findings at /tmp/bandit_mcp_hub_stage1_spec_review.json.

2026-05-10 Stage 1 code-quality review follow-up: added RED tests for preserving old adapters/tools through failed replacement build/discovery, closing failed replacement adapters, attempted total_servers counts for removals/failures, and refresh-discovery body validation/conflict handling. RED evidence: focused pytest failed with 6 failures across total_servers/runtime preservation and endpoint unknown-field/conflict validation; targeted replacement-discovery test failed because old adapter was closed. Implemented staged replacement in ExternalServerManager.reconcile_servers, typed ExternalServerDiscoveryRefreshRequest with extra=forbid and normalized server_id, and query/body conflict handling. Verification: focused pytest passed 59 tests; git diff --check passed; Bandit over touched production files passed with 0 findings at /tmp/bandit_mcp_hub_stage1_code_quality.json. No frontend edits and no commit.

2026-05-10: Stage 1 backend committed as 118004a1d (Add live MCP external discovery refresh). Verification before commit: focused MCP backend pytest 61 passed / 5 warnings; Bandit touched production files 0 findings at /tmp/bandit_mcp_hub_stage1_partial_loader.json; git diff --check clean. Spec review and final code-quality review approved after fixing atomic replacement, total_servers attempted counting, typed refresh request validation, and managed partial-loader preservation of last-known-good runtime.

2026-05-10 Stage 2 frontend-only implementation completed in .worktrees/mcp-hub-pr1-live-discovery. RED evidence: focused UI test run after adding tests failed as expected on missing external discovery refresh calls and missing Tool Catalog refresh action (plus corrected two brittle test selectors before implementation). Implemented refreshExternalServerDiscovery service helper, ExternalServersTab runtime refresh helper with mcp-tools/mcp-tool-catalogs/mcp-tool-modules/mcp-health invalidation after create/update/import/delete, non-blocking persistence success messages for refresh failures, and ToolCatalogsTab explicit Refresh Tools action with registry reload/error surfacing. Verification: bun run test src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx passed 11 tests; git diff --check passed. Frontend-only scope, so Bandit not applicable. No commit.

2026-05-10: Stage 2 frontend committed as 3a2b3bb75 (Refresh MCP Hub tools after server changes). Verification before commit: package-local Vitest for ExternalServersTab and ToolCatalogsTab passed 13 tests; git diff --check clean. Spec review passed; code-quality review requested handling resolved backend ok:false refresh responses, then final review approved after service/UI/tests were updated. Bandit not applicable to frontend-only TypeScript changes.

2026-05-10 Stage 3 frontend chat-tool payload implementation completed in .worktrees/mcp-hub-pr1-live-discovery. RED evidence: focused package-local Vitest failed after adding tests with 19 expected failures covering missing resolveChatToolRequest and missing metadata on pageAssistModel/raw preview. Implemented shared resolveChatToolRequest in chat-tools.ts, used it in pageAssistModel and normal/comparison raw preview, added ChatRequestDebugSnapshot metadata plus ChatTldw/TldwChat metadata threading, and kept omission reasons outside request bodies. Verification: focused Vitest passed 26 tests; git diff --check passed. Bandit not applicable to frontend TypeScript-only changes. No commit.

Stage 3 quality review follow-up: verified the blocking issue that TldwChatService captures chat debug metadata but TldwApiClient/chat-rag immediately overwrite the singleton debug snapshot without metadata on live send/stream paths. Adding a regression that requires debug metadata to be forwarded into lower-level chat completion options before patching the implementation.

Stage 3 quality follow-up RED/GREEN: added service tests requiring chatDebugMetadata to be forwarded to lower-level createChatCompletion/streamChatCompletion options. RED run failed with missing debugMetadata on both calls. Implemented ChatCompletionRequestOptions/ChatCompletionStreamOptions carrying debugMetadata, included metadata in TldwApiClient and duplicate chat-rag captures, and passed debugMetadata from TldwChatService. Verification: package-local Stage 3 Vitest passed 42 tests across chat-tools/pageAssistModel/raw-preview/TldwChatService; git diff --check passed. Focused quality re-review approved the fix before Stage 3 commit.

2026-05-10: Stage 3 committed as 49580bd4a (Align MCP chat tool payload debugging) after focused quality re-review approval. Starting Stage 4 readiness-gate work: degraded HTTP 200/206 API health should allow app entry while unreachable, malformed, or explicitly unhealthy responses preserve existing blocking/retry behavior.

2026-05-10 Stage 4 readiness-gate scope: add focused RED Vitest coverage in ServerReadinessGate.test.tsx for HTTP 206 degraded app entry and explicitly unhealthy retry/blocking behavior; add cheap ok/200 degraded coverage if it fits existing patterns; minimally adjust ServerReadinessGate readiness parsing to accept HTTP 200/206 with degraded/healthy/ok while preserving network/malformed/unhealthy retries; rerun focused readiness Vitest and git diff --check; no commit.

2026-05-10 Stage 4 readiness-gate implementation completed. RED focused Vitest failed as expected on HTTP 206 degraded and HTTP 200 degraded health responses staying in the retrying gate while unhealthy retry coverage passed. Implemented structured readiness parsing in ServerReadinessGate to accept only HTTP 200/206 plus body status degraded/healthy/ok; network failures, malformed JSON, non-200/206 responses, and unhealthy statuses remain non-enterable and retry/timeout through existing behavior. Verification: bun run test:run components/networking/__tests__/ServerReadinessGate.test.tsx passed 6 tests; git diff --check passed. Bandit not applicable to frontend TypeScript-only changes. No commit.

2026-05-10: Stage 4 committed as 6d9162f84 (Allow degraded API readiness entry) after spec and quality reviews approved. Beginning Stage 5 focused final verification: backend MCP pytest, frontend MCP/readiness Vitest, Bandit over touched backend production files, and git diff --check.

2026-05-10 Stage 5 final verification completed. Backend: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py -q -> 61 passed, 5 warnings. UI package: bun run test chat-tools/pageAssistModel/raw-preview/TldwChatService/MCPHub tab tests -> 6 files passed, 55 tests. Frontend readiness: bun run test:run components/networking/__tests__/ServerReadinessGate.test.tsx -> 6 passed. Bandit: 0 findings, output /tmp/bandit_mcp_hub_pr1.json. git diff --check passed.

2026-05-10: Draft PR created for the completed PR 1 slice: https://github.com/rmusser01/tldw_server/pull/1514. PR is draft pending a human-authored Change summary per the repo AI-generated PR policy.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 1 implementation completed for MCP Hub live discovery and chat payload correctness. Added backend live external discovery reconciliation so managed server changes can refresh the running MCP runtime without backend restart, including registry remapping and external federation argument validation. Added MCP Hub frontend refresh/invalidation hooks and Tool Catalog refresh feedback so setup/catalog surfaces report refresh success and degraded refresh outcomes. Shared chat MCP tool eligibility between pageAssistModel and raw preview, preserved omission reasons in debug metadata outside the wire payload, and forwarded metadata through the live chat client capture path. Updated the frontend readiness gate to allow degraded but enterable HTTP 200/206 health responses while preserving retry/blocking for unreachable, malformed, non-enterable, and unhealthy responses. Verification passed: backend focused pytest 61 passed; UI package focused Vitest 55 passed; frontend readiness Vitest 6 passed; Bandit over touched backend production files had 0 findings at /tmp/bandit_mcp_hub_pr1.json; git diff --check passed.
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
