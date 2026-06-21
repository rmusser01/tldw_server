---
id: TASK-2281
title: Add LSP-backed code intelligence MCP tools
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-06-19 03:19
labels:
- mcp
- code-intelligence
- lsp
- tools
- agentic-execution
dependencies:
- TASK-2387
references:
- https://code.claude.com/docs/en/tools-reference
- TASK-2387
documentation:
- Docs/superpowers/specs/2026-06-19-mcp-smoke-client-transport-harness-design.md
- Docs/superpowers/plans/2026-06-19-mcp-smoke-client-transport-harness-implementation-plan.md
- Docs/MCP/Unified/Smoke_Client.md
- Docs/superpowers/specs/2026-06-20-mcp-lsp-code-intelligence-tools-design.md
- Docs/superpowers/plans/2026-06-21-mcp-lsp-code-intelligence-tools-implementation-plan.md
modified_files:
- mcp_unified/lsp/__init__.py
- mcp_unified/lsp/config.py
- mcp_unified/lsp/errors.py
- mcp_unified/lsp/models.py
- mcp_unified/lsp/backends.py
- mcp_unified/lsp/router.py
- mcp_unified/lsp/service.py
- mcp_unified/lsp/jsonrpc.py
- mcp_unified/lsp/sessions.py
- mcp_unified/lsp/executables.py
- mcp_unified/lsp/filtering.py
- mcp_unified/lsp/pylsp.py
- mcp_unified/lsp/ruff.py
- mcp_unified/lsp/gateway_runtime.py
- mcp_unified/smoke/scenarios.py
- mcp_unified/smoke/cli.py
- mcp_unified/profiles/presets.py
- mcp_unified/package_metadata.py
- mcp_unified/README.md
- mcp_unified/USER_GUIDE.md
- mcp_unified/pyproject.toml
- Docs/MCP/Unified/Smoke_Client.md
- Docs/superpowers/plans/2026-06-21-mcp-lsp-code-intelligence-tools-implementation-plan.md
- tldw_Server_API/app/core/MCP_unified/modules/implementations/lsp_module.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_models.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_router.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_backends_fake.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_jsonrpc.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_sessions.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_real_backends.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_filtering.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_module_registration.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_gateway_runtime.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_smoke_scenario.py
- tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py
- tldw_Server_API/app/core/MCP_unified/tests/fixtures/fake_lsp_stdio_server.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement LSP-backed MCP tools inspired by Claude Code's LSP tool: definitions, references, hover/type info, document symbols, workspace symbol search, implementations, call hierarchy, and post-edit diagnostics. The design should support optional language-server plugins, workspace/path grants, bounded output, diagnostics after fs.edit/fs.patch/fs.write, and fallback behavior when no server is configured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Before implementing LSP-backed MCP tools, build or at least plan the MCP smoke client harness from TASK-2387 so LSP scenarios can be added on top of the baseline protocol/transport coverage.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Progress so far: Task 1 added LSP public model/config/error contracts, package metadata for mcp_unified.lsp, and targeted model contract tests. Task 2 added deterministic fake LSP backends, capability routing for the first-slice lsp.* tools, and the host-neutral service facade/status surface. Task 3 added the async stdio JSON-RPC client, fake LSP stdio fixture, and per-workspace session manager with idle eviction and exception-safe stop-all. Task 4 added executable resolution, real Ruff and pylsp backend adapters, JSON-RPC notification waiting for diagnostics, env-gated real-backend tests, and a workspace-boundary guard that rejects traversal/absolute path escapes before backend reads or URI generation. Task 5 added the LSP result path-filtering contract: read-only path lists are filtered with filtered_count metadata, while edit-preview/code-action results fail closed on denied or unknown affected paths. Task 6 added the tldw-hosted LSPModule, module-derived read path-scope candidates, conservative post-result filtering, argument validation, server opt-in registration via MCP_ENABLE_LSP_MODULE, and registration tests. Verification on 2026-06-21: focused LSP tests passed with 111 tests and 5 env-gated skips; the explicit TLDW_MCP_LSP_REAL_BACKENDS=1 run passed with 11 tests and 5 skips because ruff/pylsp are not installed on PATH; Ruff passed on touched LSP Python files/tests; Bandit on mcp_unified/lsp plus the hosted LSP module reported zero findings; git diff --check was clean. Used the repository root virtualenv because this worktree has no local .venv directory.
<!-- SECTION:FINAL_SUMMARY:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 9 complete: ran final LSP verification and backlog closeout. Focused LSP suite: 109 passed, 5 skipped. Regression-adjacent MCP suite: initial sandbox run hit local WebSocket bind PermissionError; escalated loopback rerun passed 136 tests. Explicit TLDW_MCP_LSP_REAL_BACKENDS=1 run: 11 passed, 5 skipped. LSP smoke CLI best-effort in-process scenario passed with backend_missing/capability_unavailable best-effort notes where optional backends were unavailable. Bandit report /tmp/bandit_mcp_lsp_code_intelligence.json has zero findings. Known limitations remain Python-only first slice, single workspace root, preview-only edit actions, and file-level diagnostics dependent on optional Ruff/pylsp availability. Also corrected an older formatting regression test to assert the current eval-metadata-enriched dict result contract.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
