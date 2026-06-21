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
- mcp_unified/pyproject.toml
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_models.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_router.py
- tldw_Server_API/app/core/MCP_unified/tests/test_lsp_backends_fake.py
- Docs/superpowers/plans/2026-06-21-mcp-lsp-code-intelligence-tools-implementation-plan.md
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
Progress so far: Task 1 added LSP public model/config/error contracts, package metadata for mcp_unified.lsp, and targeted model contract tests. Task 2 added deterministic fake LSP backends, capability routing for the first-slice lsp.* tools, and the host-neutral service facade/status surface. Verification on 2026-06-21: test_lsp_models.py, test_lsp_router.py, and test_lsp_backends_fake.py passed with 65 tests; Ruff passed on touched LSP Python files/tests; Bandit on mcp_unified/lsp reported zero findings; git diff --check was clean. Used the repository root virtualenv because this worktree has no local .venv directory.
<!-- SECTION:FINAL_SUMMARY:END -->
