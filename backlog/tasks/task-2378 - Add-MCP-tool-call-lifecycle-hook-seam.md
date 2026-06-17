---
id: TASK-2378
title: Add MCP tool-call lifecycle hook seam
status: Done
labels:
- mcp
- mcp-unified
- policy
- hooks
priority: medium
pr_url: https://github.com/rmusser01/tldw_server/pull/2377
documentation:
- Docs/MCP/Unified/Developer_Guide.md
- tldw_Server_API/app/core/MCP_unified/protocol.py
modified_files:
- Docs/superpowers/plans/2026-06-17-mcp-tool-call-hooks-plan.md
- mcp_unified/interfaces/__init__.py
- mcp_unified/interfaces/runtime.py
- tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py
- tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py
- tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
- Docs/MCP/Unified/Developer_Guide.md
- backlog/tasks/task-2378 - Add-MCP-tool-call-lifecycle-hook-seam.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first narrow MCP Unified tool-call hook slice: a typed pre/post hook runtime seam around tool preparation/execution that can deny before execution, observe success/failure after execution, and preserve existing explicit-deny/profile policy precedence. Keep the slice local to MCP Unified runtime tests and docs; do not add a full admin UI or shell-command hook marketplace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pre-tool hooks can observe sanitized tool metadata and return allow/deny/ask-compatible decisions without bypassing existing explicit denies.
- [x] #2 A pre-tool hook denial maps to the existing authorization error shape with structured governance/hook metadata and prevents module execution.
- [x] #3 Post-tool hooks run after successful and failed module execution with bounded metadata and cannot convert a failed tool call into success.
- [x] #4 Focused MCP protocol tests cover allow, deny, success-post, failure-post, and existing policy-deny precedence.
- [x] #5 Verification records focused pytest, import/compile check if needed, Bandit on touched MCP Python files, and git diff checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-17-mcp-tool-call-hooks-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented a host-neutral MCP tool-call hook seam through `ToolHookCallContext`, `ToolHookDecision`, `ToolCallHookManager`, and `NoopToolCallHookManager` in the standalone runtime interface. `MCPProtocol` now resolves an optional runtime hook manager, runs pre-hooks after existing policy/RBAC/path/approval/governance gates, maps pre-hook deny/ask decisions into the existing authorization response shapes, and runs post-hooks after success/failure without letting post-hooks rewrite outcomes.

Regression coverage was added for pre-hook allow/deny/ask behavior, post-hook success/failure observation, and explicit context-policy denial precedence over hooks. Extraction-boundary coverage now asserts the hook types are re-exported through the standalone and tldw compatibility interface packages and that runtime dependencies default to the no-op hook manager.

PR review follow-up addressed Qodo findings by adding missing docstrings/type annotations, detaching hook-visible metadata/tool args/scope payload to prevent nested mutation from affecting prepared execution, adding stack-trace logging with request/tool context for pre/post hook failures, and documenting that pre-hook exceptions intentionally fail closed because pre-hooks are enforcement hooks. Added regression coverage for nested hook-context mutation isolation and pre-hook unavailable fail-closed behavior.

Outside-diff review follow-up verified the requested pre-hook exception test was already present, then expanded it to assert the captured pre-hook context and absence of post-hook execution. Added missing post-hook exception coverage proving a raised post-hook preserves the successful tool result while recording the post-hook context.

Follow-up review verified the repeated hook exception test request is now stale against current code. The `ToolHookDecision.action` annotation was still a plain `str`, so it was narrowed to a public `ToolHookAction` literal alias and re-exported through standalone and compatibility interface packages. Runtime dict coercion now normalizes incoming actions before constructing `ToolHookDecision`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the first MCP Unified tool-call lifecycle hook seam for standalone embedders. The implementation keeps existing permission/profile/path/governance checks authoritative, adds a pre-execution hook decision point for allow/deny/ask, adds post-execution observation for success and failure, documents lifecycle ordering, and preserves no-op behavior when no hook manager is configured.

Verification: initial red run failed on missing hook contract; focused hook/export tests then passed. Qodo follow-up final checks passed: 39 focused pytest tests, compileall on touched MCP Python files, Ruff on touched Python files, Bandit on touched MCP production Python with 0 findings, and `git diff --check`. Outside-diff test follow-up checks passed: hook test file (8 tests), focused MCP regression suite (40 tests), Ruff on touched test file, compileall on touched test file, Bandit on touched test file with pytest `B101` skipped (`results=0`, `errors=[]`), and `git diff --check`. Hook action literal follow-up checks passed: focused MCP regression suite (40 tests), Ruff, compileall, production Bandit (`results=0`, `errors=[]`), and `git diff --check`; test Bandit findings were pre-existing `B108` temp-path warnings in unchanged `test_extraction_contracts.py` fixtures.

PR: https://github.com/rmusser01/tldw_server/pull/2377
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
