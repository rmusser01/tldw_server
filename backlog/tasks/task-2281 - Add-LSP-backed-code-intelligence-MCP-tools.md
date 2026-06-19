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

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
