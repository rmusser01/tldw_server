---
id: TASK-2296
title: Add Claude-style tool permission rule parser and evaluator
status: To Do
labels:
- mcp
- policy
- permissions
- profiles
- agentic-execution
references:
- https://code.claude.com/docs/en/tools-reference
- Docs/superpowers/specs/2026-06-07-mcp-tool-call-hooks-design.md
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement Claude-style ToolName(specifier) permission-rule parsing for MCP profiles and agentic execution. Cover command patterns for Bash/Monitor/PowerShell, path patterns for Read/Grep/Glob/LSP/Edit/Write/NotebookEdit, domain rules for WebFetch, skill name matching, agent type matching, MCP external tool names, deny/ask/allow precedence, hooks integration, and migration from existing profile grants.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
