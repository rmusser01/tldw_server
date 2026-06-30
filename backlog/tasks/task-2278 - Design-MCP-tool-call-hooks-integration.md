---
id: TASK-2278
title: Design MCP tool-call hooks integration
status: Done
labels:
- mcp
- hooks
- policy
- design
references:
- https://code.claude.com/docs/en/tools-reference#edit-tool-behavior
- https://code.claude.com/docs/en/hooks-guide
- https://code.claude.com/docs/en/hooks
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design/spec for MCP server tool-call hooks inspired by Claude Code hooks. The design must cover before/during/after tool-call lifecycle events, hook handler types, matcher semantics, permission/policy ordering, input/output schemas, redaction, timeouts, failure behavior, and integration with tool-use reporting and profile policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec defines MCP tool-call hook lifecycle events for before, during, after-success, after-failure, and batch-level observation.
- [x] #2 Spec defines hook configuration sources, matcher semantics, handler types, and security constraints.
- [x] #3 Spec defines structured hook input/output schemas, decision merge rules, timeout/failure behavior, and redaction rules.
- [x] #4 Spec covers protocol integration points for prepare_tool_call and execute_prepared_tool_call.
- [x] #5 Spec explains how hooks interact with RBAC, profiles, path scope, approvals, credential grants, and tool-use reporting.
- [x] #6 Spec references Claude Code tools/hooks behavior and records intentional differences for MCP server safety.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created Docs/superpowers/specs/2026-06-07-mcp-tool-call-hooks-design.md. The design uses Claude Code hooks as the reference model while adapting behavior for MCP server policy boundaries: hooks can tighten but not loosen policy, PreToolUse can deny/ask/rewrite, during hooks are observer-only in the first slice, post hooks cannot roll back side effects, command hooks are disabled by default, and all hook inputs/outputs are redacted and bounded.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the MCP tool-call hooks design spec at Docs/superpowers/specs/2026-06-07-mcp-tool-call-hooks-design.md. Validation: git diff --cached --check passed; placeholder scan found no outstanding marker text. Bandit was skipped because this task changed only documentation and Backlog task metadata.
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
