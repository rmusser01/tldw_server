---
id: TASK-2387
title: Design MCP smoke client transport harness
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-19 03:19'
labels:
  - mcp
  - testing
  - smoke-client
  - gateway
  - agentic-execution
dependencies: []
references:
  - TASK-2281
  - 'https://code.claude.com/docs/en/tools-reference'
documentation:
  - >-
    Docs/superpowers/specs/2026-06-19-mcp-smoke-client-transport-harness-design.md
  - >-
    Docs/superpowers/plans/2026-06-19-mcp-smoke-client-transport-harness-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a simple MCP smoke client that runs the standalone/tldw MCP server through protocol, catalog, tool-call, resources, prompts, profile filtering, denial, and error-contract scenarios before continuing LSP-backed MCP tool implementation. The client must support deterministic in-process testing plus live HTTP, WebSocket, and stdio subprocess transports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec covers in-process, live HTTP, live WebSocket, and stdio subprocess transports.
- [ ] #2 Report contract is bounded and redacted for PR, CI, and Backlog use.
- [ ] #3 Stdio subprocess execution is argv-based and never shell-based.
- [ ] #4 TASK-2281 LSP implementation references the smoke client as its prerequisite harness.
- [ ] #5 Baseline scenarios cover initialize, initialized notification, tools/list, ping, safe tools/call, unknown tool, resources, prompts, JSON-RPC batch, malformed request, profile filtering, and policy denial where fixtures support it.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-19-mcp-smoke-client-transport-harness-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Created design spec Docs/superpowers/specs/2026-06-19-mcp-smoke-client-transport-harness-design.md with deterministic in-process plus live HTTP, WebSocket, and stdio subprocess transport coverage.
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
