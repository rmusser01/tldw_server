---
id: TASK-593
title: Implement standalone MCP gateway remote runtime CLI and docs
status: To Do
assignee: []
created_date: ''
updated_date: '2026-06-02 02:20'
labels:
  - mcp-unified
  - standalone-gateway
  - cli
dependencies:
  - TASK-591
  - TASK-592
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add remote CLI commands that call a running standalone gateway for external runtime lifecycle operations, and document the end-to-end standalone admin workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CLI can call a running gateway for runtime list/start/stop/restart/refresh/reconcile/install/update operations.
- [ ] #2 CLI runtime commands require an explicit --gateway-url or MCP_UNIFIED_GATEWAY_URL.
- [ ] #3 Admin auth uses an environment-provided value such as MCP_UNIFIED_GATEWAY_ADMIN_KEY; command-line secret arguments are avoided.
- [ ] #4 CLI preserves the gateway JSON payloads and reason codes.
- [ ] #5 Docs explain local store commands versus remote runtime commands and include a safe credential-grant example.
- [ ] #6 No runtime CLI command starts an upstream process that becomes orphaned when the CLI exits.
- [ ] #7 Gateway URL semantics are explicit: the URL is the mounted gateway base path, such as http://host/mcp, and the client does not auto-add /mcp.
- [ ] #8 Remote CLI preserves JSON reason_code payloads from HTTP 4xx/5xx response bodies.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-02-mcp-gateway-remote-runtime-cli-docs.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
