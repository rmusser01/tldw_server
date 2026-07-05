---
id: TASK-2290
title: Add MCP resource discovery and read tools parity
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 23:31'
labels:
  - mcp
  - resources
  - gateway
  - tools
  - agentic-execution
dependencies: []
references:
  - 'https://code.claude.com/docs/en/tools-reference'
documentation:
  - >-
    Docs/superpowers/specs/2026-07-05-mcp-resource-discovery-read-parity-design.md
  - >-
    Docs/superpowers/plans/2026-07-05-mcp-resource-discovery-read-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement Claude-style ListMcpResourcesTool, ReadMcpResourceTool, and WaitForMcpServers parity for the standalone gateway and tldw_server MCP hub. Cover resource listing/reading across internal and external MCP servers, server readiness waiting, profile grants, pagination, redacted resource metadata, errors for unavailable servers, and hooks/tool-use reporting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Gateway resource list/read remains available for internal resources through the existing JSON-RPC gateway paths.
- [x] #2 External runtime resources from running MCP servers are exposed as redacted, namespaced resource descriptors.
- [x] #3 External resource reads route to the owning running server and return safe unavailable/not-found errors when the server or resource is missing.
- [x] #4 The bounded wait-for-MCP-servers helper reports ready/unavailable servers without starting a new background monitor.
- [x] #5 Focused pytest coverage, Bandit on touched code, and diff verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Started after TASK-12148 was closed. Existing TASK-581 already covers external server lifecycle runtime integration, so this task is scoped to resource parity over the current runtime surfaces.

Verification: PYTHONPATH=apps/mcp-unified/src python -m pytest touched MCP resource/runtime test set -q (300 passed); git diff --check (clean); Bandit on touched implementation files wrote /tmp/bandit_mcp_resource_parity.json with zero findings. Note: the full stdio transport file requires PYTHONPATH=apps/mcp-unified/src in this worktree because the subprocess import boundary test uses the main checkout venv.

Reopened for requested code-review fixes after PR #2653 review: profile grant enforcement, stable JSON-RPC errors, and stronger read payload URI redaction.

Review fixes: rebased onto latest dev; wired configured gateway bootstrap through ExternalRuntimeGatewayRuntime; applied profile effective policy to resource list/read; denied external resource reads without external_server_grants or required credential_grants; pre-filtered resource discovery so ungranted servers are not contacted; recursively redacted upstream URI/token references in read payloads; mapped external resource runtime errors to stable JSON-RPC reason_code data; added websocket resource list/read parity; cleared resource-discovery last_error after successful rediscovery; removed stale TASK-12148 churn from the exact-base diff. Verification: focused red checks failed before fixes and passed after; PYTHONPATH=apps/mcp-unified/src python -m pytest touched MCP runtime/resource suite -q (308 passed); package-boundary regression passed; Bandit touched source scope including websocket adapter wrote /tmp/bandit_mcp_resource_parity_review_fixes.json with zero findings; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented MCP resource discovery/read parity for the standalone gateway external runtime and stdio/websocket transports. External resources are exposed as redacted external:// virtual URIs, reads route back to the owning active server through profile policy, stopped/missing resources return stable runtime errors, and a bounded wait_for_servers helper reports ready/unavailable/unknown server ids without stale resource-discovery health state. Added matching package and app-side transport coverage plus design documentation.
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
