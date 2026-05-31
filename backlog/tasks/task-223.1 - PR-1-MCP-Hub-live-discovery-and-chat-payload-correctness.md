---
id: TASK-223.1
title: 'PR 1: MCP Hub live discovery and chat payload correctness'
status: Done
assignee:
  - Codex
created_date: '2026-05-10 06:13'
updated_date: '2026-05-28 20:16'
labels:
  - mcp
  - webui
  - backend
  - chat
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-10-mcp-hub-walkthrough-remediation-design.md
  - Docs/superpowers/plans/2026-05-10-mcp-hub-live-discovery-chat-plan.md
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
PR 1 implementation plan drafted at Docs/superpowers/plans/2026-05-10-mcp-hub-live-discovery-chat-plan.md.

Stages:
1. Backend runtime refresh/reconcile endpoint at POST /api/v1/mcp/hub/external-servers/refresh-discovery, live MCP singleton resolution, manager reconciliation, module registry remapping, external federation validator coverage.
2. Frontend MCP Hub refresh hooks and TanStack Query invalidation after external-server create/update/import/delete plus Tool Catalog refresh.
3. Shared chat tool eligibility resolver for pageAssistModel and normal/comparison raw preview with omission reasons outside the wire payload.
4. Server readiness gate accepts degraded HTTP 206/200 health states while preserving blocking behavior for unreachable/unhealthy states.
5. Focused pytest/Vitest/Bandit/git diff verification before PR packaging.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started implementation planning for PR 1 only. No code changes yet. Plan will resolve spec open questions for endpoint path, external federation module-id fallback, delete/disable reconciliation coverage, and verification commands before implementation.

Plan self-review tightened Stage 1 and Stage 2 details: the refresh endpoint remains POST /api/v1/mcp/hub/external-servers/refresh-discovery but should be placed before nearby parameterized routes; ExternalFederationModule.validate_tool_arguments must validate external.tools.refresh server_id and __confirm_write booleans; frontend invalidation should include the exact ["mcp-health"] query family.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified on origin/dev after PR merge that the PR 1 MCP Hub live discovery and chat payload correctness work is already implemented. Evidence: `python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py -q` passed 48 tests; `bun run test src/services/tldw/__tests__/mcp-hub.test.ts src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx src/utils/__tests__/chat-tools.test.ts src/models/__tests__/pageAssistModel.mcp-tools.test.ts src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx` passed 59 tests.
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
