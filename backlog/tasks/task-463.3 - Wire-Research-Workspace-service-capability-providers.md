---
id: TASK-463.3
title: Wire Research Workspace service capability providers
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-26 16:33'
labels:
  - research-workspace
  - workspace
  - capabilities
  - mcp
  - acp
  - sandbox
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
parent_task_id: TASK-463
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Research Workspace capability slice by deriving MCP/tool, ACP/agent, sandbox, and provider/model readiness from backend services instead of hardcoded placeholder states. Preserve fail-closed action gates and keep /research and /research-workspace distinct with no /workspace-playground route aliases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capabilities endpoint consumes a backend service capability projection for MCP, ACP, sandbox, and provider readiness instead of fixed placeholder values.
- [x] #2 MCP readiness is derived from effective MCP Hub policy for the workspace and fails closed when policy resolution is unavailable or blocks tools.
- [x] #3 ACP readiness reflects configured ACP profile/policy and approval requirements when available, and fails closed otherwise.
- [x] #4 Sandbox readiness reflects sandbox feature discovery/runtime availability and fails closed when unavailable or unknown.
- [x] #5 Provider readiness reflects configured chat/RAG provider availability and warns when only external providers are configured.
- [x] #6 Focused tests cover available, needs-approval, blocked/unavailable, and resolver-failure states without adding /workspace-playground aliases.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented service-derived Research Workspace capability projection for MCP Hub policy, ACP agent readiness, sandbox runtime discovery, and configured provider health. Added deterministic endpoint fixtures plus pure projection coverage for available, needs-approval, unavailable/blocked, degraded, resolver-failure, and configured-provider filtering states.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wired /api/v1/workspaces/{workspace_id}/capabilities and /context to backend service capability collection instead of fixed placeholder service states. MCP now reflects effective MCP Hub policy, ACP reflects configured agent readiness and approval requirements, sandbox reflects runtime discovery, and provider readiness is based on configured providers plus health/degraded/external-provider state. Added tests for the capability projection and endpoint merge behavior; verified no active /workspace-playground route alias was added.
<!-- SECTION:FINAL_SUMMARY:END -->

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
