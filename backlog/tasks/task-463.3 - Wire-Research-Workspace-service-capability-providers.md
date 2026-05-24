---
id: TASK-463.3
title: Wire Research Workspace service capability providers
status: To Do
labels:
- research-workspace
- workspace
- capabilities
- mcp
- acp
- sandbox
priority: high
parent_task_id: TASK-463
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Research Workspace capability slice by deriving MCP/tool, ACP/agent, sandbox, and provider/model readiness from backend services instead of hardcoded placeholder states. Preserve fail-closed action gates and keep /research and /research-workspace distinct with no /workspace-playground route aliases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capabilities endpoint consumes a backend service capability projection for MCP, ACP, sandbox, and provider readiness instead of fixed placeholder values.
- [ ] #2 MCP readiness is derived from effective MCP Hub policy for the workspace and fails closed when policy resolution is unavailable or blocks tools.
- [ ] #3 ACP readiness reflects configured ACP profile/policy and approval requirements when available, and fails closed otherwise.
- [ ] #4 Sandbox readiness reflects sandbox feature discovery/runtime availability and fails closed when unavailable or unknown.
- [ ] #5 Provider readiness reflects configured chat/RAG provider availability and warns when only external providers are configured.
- [ ] #6 Focused tests cover available, needs-approval, blocked/unavailable, and resolver-failure states without adding /workspace-playground aliases.
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
