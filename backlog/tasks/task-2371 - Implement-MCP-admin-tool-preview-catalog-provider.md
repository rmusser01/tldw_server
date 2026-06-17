---
id: TASK-2371
title: Implement MCP admin tool preview catalog provider
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-17 05:16'
labels:
  - mcp
  - policy
  - implementation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from the MCP effective permission explain implementation plan: add an unfiltered admin tool catalog provider and wire policy preview to include denied installed tools without changing model-facing tool discovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Admin catalog preview tests cover denied installed tools from the admin catalog.
- [ ] #2 mcp_unified/gateway/tool_discovery.py exposes a public admin catalog entry/helper that does not hide denied tools.
- [ ] #3 GatewayPolicyExplainService.preview_profile_tools uses admin_tool_catalog_provider or installed_tool_catalog fallback and preserves degraded behavior when no catalog is available.
- [ ] #4 Focused policy explain service pytest passes or blockers are documented.
- [ ] #5 Changes are committed separately for Task 2.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 2 implementation under the approved subagent-driven workflow.

Implemented Task 2 preview catalog wiring. Added AdminToolCatalogEntry and list_admin_tool_catalog without model-facing visibility filtering, wired GatewayPolicyExplainService to prefer admin_tool_catalog_provider and fall back to installed_tool_catalog/list_admin_tool_catalog, and added denied-installed admin catalog preview coverage. Verification so far: focused pytest passed (35 passed), Bandit passed with no issues, git diff --check passed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
