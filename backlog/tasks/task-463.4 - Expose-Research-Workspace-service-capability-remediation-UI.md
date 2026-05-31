---
id: TASK-463.4
title: Expose Research Workspace service capability remediation UI
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-26 17:10'
labels:
  - research-workspace
  - workspace
  - capabilities
  - frontend
  - mcp
  - acp
  - sandbox
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-24-research-workspace-service-capability-providers-plan.md
parent_task_id: TASK-463
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surface service-derived MCP, ACP, sandbox, and provider capability details in the Research Workspace composer with actionable remediation copy and management links. Preserve the hard /research-workspace route replacement constraints with no /workspace-playground aliases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Readiness remediation renders service capability labels, reasons, and bounded details in user-facing copy instead of raw reason codes only.
- [x] #2 Blocked, not-configured, needs-approval, unknown, and external-provider-warning states include concise remediation guidance and management-surface links when a route exists.
- [x] #3 Panel remains compact and readable for power users while helping first-time NotebookLM migrants understand why tools, agents, sandbox, or grounded answers are unavailable.
- [x] #4 Frontend tests cover service detail rendering, remediation copy, and management-link routing without adding /workspace-playground aliases.
- [x] #5 Implementation uses existing Research Workspace component patterns and does not introduce backend changes unless needed by tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented compact composer-level workspace readiness remediation instead of reintroducing the removed trust chrome. Added user-facing labels and remediation for MCP, ACP, sandbox, provider, external-provider, unknown, blocked, needs-approval, and grounded-answer readiness states. Management links use existing routes only: /mcp-hub, /acp-playground, /admin/runtime-config, /settings/model, and /shared; no /workspace-playground aliases or redirects were added. Verification: focused Vitest suite passed 63 tests across WorkspaceCapabilityRemediation, ChatPane stage 1, ResearchWorkspace stage 3, route metadata, route viewport paths, and workspace API client coverage. git diff --check passed. Full UI TypeScript check with default heap OOMed; retry with 8 GB heap completed and reported unrelated existing errors in CharacterListContent.design-system.test.tsx and sidepanel-flashcards.test.tsx. Bandit skipped because this slice touched frontend TypeScript/tests and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed service-derived Research Workspace capability remediation as a compact composer disclosure, not a resurrected top-level trust bar. The UI maps raw service states and reason codes into user-facing guidance, links to existing management surfaces, suppresses raw reason-code copy, preserves /research-workspace routing constraints, and passes workspace capability data from the page shell into ChatPane.
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
