---
id: TASK-463.4
title: Expose Research Workspace service capability remediation UI
status: In Progress
labels:
- research-workspace
- workspace
- capabilities
- frontend
- mcp
- acp
- sandbox
priority: high
parent_task_id: TASK-463
references:
- Docs/superpowers/plans/2026-05-24-research-workspace-service-capability-providers-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surface service-derived MCP, ACP, sandbox, and provider capability details in the Research Workspace trust panel with actionable remediation copy and management links. Preserve the hard /research-workspace route replacement constraints with no /workspace-playground aliases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Trust panel renders service capability labels, reasons, and bounded details in user-facing copy instead of raw reason codes only.
- [ ] #2 Blocked, not-configured, needs-approval, unknown, and external-provider-warning states include concise remediation guidance and management-surface links when a route exists.
- [ ] #3 Panel remains compact and readable for power users while helping first-time NotebookLM migrants understand why tools, agents, sandbox, or grounded answers are unavailable.
- [ ] #4 Frontend tests cover service detail rendering, remediation copy, and management-link routing without adding /workspace-playground aliases.
- [ ] #5 Implementation uses existing Research Workspace component patterns and does not introduce backend changes unless needed by tests.
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
