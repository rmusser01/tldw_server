---
id: TASK-2247
title: Implement MCP CDP browser inspection read tools
status: In Progress
labels:
- mcp-unified
- browser
- cdp
- profiles
priority: medium
references:
- Docs/superpowers/specs/2026-06-03-mcp-default-profile-tooling-presets-design.md
- Docs/superpowers/plans/2026-06-03-mcp-default-profile-tooling-presets-implementation-plan.md
documentation:
- Docs/superpowers/specs/2026-06-04-mcp-cdp-browser-inspection-read-tools-design.md
- Docs/superpowers/plans/2026-06-04-mcp-cdp-browser-inspection-read-tools-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-04-mcp-cdp-browser-inspection-read-tools-design.md
- Docs/superpowers/plans/2026-06-04-mcp-cdp-browser-inspection-read-tools-implementation-plan.md
- backlog/tasks/task-2247 - Implement-MCP-CDP-browser-inspection-read-tools.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first native CDP-backed read-only browser inspection MCP tool slice: status/discovery, DOM snapshot/page-state, screenshot, console, and network read surfaces. Keep browser mutation/interactions out of scope, gate availability on explicit CDP configuration, expose the tools through profile-scoped discovery for browser-capable profiles, and validate package/runtime boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-04-mcp-cdp-browser-inspection-read-tools-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Planning/design review completed before implementation. Tightened loopback endpoint validation requirements to avoid DNS-based trust, clarified explicit-disable precedence over MCP_BROWSER_CDP_URL auto-registration, and chose a dedicated browser CDP server-registration test file instead of modifying filesystem tests.
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
