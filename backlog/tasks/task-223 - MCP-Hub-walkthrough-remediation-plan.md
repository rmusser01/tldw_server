---
id: TASK-223
title: MCP Hub walkthrough remediation plan
status: In Progress
assignee: []
created_date: '2026-05-10 06:13'
updated_date: '2026-05-10 06:15'
labels:
  - mcp
  - webui
  - ux
  - planning
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-10-mcp-hub-walkthrough-remediation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and track the two-PR remediation program from the toy MCP server walkthrough. The work should make managed external server setup usable without backend restart, make chat MCP payloads honest and consistent, and then polish setup copy, catalog guidance, diagnostics, and setup isolation ergonomics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reviewed design spec exists under Docs/superpowers/specs for the two-PR remediation program.
- [x] #2 The plan separates end-to-end blocker fixes from setup polish and diagnostics.
- [x] #3 The plan includes backend, frontend, chat, readiness, error handling, and verification coverage.
- [x] #4 Follow-up implementation tasks exist for both PR-sized phases.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created two PR-sized child implementation tasks: TASK-223.1 for live discovery/chat/readiness blockers and TASK-223.2 for setup polish/diagnostics. Drafted approved design spec at Docs/superpowers/specs/2026-05-10-mcp-hub-walkthrough-remediation-design.md.
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
