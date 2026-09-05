---
id: TASK-13180
title: Surface Buddy stream outcomes and route pending approval to its live session
status: To Do
assignee: []
created_date: '2026-09-05 15:39'
labels: []
dependencies: []
references:
  - Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md
  - Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After browser transport repairs, a real Buddy greeting receives a tool_plan frame, clears the composer, and leaves the compact shell with no visible plan, reply, or approval-needed state. The shared live-control hook does not consume incoming WebSocket frames. This fails usable feedback and urgent-state expectations in the Buddy interaction PRD; approval execution must remain explicit in the full Live view.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A real Buddy text send produces visible pending, outcome, or error feedback associated with the correct persona and session.
- [ ] #2 Incoming approval-needed state is shown and opens the corresponding full Live session without automatic approval or tool execution.
- [ ] #3 Late or unrelated session frames cannot overwrite current Buddy feedback; targeted regressions and real backend UAT cover the interaction.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
