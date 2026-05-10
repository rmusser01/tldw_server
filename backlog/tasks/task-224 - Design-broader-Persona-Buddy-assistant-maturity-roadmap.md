---
id: TASK-224
title: Design broader Persona/Buddy assistant maturity roadmap
status: In Progress
assignee: []
created_date: '2026-05-10 06:26'
labels:
  - persona
  - buddy
  - roadmap
  - design
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/635'
  - 'https://github.com/rmusser01/tldw_server/issues/1388'
  - 'https://github.com/rmusser01/tldw_server/issues/1449'
documentation:
  - Docs/Design/Personas.md
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved staged design spec for the broader Persona/Buddy assistant functionality roadmap. The spec should start from the verified current state after the visual-pack epics closed, decompose Persona Chat, Persona Live, Buddy shell, Persona Garden, wake/voice, MCP persona tools, and renderer follow-up work into staged roadmap slices, and identify reliability/UX baseline as the first implementation target.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design spec documents the approved staged roadmap from audit through reliability/UX baseline, Persona Chat quality, runtime/MCP expansion, and visual/renderer future work.
- [ ] #2 Spec identifies the first implementation target as reliability/UX baseline and explains why it precedes Persona Chat quality and runtime expansion.
- [ ] #3 Spec is grounded in current repo surfaces and open/closed GitHub tracker state, including #635, #1388, #1389, #1449, and #1497.
- [ ] #4 Spec lists proposed follow-up issue slices without implementing code.
- [ ] #5 Verification and review status are recorded in the Backlog task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a design-only roadmap spec in `Docs/superpowers/specs/2026-05-10-persona-buddy-assistant-maturity-roadmap-design.md`.
2. Ground the spec in the current Persona/Buddy surfaces, closed Live Personas and visual-pack trackers, open #635 Persona Chat tracker, and #1391 VN/CYOA separation.
3. Preserve the approved staging: current-state audit, reliability/UX baseline first, Persona Chat quality, unified runtime/MCP expansion, and visual/renderer future work.
4. List follow-up issue slices and acceptance gates without changing runtime code.
5. Run documentation verification, record the review limitation for the subagent gate if no explicit subagent authorization is available, then commit the spec and task record.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added `Docs/superpowers/specs/2026-05-10-persona-buddy-assistant-maturity-roadmap-design.md` with the approved staged roadmap.
- First implementation target is Stage 1 reliability/UX baseline, preceded by Stage 0 current-state audit and issue-tree refresh.
- Spec grounds the roadmap in closed Persona/Buddy visual/runtime trackers and the open/stale #635 Persona Chat tracker.
- Verification: `git diff --check` passed.
- Bandit: skipped because this is a docs/backlog-only design change with no touched Python code.
- Spec review status: pending explicit authorization to dispatch a reviewer subagent in this environment.
<!-- SECTION:NOTES:END -->
