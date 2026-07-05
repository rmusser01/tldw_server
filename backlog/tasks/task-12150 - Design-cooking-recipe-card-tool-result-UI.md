---
id: TASK-12150
title: Design cooking recipe card tool-result UI
status: In Progress
labels:
- design
- mcp
- frontend
priority: Medium
modified_files:
- Docs/superpowers/specs/2026-07-05-cooking-recipe-card-tool-result-ui-design.md
- Docs/superpowers/plans/2026-07-05-cooking-recipe-card-tool-result-ui.md
- backlog/tasks/task-12150 - Design-cooking-recipe-card-tool-result-UI.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review a design spec for a read-only MCP tool that emits a typed recipe card UI payload and a shared frontend renderer for WebUI/browser extension chat tool results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan written at Docs/superpowers/plans/2026-07-05-cooking-recipe-card-tool-result-ui.md. Plan slices: backend CookingModule contract tests and implementation; mcp_modules.yaml registration and config tests; frontend recipe-card payload parser; shared RecipeCard component; ToolCallBlock integration; tool-result replay guard; targeted pytest/Vitest/Bandit/typecheck/visual verification. Design spec remains at Docs/superpowers/specs/2026-07-05-cooking-recipe-card-tool-result-ui-design.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec and implementation plan are written and locally reviewed. Independent spec review could not complete after three timed-out subagent attempts; plan subagent review was not retried because the review path had already failed repeatedly and the user instructed continuation. Local checks passed: plan and spec have no TODO/TBD/FIXME markers, and both are ASCII-only. Awaiting execution choice.
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
