---
id: TASK-45.44.4
title: 'Migrate design-system product state: MCP and ACP'
status: In Progress
assignee: []
created_date: '2026-05-14 03:19'
updated_date: '2026-05-14 06:35'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1661'
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub product-area migration issue. Closure requires zero current product-state baseline exceptions for the owned path map area and the verification gates recorded in the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The linked GitHub issue owns current count and public status.
- [ ] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [ ] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review-driven unblock from PR #1683: replace the unbaselined WorkspaceACPHistoryModal AntD Alert product-state error with the shared design-system recovery primitive, then rerun the full design-system verifier. Keep the broader MCP/ACP migration task open for the remaining baseline debt.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #1683 review surfaced a full-verifier blocker in WorkspaceACPHistoryModal. This narrow change is being handled inside the Chat/Playground PR only to restore verifier pass status; it does not complete the broader MCP/ACP migration area.

PR #1683 review unblock completed: after rebasing onto current dev, WorkspaceACPHistoryModal uses the shared design-system Alert primitive for the load-error product state. Full bun run verify:design-system-state exits 0. The broader MCP/ACP baseline migration remains open.
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
