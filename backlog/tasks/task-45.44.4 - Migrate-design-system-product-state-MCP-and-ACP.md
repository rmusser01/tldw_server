---
id: TASK-45.44.4
title: 'Migrate design-system product state: MCP and ACP'
status: Done
assignee: []
created_date: '2026-05-14 03:19'
updated_date: '2026-05-31 18:31'
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
- [x] #1 The linked GitHub issue owns current count and public status.
- [x] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [x] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review-driven unblock from PR #1683: replace the unbaselined WorkspaceACPHistoryModal AntD Alert product-state error with the shared design-system recovery primitive, then rerun the full design-system verifier. Keep the broader MCP/ACP migration task open for the remaining baseline debt.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #1683 review surfaced a full-verifier blocker in WorkspaceACPHistoryModal. This narrow change is being handled inside the Chat/Playground PR only to restore verifier pass status; it does not complete the broader MCP/ACP migration area.

PR #1683 review unblock completed: after rebasing onto current dev, WorkspaceACPHistoryModal uses the shared design-system Alert primitive for the load-error product state. Full bun run verify:design-system-state exits 0. The broader MCP/ACP baseline migration remains open.

Closeout verification on 2026-05-31: current `apps/packages/ui/scripts/design-system-product-state-baseline.json` contains 82 allowed repo-wide exceptions and 0 MCP/ACP-owned hits for MCP, ACP, AgentTasks, WorkspaceACP, ACPPlayground, MCPHub, `Option/MCP`, and `Option/ACP` labels. The linked GitHub issue #1661 was refreshed with the current public status and zero owned-exception count. No additional implementation child tasks are required for this closeout because the current owned bucket is already zero; the prior implementation path is recorded through PR #1683 and later merged MCP/ACP route-alignment work.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed after confirming the current design-system product-state baseline no longer contains MCP/ACP-owned entries. `bun run verify:design-system-state` passes from `apps/packages/ui` and reports 82 allowed legacy exceptions in other product areas. A targeted baseline parse reports `{ "total": 82, "mcpAcpHits": 0 }` for MCP, ACP, AgentTasks, WorkspaceACP, ACPPlayground, MCPHub, `Option/MCP`, and `Option/ACP` labels.

Updated the linked public GitHub issue #1661 with the verified 2026-05-31 status, current zero owned-exception count, and implementation PR reference (#1683). This closeout changes Backlog metadata only, so Bandit is not applicable. Known remaining work is outside this tracker: the shared-product-state baseline still contains 82 allowed exceptions owned by other queues.
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
