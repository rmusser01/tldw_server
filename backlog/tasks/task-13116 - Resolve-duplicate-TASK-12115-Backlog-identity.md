---
id: TASK-13116
title: Resolve duplicate TASK-12115 Backlog identity
status: To Do
created_date: 2026-08-24 17:49
labels:
- backlog
- data-integrity
priority: Low
references:
- backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md
- backlog/tasks/task-12115 - Make-chat-focus-mode-truly-fullscreen.md
- https://github.com/rmusser01/tldw_server/pull/2809
- https://github.com/rmusser01/tldw_server/pull/2578
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two historical Backlog records currently declare TASK-12115: the standalone HTML/JavaScript presentations rollout merged in PR #2809 and the earlier completed chat-focus-mode work merged in PR #2578. Normalize the duplicate through the official Backlog workflow without losing either task's history, references, or completion evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every active, completed, and archived Backlog record has a unique canonical task ID.
- [ ] #2 Historical references for PR #2578 and PR #2809 remain discoverable through explicit legacy notes or redirects.
- [ ] #3 Backlog MCP task_view and task_search resolve both work items deterministically.
- [ ] #4 No acceptance criteria, implementation history, or completion evidence is lost from either task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created during TASK-12115 rollout closeout. This task deliberately defers renumbering/deleting either historical record so the closeout PR remains metadata-only and reviewable.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 The duplicate identity is normalized through the official Backlog workflow.
- [ ] #2 Both affected task records link to each other or to the normalization record.
- [ ] #3 Backlog MCP resolution is verified after the change.
- [ ] #4 The change is committed and reviewed independently of product code.
<!-- DOD:END -->
