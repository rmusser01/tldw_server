---
id: TASK-463.1
title: Gate Research Workspace grounded chat on queryable sources
status: In Progress
labels:
- research-workspace
- workspace
- chat
- source-status
priority: high
parent_task_id: TASK-463
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
documentation:
- Docs/superpowers/plans/2026-05-24-research-workspace-queryable-chat-guard-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Phase A chat guardrail so grounded/RAG chat uses only queryable selected Research Workspace sources. Users with selected but still-processing or failed sources should see why grounded mode is unavailable while general chat remains usable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 RAG/grounded chat mode is enabled only when at least one selected effective source is queryable/ready.
- [ ] #2 Selected processing or failed sources are visible in the composer context but do not populate RAG media ids.
- [ ] #3 The chat input explains when selected sources are not queryable yet and keeps general chat available.
- [ ] #4 Focused tests cover selected ready, processing, and failed source combinations.
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
