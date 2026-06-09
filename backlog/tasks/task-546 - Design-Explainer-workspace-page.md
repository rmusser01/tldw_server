---
id: TASK-546
title: Design Explainer workspace page
status: Done
labels:
- frontend
- design
- explainer
priority: Medium
references:
- https://breakdowner.exe.xyz/
- TASK-546
documentation:
- Docs/superpowers/specs/2026-06-09-explainer-workspace-design.md
modified_files:
- Docs/superpowers/specs/2026-06-09-explainer-workspace-design.md
- backlog/tasks/task-546 - Design-Explainer-workspace-page.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for a persisted Explainer workspace inspired by Breakdowner, with Goal and Sources tabs, recursive explanation trees, citation-aware grounding modes, and backend persistence from day one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Write a design-only spec for the persisted Explainer workspace, review it for product/architecture risks, update the Backlog task with touched files and verification, then commit the spec and task record.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wrote the Explainer workspace design spec covering the Breakdowner-inspired UX, explicit Goal/Sources tabs, backend-persisted sessions and nodes from day one, Jobs-backed node expansion, configurable grounding modes, citation snapshots, security/privacy constraints, accessibility requirements, and frontend/backend testing strategy. Verification: reviewed the spec locally after the subagent review tool was unavailable under current delegation policy; checked the spec for ASCII-only content and confirmed the expected Backlog reference. Bandit skipped because this task only changes documentation and Backlog task metadata, not Python code.
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
