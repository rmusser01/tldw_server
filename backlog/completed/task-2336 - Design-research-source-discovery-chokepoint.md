---
id: TASK-2336
title: Design research source discovery chokepoint
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-14 03:05'
labels:
  - research
  - design
  - discovery
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a shared research discovery chokepoint seeded from Sourclip-style research sources, with open research graph source routing, OA resolution, and review-gated ingest support for standalone search and Deep Research.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared Research Discovery chokepoint architecture is documented with Media as the sole ingestion owner.
- [x] #2 Phase 1 discovery and Phase 2A PDF handoff boundaries are specified and implemented through linked follow-up tasks.
- [x] #3 Phase 2B HTML handoff remains explicitly separated for a new design and implementation cycle.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Brainstorming/spec phase only. The approved design documents a shared research discovery chokepoint, source catalog, normalized discovery contract, review-gated ingest, Deep Research integration, security/ops guardrails, tests, and rollout phases.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The foundational design was completed and reviewed in Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md. Phase 1 shipped through TASK-2338 and PR #2420. The Media-owned ingestion boundary was corrected through TASK-12082. Phase 2A planning and implementation were completed through TASK-12108 and TASK-12954; PR #2716 merged into dev as 5e5acac6d51761204e75ee1de8d41bb0d1f4eea7. Further HTML handoff work is intentionally transferred to a separate Phase 2B task. Verification for this design-only task consists of the completed spec review loops recorded in the final summary; Bandit is not applicable because this task changed planning documentation only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the foundational Research Discovery chokepoint design and its Phase 1 and Phase 2A boundary definition. The design establishes normalized multi-source discovery, persisted owner-scoped snapshots, review-gated candidate selection, and Media as the only public ingestion owner. Remaining Phase 2B HTML work is not part of this completed task and is handed off separately.
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
