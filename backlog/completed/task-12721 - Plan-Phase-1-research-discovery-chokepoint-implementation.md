---
id: TASK-12721
title: Plan Phase 1 research discovery chokepoint implementation
status: Done
labels:
- research
- planning
- discovery
documentation:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
- Docs/superpowers/plans/2026-06-20-research-discovery-chokepoint-phase1-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-20-research-discovery-chokepoint-phase1-plan.md
- backlog/tasks/task-2337 - Plan-Phase-1-research-discovery-chokepoint-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the Phase 1 implementation plan for the research discovery chokepoint: catalog, source router, standalone discovery API, normalized metadata/OA candidates, persisted discovery snapshots, and focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Phase 1 implementation plan is written and saved under Docs/superpowers/plans.
- [x] #2 Plan covers catalog, router, discovery service, standalone API, snapshot persistence, OA candidate sanitization, and tests.
- [x] #3 Plan explicitly excludes standalone ingest, Deep Research migration, compatibility delegation, and fallback site-search rollout from Phase 1.
- [x] #4 Plan decides snapshot storage/schema/default retention.
- [x] #5 Plan passes plan-document-reviewer review or records resolved findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-20-research-discovery-chokepoint-phase1-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Final review approved with no blocking issues. Advisory updates applied: empty source selection defaults to open_research_graph and implementation is tracked separately in TASK-2338.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 1 implementation plan saved at Docs/superpowers/plans/2026-06-20-research-discovery-chokepoint-phase1-plan.md. The plan covers source catalog, provider router/adapters, discovery service, standalone API, owner-scoped sanitized snapshot persistence, OA candidate sanitization including Unpaywall-style resolution, timeout/concurrency/rate-limit seams, disabled/credential-required source behavior, hard all-source failure behavior, fallback disabled-by-default behavior, and focused verification commands. Plan review loop ran four passes: the first three found blocking buildability/spec-alignment issues and the plan was patched each time; the final pass approved with no blocking issues. Advisory follow-ups were incorporated by clarifying empty source selection behavior and creating TASK-2338 as the dedicated implementation task. This is a documentation/planning-only change; Bandit is not applicable because no Python/code paths were modified.
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
