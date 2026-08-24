---
id: TASK-12125
title: Implement Stage 1 Persona ambient companion
status: In Progress
created_date: 2026-08-24 05:42
dependencies:
- TASK-12123
labels:
- persona
- persona-visuals
- buddy
- implementation
priority: High
documentation:
- Docs/superpowers/specs/2026-08-23-persona-ambient-companion-transparent-video-design.md
- Docs/superpowers/plans/2026-08-23-persona-ambient-companion-stage-1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Stage 1 implementation plan end to end using subagent-driven development and strict TDD: persistence/lifecycle hardening, behavior metadata and reviews, versioned preferences and APIs, authenticated raster asset loading, deterministic idle-only companion engine, adaptive interactions, reduced motion, grounded transient roaming, E2E coverage, documentation, Bandit, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Schema migrations, stores, behavior validation, immutable review fingerprints, and atomic activation satisfy Stage 1 plan Tasks 1-3 in SQLite and PostgreSQL-supported paths.
- [ ] #2 Frontend authenticated raster loading, deterministic companion engine, adaptive controls, reduced motion, transient grounded roaming, and focused Persona behavior satisfy Stage 1 plan Tasks 4-6.
- [ ] #3 Every implementation task records red-green TDD evidence, focused tests, an implementation commit, and independent specification/code-quality review.
- [ ] #4 Focused backend, frontend, E2E, lint/typecheck, Bandit, and diff verification required by Task 7 pass or any environment skip/blocker is explicitly documented.
- [ ] #5 Documentation is updated, the final whole-branch review is resolved, and the branch is ready for the repository integration workflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the seven tasks in Docs/superpowers/plans/2026-08-23-persona-ambient-companion-stage-1-implementation-plan.md sequentially. Use the plan's exact interfaces and commands, with the approved design as binding authority. Stage 2 video implementation is out of scope.
<!-- SECTION:PLAN:END -->

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
