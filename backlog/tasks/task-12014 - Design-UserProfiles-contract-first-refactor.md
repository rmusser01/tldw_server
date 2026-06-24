---
id: TASK-12014
title: Design UserProfiles contract-first refactor
status: Done
created_date: 2026-06-24 17:55
labels:
- userprofiles
- design
- refactor
priority: medium
updated_date: 2026-06-24 19:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved contract-first staged cleanup design for the UserProfiles module, including API orchestration, clean v2 contracts, planner/executor architecture, migration, testing, and compatibility strategy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec is written under Docs/superpowers/specs with the approved contract-first staged cleanup direction.
- [x] #2 Spec covers architecture boundaries, read/update/bulk data flows, contracts/errors, effects/audit, testing, and migration strategy.
- [x] #3 Spec self-review resolves placeholders, contradictions, ambiguity, and scope issues.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote and self-reviewed Docs/superpowers/specs/2026-06-24-userprofiles-contract-first-refactor-design.md. Placeholder scan found no TODO/TBD markers; scope remains design-only; storage schema changes are explicitly deferred to a separate future plan if needed.
Reopened after review to clarify required audit timing, v2 route default, transactional atomicity scope, bulk scope-before-disclosure, bulk diff defaults, and milestone sizing before implementation planning.
Applied second review clarifications: v2 defaults to explicit versioned routes, required audit is pre-commit/transactional, v2 atomicity covers DB mutations only, bulk performs scope-before-disclosure, bulk diffs default to before-snapshots plus masking, and implementation planning must be milestone-based.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and self-reviewed the UserProfiles contract-first refactor design spec at Docs/superpowers/specs/2026-06-24-userprofiles-contract-first-refactor-design.md. The spec captures the approved staged architecture, clean v2 contract direction, legacy compatibility strategy, planner/executor flow, pre-commit audit and post-commit effect rules, backend transaction notes, bulk security/diff defaults, testing strategy, and migration gates.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Spec file committed with the Backlog task record
- [x] #3 User asked to review the written spec before implementation planning
<!-- DOD:END -->
