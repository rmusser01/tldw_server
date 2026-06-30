---
id: TASK-470
title: Draft Persona Expressive Avatar Runtime PRD
status: Done
labels:
- persona
- visual
- avatar
- runtime
- prd
- docs
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1916
- https://github.com/rmusser01/tldw_server/issues/1902
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Draft a repo-grounded PRD for Persona Expressive Avatar Runtime covering Persona-owned visual runtime intent, 2D/3D avatar support, visemes/lip-sync boundaries, renderer capability negotiation, fallbacks, accessibility, and separation from Buddy-specific animation/runtime implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRD is grounded in current Persona Visual pack/runtime and Persona PRD contracts.
- [x] #2 Scope, non-goals, Persona-vs-Buddy boundary, runtime contract, risks, staged implementation, and validation plan are documented.
- [x] #3 Issue #1916 and tracker #1902 are referenced.
- [x] #4 Docs-only verification is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current Persona Visual pack/runtime docs and API surfaces. 2. Draft the PRD with scope, non-goals, ownership boundaries, contract shape, staged delivery, risks, and validation. 3. Run docs-only verification and update the task status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created GitHub issue #1916 for Persona Expressive Avatar Runtime and linked it to overarching tracker #1902.
- Inspected the current Persona Visual Packs PRD, Persona Visual Packs code documentation, renderer capability registry, visual manifest validation, Persona endpoint runtime visual override handling, Persona schemas, and Persona module README.
- Drafted `Docs/Product/Persona_Expressive_Avatar_Runtime_PRD.md` as a Persona-owned runtime intent contract that separates Persona visual intent from Buddy-specific animation implementation.
- Verification: `git diff --check` and `git diff --cached --check` pass. Bandit skipped because this slice changes docs/backlog only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Drafted the Persona Expressive Avatar Runtime PRD. The PRD defines renderer-neutral runtime intent, capability-gated speech/viseme support, fallback/accessibility requirements, and clear Persona-vs-Buddy ownership boundaries while preserving existing Persona Visual Pack review and activation semantics.
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
