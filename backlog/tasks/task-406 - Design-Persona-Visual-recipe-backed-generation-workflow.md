---
id: TASK-406
title: Design Persona Visual recipe-backed generation workflow
status: Done
labels:
- persona
- visual-packs
- design
priority: medium
references:
- https://github.com/rmusser01/tldw_server/issues/1765
- https://github.com/rmusser01/tldw_server/issues/1510
- https://github.com/rmusser01/tldw_server/pull/1767
documentation:
- Docs/superpowers/specs/2026-05-16-persona-visual-recipe-generation-design.md
modified_files:
- Docs/superpowers/specs/2026-05-16-persona-visual-recipe-generation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a backend-only design for connecting Persona Visual starter production_recipe metadata to the existing persona_visual_generate_candidate Jobs and generated-candidate review flow. Scope excludes WebUI, final art generation, automatic activation, renderer expansion, MCP provider execution, marketplace behavior, and VN/CYOA behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design documents recipe-output selection against starter production_recipe metadata.
- [x] #2 Design reuses existing Persona Visual generation Jobs and candidate review storage instead of introducing a parallel job system.
- [x] #3 Design defines backend validation, idempotency, trace-safety, failure semantics, and future implementation slices.
- [x] #4 Design explicitly excludes WebUI and runtime activation behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-05-16-persona-visual-recipe-generation-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/superpowers/specs/2026-05-16-persona-visual-recipe-generation-design.md`.
- Addressed PR review feedback by replacing the non-existent `talking_static_sheet` example with the fixture-backed `static_talking_reaction_sheet` key, clarifying that `prompt` remains required for V1, adding request/correlation identifiers, specifying trace/audit events, and expanding validation tests for missing prompts and overlong composed prompts.
- Verified with `git diff --check`.
- Bandit skipped because this slice only changes documentation and Backlog task metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backend-only design drafted and review feedback addressed for connecting Persona Visual starter production_recipe metadata to existing generation Jobs and generated-candidate review without adding WebUI or runtime activation behavior.
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
