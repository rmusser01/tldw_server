---
id: TASK-405
title: Add Persona Visual starter asset-production recipes
status: In Progress
labels:
- persona
- visual-packs
- backend
priority: medium
references:
- https://github.com/rmusser01/tldw_server/issues/1760
- https://github.com/rmusser01/tldw_server/issues/1510
- https://github.com/rmusser01/tldw_server/pull/1762
documentation:
- Docs/Code_Documentation/Persona_Visual_Packs.md
- Docs/superpowers/specs/2026-05-14-persona-buddy-default-catalog-state-catalog-extension-design.md
- Docs/superpowers/plans/2026-05-16-persona-visual-production-recipes-plan.md
modified_files:
- tldw_Server_API/app/core/Persona/visual_starter_fixtures.py
- tldw_Server_API/app/core/Persona/visual_starter_catalog.py
- tldw_Server_API/app/api/v1/schemas/persona.py
- tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py
- tldw_Server_API/tests/Persona/test_persona_visuals_api.py
- Docs/Code_Documentation/Persona_Visual_Packs.md
- Docs/superpowers/plans/2026-05-16-persona-visual-production-recipes-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1760 by adding bounded, structured production recipe metadata for each bundled Persona Visual starter scaffold. Recipes should describe the neutral-anchor-first asset production handoff for identity brief, neutral anchor, static talking/reaction sheets, animation outputs, and review checks without bundling final art, expanding runtime renderers, executing MCP providers, or changing copy-to-draft activation semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Starter catalog list/detail responses expose bounded production recipe metadata for each bundled starter.
- [x] #2 Recipes distinguish identity brief, neutral anchor, static talking/reaction sheet, animation outputs, and review checks.
- [x] #3 Basic, intermediate, and intricate starters have tier-appropriate production outputs without claiming final art exists.
- [x] #4 Focused backend/API tests cover the new recipe contract and copy-to-draft behavior remains unchanged.
- [x] #5 Persona Visual docs explain recipes as the authored-asset handoff after scaffold metadata.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-persona-visual-production-recipes-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added immutable `PersonaVisualStarterProductionRecipe` metadata to bundled starter fixtures and exposed it through starter catalog summaries/details. Recipes describe identity brief, neutral anchor guidance, static talking/reaction sheet guidance, expected animation outputs, and review checks for each starter. Service validation keeps recipes immutable, bounded, non-empty, and explicitly requires the neutral identity consistency review check.

Updated Persona starter response schemas and docs so the recipe contract is clear to API clients and future authored-asset generation/review flows. Review follow-up aligned the response schema with the catalog validation bounds for non-empty recipe text, bounded recipe item lists, bounded item text, and the required neutral identity consistency review check. Copy-to-draft behavior is unchanged.

Verification: focused Persona Visual starter/API pytest passed 94 tests; py_compile passed for touched backend/schema modules; Bandit JSON report for touched backend/schema modules had zero results; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Persona Visual starter catalog responses now include structured production recipes for all nine bundled starter scaffolds. The recipes make the neutral-anchor-first authored-asset handoff explicit without bundling final art, executing generation, changing renderer support, or auto-activating copied drafts.
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
