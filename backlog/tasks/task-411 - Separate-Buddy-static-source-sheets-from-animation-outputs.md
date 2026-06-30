---
id: TASK-411
title: Separate Buddy static source sheets from animation outputs
status: Done
labels:
- persona
- visual-starter
- tests
priority: medium
modified_files:
- Docs/superpowers/specs/2026-05-16-buddy-animation-pipeline-design.md
- Docs/superpowers/plans/2026-05-16-buddy-animation-pipeline-catalog-metadata-plan.md
- tldw_Server_API/app/core/Persona/visual_starter_recipe_taxonomy.py
- tldw_Server_API/app/core/Persona/visual_starter_fixtures.py
- tldw_Server_API/app/core/Persona/visual_starter_catalog.py
- tldw_Server_API/app/api/v1/schemas/persona.py
- tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py
- tldw_Server_API/tests/Persona/test_persona_visuals_api.py
- tldw_Server_API/tests/Persona/test_persona_visual_jobs.py
- tldw_Server_API/tests/Persona/test_persona_visual_candidate_provenance.py
- tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py
- Docs/Code_Documentation/Persona_Visual_Packs.md
- backlog/tasks/task-411 - Separate-Buddy-static-source-sheets-from-animation-outputs.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first executable Buddy Animation Pipeline Catalog Metadata unit by adding shared starter recipe taxonomy tests/constants and updating fixture recipe animation outputs so static talking sheets and static reaction sheets remain distinct expected source material, not timed animation outputs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared taxonomy defines expected asset groups, static/source groups, and timed animation output IDs.
- [x] #2 Static talking sheets and static reaction sheets are distinct source/expected groups and are excluded from `production_recipe.animation_outputs`.
- [x] #3 Catalog and API schema validation reject unknown expected groups, non-timed animation outputs, and outputs missing from `expected_asset_groups`.
- [x] #4 API/job/DB provenance tests use valid timed `recipe_output` IDs while retaining `static_sheet` guidance metadata.
- [x] #5 Documentation and issue tracker record the static-source versus timed-output distinction.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Buddy starter catalog metadata slice for static-sheet versus timed-animation semantics. Added shared taxonomy constants, kept static_talking_sheet and static_reaction_sheet as distinct source/expected groups while excluding them from recipe animation_outputs, enforced taxonomy and expected-group subset checks at catalog and API response boundaries, aligned API/job/DB provenance tests to use valid timed recipe_output IDs, and updated docs/plan tracking. Verification: focused pytest suite passed with 152 passed, 5 warnings; py_compile passed for touched production modules; Bandit passed with no findings on touched production modules; git diff --check passed.
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
