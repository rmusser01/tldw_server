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
Implement the first executable Buddy Animation Pipeline Catalog Metadata unit by adding shared starter recipe taxonomy tests/constants and updating fixture recipe animation outputs so static talking reaction sheets remain expected source material, not timed animation outputs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Buddy starter catalog metadata slice for static-sheet versus timed-animation semantics. Added shared taxonomy constants, removed static_talking_reaction_sheet from recipe animation_outputs while retaining it as expected source material, enforced taxonomy at catalog and API response boundaries, aligned API/job/DB provenance tests to use valid timed recipe_output IDs, and updated docs/plan tracking. Verification: focused pytest suite passed with 150 passed, 5 warnings; py_compile passed for touched production modules; Bandit passed with no findings on touched production modules; git diff --check passed.
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
