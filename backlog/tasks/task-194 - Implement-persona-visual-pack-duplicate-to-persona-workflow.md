---
id: TASK-194
title: Implement persona visual pack duplicate-to-persona workflow
status: In Progress
assignee: []
created_date: '2026-05-09 21:42'
updated_date: '2026-05-09 21:50'
labels:
  - persona
  - buddy
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1449'
  - 'https://github.com/rmusser01/tldw_server/issues/1450'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-09-persona-visual-duplicate-to-persona-design.md
  - >-
    Docs/superpowers/plans/2026-05-09-persona-visual-duplicate-to-persona-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1450: duplicate a Persona Visual pack from one same-user persona to a different same-user persona as a draft. Follow the approved plan and keep the work scoped to the Buddy/persona visual-pack system.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backend duplicate workflow copies only manifest-referenced assets and remaps the manifest.
- [ ] #2 Public duplicate endpoint returns PersonaVisualPackResponse and enforces same-user different-persona scope.
- [ ] #3 Frontend exposes duplicate-to-persona draft flow and excludes the current persona from targets.
- [ ] #4 Focused backend and frontend tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 complete: added shared persona visual manifest asset collection/remapping helpers, migrated import commit remapping to the shared helper, and added focused helper tests. Red verification: missing visual_manifest_assets module produced ModuleNotFoundError; first pytest run also confirmed collection failure. Green verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q => 28 passed, 5 warnings.

Task 2 complete: added explicit parent_persona_id support for create_persona_visual_pack so same-user duplicate creation can validate a cross-persona parent pack, exported update_persona_visual_pack_status, and added a DB-focused regression for failed-to-draft duplicate lineage. Red verification: focused test failed with unexpected parent_persona_id keyword. Green verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py::test_db_allows_explicit_cross_persona_parent_for_duplicate_path -q => 1 passed, 5 warnings.

Task 3 complete: implemented PersonaVisualService.duplicate_pack_to_persona with same-persona rejection, target persona ownership validation, manifest-referenced asset preflight, source file/checksum checks, physical asset copying through create_asset_from_upload, manifest remapping, failed-to-draft finalization, and copied-file cleanup on partial failures. Red verification: service tests failed because duplicate_pack_to_persona was missing. Green verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py -q => 14 passed, 5 warnings.

Task 4 complete: added PersonaVisualPackDuplicateRequest, mapped target_persona_not_found to 404 and stale source asset states to 409, and exposed POST /profiles/{persona_id}/visual-packs/{pack_id}/duplicate returning PersonaVisualPackResponse. Red verification: API duplicate tests returned 404 Not Found before route wiring. Green verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q => 29 passed, 5 warnings.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
