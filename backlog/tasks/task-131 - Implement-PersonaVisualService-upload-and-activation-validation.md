---
id: TASK-131
title: Implement PersonaVisualService upload and activation validation
status: Done
assignee: []
created_date: '2026-05-09 00:19'
updated_date: '2026-05-09 00:22'
labels:
  - persona
  - webui
  - implementation
  - service
dependencies:
  - TASK-130
documentation:
  - Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the PersonaVisualService backend slice for visual pack upload validation, safe per-user asset storage, manifest validation before activation, active-pack transitions, and deactivate-to-derived-buddy fallback. Keep this slice backend/core-only and do not add API endpoints yet.

Implementation plan:
1. Inspect existing core Persona and upload/storage helper patterns.
2. Write failing service tests in tldw_Server_API/tests/Persona/test_persona_visuals_core.py or a focused service test file.
3. Run the service tests to confirm expected missing-service failures.
4. Implement PersonaVisualService in tldw_Server_API/app/core/Persona/visual_service.py and add any pure upload constants/helpers to visuals.py if needed.
5. Re-run focused service tests plus existing visual manifest and DB tests.
6. Run Bandit on touched Persona service/core files.
7. Update this task and commit the slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PersonaVisualService rejects unsupported MIME types and oversized uploads without writing files
- [x] #2 PersonaVisualService writes accepted assets below DatabasePaths.get_user_persona_visuals_dir using safe persona/pack/asset storage keys
- [x] #3 PersonaVisualService activation validates manifests against available pack assets and rejects incomplete activatable manifests
- [x] #4 PersonaVisualService activation archives a previous active pack and activates the requested draft through the persistence layer
- [x] #5 PersonaVisualService deactivate leaves no active pack so Buddy rendering can fall back to derived text/profile rendering
- [x] #6 Focused tests cover upload validation, storage containment, activation validation, active-pack replacement, and deactivation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py -q --tb=short` failed during collection with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Persona.visual_service'`.

Green run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py -q --tb=short` passed 6 tests.

Regression run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -q --tb=short` passed 22 tests.

Whitespace/security checks: `git diff --check` passed; `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Persona/visual_service.py tldw_Server_API/app/core/Persona/visuals.py -f json -o /tmp/bandit_persona_visual_service.json` exited 0 with no results/errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PersonaVisualService for upload MIME and size validation, raster dimension validation, safe per-user asset storage, metadata persistence, manifest validation before activation, active-pack replacement, and deactivation fallback. Added focused service tests for invalid upload handling, storage containment, activation validation, active-pack replacement, and deactivation.
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
