---
id: TASK-132
title: Add persona visual pack API endpoints
status: Done
assignee: []
created_date: '2026-05-09 00:23'
updated_date: '2026-05-09 00:29'
labels:
  - persona
  - webui
  - implementation
  - api
dependencies:
  - TASK-131
documentation:
  - Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add persona-scoped API schemas and endpoints for visual pack draft creation, listing/detail, manifest update, asset upload, activation, deactivation, generated candidate review, and authenticated asset serving. Use PersonaVisualService and existing persona auth/dependency patterns; keep real generation jobs out of this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona schemas define visual pack, asset, candidate, manifest update, and candidate-review request/response models
- [x] #2 Endpoints support create/list/detail/update-manifest/upload/activate/deactivate for a persona's visual packs
- [x] #3 Upload endpoint rejects unsupported MIME types through service error mapping
- [x] #4 Activation endpoint rejects invalid manifests through service error mapping
- [x] #5 User/persona/pack scoping prevents cross-user access
- [x] #6 Focused API tests cover create/list/activate, upload rejection, invalid activation, cross-user denial, candidate accept/reject, and deactivate fallback
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification before implementation: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q --tb=short` failed with six 404 route failures because the visual-pack endpoints were not implemented.

Green focused verification: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q --tb=short` passed, 6 passed.

Regression verification: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Persona/test_persona_visual_service.py tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py -q --tb=short` passed, 28 passed.

Security and hygiene: `git diff --check` passed with no output. `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py -f json -o /tmp/bandit_persona_visual_api.json` passed with zero results and zero errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added persona visual-pack API schemas and persona-scoped FastAPI endpoints for pack create/list/detail, manifest update, asset upload and authenticated content serving, activation/deactivation, and generated candidate review. The route layer reuses PersonaVisualService for upload and activation validation, maps stable service error codes into API responses, and preserves user/persona/pack scoping through the existing persona profile dependency pattern.
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
