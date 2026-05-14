---
id: TASK-344
title: Add bundled Persona Visual starter pack catalog
status: Done
assignee: []
created_date: '2026-05-14 19:49'
updated_date: '2026-05-14 19:54'
labels:
  - persona-visuals
  - backend
  - starter-packs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1694'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1694: add backend-owned bundled Persona Visual starter packs that can be listed and copied into user-owned draft storage for a selected persona. The copied pack must remain inactive until the existing explicit activation flow is used. Keep the path aligned with existing Persona Visual manifest validation, personal library/storage ownership, and import/review semantics. Do not add Live2D runtime support, marketplace/library sharing, external MCP provider execution, or VN/CYOA behavior in this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend can list available bundled sprite_frames starter packs with enough metadata for custom frontends to present a default choice.
- [x] #2 Copying a bundled starter pack for a target persona creates a user-owned draft visual pack and owned asset records/storage references.
- [x] #3 Copied starter packs are not activated automatically and require the existing explicit activation step.
- [x] #4 Bundled starter pack validation rejects malformed manifest or asset data before copy-to-draft succeeds.
- [x] #5 Tests cover list, copy-to-draft, non-activation, validation success, and malformed bundled data rejection.
- [x] #6 Documentation or task notes explain that bundled defaults are copied into user-owned draft storage rather than referenced as global mutable state.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented backend Persona Visual starter catalog endpoints: GET /api/v1/persona/visual-starter-packs and POST /api/v1/persona/visual-starter-packs/{starter_pack_id}/use. The copy path creates an inactive user-owned draft by validating bundled manifest assets, copying image bytes through PersonaVisualService storage, remapping manifest asset IDs, and preserving explicit activation.

Verification: targeted starter tests passed; full test_persona_visuals_api.py passed with 49 tests; test_persona_visual_service.py passed with 14 tests; py_compile passed for touched Python files; git diff --check passed; Bandit on touched Python paths produced zero findings in /tmp/bandit_persona_visual_starter_packs.json.

Known skips/blockers: none. No frontend setup flow or E2E coverage is included here; those are intentionally left to the follow-up first-run setup and happy-path coverage issues (#1695 and #1698).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a backend-owned Persona Visual starter pack catalog and copy-to-draft path for issue #1694. Bundled defaults are listed through /api/v1/persona/visual-starter-packs and copied through /api/v1/persona/visual-starter-packs/{starter_pack_id}/use into user-owned draft visual pack storage. The copied draft remains inactive until explicit activation, uses existing manifest validation/storage semantics, and documents that bundled defaults are copied rather than referenced as global mutable state.
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
