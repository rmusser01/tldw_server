---
id: TASK-345
title: Add bundled Persona Visual starter pack catalog
status: Done
assignee:
  - '@codex'
created_date: '2026-05-14 19:57'
updated_date: '2026-05-14 20:13'
labels:
  - persona
  - persona-visual
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1694'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/pull/1701'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1694: provide bundled sprite_frames Persona Visual starter packs that the backend can list and copy into a user-owned draft for a selected persona. The copied pack must align with existing Persona Visual import and manifest validation behavior, preserve the personal visual library reference-backed model, and require explicit activation after copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A bundled default sprite_frames pack can be listed by the backend.
- [x] #2 Copying a bundled pack creates a user-owned draft attached to the target persona.
- [x] #3 Copied drafts are not activated automatically and still require the existing explicit activation step.
- [x] #4 Validation covers bundled pack manifests/assets and rejects malformed bundled pack data.
- [x] #5 Docs or task notes explain that bundled defaults are copied into user-owned draft storage rather than referenced as global mutable state.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan artifact: `Docs/superpowers/plans/2026-05-14-persona-visual-starter-catalog-plan.md`

Stage 1: Service Contract And Fixture Validation.
Stage 2: REST Schemas And Endpoints.
Stage 3: Docs, Tracker, And Verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Planning started in isolated worktree `.worktrees/persona-visual-starter-catalog` on branch `codex/persona-visual-starter-catalog` from `origin/dev`. Key implementation direction: bundled starter packs create normal user-owned draft pack and asset rows through existing storage/validation paths rather than global mutable pack references or auto-activation.

Stage 1 green: service tests cover listing the bundled starter fixture, copying into an inactive user-owned draft, preserving an existing active pack, remapping fixture asset keys, and rejecting malformed fixture manifests.

Stage 2 green: API tests cover catalog list/detail, copy-to-draft without activation, unknown starter errors, and cross-user target rejection.

Verification completed: `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q --tb=short --disable-warnings` -> 53 passed; `python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_catalog.py tldw_Server_API/app/core/Persona/visual_starter_fixtures.py` -> passed; Bandit touched Python scope -> 0 findings in `/tmp/bandit_persona_visual_starter_catalog.json`; `git diff --check` -> passed.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1701.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented bundled Persona Visual starter-pack catalog support for issue #1694. Added immutable fixture definitions, a starter catalog service that validates fixture manifests/assets and copies them through existing PersonaVisualService storage into inactive user-owned draft packs, REST schemas/endpoints for list/detail/copy, docs for starter-copy semantics, and focused service/API regression coverage. Verification: 53 focused Persona Visual tests passed, py_compile passed for new modules, Bandit touched Python scope reported 0 findings, and staged whitespace check passed.
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
