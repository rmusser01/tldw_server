---
id: TASK-356
title: Implement Persona Visual state catalog backend validation
status: Done
assignee: []
created_date: '2026-05-15 02:15'
updated_date: '2026-05-15 02:29'
labels:
  - persona
  - buddy
  - visual-packs
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1713'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-14-persona-buddy-default-catalog-state-catalog-extension-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend validation slice from TASK-347 Stage 2. Persona Visual sprite packs must remain manifest_version: 1, while optionally supporting a V1-compatible state_catalog extension for declared custom state IDs. Validation should accept declared custom states in states, fallbacks, and authored_triggers, reject unknown or unsafe custom-state references, preserve required built-in activation behavior, and avoid changing non-sprite Manifest V2 semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plain Manifest V1 sprite_frames manifests continue to validate without state_catalog.
- [x] #2 Declared state_catalog custom states can be referenced from states, fallbacks, and authored_triggers while required built-in activation behavior remains unchanged.
- [x] #3 Unknown custom states, reserved built-in redeclarations, unsafe custom IDs, missing/invalid kind, excessive custom state count, excessive trigger count, fallback depth over 8, and fallback cycles are rejected.
- [x] #4 Focused Persona Visual backend tests cover the new validation behavior and existing renderer manifest-version semantics remain unchanged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented optional state_catalog validation in Persona Visual sprite frame manifests with declared custom states, reserved built-in protection, safe identifier/kind/label/tag/description bounds, trigger count cap, tool_name trigger support, and fallback depth/cycle validation.

Verification: pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -q; pytest focused Persona visual import/manifest/service/starter suite -q; git diff --check; bandit -r tldw_Server_API/app/core/Persona/visuals.py -f json.

Draft PR: https://github.com/rmusser01/tldw_server/pull/1713
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added backend validation for V1 sprite_frames state_catalog custom states while preserving required built-in activation behavior and renderer manifest-version semantics. Added focused Persona Visual tests for custom state references, invalid catalog entries, bounds, trigger caps, fallback depth, and fallback cycles.
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
