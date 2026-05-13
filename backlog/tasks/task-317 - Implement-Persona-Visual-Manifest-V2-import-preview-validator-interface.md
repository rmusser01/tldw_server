---
id: TASK-317
title: Implement Persona Visual Manifest V2 import-preview validator interface
status: Done
assignee:
  - codex
created_date: '2026-05-13 14:28'
updated_date: '2026-05-13 14:36'
labels:
  - persona
  - buddy
  - visual-packs
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1632'
documentation:
  - Docs/Design/2026-05-13-persona-visual-manifest-v2-contract.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Manifest V2 foundation slice for Persona/Buddy visual packs: a backend renderer-specific import-preview validator interface with fixture-only validation diagnostics. This must not add an archive parser, runtime renderer, activatable non-sprite pack, asset writes, external MCP provider behavior, VN/CYOA behavior, or live response mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A backend renderer import-preview validator interface exists and is covered by focused tests.
- [x] #2 Fixture-only V2 preview validation can report unsupported or blocked renderer diagnostics without committing assets.
- [x] #3 sprite_frames import and activation behavior remains unchanged.
- [x] #4 live2d or other V2 future renderer preview state remains non-activatable and clearly blocked.
- [x] #5 Tests cover known renderer, unknown renderer, missing required fallback/source categories, and fixture diagnostics where applicable.
- [x] #6 Docs explain this is a validation seam only, not archive parsing or runtime renderer support.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan created: Docs/superpowers/plans/2026-05-13-persona-visual-v2-import-preview-validator.md. Plan reviewer subagent was not dispatched because this session only allows subagents when explicitly requested; proceeding with an inline self-review and TDD implementation.

Implemented fixture-only renderer import-preview validator and documented the boundary. Verification: initial red pytest failed on missing visual_import_preview_validators module; focused validator tests passed; persona visual boundary suite passed; git diff --check passed; Bandit JSON reported zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a backend Persona Visual renderer import-preview validator interface for normalized manifest/assets metadata. The slice reports known disabled live2d blockers, unknown renderer diagnostics, required role-category gaps, and supported sprite_frames eligibility without parsing archives, writing assets, activating packs, or changing the existing V1 portability path.
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
