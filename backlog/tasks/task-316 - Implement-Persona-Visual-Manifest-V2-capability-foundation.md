---
id: TASK-316
title: Implement Persona Visual Manifest V2 capability foundation
status: Done
assignee:
  - codex
created_date: '2026-05-13 06:25'
updated_date: '2026-05-13 06:32'
labels:
  - persona
  - buddy
  - visual-packs
  - backend
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1628'
  - 'https://github.com/rmusser01/tldw_server/pull/1630'
documentation:
  - Docs/Design/2026-05-13-persona-visual-manifest-v2-contract.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first Manifest V2 follow-up from the Persona/Buddy epic: additive renderer capability metadata and schema coverage for non-sprite Persona Visual renderers without enabling a new runtime renderer. Preserve the existing visual-renderers API fields, keep sprite_frames as the only activatable Buddy-runtime renderer, and expose future/non-sprite renderer state explicitly as disabled or unsupported rather than as a support claim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing GET /api/v1/persona/visual-renderers response fields remain present and compatible.
- [x] #2 Additive V2 capability metadata is exposed from one backend renderer registry source of truth.
- [x] #3 sprite_frames remains the only activatable Buddy-runtime renderer.
- [x] #4 Future/non-sprite renderer state is explicit and non-activatable rather than presented as supported runtime behavior.
- [x] #5 Backend and frontend type/test coverage reflects the expanded capability contract.
- [x] #6 Docs explain the new fields and unsupported/future renderer boundary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-05-13-persona-visual-v2-capability-foundation.md

Stage 1 extends the backend renderer capability registry and visual-renderers API response with additive Manifest V2 metadata while preserving all existing response fields. It keeps sprite_frames as the only activatable Buddy-runtime renderer and adds explicit disabled/future renderer metadata for non-sprite Manifest V2 work.

Stage 2 aligns shared WebUI TypeScript types with the expanded capability contract without adding runtime UI behavior.

Stage 3 updates Persona Visual docs/PRD language, records verification, and packages the PR for GitHub issue #1628 under epic #1510.

Verification targets: focused pytest for persona visual core/API capability behavior, relevant frontend test/type checks if shared UI changes require them, git diff --check, and Bandit on touched backend files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented additive renderer capability metadata in the backend registry/API, including explicit non-activatable live2d future-state metadata, while preserving existing sprite_frames response fields and runtime support boundaries.

Verification: pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q passed with 59 passed; bun run test -- src/services/__tests__/persona-visuals.test.ts passed with 2 tests; git diff --check passed; Bandit on touched backend files reported 0 findings.

Fresh verification on 2026-05-12: python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q passed with 59 passed; bun run test -- src/services/__tests__/persona-visuals.test.ts passed with 2 passed; git diff --check passed; Bandit JSON results length was 0 with all severity/confidence counts 0.

Known skips/blockers: none. Final frontend verification used the package-local bun run test command so Vitest loaded the workspace config and aliases correctly.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Manifest V2 renderer capability foundation for Persona/Buddy visual packs. The backend registry and /api/v1/persona/visual-renderers response now expose additive renderer metadata for contract versions, asset roles, role-category mapping, limits, setup status/blockers, and fallback/license requirements while preserving existing response fields. sprite_frames remains the only activatable Buddy-runtime renderer, and live2d is represented only as an explicit disabled future-state capability. Shared UI types, service coverage, docs, and the PRD now reflect that boundary.

Draft PR: https://github.com/rmusser01/tldw_server/pull/1630
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
