---
id: TASK-296
title: Implement Persona Buddy renderer capability registry
status: In Progress
assignee: []
created_date: '2026-05-12 05:15'
updated_date: '2026-05-12 05:36'
labels:
  - persona
  - buddy
  - visual-packs
  - implementation
dependencies:
  - TASK-293
  - TASK-294
documentation:
  - >-
    Docs/superpowers/specs/2026-05-12-persona-buddy-renderer-capability-registry-design.md
  - >-
    Docs/superpowers/plans/2026-05-12-persona-buddy-renderer-capability-registry-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Persona/Buddy renderer capability registry thin slice. This should add backend renderer capability registry and Persona API reporting, wire backend manifest validation to the registry, add frontend capability service types/helper, route Buddy display and diagnostics through a frontend renderer registry, and add focused backend/frontend tests. Keep sprite_frames as the only enabled V1 renderer, keep draft manifest saves permissive, fail closed at activation/import-preview validation boundaries for unsupported renderers, avoid renderer-level asset-role enforcement, and do not implement Live2D, Persona Chat, VN, CYOA, or external MCP provider behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backend registry exposes only enabled sprite_frames in V1 and validator uses it for renderer/version checks.
- [x] #2 Persona API exposes authenticated visual-renderer capabilities without requiring a persona-specific lookup.
- [x] #3 Draft manifest save remains permissive while activation and import-preview reject unsupported renderers.
- [x] #4 Frontend service types/helper can fetch renderer capabilities and Buddy runtime uses a local renderer registry for renderability.
- [ ] #5 Buddy diagnostics and dock rendering use the same frontend registry and preserve text fallback for unsupported renderers.
- [ ] #6 Focused backend and frontend tests pass, diff check passes, and Bandit on touched backend production scope reports no new findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the reviewed implementation plan in Docs/superpowers/plans/2026-05-12-persona-buddy-renderer-capability-registry-implementation-plan.md using subagent-driven development.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/persona-buddy-renderer-capability-spec

Branch: codex/persona-buddy-renderer-capability-spec

Task 1 backend registry/core validation completed in commit 5a4b44775. Focused core tests passed with 16 passed and 5 warnings. Spec compliance review passed. Code quality review found one whitespace-padded renderer fail-closed regression, fixed in the amended commit, and final code quality review approved with no findings.

Task 2 backend API slice plan: add requested visual-renderer capability API tests, run targeted tests to confirm the missing route/schema failure, add Persona visual renderer capability response schemas, add authenticated GET /api/v1/persona/visual-renderers using the existing Persona feature flag, request-user auth, rate limit, and registry helper, rerun the targeted tests, run git diff --check, and commit the assigned files.

Task 2 backend API verification: red run produced expected 404 on test_list_persona_visual_renderer_capabilities before route/schema implementation; green run passed 2 targeted tests. git diff --check passed. Bandit on tldw_Server_API/app/api/v1/schemas/persona.py and tldw_Server_API/app/api/v1/endpoints/persona.py wrote /tmp/bandit_task296_task2.json with zero findings.

Task 3 validation-boundary regression plan: add API test proving draft manifest PATCH permits unsupported future renderer while activate rejects invalid_manifest; add portability preview test by mutating exported metadata/pack.json to renderer_type live2d and updating metadata/pack.json checksum so validation reaches malformed_visual_manifest; run the requested focused pytest command, git diff --check against 473ca0ceb..HEAD, and commit test-only changes.

Task 3 verification: focused pytest command passed with 9 passed and 5 warnings. Red phase was not possible without reverting existing production behavior because the newly added regression tests passed immediately against current code. git diff --check 473ca0ceb..HEAD passed. No Bandit run needed for Task 3 because only tests and Backlog notes were changed.

Task 4 frontend registry/service plan: add typed capability response models, add getPersonaVisualRendererCapabilities() against /api/v1/persona/visual-renderers with malformed renderers normalized to [], add a local Buddy renderer registry with only sprite_frames mapped to SpriteFrameRenderer, and add focused Vitest coverage for service fetching and renderer resolution/fallback.

Task 4 verification: red Vitest run failed before implementation because personaVisualRenderers.tsx and getPersonaVisualRendererCapabilities() were missing. Green run passed 2 files and 6 tests for personaVisualRenderers and persona-visuals service.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
