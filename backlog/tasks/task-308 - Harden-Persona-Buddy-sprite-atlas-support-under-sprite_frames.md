---
id: TASK-308
title: Harden Persona Buddy sprite atlas support under sprite_frames
status: Done
assignee: []
created_date: '2026-05-13 01:02'
updated_date: '2026-05-13 01:18'
labels:
  - persona
  - buddy
  - visual-packs
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1611'
documentation:
  - Docs/Code_Documentation/Persona_Visual_Packs.md
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
  - >-
    Docs/Design/2026-05-10-persona-visual-renderer-provider-adapter-evaluation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue 1611 as a narrow V1.1 hardening slice. Keep sprite atlas support under renderer_type sprite_frames using sprite_sheet asset roles plus frame-level region rectangles. Do not add a sprite_sheet renderer, manifest version bump, Live2D adapter, external provider behavior, VN/CYOA behavior, image generation, automatic atlas packing, or marketplace/shared-library behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Valid sprite_frames manifests can use a sprite_sheet atlas asset with frames[].region and remain activatable when dimensions are known.
- [x] #2 Backend validation coverage rejects invalid atlas regions when dimensions are known and documents fail-open missing-dimension behavior.
- [x] #3 Buddy WebUI renderer coverage proves atlas-backed frames render cropped through the existing renderer registry path.
- [x] #4 Diagnostics and fallback coverage stays fail-soft for missing assets or unsupported regions.
- [x] #5 Docs include a minimal atlas-backed sprite_frames manifest example and clearly state sprite_sheet is an asset role, not a separate renderer in this slice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan saved at Docs/superpowers/plans/2026-05-13-persona-buddy-sprite-atlas-v11.md. Scope: backend atlas validation contract tests, Buddy renderer and diagnostics tests, and docs examples. No new renderer type, manifest version, Live2D adapter, external provider behavior, VN/CYOA behavior, or automatic atlas generation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan after inspecting current backend validation and Buddy renderer support. Current code already supports frames[].region cropping and known-dimension bounds checks, so this slice should mostly harden contract tests and docs unless focused tests expose a gap.

Completed implementation as a contract/docs hardening slice. Added backend coverage for activatable atlas manifests with known dimensions, retained existing rejection coverage for out-of-bounds regions, and added missing-dimension fail-open coverage for draft/import metadata gaps. Added Buddy renderer registry-path atlas rendering coverage and unsupported-region diagnostics coverage. Added docs example defining sprite_sheet as an asset role under sprite_frames, not a renderer_type.

Verification: git diff --check passed. Backend focused test passed with 20 tests. Frontend focused test passed with 17 tests after running from apps/packages/ui so Vitest used the UI package dependency context. Bandit against Persona visual source files reported zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Persona Buddy sprite atlas support under the existing sprite_frames renderer. Added backend tests for activatable atlas manifests and missing-dimension region behavior, WebUI tests for cropped atlas rendering through the renderer registry, and diagnostics coverage for unsupported regions. Updated docs and the renderer evaluation to show sprite_sheet as an asset role with frames[].region rectangles, not a separate renderer or manifest version. Verification: git diff --check passed, tldw_Server_API/tests/Persona/test_persona_visuals_core.py passed with 20 tests, PersonaBuddy Vitest focus passed with 17 tests, and Bandit reported zero issues across Persona visual source files.
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
