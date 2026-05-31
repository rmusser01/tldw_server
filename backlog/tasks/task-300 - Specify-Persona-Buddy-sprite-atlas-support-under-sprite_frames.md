---
id: TASK-300
title: Specify Persona Buddy sprite atlas support under sprite_frames
status: Done
assignee: []
created_date: '2026-05-12 14:33'
updated_date: '2026-05-12 14:40'
labels:
  - persona-buddy
  - visual-packs
  - design
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1611'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
documentation:
  - Docs/superpowers/specs/2026-05-12-persona-buddy-sprite-atlas-v1-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved design spec for the Persona/Buddy sprite atlas V1.1 slice. Scope keeps atlas support under renderer_type: sprite_frames, using sprite_sheet assets plus frame-level regions without introducing a new sprite_sheet renderer or manifest version bump.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design documents sprite atlas support under sprite_frames without creating a separate sprite_sheet renderer.
- [x] #2 Design covers backend validation, WebUI Buddy rendering, diagnostics, docs, and focused tests.
- [x] #3 Design preserves explicit activation, fail-soft Buddy fallback, and current renderer capability boundaries.
- [x] #4 Design is reviewed for scope drift before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-12: Wrote Docs/superpowers/specs/2026-05-12-persona-buddy-sprite-atlas-v1-design.md for issue #1611. Scope keeps sprite atlas support under renderer_type: sprite_frames with asset_role: sprite_sheet and frames[].region rectangles. No sprite_sheet renderer, manifest version bump, Live2D, VN/CYOA, image generation, or marketplace behavior is included.

2026-05-12 review: Self-reviewed the spec against the merged PR #1608 renderer capability contract and the user-approved scope decision. The spec preserves the existing renderer registry, fail-soft Buddy fallback, explicit activation, and backend validation boundaries. Subagent review was not run because this turn did not include explicit authorization to spawn a reviewer agent.

2026-05-12 verification: Documentation-only design slice; Bandit is not applicable because no Python/runtime code changed. Ran targeted text review for scope terms and will run git diff --check before commit.

2026-05-12 follow-up design review: Tightened the spec before implementation planning. Clarified that V1.1 is not a manifest version, atlas previews should use preview_frame because preview_asset_id is ambiguous when many frames share one atlas asset, missing dimensions can remain permissive for finite positive regions, and registry renderability should stay coarse so SpriteFrameRenderer can mount and emit unsupported_region diagnostics.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the design spec for Persona/Buddy sprite atlas V1.1 support under the existing sprite_frames renderer. The spec defines atlas packs as sprite_frames manifests with sprite_sheet assets and frame-level regions, preserves PR #1608 renderer capability boundaries, and scopes the eventual implementation to docs, tests, validation coverage, and small fixes only where current atlas behavior is incomplete.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Spec file committed in Docs/superpowers/specs.
- [x] #8 Backlog task updated with design path and verification notes.
- [x] #9 Known non-goals and future work are documented.
<!-- DOD:END -->
