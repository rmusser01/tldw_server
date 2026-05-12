---
id: TASK-221
title: Evaluate Persona Visual renderer and provider adapters
status: Done
assignee: []
created_date: '2026-05-10 05:30'
labels:
  - persona
  - visual-packs
  - research
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1449'
  - 'https://github.com/rmusser01/tldw_server/issues/1497'
documentation:
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Research GitHub issue #1497 for the Persona/Buddy visual-pack system. Produce a repo-grounded evaluation of future renderer/provider adapter options, including Live2D-style model support and other local/self-hosted renderer paths, without implementing an adapter. The evaluation must preserve existing user-owned, manifest-backed, review-before-activation semantics and stay separate from VN Play asset-pack runtime work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Evaluation names viable and rejected renderer/provider options with concrete tradeoffs.
- [x] #2 Recommendation preserves user-owned, manifest-backed, review-first Persona/Buddy pack semantics.
- [x] #3 Required manifest/API extension points are identified and scoped for future issues.
- [x] #4 Runtime, licensing, security, portability, asset-size, dependency, and fallback risks are explicit enough to decide whether Live2D or another adapter should be pursued.
- [x] #5 Documentation is committed and linked to GitHub issue #1497.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Ground the evaluation in the current Persona/Buddy visual-pack code and docs, especially manifest validation, Buddy runtime rendering, import/export preview, personal library semantics, and MCP durable/runtime boundaries.
2. Check current external renderer/provider facts from primary sources where licensing or runtime behavior can drift.
3. Add a docs-only renderer/provider adapter evaluation that names viable and rejected options, recommends sequencing, and identifies manifest/API extension points without implementing adapters.
4. Update existing Persona Visual docs only where they need to link the evaluation or correct stale wording that conflicts with current reference-backed/no-snapshot library semantics.
5. Run documentation verification, update this task with evidence, then commit the research slice for a PR linked to #1497.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added `Docs/Design/2026-05-10-persona-visual-renderer-provider-adapter-evaluation.md` with a repo-grounded renderer/provider evaluation for #1497.
- Recommended keeping `sprite_frames` as the only V1 activatable renderer while adding a renderer capability contract before non-sprite manifests.
- Identified Live2D as the best expressive 2D persona candidate, gated by licensing, official SDK packaging, archive validation, dependency, and fallback requirements.
- Deferred Rive/Lottie to later registry-backed renderer paths and rejected Spine plus arbitrary executable HTML/SVG/JS renderer packs for this slice.
- Corrected stale PRD wording so the personal visual library remains reference-backed with no display snapshots.
- Verification: `git diff --check` passed.
- Bandit: skipped because this is a docs/backlog-only research change with no touched Python code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the #1497 research slice by documenting renderer/provider adapter options, recommended sequencing, manifest/API extension points, and runtime/licensing/security/portability risks for future Persona/Buddy visual-pack renderer support.
<!-- SECTION:FINAL_SUMMARY:END -->
