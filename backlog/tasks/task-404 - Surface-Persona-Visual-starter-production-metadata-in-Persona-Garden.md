---
id: TASK-404
title: Surface Persona Visual starter production metadata in Persona Garden
status: Done
labels:
- persona
- visual-packs
- webui
priority: medium
references:
- https://github.com/rmusser01/tldw_server/issues/1752
- https://github.com/rmusser01/tldw_server/pull/1754
- https://github.com/rmusser01/tldw_server/pull/1755
documentation:
- Docs/Code_Documentation/Persona_Visual_Packs.md
modified_files:
- apps/packages/ui/src/types/persona-visuals.ts
- apps/packages/ui/src/components/PersonaGarden/VisualBuddySetupChoiceCard.tsx
- apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx
- apps/packages/ui/src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx
- apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
- apps/packages/ui/src/services/__tests__/persona-visuals.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surface the Persona Visual starter catalog production-readiness metadata added in #1741 / PR #1744 inside Persona Garden setup and bundled-default picker, without changing copy-to-draft or activation behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona Garden starter setup surfaces scaffold/production status and complexity tier for the recommended starter
- [x] #2 Bundled-default picker shows neutral-anchor/expected asset group/animation coverage hints for each starter
- [x] #3 Shared TypeScript starter-pack types include the backend production-readiness metadata fields
- [x] #4 Focused Vitest coverage verifies the new display and preserves copy-to-draft actions
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Persona Garden starter production metadata display for the recommended default card and bundled-default picker. Added shared UI starter metadata types for complexity_tier, production_status, neutral_anchor_required, expected_asset_groups, and animation_coverage_notes. PR review follow-up: moved new readiness labels through the existing translation function and cached formatted picker metadata per starter row. Verification: focused Vitest suite passed 73 tests; git diff --check passed. Full package tsc --noEmit was attempted and failed on pre-existing repo-wide UI type-test debt outside this slice. Bandit is not applicable because this is a frontend-only TypeScript/UI change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Persona Garden now surfaces starter scaffold readiness context from the backend starter catalog so users can see scaffold status, complexity tier, neutral-anchor requirements, expected authored asset groups, and coverage notes before copying a bundled starter as an inactive draft. PR #1755 now rebases cleanly over the merged #1754 implementation and adds the remaining service normalization plus localization review fixes.
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
