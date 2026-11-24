---
id: TASK-15634
title: Finish the intermediate and intricate Buddy starter artwork
status: In Progress
created_date: 2026-11-24 16:51
labels:
- buddy
- persona
references:
- https://github.com/rmusser01/tldw_server/issues/1803
- https://github.com/rmusser01/tldw_server/issues/1806
- https://github.com/rmusser01/tldw_server/issues/1805
documentation:
- Docs/Design/Buddy_Starter_Artwork.md
- Docs/superpowers/specs/2026-05-16-buddy-animation-pipeline-design.md
updated_date: 2026-11-24 17:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace six bundled Persona Visual placeholder packs with cohesive original artwork and previewable state animations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All six starter IDs ship art-ready transparent assets with distinct required-state animations.
- [x] #2 Custom state IDs, trigger conditions and fallback priorities are preserved and all intricate loops use validated atlas regions.
- [x] #3 Each pack copies as a user-owned inactive draft and can be reviewed and activated explicitly.
- [x] #4 The catalog documents bundled availability and includes source and visual review evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the six identities, runtime states, triggers and fallbacks against the existing catalog.
2. Compile transparent, anchor-aligned loops and keep separate static source poses.
3. Map PNG sequences and atlas regions onto the existing manifests.
4. Test catalog geometry, copy/review/activation and both render paths. Record visual evidence.
ADR required: no
ADR path: N/A
Reason: Reuses existing starter, state, sprite asset and activation contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added two-pose transparent loops for six original tldw-project Buddies, with separate frame PNGs for intermediate packs and region atlases for intricate packs. Added four state fallbacks to the bundled Basic and new packs for explicit activation. Kept custom triggers and fallbacks, credited the new artwork AGPL-3.0-only, and clipped atlas catalog previews to their selected cells.
Verified 94 catalog/API cases, final asset alignment and activation in 19 cases, and 45 UI tests.
ADR required: no. The existing Persona Visual manifest, storage and review contracts govern the work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
