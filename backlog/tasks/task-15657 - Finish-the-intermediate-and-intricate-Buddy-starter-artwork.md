---
id: TASK-15657
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
Renumbered the Buddy starter task after TASK-15634 was assigned to stored media source hashes on dev.
Rebased cleanly on dev at 1d8589b7c4. The rebase includes the upstream authenticated Notes test fix and current Bun lockfile. Frozen install passed with Bun 1.3.2 and Node 20.20.2. The previously failing coverage shard passed 1,246 tests, and all four Notes browser cases passed against a disposable local Server.
Independent review reproduced adjacent atlas pixels in fitted 92 × 128 avatar cards. The fitted SVG now clips to its current frame. A focused regression failed before the fix and passed afterward. All 121 focused UI tests, 85 catalog tests, Ruff, and Bandit passed.
Live Advanced WebUI verification used the repository authentication fixture with a disposable real Server and SQLite. All six catalog images loaded, copied as inactive drafts, played two idle frames at a fixed position, recorded review, and activated. Each exact pack stayed active after reload. Separate 320-pixel Avatar settings verification measured no adjacent-frame pixels in any of the three intricate previews.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
All six bundled starter packs preview, animate, and activate in the live WebUI. PR #3605 is pending GitHub CI and maintainer review.
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
