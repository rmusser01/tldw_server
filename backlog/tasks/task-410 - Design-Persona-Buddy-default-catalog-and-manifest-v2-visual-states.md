---
id: TASK-410
title: Design Persona Buddy default catalog and manifest v2 visual states
status: In Progress
labels:
- persona
- buddy
- visual-packs
- design
priority: medium
ordinal: 347
documentation:
- Docs/superpowers/specs/2026-05-16-buddy-animation-pipeline-design.md
- Docs/superpowers/plans/2026-05-16-buddy-animation-pipeline-catalog-metadata-plan.md
modified_files:
- Docs/superpowers/specs/2026-05-16-buddy-animation-pipeline-design.md
- Docs/superpowers/plans/2026-05-16-buddy-animation-pipeline-catalog-metadata-plan.md
references:
- https://github.com/rmusser01/tldw_server/issues/1787
- https://github.com/rmusser01/tldw_server/issues/1510
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a repo-backed design spec for the Persona Buddy default visual catalog and user-facing asset creation workflow. The design must capture the approved 9-default catalog structure, the neutral-pose-first asset generation pipeline adapted from Puzzle Attack, and a manifest v2 extension for bounded custom state IDs and per-tool animation variants while preserving the existing Persona Visual pack/runtime contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec documents 9 bundled default buddies across basic, intermediate, and intricate complexity tiers.
- [ ] #2 Spec defines the neutral identity anchor workflow and distinguishes static talking/reaction sheets from generated animation frames.
- [ ] #3 Spec defines manifest v2 custom state catalog semantics, bounds, fallback behavior, trigger matching, and compatibility with existing required states.
- [ ] #4 Spec outlines staged implementation slices for backend validation, MCP/runtime trigger handling, editor UX, starter catalog fixtures, and verification.
- [ ] #5 Spec references the current Persona Visual pack contract and Puzzle Attack asset-generation process enough for future implementers to work without prior chat context.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Buddy animation pipeline design spec and GitHub tracker issue #1787 under epic #1510. The spec covers the nine default Buddy catalog tiers, neutral-anchor-first workflow, static sheet versus timed animation separation, state_catalog/authored_triggers custom-state semantics, staged implementation slices, and verification expectations. Validation so far is documentation-focused: git diff --check passed. Bandit is not applicable because this slice only changes Markdown/Backlog tracking.
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
