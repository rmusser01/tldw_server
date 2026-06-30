---
id: TASK-410
title: Design Persona Buddy default catalog and manifest v2 visual states
status: Done
updated_date: '2026-05-23'
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
Create a repo-backed design spec for the Persona Buddy default visual catalog and user-facing asset creation workflow. The design captured the original approved default-catalog structure, the neutral-pose-first asset generation pipeline adapted from Puzzle Attack, and a manifest v2 extension for bounded custom state IDs and per-tool animation variants while preserving the existing Persona Visual pack/runtime contract. TASK-419 later reconciled the catalog wording to the current six-basic/twelve-starter source of truth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the bundled default Buddy catalog across basic, intermediate, and intricate complexity tiers.
- [x] #2 Spec defines the neutral identity anchor workflow and distinguishes static talking/reaction sheets from generated animation frames.
- [x] #3 Spec defines manifest v2 custom state catalog semantics, bounds, fallback behavior, trigger matching, and compatibility with existing required states.
- [x] #4 Spec outlines staged implementation slices for backend validation, MCP/runtime trigger handling, editor UX, starter catalog fixtures, and verification.
- [x] #5 Spec references the current Persona Visual pack contract and Puzzle Attack asset-generation process enough for future implementers to work without prior chat context.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closeout 2026-05-23: PR #1708 merged the design task into `dev` at `e66f1e75ed1d92c7c7187c118fc780a944807ab8`, and TASK-419 / PR #1818 reconciled the later Codex Buddy catalog expansion so the current source of truth is the six-basic/twelve-starter catalog described in issue #1787. The original "9 bundled default buddies" acceptance wording is superseded by the current catalog correction; this design task is closed because the spec, tracker, and reconciliation task now carry the current contract.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Buddy animation pipeline design spec and GitHub tracker issue #1787 under epic #1510. The spec originally tracked the nine-default Buddy catalog tiers; TASK-419 reconciles that with the later six-basic Codex Buddy expansion and current twelve-ID starter catalog. The still-current design covers the neutral-anchor-first workflow, static sheet versus timed animation separation, state_catalog/authored_triggers custom-state semantics, staged implementation slices, and verification expectations. Validation so far is documentation-focused: git diff --check passed. Bandit is not applicable because this slice only changes Markdown/Backlog tracking.
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
