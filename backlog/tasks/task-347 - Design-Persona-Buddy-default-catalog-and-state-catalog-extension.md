---
id: TASK-347
title: Design Persona Buddy default catalog and state catalog extension
status: Done
assignee: []
created_date: '2026-05-14 22:41'
updated_date: '2026-05-14 18:30'
labels:
  - persona
  - buddy
  - visual-packs
  - design
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-14-persona-buddy-default-catalog-state-catalog-extension-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a repo-backed design spec for the Persona Buddy default visual catalog and user-facing asset creation workflow. The design must capture the approved 9-default catalog structure, the neutral-pose-first asset generation pipeline adapted from Puzzle Attack, and a V1-compatible state catalog extension for bounded custom state IDs and per-tool animation variants while preserving the existing Persona Visual pack/runtime contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents 9 bundled default buddies across basic, intermediate, and intricate complexity tiers.
- [x] #2 Spec defines the neutral identity anchor workflow and distinguishes static talking/reaction sheets from generated animation frames.
- [x] #3 Spec defines custom state catalog semantics, bounds, fallback behavior, trigger matching, and compatibility with existing required states.
- [x] #4 Spec outlines staged implementation slices for backend validation, MCP/runtime trigger handling, editor UX, starter catalog fixtures, and verification.
- [x] #5 Spec references the current Persona Visual pack contract and Puzzle Attack asset-generation process enough for future implementers to work without prior chat context.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/superpowers/specs/2026-05-14-persona-buddy-default-catalog-state-catalog-extension-design.md. The design keeps `sprite_frames` as the renderer, keeps sprite manifests on `manifest_version: 1`, treats `sprite_sheet` as an asset role, defines a neutral-anchor-first creation pipeline adapted from Puzzle Attack, documents the 9 default starter buddy catalog, and stages state catalog work across backend validation, frontend resolver/editor, MCP/generation jobs, asset pipeline, starter fixtures, docs, and verification.

Verification recorded: `git diff --check` passed for the new docs/task files; ASCII scan passed with no non-ASCII matches. Bandit is not applicable because this slice only changes Markdown documentation.

PR review follow-up: addressed Qodo and Gemini feedback by reframing the work as a `sprite_frames` Manifest V1 state catalog extension instead of overloading repo-wide Manifest V2, changing the example manifest to `manifest_version: 1`, making Puzzle Attack references portable, removing the redundant lowercase rule, making `state_catalog.kind` mandatory, adding fallback-depth cap 8 to Stage 2 success criteria, and requiring state catalog capability advertisement only after validation and Buddy runtime support exist.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Designed the Persona/Buddy default visual catalog and V1-compatible state catalog extension. The spec defines 3 basic, 3 intermediate, and 3 intricate starter buddies, a neutral-pose-first asset pipeline, static talking sheet versus animation-strip boundaries, bounded custom state IDs, exact per-tool trigger matching, MCP/runtime behavior, editor UX, implementation stages, and verification expectations.
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
