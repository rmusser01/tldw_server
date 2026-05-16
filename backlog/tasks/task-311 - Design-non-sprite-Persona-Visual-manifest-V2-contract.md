---
id: TASK-311
title: Design non-sprite Persona Visual manifest V2 contract
status: Done
assignee: []
created_date: '2026-05-13 02:09'
updated_date: '2026-05-13 02:33'
labels:
  - persona
  - buddy
  - visual-packs
  - design
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1623'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/pull/1624'
documentation:
  - >-
    Docs/Design/2026-05-10-persona-visual-renderer-provider-adapter-evaluation.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
  - Docs/Design/2026-05-13-persona-visual-manifest-v2-contract.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design slice for issue #1623 under the Persona/Buddy epic. Define a non-sprite manifest V2 boundary, renderer-specific asset roles, fallback requirements, import-preview validation hooks, capability states, and staged follow-up slices without implementing a non-sprite renderer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design document specifies the non-sprite manifest V2 contract and V1 migration boundary.
- [x] #2 Design covers renderer-specific asset roles, fallback requirements, import-preview hooks, capability states, security/portability rules, and phased implementation slices.
- [x] #3 Design explicitly explains how Live2D and external MCP provider work are unblocked without being implemented.
- [x] #4 Issue #1623 and epic #1510 are updated with the design PR once opened.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created GitHub issue #1623 and draft PR #1624. Added the Manifest V2 contract design doc and linked it from the renderer/provider evaluation and Persona Visual Packs PRD. Verification: git diff --check passed. Tests and Bandit skipped because this is docs-only plus Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Defined the non-sprite Persona Visual Manifest V2 design boundary for future Live2D/external-provider work, covering renderer asset roles, static fallback requirements, import-preview validation hooks, capability/setup states, MCP/provider boundaries, security rules, and staged follow-up slices while preserving V1 sprite_frames behavior.
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
