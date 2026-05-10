---
id: TASK-218
title: Add reusable visual pack MCP/tool semantics
status: Done
assignee: []
created_date: '2026-05-10 04:16'
labels:
  - persona
  - mcp
  - visual-packs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1449'
  - 'https://github.com/rmusser01/tldw_server/issues/1496'
documentation:
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1496: extend the existing internal persona_visuals MCP/tool surface so user-owned reusable Persona/Buddy visual packs can be discovered and reused through the review-first personal-library semantics. This should build on the merged duplicate-to-persona, personal visual library, import/export conflict-choice, and Persona Garden affordance flows. Keep the scope Persona/Buddy specific; do not add cross-user sharing, marketplace behavior, external renderer adapters, automatic activation, or VN Play asset-pack runtime changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP/tool callers can inspect available reusable visual-pack capabilities without bypassing user/persona ownership checks.
- [x] #2 Reusing a personal library entry through MCP creates a target-persona draft/review artifact and does not mutate active packs.
- [x] #3 Tool schemas and documentation distinguish personal-library reuse from transient visual-state overrides.
- [x] #4 Empty, missing, unauthorized, and stale-source cases fail closed with clear errors.
- [x] #5 Focused backend tests cover successful reuse and the main rejection paths.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- `Docs/superpowers/plans/2026-05-10-persona-mcp-reuse-semantics.md`
<!-- SECTION:PLAN:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Added `persona_visuals.library_items` for read-only reference-backed personal library discovery.
- Added `persona_visuals.use_library_item` for creating inactive target-persona drafts through `PersonaVisualLibraryService`.
- Updated Persona Visual Packs docs and PRD MCP contract copy, including removing stale display-snapshot wording.
- Verification passed:
  - `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py -q`
  - `git diff --check`
  - `python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py -f json -o /tmp/bandit_persona_mcp_reuse.json`
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented reusable Persona Visual pack MCP semantics for #1496. The internal `persona_visuals` module now lists user-scoped personal library entries and can reuse one by creating a reviewable inactive draft for a target persona through existing duplicate/library service semantics. Focused backend tests cover discovery, successful reuse, missing target context, missing items, and unavailable sources; docs now distinguish library reuse from transient runtime overrides.
<!-- SECTION:FINAL_SUMMARY:END -->
