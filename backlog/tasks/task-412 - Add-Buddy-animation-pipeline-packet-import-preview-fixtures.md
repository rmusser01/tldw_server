---
id: TASK-412
title: Add Buddy animation pipeline packet import-preview fixtures
status: Done
labels:
- persona
- buddy
- visual-packs
- backend
- issue-1787
priority: medium
references:
- https://github.com/rmusser01/tldw_server/issues/1787
- https://github.com/rmusser01/tldw_server/issues/1510
modified_files:
- tldw_Server_API/tests/Persona/test_persona_visual_portability.py
- tldw_Server_API/app/core/Persona/visual_portability/preview.py
- Docs/Code_Documentation/Persona_Visual_Packs.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next backend-first slice for GitHub issue #1787: add deterministic Persona/Buddy animation pipeline packet fixtures and focused import-preview coverage for neutral anchors, distinct static talking/reaction sheets, animation strips/atlas outputs, and compiled manifests. Reuse the existing Persona Visual portability/import-preview path and preserve draft/review-before-activation semantics. Scope excludes WebUI changes, final art generation, automatic activation, new renderer support, MCP provider execution/resource download, marketplace/shared-library behavior, and VN/CYOA behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deterministic test fixture represents a Buddy pipeline packet with neutral anchor, static_talking_sheet, static_reaction_sheet, strips/atlas outputs, and a compiled manifest using current Persona Visual contracts.
- [x] #2 Import-preview validation accepts the packet as a previewable Persona Visual archive without activating or mutating packs.
- [x] #3 Preview diagnostics expose enough staged asset/manifest information to distinguish source sheets from timed runtime animation outputs.
- [x] #4 Existing malformed archive/import-preview tests continue to pass without loosening manifest validation.
- [x] #5 Focused pytest, py_compile, git diff --check, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added a deterministic Buddy pipeline portable archive fixture covering
`neutral_anchor`, `static_talking_sheet`, `static_reaction_sheet`,
`animation_strips`, `animation_atlas`, custom state catalog entries, exact tool
triggers, and timed sprite-frame manifest output. Import preview now reports
`manifest_asset_references` and per-asset `asset_group` / `manifest_referenced`
diagnostics so review surfaces can distinguish source sheets from runtime
outputs without committing or activating the archive.

Verification:

- RED focused test failed with `KeyError: 'manifest_asset_references'` before implementation.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q` passed with 14 tests.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/tests/Persona/test_persona_visual_portability.py` passed.
- `git diff --check` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/tests/Persona/test_persona_visual_portability.py -f json -o /tmp/bandit_persona_visual_pipeline.json` passed with 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Buddy animation pipeline packet import-preview coverage and lightweight preview diagnostics for source-vs-runtime assets. The preview remains review-only and uses the existing Persona Visual portable archive path.
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
