---
id: TASK-319
title: Implement Persona Visual Manifest V2 archive import-preview diagnostics
status: Done
assignee:
  - codex
created_date: '2026-05-13 15:09'
updated_date: '2026-05-14 00:29'
labels:
  - persona
  - buddy
  - visual-packs
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1638'
documentation:
  - Docs/Design/2026-05-13-persona-visual-manifest-v2-contract.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire Manifest V2/non-sprite Persona Visual archive metadata into the backend import-preview path so review diagnostics can be returned before asset rows, pack rows, activation, runtime renderer support, MCP provider exposure, or UI changes exist. Preserve the existing V1 sprite_frames preview and commit behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 V2-style archive preview can report unsupported or blocked renderer diagnostics without the old generic malformed_visual_manifest failure
- [x] #2 Known disabled renderers such as live2d return renderer preview diagnostics, normalized role categories, and non-activation status
- [x] #3 Unknown renderers return safe unsupported diagnostics
- [x] #4 Existing V1 sprite_frames archive preview and validator tests still pass
- [x] #5 Tests cover V2 disabled renderer, unknown renderer, missing required role category, and V1 regression behavior
- [x] #6 Docs or task notes clarify this slice still does not commit assets, activate packs, add runtime renderers, MCP provider behavior, UI changes, or VN/CYOA behavior
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Manifest V2 archive preview routing through the existing renderer import-preview validator for non-V1 manifests. Known disabled renderers such as live2d now return proposed_plan.renderer_import_preview diagnostics and blocked preview status instead of malformed_visual_manifest.

Updated the import-preview worker to persist result.status so renderer-blocked previews remain non-committable through the existing commit status gate.

Verification: python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py tldw_Server_API/tests/Persona/test_persona_visual_import_preview_validators.py tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py -q => 29 passed, 5 warnings.

Verification: git diff --check => passed. Bandit scanned the touched backend files tldw_Server_API/app/core/Persona/visual_portability/preview.py and tldw_Server_API/app/core/Persona/visual_jobs_worker.py with python -m bandit -r tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_jobs_worker.py -f json -o /tmp/bandit_persona_visual_v2_archive_preview.json => 0 findings.

No runtime activation, MCP provider execution, frontend, or VN/CYOA behavior was added in this slice.

Review follow-up started for PR #1642: verifying Qodo, CodeRabbit, and Gemini comments against current branch before minimal fixes.

Review follow-up: fixed Qodo version-coercion finding by accepting integer-like floats and signed ASCII integer strings for archive preview routing, avoiding isdigit Unicode edge cases. Added regression coverage for manifest_version 2.0 and +2 routing to renderer diagnostics.

Review follow-up: updated Backlog verification notes to use reproducible python -m commands and to clarify Bandit scanned preview.py and visual_jobs_worker.py specifically.

Review verification: python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py tldw_Server_API/tests/Persona/test_persona_visual_import_preview_validators.py tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py -q => 31 passed, 5 warnings. git diff --check => passed. python -m bandit -r tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_jobs_worker.py -f json -o /tmp/bandit_persona_visual_v2_archive_preview_review_fix.json => 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented backend-only Persona Visual Manifest V2 archive import-preview diagnostics. V2 renderer metadata now uses the renderer capability import-preview validator, returns structured diagnostics under proposed_plan.renderer_import_preview, marks non-committable renderer previews as blocked, and preserves V1 sprite_frames preview behavior. Added focused previewer and worker regression tests plus documentation for the review-only boundary.

Review follow-up fixed manifest version routing for integer-like floats and signed strings, added regression tests, and normalized task verification wording.
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
