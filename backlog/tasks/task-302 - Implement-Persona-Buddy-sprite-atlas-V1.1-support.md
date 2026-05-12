---
id: TASK-302
title: Implement Persona Buddy sprite atlas V1.1 support
status: In Progress
assignee:
  - codex
created_date: '2026-05-12 14:50'
updated_date: '2026-05-12 14:55'
labels:
  - persona-buddy
  - visual-packs
  - implementation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1611'
documentation:
  - Docs/superpowers/specs/2026-05-12-persona-buddy-sprite-atlas-v1-design.md
  - >-
    Docs/superpowers/plans/2026-05-12-persona-buddy-sprite-atlas-v1-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Persona/Buddy sprite atlas V1.1 hardening slice under sprite_frames. Scope is focused tests, docs, and minimal fixes only if current atlas behavior has gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend atlas manifest validation characterization covers known dimensions, missing dimensions, malformed regions, and required-state activation.
- [ ] #2 WebUI renderer and diagnostics characterization covers atlas preview_frame, coarse registry renderability, and unsupported_region fallback.
- [ ] #3 Persona Visual Packs documentation explains sprite atlas packs under sprite_frames and rejects sprite_sheet as a renderer.
- [ ] #4 Focused backend/frontend/security verification is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 1 backend atlas manifest characterization

1. Inspect existing Persona visual manifest tests and validation code around sprite_frames regions.
2. Add focused regression tests in tldw_Server_API/tests/Persona/test_persona_visuals_core.py for atlas regions without known dimensions during activation and malformed region fields.
3. Run the focused Persona visuals pytest target.
4. Patch tldw_Server_API/app/core/Persona/visuals.py only if the new tests expose a real validation gap, preserving renderer_type sprite_frames and asset_role sprite_sheet.
5. Record verification in TASK-302 and commit only owned changes with message: test: cover persona visual sprite atlas validation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 backend characterization complete. Added atlas activation coverage for frames[].region without known asset dimensions and malformed region validation coverage in tldw_Server_API/tests/Persona/test_persona_visuals_core.py. Focused pytest passed: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -q (23 passed, 5 warnings). visuals.py was not changed because existing validation already satisfies the characterization. Bandit hygiene: raw test-file run reports expected pytest assert B101 findings; B101-skipped test-scope run exited 0 and wrote /tmp/bandit_task302_task1_skip_b101.json.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
