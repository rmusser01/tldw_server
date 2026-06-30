---
id: TASK-129
title: Implement persona visual manifest validation core
status: Done
assignee: []
created_date: '2026-05-09 00:06'
updated_date: '2026-05-09 00:09'
labels:
  - persona
  - webui
  - implementation
dependencies:
  - TASK-126
documentation:
  - Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first code slice from the accepted Persona Visual Packs plan: a pure backend manifest validation module for sprite/frame visual packs. This slice should establish the V1 manifest contract, required state resolution, fallback validation, ordered frames, sprite-sheet region checks, authored trigger validation, and focused unit tests without adding DB/API/frontend behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pure module validates V1 sprite_frames manifests and normalizes shorthand asset_ids into ordered frames
- [x] #2 Activation validation requires idle/listening/thinking/speaking/error to resolve to valid animations or fallback chains
- [x] #3 Validation rejects unknown assets, fallback cycles, invalid frame rates, too many frames, invalid sprite-sheet regions, invalid preview_frame values, and invalid authored triggers
- [x] #4 Focused pytest coverage exercises valid manifests, missing required states, fallback cycles, sprite-sheet regions, preview frames, and authored trigger validation
- [x] #5 Verification commands and Bandit result for touched backend scope are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Work in /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/persona-visual-packs-plan on branch codex/persona-visual-packs-plan.

1. Write failing pytest coverage in tldw_Server_API/tests/Persona/test_persona_visuals_core.py for valid manifests, activation-required states, fallback cycles, ordered frames/sprite-sheet regions, invalid preview frames, and authored trigger validation.
2. Run the focused test to observe the expected import/module failure.
3. Implement tldw_Server_API/app/core/Persona/visuals.py as a pure validation module.
4. Re-run focused tests until passing.
5. Run Bandit on tldw_Server_API/app/core/Persona/visuals.py and record results.
6. Commit this slice with TASK-129 metadata.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification: running `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -v` in the clean worktree initially failed because `tldw_Server_API.app.core.Persona.visuals` did not exist. A direct import confirmed `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Persona.visuals'`.

Implementation: added pure `tldw_Server_API/app/core/Persona/visuals.py` and focused tests in `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`. The validator normalizes `asset_ids` to ordered `frames`, validates required state resolution, fallback cycles, frame rates, max frame count, sprite-sheet regions, preview frames, unknown assets, and authored triggers.

Verification: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -q --tb=short` passed with 9 passed and 5 warnings. `git diff --check` passed. Bandit command `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Persona/visuals.py -f json -o /tmp/bandit_persona_visuals_core.json` completed with no errors and no results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first Persona Visual Packs code slice: a pure manifest validator for V1 sprite/frame packs plus focused unit tests. The validator enforces renderer/version shape, required state resolution, fallback-cycle rejection, ordered frame normalization, unknown-asset rejection, frame-rate/frame-count bounds, sprite-sheet region bounds against known asset dimensions, preview_frame validation, and authored trigger validation. Focused pytest, diff check, and Bandit all passed.
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
